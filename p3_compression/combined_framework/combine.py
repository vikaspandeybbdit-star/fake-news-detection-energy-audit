"""RO2's "propose a framework" deliverable (Stage 2 milestone M10 in the original plan
at `C:\\Users\\Administrator\\.claude\\plans\\after-the-survey-start-smooth-pinwheel.md`:
"Extend whichever of quant/prune/distill/EBT worked to all three datasets, run
combine.py + search.py, pick one Pareto-favorable config as the proposed framework.").
Not started until now — every P3 milestone so far (quantize/prune/distill/M9-M13's
exploratory architectures) evaluated ONE technique at a time. This module composes the
two techniques that actually reduced energy on their own (Section 4.2/4.3 of
P3_compression_draft.md: pruning -11-14%, distillation -46-49%) into a single pipeline,
since neither paper nor either technique's own literature tests stacking them together.

Not a novel mechanism — a novel COMBINATION, evaluated honestly. If pruning the
already-distilled student compounds the two techniques' savings (plausible: distillation
shrinks the whole model via knowledge transfer, pruning removes head-level redundancy
*within* whatever model it's applied to — different redundancy axes, not obviously
overlapping), that combination is the closest thing to original novelty in this
project's PhD series to date, worth stating plainly rather than oversold as more.
Equally legitimate if it turns out NOT to compound (diminishing or negative returns) --
report that as honestly as this project has reported every prior negative/mixed result.

Quantization is deliberately excluded from the search space: it runs on CPU only
(Section 3.2), which is not energy-comparable to the GPU-measured prune/distill/baseline
figures this search optimizes against (same scope caveat as the standalone quantize
result). Including it would either force an apples-to-oranges energy axis or require
building a second, CPU-only search entirely — out of scope for this milestone.

DistilBERT pruning note: `prune.py`'s existing head-pruning logic
(`heads_to_prune_by_magnitude`/`prune_layer_heads`) is BERT-specific
(`layer.attention.self.query/key/value` + `layer.attention.output.dense`) and does not
apply to a distilled student, which uses a structurally different module layout
(confirmed by reading `DistilBertSelfAttention` source directly before writing this
file, not assumed): `layer.attention.{q_lin,k_lin,v_lin,out_lin}`, no separate
"output" wrapper submodule -- `out_lin` IS the attention output projection, living
directly on the attention module itself. `prune_distilbert_heads` below is a parallel,
DistilBERT-specific implementation of the same magnitude-based head-pruning idea, not a
reuse of `prune.py`'s BERT-specific functions.

STATUS: written but not yet run on Kaggle (no GPU access from this local machine for a
9-run grid search -- see search.py's docstring for the run plan). Smoke-test on Kaggle
before spending quota on the full grid, per this project's own established discipline
(every prior M-numbered milestone was smoke-tested first).

Usage (single config, for debugging one cell of the grid manually):
    python combine.py --dataset liar --prune_ratio 0.5 --distill
"""

import argparse
import csv
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "p1_baseline"))
from config import BATCH_SIZE, MAX_SEQ_LEN, PRECISION, SEED  # noqa: E402
from profile_energy import run_profiled_inference  # noqa: E402
from train import load_split, compute_metrics  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from quantize import evaluate_accuracy  # noqa: E402
from distill import DistillationTrainer, STUDENT_MODEL, KD_TEMPERATURE, KD_ALPHA  # noqa: E402,F401

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NUM_HEADS = 12
HEAD_DIM = 64


# --- BERT-checkpoint pruning: thin wrapper around prune.py's own (already-verified,
# already-run-on-Kaggle) functions, not a reimplementation --------------------------

def prune_bert_heads(model, prune_ratio: float):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from prune import heads_to_prune_by_magnitude, prune_layer_heads
    for layer in model.bert.encoder.layer:
        heads = heads_to_prune_by_magnitude(layer, prune_ratio)
        prune_layer_heads(layer, heads)
    return model


# --- DistilBERT-checkpoint pruning: new, DistilBERT's own module layout is different
# from BERT's (see module docstring) -------------------------------------------------

def _distilbert_heads_to_prune_by_magnitude(layer, prune_ratio: float):
    weight = layer.attention.out_lin.weight  # [dim, dim], input dim split by head
    scores = []
    for h in range(NUM_HEADS):
        col_slice = weight[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        scores.append((col_slice.abs().sum().item(), h))
    scores.sort()
    n_prune = int(NUM_HEADS * prune_ratio)
    return [h for _, h in scores[:n_prune]]


def _prune_distilbert_layer_heads(layer, heads_to_prune):
    attn = layer.attention
    device = attn.q_lin.weight.device
    keep_heads = [h for h in range(NUM_HEADS) if h not in heads_to_prune]
    keep_dims = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in keep_heads])

    for name in ("q_lin", "k_lin", "v_lin"):
        old = getattr(attn, name)
        new = nn.Linear(old.in_features, len(keep_dims)).to(device)
        new.weight.data = old.weight.data[keep_dims, :].clone()
        new.bias.data = old.bias.data[keep_dims].clone()
        setattr(attn, name, new)

    attn.n_heads = len(keep_heads)

    old_out = attn.out_lin
    new_out = nn.Linear(len(keep_dims), old_out.out_features).to(device)
    new_out.weight.data = old_out.weight.data[:, keep_dims].clone()
    new_out.bias.data = old_out.bias.data.clone()
    attn.out_lin = new_out


def prune_distilbert_heads(model, prune_ratio: float):
    for layer in model.distilbert.transformer.layer:
        heads = _distilbert_heads_to_prune_by_magnitude(layer, prune_ratio)
        _prune_distilbert_layer_heads(layer, heads)
    return model


# --- distillation step: reuses distill.py's own DistillationTrainer verbatim -------

def distill_to_student(dataset: str, device: str, smoke_test: bool = False):
    teacher_dir = os.path.join(P1_RESULTS_DIR, dataset, "best")
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL, num_labels=2)
    if torch.cuda.is_available():
        teacher, student = teacher.to(device), student.to(device)

    def tokenize(batch):
        return student_tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    train_ds = load_split(dataset, "train")
    val_ds = load_split(dataset, "val")
    if smoke_test:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        val_ds = val_ds.select(range(min(8, len(val_ds))))
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    out_dir = os.path.join(RESULTS_DIR, dataset, "combined_distill_ckpt")
    training_args = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, num_train_epochs=1 if smoke_test else 3,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", fp16=(PRECISION == "fp16"),
        seed=SEED, logging_steps=50, report_to="none",
    )
    trainer = DistillationTrainer(teacher_model=teacher, model=student, args=training_args,
                                   train_dataset=train_ds, eval_dataset=val_ds,
                                   compute_metrics=compute_metrics)
    trainer.train()
    return student, student_tokenizer


def recovery_finetune(model, tokenizer, dataset: str, out_dir: str, epochs: int = 1, smoke_test: bool = False):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    train_ds = load_split(dataset, "train")
    val_ds = load_split(dataset, "val")
    if smoke_test:
        train_ds = train_ds.select(range(min(8, len(train_ds))))
        val_ds = val_ds.select(range(min(8, len(val_ds))))
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    training_args = TrainingArguments(
        output_dir=out_dir, per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, num_train_epochs=1 if smoke_test else epochs,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1", fp16=(PRECISION == "fp16"),
        seed=SEED, logging_steps=50, report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                       eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    return model


def run_config(dataset: str, prune_ratio: float, distill: bool, device: str, smoke_test: bool = False) -> dict:
    """Runs ONE cell of the search grid: optionally distill, optionally prune (with
    one epoch of recovery fine-tuning if pruning was applied, matching prune.py's own
    established practice -- Section 3.3, one-shot pruning without recovery collapses
    the classifier). Returns accuracy/energy metrics plus the config's own identity."""
    tag = f"distill={distill}_prune={prune_ratio:.0%}"
    print(f"=== config: {tag} ===")

    if distill:
        model, tokenizer = distill_to_student(dataset, device, smoke_test=smoke_test)
        pruner = prune_distilbert_heads
    else:
        ckpt_dir = os.path.join(P1_RESULTS_DIR, dataset, "best")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
        if torch.cuda.is_available():
            model = model.to(device)
        pruner = prune_bert_heads

    if prune_ratio > 0:
        pruner(model, prune_ratio)
        recovery_out = os.path.join(RESULTS_DIR, dataset, f"combined_recovery_{tag}")
        model = recovery_finetune(model, tokenizer, dataset, recovery_out, epochs=1, smoke_test=smoke_test)

    metrics = evaluate_accuracy(model, tokenizer, dataset)

    val_path = os.path.join(P1_RESULTS_DIR, "data", f"{dataset}_val.parquet")
    sample_texts = pd.read_parquet(val_path)["text"].tolist()[:BATCH_SIZE]
    if len(sample_texts) < BATCH_SIZE:
        sample_texts = (sample_texts * (BATCH_SIZE // len(sample_texts) + 1))[:BATCH_SIZE]
    encoded = tokenizer(sample_texts, truncation=True, max_length=MAX_SEQ_LEN,
                         padding="max_length", return_tensors="pt")
    energy = run_profiled_inference(model, encoded["input_ids"], encoded["attention_mask"])

    result = {
        "config": tag, "dataset": dataset, "distill": distill, "prune_ratio": prune_ratio,
        "accuracy": metrics["accuracy"], "precision": metrics["precision"],
        "recall": metrics["recall"], "f1": metrics["f1"], "mcc": metrics["mcc"],
        "energy_j_total": energy["energy_j_total"], "energy_method": energy["energy_method"],
    }
    print(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liar", choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--prune_ratio", type=float, default=0.0)
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny run (8 examples, 1 epoch per stage) to check the pipeline doesn't crash.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = run_config(args.dataset, args.prune_ratio, args.distill, device, smoke_test=args.smoke_test)

    if args.smoke_test:
        print("Smoke test finished without crashing:", result)
        return

    results_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"combined_{result['config']}_metrics.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in result.items():
            writer.writerow([k, v])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
