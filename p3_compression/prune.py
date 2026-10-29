"""Stage 2 / RO2 — structured attention-head pruning of a P1 baseline checkpoint.

Structured (not unstructured) pruning is used deliberately: unstructured L1 pruning
zeros weights but doesn't reduce wall-clock/energy on dense GPU kernels without
Ampere+ 2:4 sparse tensor core support (the Kaggle P100/T4 in this project don't have
it) — it only produces an accuracy-vs-sparsity curve, not an energy win.

Pruning is implemented manually (resizing the Q/K/V and output-projection Linear
layers, updating the self-attention module's num_attention_heads/all_head_size
attributes directly) rather than via HF's prune_heads() utility — confirmed absent
from transformers 5.x entirely (a v5 API removal, not a rename: hasattr checks on both
PreTrainedModel and BertModel return False). This only supports pruning the SAME number
of heads per layer (true here — prune_ratio is uniform across layers), which is what
lets a single shared num_attention_heads value stay valid model-wide.

Head importance: L1 norm of each head's slice of the attention output projection
weight (columns [h*head_dim:(h+1)*head_dim] of attention.output.dense.weight) — a
standard, cheap magnitude-based importance proxy. Lowest prune_ratio fraction pruned
per layer.

Post-prune recovery fine-tuning (1 epoch, on by default): one-shot structured pruning
without any recovery step routinely collapses a fine-tuned classifier to a degenerate
constant-output solution — confirmed empirically here (50% head pruning, no recovery,
produced recall=1.0 i.e. predicting one class always). This is standard practice in the
pruning literature, not optional polish; --no_recovery exists only to reproduce/compare
against that pre-recovery baseline number.

Usage: python prune.py --dataset liar --prune_ratio 0.5
"""

import argparse
import csv
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "p1_baseline"))
from config import BATCH_SIZE, MAX_SEQ_LEN, MODEL_NAME, PRECISION, SEED  # noqa: E402
from profile_energy import run_profiled_inference  # noqa: E402
from train import load_split, compute_metrics  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from quantize import evaluate_accuracy  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

NUM_HEADS = 12
HEAD_DIM = 64


def heads_to_prune_by_magnitude(layer, prune_ratio: float):
    weight = layer.attention.output.dense.weight  # [hidden, hidden], input dim split by head
    scores = []
    for h in range(NUM_HEADS):
        col_slice = weight[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
        scores.append((col_slice.abs().sum().item(), h))
    scores.sort()
    n_prune = int(NUM_HEADS * prune_ratio)
    return [h for _, h in scores[:n_prune]]


def prune_layer_heads(layer, heads_to_prune):
    sa = layer.attention.self
    device = sa.query.weight.device
    keep_heads = [h for h in range(NUM_HEADS) if h not in heads_to_prune]
    keep_dims = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in keep_heads])

    for name in ("query", "key", "value"):
        old = getattr(sa, name)
        new = torch.nn.Linear(old.in_features, len(keep_dims)).to(device)
        new.weight.data = old.weight.data[keep_dims, :].clone()
        new.bias.data = old.bias.data[keep_dims].clone()
        setattr(sa, name, new)

    sa.num_attention_heads = len(keep_heads)
    sa.all_head_size = len(keep_heads) * HEAD_DIM

    old_out = layer.attention.output.dense
    new_out = torch.nn.Linear(len(keep_dims), old_out.out_features).to(device)
    new_out.weight.data = old_out.weight.data[:, keep_dims].clone()
    new_out.bias.data = old_out.bias.data.clone()
    layer.attention.output.dense = new_out


def recovery_finetune(model, tokenizer, dataset: str, out_dir: str, epochs: int = 1):
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")

    train_ds = load_split(dataset, "train").map(tokenize, batched=True)
    val_ds = load_split(dataset, "val").map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=(PRECISION == "fp16"),
        seed=SEED,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                       eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--prune_ratio", type=float, default=0.5)
    parser.add_argument("--no_recovery", action="store_true",
                         help="Skip post-prune fine-tuning; reproduces the pre-recovery collapsed baseline")
    args = parser.parse_args()

    ckpt_dir = os.path.join(P1_RESULTS_DIR, args.dataset, "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")

    for layer in model.bert.encoder.layer:
        heads = heads_to_prune_by_magnitude(layer, args.prune_ratio)
        prune_layer_heads(layer, heads)
    print(f"Pruned {args.prune_ratio:.0%} of heads per layer ({MODEL_NAME}, {NUM_HEADS} heads/layer)")

    tag = "no_recovery" if args.no_recovery else "recovered"
    if not args.no_recovery:
        recovery_out = os.path.join(RESULTS_DIR, args.dataset, "prune_recovery_ckpt")
        model = recovery_finetune(model, tokenizer, args.dataset, recovery_out)

    metrics = evaluate_accuracy(model, tokenizer, args.dataset)
    print(f"{args.dataset} pruned({args.prune_ratio:.0%}, {tag}) metrics: {metrics}")

    val_path = os.path.join(P1_RESULTS_DIR, "data", f"{args.dataset}_val.parquet")
    sample_texts = pd.read_parquet(val_path)["text"].tolist()[:BATCH_SIZE]
    if len(sample_texts) < BATCH_SIZE:
        sample_texts = (sample_texts * (BATCH_SIZE // len(sample_texts) + 1))[:BATCH_SIZE]
    encoded = tokenizer(sample_texts, truncation=True, max_length=MAX_SEQ_LEN,
                         padding="max_length", return_tensors="pt")
    result = run_profiled_inference(model, encoded["input_ids"], encoded["attention_mask"])
    print(result)

    out_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"prune_{args.prune_ratio:.0%}_{tag}_metrics.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        writer.writerow(["energy_j_total", result["energy_j_total"]])
        writer.writerow(["energy_method", result["energy_method"]])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
