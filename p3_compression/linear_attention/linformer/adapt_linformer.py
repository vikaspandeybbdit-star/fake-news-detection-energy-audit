"""Linformer (Wang et al. 2020, "Linformer: Self-Attention with Linear Complexity",
arXiv:2006.04768) adapted to text classification on LIAR.

M11 of the exploratory (beyond-P1-P4) scope. Real mechanism verified by reading
tatp22/linformer-pytorch's LinearAttentionHead.forward directly (linformer_pytorch.py,
github.com/tatp22/linformer-pytorch), not recalled from the paper alone: K and V are
projected along the SEQUENCE axis by learned matrices E, F (shape seq_len x k, k <<
seq_len) before the Q@K^T score computation, turning attention from O(S^2) to O(S*k) per
head. Q is left full-length -- only K/V are compressed.

Deliberate scope decisions, stated explicitly:

- Retrofit, not from-scratch: starts from the P1 baseline's already-fine-tuned LIAR
  checkpoint (bert-base-uncased/best), not a freshly pretrained model. Only each layer's
  `attention.self` module is replaced -- query/key/value Linear weights are inherited
  unchanged from the fine-tuned checkpoint (already adapted to this task), only E and F
  are new randomly-initialized parameters. This mirrors prune.py's own recovery-finetune
  philosophy (structurally modify attention, then fine-tune to recover) rather than EBT/
  SNN's from-scratch training. It means this tests "does swapping in low-rank K/V
  projection work when retrofitted onto an already-adapted model", not "does a
  Linformer-pretrained-from-scratch model reach the same accuracy" -- a narrower but
  well-defined and honestly-stated claim.
- No fixed-protocol deviations otherwise: unlike EBT (needed fp32) and SNN (needed
  seq_len=64/batch=8), this trains and profiles in the same fp16 / batch=16 / seq=256 as
  the P1 baseline and other P3 techniques -- E/F are just nn.Linear(MAX_SEQ_LEN, proj_k),
  no double-backward or exotic memory pattern involved.
- proj_k=64 (4x reduction from MAX_SEQ_LEN=256) is a single fixed choice, not swept --
  matches the paper's own reported low-loss range (k as low as 128 for seq=512 with
  <1% degradation; 64 for a 256-length sequence is proportionally similar). This project's
  fixed seq_len=256 (dictated by p1_baseline/config.py, not by this technique) is short by
  Linformer's own standards -- the paper's efficiency case is strongest at seq>=512, so any
  efficiency win measured here is a floor, not a representative best case, and this is
  stated plainly wherever the result is reported rather than implied to generalize.
- K/V are masked at pad positions (zeroed) before the E/F projection, matching the
  reference implementation -- otherwise padding tokens would leak into the low-rank
  summary regardless of the query's own attention weights. The raw boolean pad mask is
  captured via a forward_pre_hook on the top-level model (see _PadMaskBox below) rather
  than parsed from transformers' internal extended-attention-mask object, since that
  object's exact shape/dtype is a transformers-version-internal implementation detail
  this project shouldn't depend on.

Time-boxed single go/no-go attempt, matching M9/M10's discipline: if this doesn't train
to a usable, evaluable classifier within one Kaggle run, "evaluated, found impractical to
adapt within scope" is the citable finding, stated plainly in
results/liar/linformer_metrics.csv, not forced.

Usage: python adapt_linformer.py --dataset liar
"""

import argparse
import csv
import os
import sys

import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "p1_baseline"))
from config import BATCH_SIZE, MAX_SEQ_LEN, PRECISION, SEED  # noqa: E402
from profile_energy import run_profiled_inference  # noqa: E402
from train import load_split, compute_metrics  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "p3_compression"))
from quantize import evaluate_accuracy  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results")


class _PadMaskBox:
    """Shared mutable holder so every LinformerSelfAttention instance in the model sees
    the current batch's padding mask without threading it through transformers' own
    attention_mask kwarg (whose internal extended-mask format is version-dependent)."""

    def __init__(self):
        self.mask = None  # (B, S) bool, True = pad position


class LinformerSelfAttention(nn.Module):
    """Drop-in replacement for BertSelfAttention. Reuses the original query/key/value
    Linear modules verbatim (pretrained/fine-tuned weights preserved); only E and F are
    new. See module docstring for the projection math, verified against
    tatp22/linformer-pytorch's LinearAttentionHead.forward."""

    def __init__(self, orig_self_attn, proj_k: int, max_seq_len: int, pad_mask_box: _PadMaskBox):
        super().__init__()
        self.num_attention_heads = orig_self_attn.num_attention_heads
        self.attention_head_size = orig_self_attn.attention_head_size
        self.all_head_size = orig_self_attn.all_head_size
        self.scaling = orig_self_attn.scaling
        self.query = orig_self_attn.query
        self.key = orig_self_attn.key
        self.value = orig_self_attn.value
        self.dropout = orig_self_attn.dropout
        self.E = nn.Linear(max_seq_len, proj_k, bias=False)
        self.F = nn.Linear(max_seq_len, proj_k, bias=False)
        self.pad_mask_box = pad_mask_box

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]  # (B, S)
        hidden_shape = (*input_shape, self.num_attention_heads, self.attention_head_size)

        q = self.query(hidden_states).view(hidden_shape).transpose(1, 2)  # (B,H,S,d)
        k = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        pad_mask = self.pad_mask_box.mask
        if pad_mask is not None:
            m = pad_mask[:, None, :, None]  # (B,1,S,1), broadcasts over heads/head_dim
            k = k.masked_fill(m, 0.0)
            v = v.masked_fill(m, 0.0)

        k_proj = self.E(k.transpose(2, 3))  # (B,H,d,S) -> (B,H,d,proj_k)
        scores = torch.matmul(q, k_proj) * self.scaling  # (B,H,S,proj_k)
        probs = scores.softmax(dim=-1)
        probs = self.dropout(probs)

        v_proj = self.F(v.transpose(2, 3)).transpose(2, 3)  # (B,H,d,S)->(B,H,d,k)->(B,H,k,d)
        attn_output = torch.matmul(probs, v_proj)  # (B,H,S,d)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, self.all_head_size).contiguous()
        return attn_output, probs


def linformerize(model, proj_k: int, max_seq_len: int) -> _PadMaskBox:
    pad_mask_box = _PadMaskBox()

    def _capture_pad_mask(module, args, kwargs):
        am = kwargs.get("attention_mask")
        if am is None and len(args) > 1:
            am = args[1]
        if am is not None:
            pad_mask_box.mask = (am == 0)

    model.register_forward_pre_hook(_capture_pad_mask, with_kwargs=True)

    for layer in model.bert.encoder.layer:
        orig = layer.attention.self
        layer.attention.self = LinformerSelfAttention(orig, proj_k, max_seq_len, pad_mask_box)

    return pad_mask_box


def recovery_finetune(model, tokenizer, dataset: str, out_dir: str, epochs: int, smoke_test: bool = False):
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
    parser.add_argument("--dataset", default="liar", choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--proj_k", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny run (few examples, 1 epoch) to check the pipeline doesn't crash.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    ckpt_dir = os.path.join(P1_RESULTS_DIR, args.dataset, "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    linformerize(model, args.proj_k, MAX_SEQ_LEN)
    print(f"Replaced attention.self in {len(model.bert.encoder.layer)} layers "
          f"with Linformer (proj_k={args.proj_k}, max_seq_len={MAX_SEQ_LEN})")
    # linformerize() constructs new E/F Linear layers on CPU (nn.Linear's default) --
    # must move the WHOLE model (original weights + new layers) to device in one shot
    # after replacement, not before, or the new layers stay stranded on CPU.
    if torch.cuda.is_available():
        model = model.to("cuda")

    epochs = 1 if args.smoke_test else args.epochs
    recovery_out = os.path.join(RESULTS_DIR, args.dataset, "linformer_recovery_ckpt")
    model = recovery_finetune(model, tokenizer, args.dataset, recovery_out, epochs, smoke_test=args.smoke_test)

    if args.smoke_test:
        metrics = evaluate_accuracy(model, tokenizer, args.dataset)
        print("Smoke test finished without crashing:", metrics)
        return

    metrics = evaluate_accuracy(model, tokenizer, args.dataset)
    print(f"{args.dataset} Linformer(proj_k={args.proj_k}) metrics: {metrics}")

    if PRECISION == "fp16" and torch.cuda.is_available():
        model.half()

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
    ckpt_path = os.path.join(out_dir, "linformer_checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    with open(os.path.join(out_dir, "linformer_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        writer.writerow(["energy_j_total", result["energy_j_total"]])
        writer.writerow(["energy_method", result["energy_method"]])
        writer.writerow(["proj_k", args.proj_k])
        writer.writerow(["max_seq_len", MAX_SEQ_LEN])
        writer.writerow(["epochs", epochs])
        writer.writerow(["note",
                          "retrofit onto P1's already-fine-tuned checkpoint (Q/K/V inherited, "
                          "only E/F projections are new); seq_len=256 is short by Linformer's "
                          "own standards (paper's efficiency case is strongest at seq>=512), "
                          "so this energy figure is a floor, not Linformer's best case"])
    print(f"Wrote {out_dir}/linformer_metrics.csv")


if __name__ == "__main__":
    main()
