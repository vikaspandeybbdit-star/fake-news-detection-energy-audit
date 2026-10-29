"""Stage 2 / RO2 — dynamic INT8 quantization of a P1 baseline checkpoint, evaluated for
accuracy + CPU wall-clock latency under the same fixed batch/seq protocol as P1.

Uses PyTorch's native torch.quantization.quantize_dynamic(), not bitsandbytes:
bitsandbytes' INT8 path failed two different ways on two different GPUs (a
"cublasLt ran into an error!" RuntimeError on the Kaggle P100 — plausibly a Pascal/
compute-capability-6.0 incompatibility, since bitsandbytes' documented INT8 matmul
support starts at Turing/7.5 — and a separate "'Parameter' object has no attribute
'CB'" AttributeError even on a local Turing/7.5 GPU, traced to the freshly-initialized
classification head's Linear layer not going through bnb's on-the-fly quantization
conversion the same way a pretrained-and-loaded Linear does). PyTorch's own dynamic
quantization is CPU-only but well-supported and simple — the right tradeoff here.

IMPORTANT methodological note: because this technique runs on CPU, NOT GPU,
profile_energy.py's NVML-based energy measurement is NOT used here — NVML polls GPU
power draw, and while the CPU does the actual work the GPU would just be idle, so an
NVML reading would misattribute GPU idle power to CPU-executed computation. Only CPU
wall-clock latency is reported for this technique, explicitly not an energy/joules
figure, and this scope difference must be stated plainly wherever this result is used.

Usage: python quantize.py --dataset liar
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "p1_baseline"))
from config import BATCH_SIZE, MAX_SEQ_LEN  # noqa: E402

P1_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "p1_baseline", "results")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def evaluate_accuracy(model, tokenizer, dataset: str) -> dict:
    """Shared by quantize.py (CPU model) and prune.py (GPU model) — moves inputs to
    whatever device the model is actually on."""
    from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_recall_fscore_support

    val_path = os.path.join(P1_RESULTS_DIR, "data", f"{dataset}_val.parquet")
    df = pd.read_parquet(val_path)
    device = next(model.parameters()).device

    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i + BATCH_SIZE]
            enc = tokenizer(batch["text"].tolist(), truncation=True, max_length=MAX_SEQ_LEN,
                             padding="max_length", return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["label"].tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": matthews_corrcoef(labels, preds),
    }


def cpu_latency_seconds(model, input_ids, attention_mask, n_iters: int = 20) -> float:
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(input_ids=input_ids, attention_mask=attention_mask)
        return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["liar", "isot", "fakenewsnet"])
    args = parser.parse_args()

    ckpt_dir = os.path.join(P1_RESULTS_DIR, args.dataset, "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    metrics = evaluate_accuracy(model, tokenizer, args.dataset)
    print(f"{args.dataset} INT8 (CPU, dynamic) metrics: {metrics}")

    val_path = os.path.join(P1_RESULTS_DIR, "data", f"{args.dataset}_val.parquet")
    sample_texts = pd.read_parquet(val_path)["text"].tolist()[:BATCH_SIZE]
    if len(sample_texts) < BATCH_SIZE:
        sample_texts = (sample_texts * (BATCH_SIZE // len(sample_texts) + 1))[:BATCH_SIZE]
    encoded = tokenizer(sample_texts, truncation=True, max_length=MAX_SEQ_LEN,
                         padding="max_length", return_tensors="pt")
    latency_s = cpu_latency_seconds(model, encoded["input_ids"], encoded["attention_mask"])
    print(f"CPU latency (20 iters, batch={BATCH_SIZE}, seq={MAX_SEQ_LEN}): {latency_s:.3f}s")

    out_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "quantize_int8_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
        writer.writerow(["cpu_latency_s_20iters", latency_s])
        writer.writerow(["note", "CPU-only technique; not NVML-energy-comparable to GPU baseline"])
    print(f"Wrote {out_dir}/quantize_int8_metrics.csv")


if __name__ == "__main__":
    main()
