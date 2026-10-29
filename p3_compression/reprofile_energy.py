"""Controlled FP16 energy re-profile of every architecture used in the papers.

Why: Stage 2.5 integrity check (JournalPaper2) found that prune.py, distill.py and
combine.py profile with run_profiled_inference() on an FP32 model (only
p1_baseline/profile_energy.py's main() and the retrofit scripts call .half()), and that
runs were spread over different Kaggle GPUs. Inference energy depends only on the
architecture and the fixed input shape, not on trained weights, so every architecture
is rebuilt here with untrained weights, cast to FP16, and profiled on ONE GPU in ONE
session, interleaved round-robin so drift hits all architectures equally.

Each measurement is taken at the protocol's 20 iterations and also at 200 iterations
(lower NVML sampling noise); both are reported per 320 samples.

Run: python reprofile_energy.py [--repeats 10] [--smoke]
Writes: results/reprofile_energy_fp16.csv
"""
import argparse
import csv
import gc
import os
import platform
import random
import sys
import time

import pandas as pd  # before torch: on Windows, pyarrow loaded after torch segfaults
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "p1_baseline"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "combined_framework"))
sys.path.insert(0, os.path.join(HERE, "linear_attention", "linformer"))
sys.path.insert(0, os.path.join(HERE, "linear_attention", "performer"))
sys.path.insert(0, os.path.join(HERE, "linear_attention", "charformer"))

from config import BATCH_SIZE, MAX_SEQ_LEN, SEED  # noqa: E402
from profile_energy import PowerSampler, run_profiled_inference  # noqa: E402
from prune import heads_to_prune_by_magnitude, prune_layer_heads  # noqa: E402
from combine import prune_distilbert_heads  # noqa: E402
from adapt_linformer import linformerize  # noqa: E402
from adapt_performer import performerize  # noqa: E402
from adapt_charformer import CharformerClassifier, encode_bytes  # noqa: E402

BERT = "bert-base-uncased"
DISTIL = "distilbert-base-uncased"
SAMPLES_PER_RUN = 320  # protocol: 20 iterations x batch 16


def untrained(name):
    # Weights do not affect inference cost; from_config avoids any checkpoint dependency.
    return AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(name, num_labels=2))


def build(arch):
    if arch == "bert_base":
        return untrained(BERT)
    if arch.startswith("bert_prune"):
        m = untrained(BERT)
        ratio = int(arch.rsplit("_", 1)[1]) / 100
        for layer in m.bert.encoder.layer:
            prune_layer_heads(layer, heads_to_prune_by_magnitude(layer, ratio))
        return m
    if arch.startswith("distil_prune"):
        m = untrained(DISTIL)
        ratio = int(arch.rsplit("_", 1)[1]) / 100
        return prune_distilbert_heads(m, ratio) if ratio > 0 else m
    if arch == "linformer":
        m = untrained(BERT)
        linformerize(m, proj_k=64, max_seq_len=MAX_SEQ_LEN)
        return m
    if arch == "performer":
        m = untrained(BERT)
        performerize(m, nb_features=64)
        return m
    if arch == "charformer":
        return CharformerClassifier()
    raise ValueError(arch)


ARCHS = ["bert_base", "bert_prune_25", "bert_prune_50", "bert_prune_75",
         "distil_prune_0", "distil_prune_25", "distil_prune_50", "distil_prune_75",
         "linformer", "performer", "charformer"]
# FP32 controls reproduce the precision the compression scripts actually profiled in.
FP32_CONTROLS = ["bert_base", "distil_prune_0", "bert_prune_50"]


def profile_charformer(model, ids, mask, n_iters):
    model.eval()
    with torch.no_grad(), PowerSampler() as sampler:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(ids, mask)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    return {"energy_j_total": sampler.energy_joules(), "total_time_s": total,
            "energy_method": sampler.method, "components": {}}


def measure(arch, model, inputs, n_iters):
    if arch == "charformer":
        return profile_charformer(model, inputs["bytes"], inputs["byte_mask"], n_iters)
    return run_profiled_inference(model, inputs["input_ids"], inputs["attention_mask"], n_iters=n_iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--smoke", action="store_true", help="1 repeat, 2 iterations, no CSV")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    random.seed(SEED)
    torch.manual_seed(SEED)
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)

    # Same input as the original profiling: first BATCH_SIZE LIAR validation texts, padded to 256.
    val = os.path.join(HERE, "..", "p1_baseline", "results", "data", "liar_val.parquet")
    texts = pd.read_parquet(val)["text"].tolist()[:BATCH_SIZE]
    tok = AutoTokenizer.from_pretrained(BERT)
    enc = tok(texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors="pt")
    b_ids, b_mask = encode_bytes(texts, 1024)
    inputs = {"input_ids": enc["input_ids"].to(dev), "attention_mask": enc["attention_mask"].to(dev),
              "bytes": b_ids.to(dev), "byte_mask": b_mask.to(dev)}

    jobs = [(a, "fp16") for a in ARCHS] + [(a, "fp32") for a in FP32_CONTROLS]
    repeats, iters = (1, [2]) if args.smoke else (args.repeats, [20, 200])
    rows = []
    for r in range(repeats):
        order = jobs[:]
        random.shuffle(order)  # interleave so slow GPU drift is spread over all architectures
        for arch, prec in order:
            torch.manual_seed(SEED)
            model = build(arch).to(dev)  # move AFTER attention replacement so new modules move too
            if prec == "fp16":
                model = model.half()
            model.eval()
            with torch.no_grad():  # warm-up: kernels, allocator, clocks
                for _ in range(3):
                    (model(inputs["bytes"], inputs["byte_mask"]) if arch == "charformer"
                     else model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]))
            for n in iters:
                res = measure(arch, model, inputs, n)
                comp = res.get("components", {})
                rows.append({
                    "arch": arch, "precision": prec, "repeat": r, "n_iters": n,
                    "energy_j": res["energy_j_total"],
                    "energy_j_per_320": res["energy_j_total"] * SAMPLES_PER_RUN / (n * BATCH_SIZE),
                    "time_s": res["total_time_s"], "energy_method": res["energy_method"],
                    "attn_share": comp.get("self_attention", {}).get("time_share", ""),
                    "ffn_share": comp.get("feed_forward", {}).get("time_share", ""),
                    "n_params": sum(p.numel() for p in model.parameters()), "gpu": gpu,
                })
                print(f"r{r} {arch:16s} {prec} n={n:3d} E/320={rows[-1]['energy_j_per_320']:8.2f} J "
                      f"({res['energy_method']})", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if args.smoke:
        print("smoke OK:", len(rows), "measurements on", gpu)
        return
    out = os.path.join(HERE, "results", "reprofile_energy_fp16.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, len(rows), "rows;", gpu, "torch", torch.__version__, platform.python_version())


if __name__ == "__main__":
    main()
