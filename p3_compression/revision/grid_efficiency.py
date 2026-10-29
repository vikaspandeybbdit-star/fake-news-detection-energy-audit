"""Same-session latency, throughput, peak memory and energy for every configuration of the combined-framework grid
(IEEE Access R1#8, R2#1): BERT-base with structured head pruning 0/25/50/75%, DistilBERT with 0/25/50/75%, and TinyBERT.
FP16, batch 16, 256 tokens (random ids; cost depends on shape, not content), windows of 20 and 100 iterations, 5 repeats each.
Pruned configurations are built from pretrained weights by magnitude-based head removal (shape, not accuracy, is measured here).
Writes /kaggle/working/out/grid_efficiency.csv."""
import copy
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
for p in (os.path.join(P3, "..", "p1_baseline"), P3, os.path.join(P3, "combined_framework")):
    sys.path.insert(0, p)

from profile_energy import PowerSampler  # noqa: E402
from prune import heads_to_prune_by_magnitude, prune_layer_heads  # noqa: E402
from combine import prune_distilbert_heads  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

DEV = "cuda"
B, L, VOCAB = 16, 256, 30522
ids = torch.randint(1000, VOCAB, (B, L), device=DEV)
mask = torch.ones(B, L, dtype=torch.long, device=DEV)
rows = []


def build(kind, r):
    if kind == "bert":
        m = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        if r:
            for layer in m.bert.encoder.layer:
                prune_layer_heads(layer, heads_to_prune_by_magnitude(layer, r))
        return m
    if kind == "distil":
        m = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        return prune_distilbert_heads(copy.deepcopy(m), r) if r else m
    return AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)


configs = [("BERT-base, prune %d%%" % int(r * 100), "bert", r) for r in (0, 0.25, 0.5, 0.75)] + \
          [("DistilBERT, prune %d%%" % int(r * 100), "distil", r) for r in (0, 0.25, 0.5, 0.75)] + [("TinyBERT-4L", "tiny", 0)]
for name, kind, r in configs:
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model = build(kind, r).to(DEV).eval().half()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / 2**20
    with torch.no_grad():
        for _ in range(10):
            model(input_ids=ids, attention_mask=mask)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        for n_iters in (20, 100):
            e, t = [], []
            for _ in range(5):
                with PowerSampler() as s:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(n_iters):
                        model(input_ids=ids, attention_mask=mask)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                e.append(s.energy_joules() / n_iters)
                t.append(dt / n_iters)
            rows.append(dict(config=name, n_iters=n_iters, params_M=round(n_params, 2), weights_MB=round(weights, 1), peak_MB=round(peak, 1),
                             j_per_iter=round(float(np.mean(e)), 4), j_sd=round(float(np.std(e)), 4), s_per_iter=round(float(np.mean(t)), 5),
                             throughput_ex_s=round(B / float(np.mean(t)), 1)))
            print(rows[-1], flush=True)
    del model

os.makedirs("/kaggle/working/out", exist_ok=True)
with open("/kaggle/working/out/grid_efficiency.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
