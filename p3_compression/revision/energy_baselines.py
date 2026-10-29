"""Same-session inference energy for the compression baselines (IEEE Access R1#2/R2#5): BERT-base, DistilBERT,
TinyBERT, FP16, batch 16, 256 tokens, 20- and 100-iteration windows, 5 repeats each. Writes energy_baselines.csv."""
import csv
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "..", "p1_baseline"))
from profile_energy import PowerSampler  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402

DEV = "cuda"
B, L, VOCAB = 16, 256, 30522
ids = torch.randint(1000, VOCAB, (B, L), device=DEV)
mask = torch.ones(B, L, dtype=torch.long, device=DEV)
rows = []
for name, ckpt in (("BERT-base", "bert-base-uncased"), ("DistilBERT", "distilbert-base-uncased"),
                   ("TinyBERT-4L", "huawei-noah/TinyBERT_General_4L_312D")):
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2).to(DEV).eval().half()
    with torch.no_grad():
        for _ in range(10):
            model(input_ids=ids, attention_mask=mask)
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
            rows.append(dict(model=name, n_iters=n_iters, j_per_iter_mean=round(float(np.mean(e)), 4),
                             j_per_iter_sd=round(float(np.std(e)), 4), s_per_iter=round(float(np.mean(t)), 4)))
            print(rows[-1], flush=True)
    del model
    torch.cuda.empty_cache()
os.makedirs("/kaggle/working/out", exist_ok=True)
with open("/kaggle/working/out/energy_baselines.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
