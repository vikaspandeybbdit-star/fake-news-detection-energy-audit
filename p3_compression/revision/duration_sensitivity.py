"""Energy-profiling sensitivity to window length, plus idle GPU power (IEEE Access R2#1).
BERT-base (FP16, batch 16, 256 tokens) and EBT (T=3, FP16): for window lengths of 20, 50, 100
and 200 iterations, 5 repeats each, report J per iteration. Idle power: PowerSampler over 10 s
with no workload. Writes duration_sensitivity.csv and idle_power.txt in /kaggle/working/out."""
import csv
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "ebt"))
sys.path.insert(0, os.path.join(P3, "..", "p1_baseline"))

import numpy as np  # noqa: E402
from profile_energy import PowerSampler  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402
from adapt_classification import EnergyClassifier  # noqa: E402

DEV = "cuda"
B, L, VOCAB = 16, 256, 30522
out_dir = "/kaggle/working/out"
os.makedirs(out_dir, exist_ok=True)

ids = torch.randint(1000, VOCAB, (B, L), device=DEV)
mask = torch.ones(B, L, dtype=torch.long, device=DEV)

with PowerSampler() as s:
    time.sleep(10)
idle_w = s.energy_joules() / 10.0
open(os.path.join(out_dir, "idle_power.txt"), "w").write(f"idle_power_w={idle_w:.2f}\n")
print("idle power W", idle_w, flush=True)

rows = []
models = (("BERT-base (FP16)", lambda: AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2),
           lambda m: m(input_ids=ids, attention_mask=mask)),
          ("EBT T=3 (FP16)", lambda: EnergyClassifier(mcmc_num_steps=3, mcmc_step_size=0.1, langevin_noise=0.02, energy_layers=2),
           lambda m: m(ids, mask, training=False)))
for name, build, fwd in models:
    model = build().to(DEV).eval().half()
    with torch.no_grad():
        for _ in range(5):
            fwd(model)
        for n_iters in (20, 50, 100, 200):
            per_iter, times = [], []
            for _ in range(5):
                with PowerSampler() as sampler:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(n_iters):
                        fwd(model)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                per_iter.append(sampler.energy_joules() / n_iters)
                times.append(dt / n_iters)
            rows.append(dict(model=name, n_iters=n_iters, j_per_iter_mean=round(float(np.mean(per_iter)), 4),
                             j_per_iter_sd=round(float(np.std(per_iter)), 4), s_per_iter=round(float(np.mean(times)), 4),
                             mean_power_w=round(float(np.mean(per_iter)) / float(np.mean(times)), 1)))
            print(rows[-1], flush=True)
    del model
    torch.cuda.empty_cache()

with open(os.path.join(out_dir, "duration_sensitivity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
