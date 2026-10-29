"""One-session energy, latency, throughput and peak-memory measurement for EVERY model in the paper (IEEE Access revision).
All numbers in the paper's energy tables come from this script: one GPU, one session, one protocol.

Models: BERT-base with structured head pruning 0/25/50/75%, DistilBERT with 0/25/50/75%, TinyBERT-4L, EBT (T=1,2,3; FP16 and
FP32 inference), SpikeBERT (T=16) and the dense control (64 tokens, FP32), plus BERT-base at 64 tokens (length control).
Protocol per configuration: random configuration order (fixed seed); 30 s warm-up under load; then two windows
(20 iterations = the paper's original protocol window; 200 iterations = steady state), 10 repeats each (5 for the slow models);
energy = NVML integrated power (PowerSampler); idle power measured before and after the whole session.
Writes /kaggle/working/out/energy_unified.csv, idle_power.txt, gpu_info.txt."""
import csv
import gc
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
for p in (os.path.join(P3, "ebt"), os.path.join(P3, "snn"), os.path.join(P3, "..", "p1_baseline"), P3, os.path.join(P3, "combined_framework"), HERE):
    sys.path.insert(0, p)

from profile_energy import PowerSampler  # noqa: E402
from prune import heads_to_prune_by_magnitude, prune_layer_heads  # noqa: E402
from combine import prune_distilbert_heads  # noqa: E402
from adapt_classification import EnergyClassifier  # noqa: E402
from adapt_spiking import SpikingClassifier  # noqa: E402
from run_snn import DenseClassifier  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402

DEV = "cuda"
B, VOCAB = 16, 30522
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
random.seed(7)


def idle_power(seconds=10):
    with PowerSampler() as s:
        time.sleep(seconds)
    return s.energy_joules() / seconds


idle_before = idle_power()
info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit,temperature.gpu", "--format=csv"], capture_output=True, text=True).stdout
open(os.path.join(OUT, "gpu_info.txt"), "w").write(info)
print(info, flush=True)


def hf(ckpt, kind="bert", r=0.0):
    def build():
        m = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
        if kind == "bert" and r:
            for layer in m.bert.encoder.layer:
                prune_layer_heads(layer, heads_to_prune_by_magnitude(layer, r))
        if kind == "distil" and r:
            import copy
            m = prune_distilbert_heads(copy.deepcopy(m), r)
        return m
    return build


def hf_fwd(L):
    ids = torch.randint(1000, VOCAB, (B, L), device=DEV)
    mask = torch.ones(B, L, dtype=torch.long, device=DEV)
    return lambda m: m(input_ids=ids, attention_mask=mask)


def ebt_fwd(L):
    ids = torch.randint(1000, VOCAB, (B, L), device=DEV)
    mask = torch.ones(B, L, dtype=torch.long, device=DEV)
    return lambda m: m(ids, mask, training=False)


def snn_fwd(L):
    ids = torch.randint(1000, VOCAB, (B, L), device=DEV)

    def f(m):
        out = m(ids).mean(dim=1)
        functional.reset_net(m)
        return out
    return f


def dense_fwd(L):
    ids = torch.randint(1000, VOCAB, (B, L), device=DEV)
    mask = torch.ones(B, L, dtype=torch.long, device=DEV)
    return lambda m: m(ids, key_padding_mask=(mask == 0))


# name, builder, forward factory, seq len, half?
CONFIGS = []
for r in (0, 0.25, 0.5, 0.75):
    CONFIGS.append((f"BERT-base prune {int(r * 100)}%", hf("bert-base-uncased", "bert", r), hf_fwd(256), 256, True))
    CONFIGS.append((f"DistilBERT prune {int(r * 100)}%", hf("distilbert-base-uncased", "distil", r), hf_fwd(256), 256, True))
CONFIGS.append(("TinyBERT-4L", hf("huawei-noah/TinyBERT_General_4L_312D", "tiny", 0), hf_fwd(256), 256, True))
CONFIGS.append(("BERT-base 64 tokens", hf("bert-base-uncased", "bert", 0), hf_fwd(64), 64, True))
for T in (1, 2, 3):
    CONFIGS.append((f"EBT T={T} FP16", (lambda T=T: EnergyClassifier(mcmc_num_steps=T, mcmc_step_size=0.1, langevin_noise=0.02, energy_layers=2)), ebt_fwd(256), 256, True))
CONFIGS.append(("EBT T=3 FP32", (lambda: EnergyClassifier(mcmc_num_steps=3, mcmc_step_size=0.1, langevin_noise=0.02, energy_layers=2)), ebt_fwd(256), 256, False))
CONFIGS.append(("SpikeBERT T=16 FP32", (lambda: SpikingClassifier(vocab_size=VOCAB, dim=768, depths=6, T=16, tau=10.0, common_thr=1.0)), snn_fwd(64), 64, False))
CONFIGS.append(("Dense control FP32", (lambda: DenseClassifier(vocab_size=VOCAB, dim=768, depths=6)), dense_fwd(64), 64, False))
random.shuffle(CONFIGS)
print("order:", [c[0] for c in CONFIGS], flush=True)

rows = []
for name, build, fwd_factory, L, half in CONFIGS:
    gc.collect()
    torch.cuda.empty_cache()
    model = build().to(DEV).eval()
    if half:
        model = model.half()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / 2**20
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    fwd = fwd_factory
    with torch.no_grad():
        t0 = time.time()
        while time.time() - t0 < 30:  # warm-up under load: steady clocks and temperature
            fwd(model)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            fwd(model)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        t1 = time.perf_counter()
        for _ in range(10):
            fwd(model)
        torch.cuda.synchronize()
        per_iter = (time.perf_counter() - t1) / 10
        reps = 10 if per_iter < 0.06 else 5
        for n_iters in (20, 200):
            e, t = [], []
            for _ in range(reps):
                with PowerSampler() as s:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(n_iters):
                        fwd(model)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                e.append(s.energy_joules() / n_iters)
                t.append(dt / n_iters)
            rows.append(dict(config=name, seq_len=L, precision="fp16" if half else "fp32", n_iters=n_iters, repeats=reps, params_M=round(n_params, 2),
                             weights_MB=round(weights, 1), peak_MB=round(peak, 1), j_per_iter=round(float(np.mean(e)), 5), j_sd=round(float(np.std(e)), 5),
                             s_per_iter=round(float(np.mean(t)), 5), throughput_ex_s=round(B / float(np.mean(t)), 1)))
            print(rows[-1], flush=True)
    del model

idle_after = idle_power()
open(os.path.join(OUT, "idle_power.txt"), "w").write(f"idle_before_w={idle_before:.2f}\nidle_after_w={idle_after:.2f}\n")
with open(os.path.join(OUT, "energy_unified.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("done; idle", idle_before, idle_after)
