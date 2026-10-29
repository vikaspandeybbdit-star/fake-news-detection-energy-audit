"""Training-cost audit: energy, time and peak memory per TRAINING step for every model family in the paper, one session, one protocol.
Complements energy_unified.py (inference). Each configuration runs real optimizer steps (forward, backward, AdamW update) on a fixed
random batch of 16 inputs, in the precision each model is trained in (BERT family: FP16 autocast; EBT, SpikeBERT, dense control: FP32).
Protocol: fixed shuffled order (seed 7); 30 s warm-up under load; peak memory measured over the first steps after optimizer state exists;
energy = NVML integrated power over windows of 10 and 50 steps, repeats 5 (3 for slow models); idle power before and after.
Distillation students include the resident BERT-base teacher forward pass. Writes /kaggle/working/out/train_energy.csv, idle_power.txt, gpu_info.txt."""
import csv
import gc
import os
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
for p in (os.path.join(P3, "ebt"), os.path.join(P3, "snn"), os.path.join(P3, "..", "p1_baseline"), P3, os.path.join(P3, "combined_framework"), HERE):
    sys.path.insert(0, p)

from profile_energy import PowerSampler  # noqa: E402
from prune import heads_to_prune_by_magnitude, prune_layer_heads  # noqa: E402
from adapt_classification import EnergyClassifier, ebt_loss  # noqa: E402
from adapt_spiking import SpikingClassifier  # noqa: E402
from run_snn import DenseClassifier  # noqa: E402
from transformers import AutoModelForSequenceClassification  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402

DEV = "cuda"
B, VOCAB = 16, 30522
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
random.seed(7)
KD_T, KD_A = 2.0, 0.5


def idle_power(seconds=10):
    with PowerSampler() as s:
        time.sleep(seconds)
    return s.energy_joules() / seconds


idle_before = idle_power()
info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit,temperature.gpu", "--format=csv"], capture_output=True, text=True).stdout
open(os.path.join(OUT, "gpu_info.txt"), "w").write(info)
print(info, flush=True)


def batch(L):
    ids = torch.randint(1000, VOCAB, (B, L), device=DEV)
    mask = torch.ones(B, L, dtype=torch.long, device=DEV)
    labels = torch.randint(0, 2, (B,), device=DEV)
    return ids, mask, labels


def hf_model(ckpt, prune=0.0):
    m = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
    if prune:
        for layer in m.bert.encoder.layer:
            prune_layer_heads(layer, heads_to_prune_by_magnitude(layer, prune))
    return m


def make_hf(ckpt, L, prune=0.0, teacher=False):
    """AdamW lr 5e-5, FP16 autocast with gradient scaling; optional resident BERT-base teacher (logit distillation)."""
    ids, mask, labels = batch(L)
    model = hf_model(ckpt, prune).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler()
    tch = None
    if teacher:
        tch = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(DEV).half().eval()

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            if tch is not None:
                with torch.no_grad():
                    tl = tch(input_ids=ids, attention_mask=mask).logits.float()
                kd = F.kl_div(F.log_softmax(out.logits.float() / KD_T, dim=-1), F.softmax(tl / KD_T, dim=-1), reduction="batchmean") * KD_T ** 2
                loss = KD_A * kd + (1 - KD_A) * F.cross_entropy(out.logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    return model, opt, step, sum(p.numel() for p in model.parameters()) / 1e6


def make_ebt():
    ids, mask, labels = batch(256)
    model = EnergyClassifier(mcmc_num_steps=3, mcmc_step_size=0.1, langevin_noise=0.02, energy_layers=2).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    def step():
        dists, _ = model(ids, mask, training=True)
        loss = ebt_loss(dists, labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    return model, opt, step, sum(p.numel() for p in model.parameters()) / 1e6


def make_snn():
    ids, mask, labels = batch(64)
    model = SpikingClassifier(vocab_size=VOCAB, dim=768, depths=6, T=16, tau=10.0, common_thr=1.0).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)

    def step():
        logits = model(ids).mean(dim=1)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        functional.reset_net(model)

    return model, opt, step, sum(p.numel() for p in model.parameters()) / 1e6


def make_dense():
    ids, mask, labels = batch(64)
    model = DenseClassifier(vocab_size=VOCAB, dim=768, depths=6).to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        logits = model(ids, key_padding_mask=(mask == 0))
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    return model, opt, step, sum(p.numel() for p in model.parameters()) / 1e6


# name, factory, tokens, precision
CONFIGS = [
    ("BERT-base fine-tuning", lambda: make_hf("bert-base-uncased", 256), 256, "FP16"),
    ("BERT-base, 64 tokens", lambda: make_hf("bert-base-uncased", 64), 64, "FP16"),
    ("BERT-base prune 75% (recovery)", lambda: make_hf("bert-base-uncased", 256, prune=0.75), 256, "FP16"),
    ("DistilBERT distillation (with teacher)", lambda: make_hf("distilbert-base-uncased", 256, teacher=True), 256, "FP16"),
    ("TinyBERT-4L distillation (with teacher)", lambda: make_hf("huawei-noah/TinyBERT_General_4L_312D", 256, teacher=True), 256, "FP16"),
    ("EBT T=3 training", make_ebt, 256, "FP32"),
    ("SpikeBERT T=16 training", make_snn, 64, "FP32"),
    ("Dense control training", make_dense, 64, "FP32"),
]
random.shuffle(CONFIGS)
print("order:", [c[0] for c in CONFIGS], flush=True)

CSV = os.path.join(OUT, "train_energy.csv")
FIELDS = ["config", "tokens", "precision", "params_M", "peak_MB", "n_steps", "repeats", "j_per_step", "j_sd", "s_per_step", "status"]
rows = []


def flush_rows():
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


for name, factory, L, prec in CONFIGS:
    gc.collect()
    torch.cuda.empty_cache()
    try:
        model, opt, step, n_params = factory()
        t0 = time.time()
        while time.time() - t0 < 30:  # warm-up under load; also builds optimizer state
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        t1 = time.perf_counter()
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        per_step = (time.perf_counter() - t1) / 3
        reps = 5 if per_step < 0.5 else 3
        for n_steps in (10, 50):
            e, t = [], []
            for _ in range(reps):
                with PowerSampler() as s:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(n_steps):
                        step()
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                e.append(s.energy_joules() / n_steps)
                t.append(dt / n_steps)
            rows.append(dict(config=name, tokens=L, precision=prec, params_M=round(n_params, 2), peak_MB=round(peak, 1), n_steps=n_steps, repeats=reps,
                             j_per_step=round(float(np.mean(e)), 4), j_sd=round(float(np.std(e)), 4), s_per_step=round(float(np.mean(t)), 4), status="ok"))
            print(rows[-1], flush=True)
        flush_rows()
        del model, opt, step
    except Exception:  # keep going: one failing configuration must not lose the others
        traceback.print_exc()
        rows.append(dict(config=name, tokens=L, precision=prec, params_M="", peak_MB="", n_steps="", repeats="", j_per_step="", j_sd="", s_per_step="", status="failed"))
        flush_rows()

idle_after = idle_power()
open(os.path.join(OUT, "idle_power.txt"), "w").write(f"idle_before_w={idle_before:.2f}\nidle_after_w={idle_after:.2f}\n")
flush_rows()
print("done; idle", idle_before, idle_after)
