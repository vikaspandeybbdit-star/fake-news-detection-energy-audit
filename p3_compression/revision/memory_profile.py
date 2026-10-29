"""Peak GPU inference-memory profile (IEEE Access R1#8). Activation memory depends on
architecture and input shape, not on learned weight values, so models are profiled at their
real inference configuration without training. Batch 16; 256 tokens for BERT-family/EBT,
64 for SpikeBERT/dense control (the lengths each was trained and evaluated at).
Writes memory_profile.csv.
"""
import csv
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(P3, "ebt"))
sys.path.insert(0, os.path.join(P3, "snn"))
sys.path.insert(0, os.path.join(P3, "..", "p1_baseline"))
sys.path.insert(0, HERE)

from transformers import AutoModelForSequenceClassification  # noqa: E402

DEV = "cuda"
B = 16
VOCAB = 30522
rows = []


def measure(name, build, make_inputs, fwd, half):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model = build().to(DEV).eval()
    if half:
        model = model.half()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() / 2**20
    inputs = make_inputs()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(5):
            fwd(model, inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**20
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    rows.append(dict(model=name, precision="fp16" if half else "fp32", batch=B,
                     seq_len=inputs[0].shape[1], params_M=round(n_params, 2),
                     weights_MB=round(weights, 1), peak_MB=round(peak, 1),
                     activation_MB=round(peak - weights, 1)))
    print(rows[-1], flush=True)
    del model
    torch.cuda.empty_cache()


def ids(n):
    return lambda: (torch.randint(1000, VOCAB, (B, n), device=DEV), torch.ones(B, n, dtype=torch.long, device=DEV))


def hf_fwd(m, x):
    return m(input_ids=x[0], attention_mask=x[1])


for name, ckpt in (("BERT-base (baseline)", "bert-base-uncased"),
                   ("DistilBERT", "distilbert-base-uncased"),
                   ("TinyBERT-4L", "huawei-noah/TinyBERT_General_4L_312D")):
    measure(name, lambda ckpt=ckpt: AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2), ids(256), hf_fwd, True)

from adapt_classification import EnergyClassifier  # noqa: E402
for T in (1, 2, 3):
    measure(f"EBT (T={T})", lambda T=T: EnergyClassifier(mcmc_num_steps=T, mcmc_step_size=0.1, langevin_noise=0.02, energy_layers=2),
            ids(256), lambda m, x: m(x[0], x[1], training=False), True)

from adapt_spiking import SpikingClassifier  # noqa: E402
from run_snn import DenseClassifier  # noqa: E402
from spikingjelly.activation_based import functional  # noqa: E402


def snn_fwd(m, x):
    out = m(x[0]).mean(dim=1)
    functional.reset_net(m)
    return out


measure("SpikeBERT (T=16)", lambda: SpikingClassifier(vocab_size=VOCAB, dim=768, depths=6, T=16, tau=10.0, common_thr=1.0), ids(64), snn_fwd, False)
measure("Dense control", lambda: DenseClassifier(vocab_size=VOCAB, dim=768, depths=6), ids(64), lambda m, x: m(x[0], key_padding_mask=(x[1] == 0)), False)

os.makedirs("/kaggle/working/out", exist_ok=True)
with open("/kaggle/working/out/memory_profile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
