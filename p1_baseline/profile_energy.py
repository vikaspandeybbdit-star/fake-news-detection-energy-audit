"""IrEne-style component-wise energy/latency profiling: direct forward-hook timing per
BERT block (module-level attribution) + NVML power polling (falls back to a fixed TDP
estimate if NVML power-draw queries are blocked, as is common on virtualized/cloud GPUs).

Module-level timing uses forward hooks rather than torch.profiler's with_modules=True:
empirically (both locally and on Kaggle's P100), with_modules=True produces zero
module-scoped events for this model/torch combination — only leaf aten::* ops show up
in both key_averages() and the raw event list. Hooks measure exactly what we need
directly and don't depend on that profiler internal.

Hard constraint: batch size, sequence length, and precision come from config.py and are
NOT overridable from the CLI here — this is the one thing that must never drift between
runs (see P2 survey Section 3.3 on serving-config confounds).

Usage: python profile_energy.py --dataset liar
Exposes run_profiled_inference() for reuse by p3_compression/ and p4_benchmark/.
"""

import argparse
import csv
import os
import threading
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import BATCH_SIZE, MAX_SEQ_LEN, PRECISION

try:
    import pynvml

    _NVML_IMPORTED = True
except ImportError:
    _NVML_IMPORTED = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Manufacturer board-power figures (W), used only if NVML power-draw is unavailable.
GPU_TDP_FALLBACK_W = {
    "Tesla T4": 70.0,
    "Tesla P100-PCIE-16GB": 250.0,
}


class PowerSampler:
    """Polls NVML power draw on a background thread during the `with` block. Falls back
    to a constant TDP-based estimate (clearly labeled) if NVML power queries are blocked."""

    def __init__(self, poll_hz: float = 10.0):
        self.poll_hz = poll_hz
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self.nvml_ok = False
        self.fallback_watts = None
        self._handle = None

        if _NVML_IMPORTED and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetPowerUsage(self._handle)  # probe — raises if blocked
                self.nvml_ok = True
            except Exception:
                self.nvml_ok = False

        if not self.nvml_ok and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            self.fallback_watts = GPU_TDP_FALLBACK_W.get(name, 70.0)

    def _poll(self):
        while not self._stop.is_set():
            try:
                watts = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                self._samples.append((time.time(), watts))
            except Exception:
                break
            time.sleep(1.0 / self.poll_hz)

    def __enter__(self):
        self._t0 = time.time()
        if self.nvml_ok:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._t1 = time.time()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def energy_joules(self) -> float:
        duration_s = self._t1 - self._t0
        if self.nvml_ok and len(self._samples) >= 2:
            energy = 0.0
            for (t0, w0), (t1, w1) in zip(self._samples, self._samples[1:]):
                energy += (w0 + w1) / 2.0 * (t1 - t0)
            return energy
        return (self.fallback_watts or 0.0) * duration_s

    @property
    def method(self) -> str:
        return "nvml_measured" if self.nvml_ok else "tdp_estimated"


class ModuleTimer:
    """Forward-hook-based timer per BERT block. `layer.attention` wraps BertSelfAttention
    + BertSelfOutput internally (one hook captures both, no double counting); FFN sums
    intermediate + output since they're siblings, not nested. Syncs CUDA at each hook
    boundary for accurate GPU timing — adds overhead, acceptable for a profiling run."""

    def __init__(self, model):
        self.times = {"self_attention": 0.0, "feed_forward": 0.0, "embedding": 0.0}
        self._starts = {}
        self._handles = []

        if hasattr(model, "distilbert"):
            # DistilBERT's hierarchy differs from BERT's: transformer.layer instead of
            # encoder.layer, and each block exposes .attention + .ffn directly (no
            # separate intermediate/output submodules).
            base = model.distilbert
            self._hook(base.embeddings, "embedding")
            for layer in base.transformer.layer:
                self._hook(layer.attention, "self_attention")
                self._hook(layer.ffn, "feed_forward")
        else:
            base = getattr(model, "bert", model)
            self._hook(base.embeddings, "embedding")
            for layer in base.encoder.layer:
                self._hook(layer.attention, "self_attention")
                self._hook(layer.intermediate, "feed_forward")
                self._hook(layer.output, "feed_forward")

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _hook(self, module, bucket):
        def pre(mod, inp):
            self._sync()
            self._starts[id(mod)] = time.perf_counter()

        def post(mod, inp, out):
            self._sync()
            self.times[bucket] += time.perf_counter() - self._starts[id(mod)]

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def remove(self):
        for h in self._handles:
            h.remove()


def run_profiled_inference(model, input_ids, attention_mask, n_iters: int = 20):
    """Runs n_iters forward passes on the given (already batch/seq-fixed) inputs and
    returns a component-wise time-share/energy breakdown. Reused by p3_compression and
    p4_benchmark so every stage measures the same way."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    timer = ModuleTimer(model)
    try:
        with torch.no_grad(), PowerSampler() as sampler:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                model(input_ids=input_ids, attention_mask=attention_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
    finally:
        timer.remove()

    attn_time = timer.times["self_attention"]
    ffn_time = timer.times["feed_forward"]
    embed_time = timer.times["embedding"]
    other_time = max(total_time - attn_time - ffn_time - embed_time, 0.0)

    energy_j = sampler.energy_joules()

    def share(t):
        return t / total_time if total_time else 0.0

    return {
        "n_iters": n_iters,
        "batch_size": input_ids.shape[0],
        "seq_len": input_ids.shape[1],
        "total_time_s": total_time,
        "energy_j_total": energy_j,
        "energy_method": sampler.method,
        "components": {
            "self_attention": {"time_share": share(attn_time), "energy_j": energy_j * share(attn_time)},
            "feed_forward": {"time_share": share(ffn_time), "energy_j": energy_j * share(ffn_time)},
            "embedding": {"time_share": share(embed_time), "energy_j": energy_j * share(embed_time)},
            "other_overhead": {"time_share": share(other_time), "energy_j": energy_j * share(other_time)},
        },
    }


def _write_table(dataset: str, result: dict):
    out_path = os.path.join(RESULTS_DIR, dataset, "energy_table.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["component", "time_share", "energy_j", "energy_method",
                          "batch_size", "seq_len", "n_iters"])
        for name, vals in result["components"].items():
            writer.writerow([name, vals["time_share"], vals["energy_j"], result["energy_method"],
                              result["batch_size"], result["seq_len"], result["n_iters"]])
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--n_iters", type=int, default=20)
    args = parser.parse_args()

    ckpt_dir = os.path.join(RESULTS_DIR, args.dataset, "best")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
        if PRECISION == "fp16":
            model = model.half()

    val_path = os.path.join(RESULTS_DIR, "data", f"{args.dataset}_val.parquet")
    sample_texts = pd.read_parquet(val_path)["text"].tolist()[:BATCH_SIZE]
    if len(sample_texts) < BATCH_SIZE:
        sample_texts = (sample_texts * (BATCH_SIZE // len(sample_texts) + 1))[:BATCH_SIZE]

    encoded = tokenizer(sample_texts, truncation=True, max_length=MAX_SEQ_LEN,
                         padding="max_length", return_tensors="pt")

    result = run_profiled_inference(model, encoded["input_ids"], encoded["attention_mask"], args.n_iters)
    print(result)
    _write_table(args.dataset, result)


if __name__ == "__main__":
    main()
