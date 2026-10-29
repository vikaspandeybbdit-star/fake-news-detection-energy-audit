"""Controlled energy session v2 (JournalPaper2 Stage-3 review response).

Changes vs reprofile_energy.py, each answering a reviewer point:
- NO timing hooks during energy measurement (hooks synchronise the GPU at every block);
  attention time share is taken in a separate hooked pass ("share" rows).
- Baseline BERT measured with both the fused library attention (sdpa) and eager attention,
  so the retrofits (unfused, eager-style ops) have a like-for-like control.
- Performer measured as trained (feature map in FP32) AND fully in FP16 (energy only).
- Energy from the GPU's hardware energy counter (nvmlDeviceGetTotalEnergyConsumption),
  cross-checked against the 10 Hz power-sample integral; latency, mean power, SM clock and
  temperature recorded; one burn-in round discarded; library versions recorded.
- Each measurement runs for ~10 s of GPU work (iterations chosen per architecture).
- Sweeps: sequence length n in {128,256,512,1024,2048} (batch 16) and batch size
  b in {1,4,16,64} (n=256); dynamic padding on real validation-length distributions.

Weights are untrained (inference cost does not depend on weight values; models are
built by the same code as the trained ones).

Run: python reprofile_energy_v2.py [--repeats 10 --sweep_repeats 5] [--smoke]
Writes: results/reprofile_energy_v2.csv and results/reprofile_energy_v2_env.json
"""
import argparse
import csv
import gc
import json
import math
import os
import platform
import random
import sys
import threading
import time

import pandas as pd  # before torch (Windows pyarrow/torch DLL clash)
import torch
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
for p in (os.path.join(P3, "..", "p1_baseline"), P3, os.path.join(P3, "combined_framework"),
          os.path.join(P3, "linear_attention", "linformer"), os.path.join(P3, "linear_attention", "performer"),
          os.path.join(P3, "linear_attention", "charformer"), HERE):
    sys.path.insert(0, p)

from profile_energy import ModuleTimer  # noqa: E402
from prune import heads_to_prune_by_magnitude, prune_layer_heads  # noqa: E402
from combine import prune_distilbert_heads  # noqa: E402
from adapt_linformer import linformerize  # noqa: E402
from adapt_performer import performerize, PerformerSelfAttention, softmax_kernel, linear_attention  # noqa: E402
from adapt_charformer import CharformerClassifier, encode_bytes  # noqa: E402
from run_seed import meanattnize  # noqa: E402

import pynvml  # noqa: E402

DATA_DIR = os.path.join(P3, "..", "p1_baseline", "results", "data")
OUT_DIR = os.path.join(P3, "results")
SEED = 42
TARGET_S = 10.0


# ---------------------------------------------------------------- models
class PerformerFP16(PerformerSelfAttention):
    """Same mechanism without the FP32 island (energy-only variant; may be numerically
    unstable for training, which is why the trained model keeps FP32)."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        shp = (*hidden_states.shape[:-1], self.num_attention_heads, self.attention_head_size)
        q = self.query(hidden_states).view(shp).transpose(1, 2)
        k = self.key(hidden_states).view(shp).transpose(1, 2)
        v = self.value(hidden_states).view(shp).transpose(1, 2)
        qp = softmax_kernel(q, self.projection_matrix, is_query=True)
        kp = softmax_kernel(k, self.projection_matrix, is_query=False)
        if self.pad_mask_box.mask is not None:
            m = self.pad_mask_box.mask[:, None, :, None]
            kp, v = kp.masked_fill(m, 0.0), v.masked_fill(m, 0.0)
        out = linear_attention(qp, kp, v)
        return out.transpose(1, 2).reshape(*hidden_states.shape[:-1], self.all_head_size).contiguous(), None


def bert(attn="sdpa", n_max=512):
    cfg = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2,
                                     max_position_embeddings=max(512, n_max))
    return AutoModelForSequenceClassification.from_config(cfg, attn_implementation=attn)


def build(arch, n_max=512):
    if arch in ("bert_sdpa", "bert_eager"):
        return bert(arch.split("_")[1], n_max)
    if arch.startswith("bert_prune"):
        m = bert("sdpa", n_max)
        r = int(arch[len("bert_prune"):]) / 100
        for layer in m.bert.encoder.layer:
            prune_layer_heads(layer, heads_to_prune_by_magnitude(layer, r))
        return m
    if arch.startswith("distil_prune"):
        m = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(
            "distilbert-base-uncased", num_labels=2, max_position_embeddings=max(512, n_max)))
        r = int(arch[len("distil_prune"):]) / 100
        return prune_distilbert_heads(m, r) if r > 0 else m
    if arch == "linformer":
        m = bert("eager", n_max)
        linformerize(m, proj_k=64, max_seq_len=n_max)
        return m
    if arch in ("performer", "performer_fp16"):
        m = bert("eager", n_max)
        box = performerize(m, nb_features=64)
        if arch == "performer_fp16":
            for layer in m.bert.encoder.layer:
                old = layer.attention.self
                new = PerformerFP16.__new__(PerformerFP16)
                new.__dict__ = old.__dict__  # same submodules and buffer, different forward
                layer.attention.self = new
        return m
    if arch == "meanattn":
        m = bert("eager", n_max)
        meanattnize(m)
        return m
    if arch == "charformer":
        return CharformerClassifier()
    raise ValueError(arch)


# ---------------------------------------------------------------- measurement
class Meter:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.counter = True
        except Exception:
            self.counter = False

    def __enter__(self):
        self.samples, self.stop = [], threading.Event()

        def poll():
            while not self.stop.is_set():
                self.samples.append((time.perf_counter(), pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                                     pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                                     pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU)))
                time.sleep(0.1)

        torch.cuda.synchronize()
        self.e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.counter else None
        self.th = threading.Thread(target=poll, daemon=True)
        self.th.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.t1 = time.perf_counter()
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h) if self.counter else None
        self.stop.set()
        self.th.join()
        s = self.samples
        mean_w = sum(x[1] for x in s) / len(s)
        self.result = dict(
            time_s=self.t1 - self.t0,
            energy_counter_j=(e1 - self.e0) / 1000.0 if self.counter else float("nan"),
            energy_sampled_j=mean_w * (self.t1 - self.t0),  # mean power x full window (edge-corrected)
            mean_power_w=mean_w, mean_sm_clock_mhz=sum(x[2] for x in s) / len(s),
            max_temp_c=max(x[3] for x in s), n_power_samples=len(s))


def forward_fn(arch, model, batch):
    if arch == "charformer":
        return lambda: model(batch["bytes"], batch["byte_mask"])
    return lambda: model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])


def calibrate(fn):
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / 3
    return max(20, math.ceil(TARGET_S / per))


def measure(meter, fn, iters):
    with torch.no_grad(), meter:
        for _ in range(iters):
            fn()
    return meter.result


def attn_share(model, batch):
    timer = ModuleTimer(model)
    try:
        with torch.no_grad():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            torch.cuda.synchronize()
        total = time.perf_counter() - t0
    finally:
        timer.remove()
    return timer.times["self_attention"] / total, timer.times["feed_forward"] / total


# ---------------------------------------------------------------- inputs
def texts(ds, k):
    p = os.path.join(DATA_DIR, f"{ds}_val.parquet")
    return pd.read_parquet(p)["text"].tolist()[:k] if os.path.exists(p) else None


def make_batch(tok, txt, n, b, pad="max_length"):
    txt = (txt * (b // len(txt) + 1))[:b]
    enc = tok(txt, truncation=True, max_length=n, padding=pad, return_tensors="pt")
    if pad == "max_length" and n > 512:  # untrained long-context models: tile real tokens to n
        enc = tok(txt, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        rep = math.ceil(n / 512)
        enc = {k: v.repeat(1, rep)[:, :n] for k, v in enc.items()}
    by, bm = encode_bytes(txt, 4 * n)
    return {"input_ids": enc["input_ids"].cuda(), "attention_mask": enc["attention_mask"].cuda(),
            "bytes": by.cuda(), "byte_mask": bm.cuda()}


# ---------------------------------------------------------------- main
MAIN = ["bert_sdpa", "bert_eager", "linformer", "performer", "performer_fp16", "meanattn",
        "bert_prune25", "bert_prune50", "bert_prune75",
        "distil_prune0", "distil_prune25", "distil_prune50", "distil_prune75", "charformer"]
SEQ_ARCHS = ["bert_sdpa", "bert_eager", "linformer", "performer", "performer_fp16", "distil_prune75"]
SEQS = [128, 256, 512, 1024, 2048]
BATCH_ARCHS = ["bert_sdpa", "linformer", "performer", "distil_prune75"]
BATCHES = [1, 4, 16, 64]
DYN_ARCHS = ["bert_sdpa", "distil_prune0", "distil_prune75"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--sweep_repeats", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    global TARGET_S
    if a.smoke:
        TARGET_S, a.repeats, a.sweep_repeats = 0.5, 1, 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    meter = Meter()
    base_txt = texts("liar", 16)
    env = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, transformers=transformers.__version__,
               cuda=torch.version.cuda, driver=pynvml.nvmlSystemGetDriverVersion(), python=platform.python_version(),
               power_limit_w=pynvml.nvmlDeviceGetEnforcedPowerLimit(meter.h) / 1000.0, energy_counter=meter.counter,
               attn_impl_bert_sdpa=build("bert_sdpa").config._attn_implementation,
               attn_impl_bert_eager=build("bert_eager").config._attn_implementation)
    json.dump(env, open(os.path.join(OUT_DIR, "reprofile_energy_v2_env.json"), "w"), indent=2)
    print(env, flush=True)

    # (study, arch, n, b, padding-dataset)
    jobs = [("main", x, 256, 16, None) for x in MAIN]
    jobs += [("seq", x, n, 16, None) for x in SEQ_ARCHS for n in SEQS if n != 256]
    jobs += [("batch", x, 256, b, None) for x in BATCH_ARCHS for b in BATCHES if b != 16]
    jobs += [("dynamic", x, 256, 16, ds) for x in DYN_ARCHS for ds in ("liar", "fakenewsnet", "isot") if texts(ds, 1)]
    if a.smoke:
        jobs = [j for j in jobs if j[0] == "main"] + [("seq", "bert_eager", 2048, 16, None), ("seq", "linformer", 2048, 16, None),
                                                     ("batch", "performer_fp16", 256, 64, None)] + [j for j in jobs if j[0] == "dynamic"][:1]

    rows = []
    out = os.path.join(OUT_DIR, "reprofile_energy_v2.csv")

    def run_round(r, burn_in=False):
        order = jobs[:]
        random.shuffle(order)
        for study, arch, n, b, ds in order:
            reps = a.repeats if study == "main" else a.sweep_repeats
            if r >= reps and not burn_in:
                continue
            torch.manual_seed(SEED)
            model = build(arch, n_max=n).cuda().half().eval()
            if ds:  # dynamic padding: 20 consecutive batches of real texts, padded to the batch's longest
                txt = texts(ds, 16 * 20)
                batches = [make_batch(tok, txt[i:i + 16], n, 16, pad="longest") for i in range(0, len(txt), 16)]
                fns = [forward_fn(arch, model, bt) for bt in batches]
                fn = lambda: [f() for f in fns]  # noqa: E731
                per_call = 16 * len(batches)
            else:
                batch = make_batch(tok, base_txt, n, b)
                fn = forward_fn(arch, model, batch)
                per_call = b
            iters = calibrate(fn)
            res = measure(meter, fn, iters)
            row = dict(study=study, arch=arch, seq_len=n, batch=b, dataset=ds or "", repeat=r, burn_in=burn_in,
                       iters=iters, samples=iters * per_call, **res,
                       energy_j_per_320=res["energy_counter_j"] * 320 / (iters * per_call) if meter.counter
                       else res["energy_sampled_j"] * 320 / (iters * per_call),
                       latency_ms_per_sample=1000 * res["time_s"] / (iters * per_call),
                       n_params=sum(p.numel() for p in model.parameters()), attn_share="", ffn_share="")
            if study == "main" and r == 0 and not burn_in and arch != "charformer":
                row["attn_share"], row["ffn_share"] = attn_share(model, batch)
            rows.append(row)
            print(f"r{r}{'B' if burn_in else ''} {study:7s} {arch:15s} n={n:4d} b={b:2d} {ds or '':11s} "
                  f"E/320={row['energy_j_per_320']:8.2f} J P={res['mean_power_w']:5.1f} W "
                  f"lat={row['latency_ms_per_sample']:6.2f} ms", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out, index=False)

    if not a.smoke:
        run_round(-1, burn_in=True)
    for r in range(max(a.repeats, a.sweep_repeats)):
        run_round(r)
    print("wrote", out, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
