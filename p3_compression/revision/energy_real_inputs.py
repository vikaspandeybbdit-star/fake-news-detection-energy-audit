"""Energy on REAL test inputs under four padding policies (answers: is the 64-token saving a padding effect on synthetic inputs?).
Models (inference, FP16, batch 16): BERT-base, DistilBERT, TinyBERT-4L (architecture only; energy does not depend on trained weights).
Policies: pad256 = every input padded to 256 (the paper's protocol); trunc64 = truncated and padded to 64; dyn = truncated to 256 and
padded to the longest input of each batch (dataset order); sorted = same but inputs first sorted by length (length bucketing).
Per (dataset, model, policy): 100 batches of real test examples (fixed sample), 10 s warm-up, 5 timed passes over all batches, NVML energy.
One session, fixed shuffled order, idle power before/after. Writes /kaggle/working/out/energy_real.csv, token_lengths.csv, idle_power.txt."""
import csv
import gc
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.dirname(HERE)
for p in (os.path.join(P3, "..", "p1_baseline"), P3, HERE):
    sys.path.insert(0, p)

from profile_energy import PowerSampler  # noqa: E402
from run_seed import DATA_DIR  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

DEV = "cuda"
B, NB = 16, 100
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)
random.seed(7)
MODELS = [("BERT-base", "bert-base-uncased"), ("DistilBERT", "distilbert-base-uncased"), ("TinyBERT-4L", "huawei-noah/TinyBERT_General_4L_312D")]
DATASETS = ["liar", "fakenewsnet", "isot"]
POLICIES = ["pad256", "trunc64", "dyn", "sorted"]


def idle_power(seconds=10):
    with PowerSampler() as s:
        time.sleep(seconds)
    return s.energy_joules() / seconds


idle_before = idle_power()
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
rng = np.random.default_rng(0)

# ---- tokenized samples and batches per (dataset, policy)
lens_rows, batches = [], {}
for ds in DATASETS:
    df = pd.read_parquet(os.path.join(DATA_DIR, f"{ds}_test.parquet"))
    df = df.iloc[rng.permutation(len(df))[: B * NB]].reset_index(drop=True)
    ids_all = tok(df["text"].astype(str).tolist(), truncation=True, max_length=256)["input_ids"]
    L = np.array([len(x) for x in ids_all])
    full = [len(x) for x in tok(df["text"].astype(str).tolist(), truncation=False)["input_ids"]]
    lens_rows.append(dict(dataset=ds, n=len(df), mean_len=float(np.mean(full)), median_len=float(np.median(full)), p95_len=float(np.percentile(full, 95)),
                          share_over_64=float(np.mean(np.array(full) > 64)), share_over_256=float(np.mean(np.array(full) > 256))))

    def make(order, pad_to=None, trunc=256):
        out = []
        for i in range(0, len(order), B):
            idx = order[i:i + B]
            seqs = [ids_all[j][:trunc] for j in idx]
            m = pad_to if pad_to else max(len(s) for s in seqs)
            ids = torch.zeros(len(seqs), m, dtype=torch.long)
            mask = torch.zeros(len(seqs), m, dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, :len(s)] = torch.tensor(s)
                mask[r, :len(s)] = 1
            out.append((ids.to(DEV), mask.to(DEV)))
        return out

    natural = list(range(len(df)))
    by_len = list(np.argsort(L))
    batches[(ds, "pad256")] = make(natural, pad_to=256)
    batches[(ds, "trunc64")] = make(natural, pad_to=64, trunc=64)
    batches[(ds, "dyn")] = make(natural)
    batches[(ds, "sorted")] = make(by_len)
pd.DataFrame(lens_rows).to_csv(os.path.join(OUT, "token_lengths.csv"), index=False)
print(pd.DataFrame(lens_rows), flush=True)

combos = [(ds, mn, ck, pol) for ds in DATASETS for mn, ck in MODELS for pol in POLICIES]
random.shuffle(combos)
rows = []
cur_model, cur_name = None, None
combos.sort(key=lambda c: c[2])  # group by model to avoid reloading; order inside a model stays shuffled
for ds, mn, ck, pol in combos:
    if cur_name != ck:
        del cur_model
        gc.collect()
        torch.cuda.empty_cache()
        cur_model = AutoModelForSequenceClassification.from_pretrained(ck, num_labels=2).to(DEV).half().eval()
        cur_name = ck
    bl = batches[(ds, pol)]
    with torch.no_grad():
        t0 = time.time()
        while time.time() - t0 < 10:
            for ids, mask in bl[:20]:
                cur_model(input_ids=ids, attention_mask=mask)
        torch.cuda.synchronize()
        e, t = [], []
        for _ in range(5):
            with PowerSampler() as s:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                for ids, mask in bl:
                    cur_model(input_ids=ids, attention_mask=mask)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t1
            e.append(s.energy_joules())
            t.append(dt)
    n_ex = sum(x[0].shape[0] for x in bl)
    mean_pad = float(np.mean([x[0].shape[1] for x in bl]))
    rows.append(dict(dataset=ds, model=mn, policy=pol, n_examples=n_ex, mean_padded_len=round(mean_pad, 1),
                     j_per_1000_ex=round(float(np.mean(e)) / n_ex * 1000, 3), j_sd_pct=round(float(np.std(e) / np.mean(e) * 100), 2),
                     s_per_1000_ex=round(float(np.mean(t)) / n_ex * 1000, 4)))
    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "energy_real.csv"), index=False)

idle_after = idle_power()
open(os.path.join(OUT, "idle_power.txt"), "w").write(f"idle_before_w={idle_before:.2f}\nidle_after_w={idle_after:.2f}\n")
print("done; idle", idle_before, idle_after)
