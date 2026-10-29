"""95% bootstrap confidence intervals and bootstrap Pareto-frontier stability from saved per-example test predictions
(IEEE Access R2#2). Test examples are resampled with replacement (B draws, shared across models so differences are paired);
for each draw the statistic of a model is the mean over its seeds. Energy ordering of the 8 grid configurations as in
pareto_stability.py. Prints a summary and LaTeX rows."""
import glob
import os
import statistics as st

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN = os.path.join(ROOT, "results_rev_final")
PULL = os.path.join(ROOT, "results_rev_altarch_pull")
B = 4000
rng = np.random.default_rng(0)
CFG = ["bert_base", "bert_prune25", "bert_prune50", "bert_prune75", "distil_prune0", "distil_prune25", "distil_prune50", "distil_prune75"]
ENERGY = {  # J, paper grid ordering
    "liar": [486.92, 478.50, 404.58, 381.42, 246.24, 225.17, 200.07, 177.46],
    "isot": [491.82, 458.93, 387.95, 368.70, 237.01, 215.09, 190.30, 166.66],
    "fakenewsnet": [487.00, 467.59, 414.60, 368.79, 239.11, 218.45, 193.81, 171.58],
}
GRID_DIRS = {"liar": ["rev-liar"], "isot": ["rev-isota", "rev-isotb"], "fakenewsnet": ["rev-fnn"]}


def grid_pred(ds, seed, cfg):
    for d in GRID_DIRS[ds]:
        for f in glob.glob(os.path.join(FIN, f"{d}-s{seed}", "**", f"preds_{cfg}_test.csv"), recursive=True):
            return pd.read_csv(f)
    return None


def first(pattern):
    for f in glob.glob(os.path.join(PULL, pattern), recursive=True):
        if "altarch-smoke" not in f:
            return pd.read_csv(f)
    return None


def mcc_from_counts(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def stat(W, df, kind="mcc"):
    y = df["label"].to_numpy().astype(int)
    p = df["pred"].to_numpy().astype(int)
    tp = W @ ((y == 1) & (p == 1)).astype(float)
    tn = W @ ((y == 0) & (p == 0)).astype(float)
    fp = W @ ((y == 0) & (p == 1)).astype(float)
    fn = W @ ((y == 1) & (p == 0)).astype(float)
    return mcc_from_counts(tp, tn, fp, fn) if kind == "mcc" else (tp + tn) / (tp + tn + fp + fn)


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def fmt(x):
    lo, hi = ci(x)
    return f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}]"


BS = chr(92)
latex_rows = []
import json
RES = {}
for ds in ("liar", "isot", "fakenewsnet"):
    dfs = {c: [grid_pred(ds, s, c) for s in (42, 43, 44)] for c in CFG}
    n = len(dfs["bert_base"][0])
    W = rng.multinomial(n, np.ones(n) / n, size=B).astype(float)
    mcc = {c: np.mean([stat(W, d) for d in dfs[c]], axis=0) for c in CFG}
    ebt = [first(f"altarch-ebt-{ds}-s{s}*/out/ebt/{ds}/s{s}/preds_test.csv") for s in (42, 43, 44)]
    snn = [first(f"altarch-snn-{ds}-s{s}*/out/snn/spiking/{ds}/s{s}/preds_test.csv") for s in (42, 43, 44)]
    tiny = first(f"altarch-tinybert-{ds}/out/tinybert_raw/{ds}/s42/preds_tinybert_test.csv")
    assert all(len(d) == n for d in ebt + snn), (ds, [len(d) for d in ebt + snn], n)
    mcc["ebt"] = np.mean([stat(W, d) for d in ebt], axis=0)
    mcc["snn"] = np.mean([stat(W, d) for d in snn], axis=0)
    mcc["tiny"] = stat(W, tiny)
    print(f"\n== {ds} (n_test={n}, B={B}): 3-seed mean test MCC [95% bootstrap CI]")
    for c in CFG + ["ebt", "snn", "tiny"]:
        print(f"  {c:16s} {fmt(mcc[c])}")
    print("  paired differences vs BERT-base (mean over seeds):")
    for c, lab in (("distil_prune75", "Distill+75% prune"), ("distil_prune0", "Distill 0%"), ("ebt", "EBT"), ("snn", "SpikeBERT"), ("tiny", "TinyBERT (seed 42)")):
        d = mcc[c] - mcc["bert_base"]
        print(f"    {lab:22s} {fmt(d)}")
    # frontier frequency under bootstrap
    E = np.array(ENERGY[ds])
    M = np.stack([mcc[c] for c in CFG], axis=1)  # B x 8
    freq = np.zeros(8)
    for b in range(B):
        for i in range(8):
            dom = np.any((E <= E[i]) & (M[b] >= M[b, i]) & ((E < E[i]) | (M[b] > M[b, i])))
            freq[i] += (not dom)
    freq /= B
    print("  P(on frontier):", {c: round(f, 2) for c, f in zip(CFG, freq)})
    latex_rows.append((ds, {c: fmt(mcc[c]) for c in CFG + ["ebt", "snn", "tiny"]}, freq))
    RES[ds] = {"n": n, "mcc": {c: [float(mcc[c].mean())] + [float(v) for v in ci(mcc[c])] for c in CFG + ["ebt", "snn", "tiny"]},
               "diff": {c: [float((mcc[c] - mcc["bert_base"]).mean())] + [float(v) for v in ci(mcc[c] - mcc["bert_base"])] for c in ("distil_prune75", "distil_prune0", "ebt", "snn", "tiny")},
               "freq": {c: float(f) for c, f in zip(CFG, freq)}}

np.save(os.path.join(ROOT, "revision", "_bootstrap_freq.npy"), np.array([f for _, _, f in latex_rows]))

json.dump(RES, open(os.path.join(ROOT, "revision", "bootstrap_results.json"), "w"), indent=1)
