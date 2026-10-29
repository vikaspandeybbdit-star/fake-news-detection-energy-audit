"""Final analysis for the consistency rebuild: single-session relative energy, 3-seed test-split MCC, per-seed Pareto
frontier membership, and paired 95% bootstrap intervals (4000 resamples of the test split, shared across models).
Writes final_analysis.json (used by the figure and table builders)."""
import glob
import json
import os
import statistics as st

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN = os.path.join(ROOT, "results_rev_final")
PULL = os.path.join(ROOT, "results_rev_altarch_pull")
B = 4000
rng = np.random.default_rng(0)
DS = ["liar", "isot", "fakenewsnet"]
SEEDS = [42, 43, 44]
GRID_DIRS = {"liar": ["rev-liar"], "isot": ["rev-isota", "rev-isotb"], "fakenewsnet": ["rev-fnn"]}
GRID = ["bert_base", "bert_prune25", "bert_prune50", "bert_prune75", "distil_prune0", "distil_prune25", "distil_prune50", "distil_prune75"]
SUITE = ["tinybert", "bert_len64"]
FRONT = GRID + SUITE

# relative energy to BERT-base, mean of two single-session replicates (200-iteration window)
CFGNAME = {"bert_base": "BERT-base prune 0%", "bert_prune25": "BERT-base prune 25%", "bert_prune50": "BERT-base prune 50%",
           "bert_prune75": "BERT-base prune 75%", "distil_prune0": "DistilBERT prune 0%", "distil_prune25": "DistilBERT prune 25%",
           "distil_prune50": "DistilBERT prune 50%", "distil_prune75": "DistilBERT prune 75%", "tinybert": "TinyBERT-4L",
           "bert_len64": "BERT-base 64 tokens", "ebt": "EBT T=3 FP16", "snn": "SpikeBERT T=16 FP32", "dense_preln": "Dense control FP32"}
rel = []
for k in ("a", "b"):
    e = pd.read_csv(os.path.join(PULL, f"energy-unified-{k}", "out", "energy_unified.csv"))
    e = e[e.n_iters == 200].set_index("config").j_per_iter
    rel.append(e / e["BERT-base prune 0%"])
REL = pd.concat(rel, axis=1).mean(axis=1).to_dict()
E = {m: REL[n] for m, n in CFGNAME.items()}


def rd(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def pred(model, ds, seed):
    if model in GRID:
        for d in GRID_DIRS[ds]:
            for f in glob.glob(os.path.join(FIN, f"{d}-s{seed}", "**", f"preds_{model}_test.csv"), recursive=True):
                return pd.read_csv(f)
    if model in SUITE or model == "suite_bert_base" or model == "quant_int8":
        name = {"tinybert": "tinybert", "bert_len64": "bert_len64", "suite_bert_base": "bert_base", "quant_int8": "quant_int8"}[model]
        fn = {"quant_int8": "preds_quant_int8_test.csv"}.get(model, f"preds_{name}_test.csv")
        return rd(os.path.join(PULL, f"suite-{ds}-s{seed}", "out", "suite_raw", ds, f"s{seed}", fn))
    if model == "ebt":
        for f in glob.glob(os.path.join(PULL, f"altarch-ebt-{ds}-s{seed}", "out", "ebt", ds, f"s{seed}", "preds_test.csv")):
            return pd.read_csv(f)
    if model == "snn":
        for f in glob.glob(os.path.join(PULL, f"altarch-snn-{ds}-s{seed}*", "out", "snn", "spiking", ds, f"s{seed}", "preds_test.csv")):
            return pd.read_csv(f)
    if model == "dense_preln":
        for f in glob.glob(os.path.join(PULL, f"densemulti-{ds}-s{seed}", "out", "snn", "dense_preln", ds, f"s{seed}", "preds_test.csv")) + \
                glob.glob(os.path.join(PULL, f"altarch-snn-dense-preln-{ds}", "out", "snn", "dense_preln", ds, f"s{seed}", "preds_test.csv")):
            return pd.read_csv(f)
    return None


def mcc_from_counts(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)


def stat(W, df):
    y = df["label"].to_numpy().astype(int)
    p = df["pred"].to_numpy().astype(int)
    tp = W @ ((y == 1) & (p == 1)).astype(float)
    tn = W @ ((y == 0) & (p == 0)).astype(float)
    fp = W @ ((y == 0) & (p == 1)).astype(float)
    fn = W @ ((y == 1) & (p == 0)).astype(float)
    return mcc_from_counts(tp, tn, fp, fn)


def point_mcc(df):
    return float(stat(np.ones((1, len(df))), df)[0])


def ci(x):
    return [float(x.mean()), float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def frontier(mcc, keys):
    out = set()
    for a in keys:
        if not any(E[b] <= E[a] and mcc[b] >= mcc[a] and (E[b] < E[a] or mcc[b] > mcc[a]) for b in keys if b != a):
            out.add(a)
    return out


ALL = FRONT + ["quant_int8", "suite_bert_base", "ebt", "snn", "dense_preln"]
OUT = {"rel_energy": E, "rel_by_config": REL}
for ds in DS:
    P = {m: [pred(m, ds, s) for s in SEEDS] for m in ALL}
    miss = [m for m in ALL if any(x is None for x in P[m])]
    assert not miss, (ds, miss)
    n = len(P["bert_base"][0])
    for m in ALL:
        assert all(len(x) == n for x in P[m]), (ds, m)
    # per-seed point MCC and frontier membership
    pm = {m: [point_mcc(x) for x in P[m]] for m in ALL}
    fr = {s: frontier({m: pm[m][i] for m in FRONT}, FRONT) for i, s in enumerate(SEEDS)}
    meanfr = frontier({m: st.mean(pm[m]) for m in FRONT}, FRONT)
    W = rng.multinomial(n, np.ones(n) / n, size=B).astype(float)
    boot = {m: np.mean([stat(W, x) for x in P[m]], axis=0) for m in ALL}
    # bootstrap frontier probability
    M = np.stack([boot[m] for m in FRONT], axis=1)
    en = np.array([E[m] for m in FRONT])
    freq = np.zeros(len(FRONT))
    for b in range(B):
        for i in range(len(FRONT)):
            dom = np.any((en <= en[i]) & (M[b] >= M[b, i]) & ((en < en[i]) | (M[b] > M[b, i])))
            freq[i] += not dom
    freq /= B
    pairs = {"distil_prune75": "bert_base", "distil_prune0": "bert_base", "bert_prune75": "bert_base", "ebt": "bert_base", "snn": "bert_base",
             "dense_preln": "bert_base", "tinybert": "suite_bert_base", "bert_len64": "suite_bert_base", "quant_int8": "suite_bert_base"}
    OUT[ds] = {
        "n": n,
        "mcc_mean": {m: st.mean(pm[m]) for m in ALL}, "mcc_sd": {m: st.stdev(pm[m]) for m in ALL}, "mcc_seeds": pm,
        "boot": {m: ci(boot[m]) for m in ALL},
        "diff": {m: ci(boot[m] - boot[r]) | {} if False else ci(boot[m] - boot[r]) for m, r in pairs.items()},
        "diff_ref": pairs,
        "frontier_seeds": {m: sum(m in fr[s] for s in SEEDS) for m in FRONT},
        "frontier_mean": sorted(meanfr),
        "frontier_boot": {m: float(f) for m, f in zip(FRONT, freq)},
    }
    print(f"== {ds} n={n}")
    for m in FRONT:
        print(f"  {m:15s} E={E[m]:.3f} MCC={OUT[ds]['mcc_mean'][m]:.3f}+-{OUT[ds]['mcc_sd'][m]:.3f} seeds-on-frontier={OUT[ds]['frontier_seeds'][m]}/3 P(front)={freq[FRONT.index(m)]:.2f}")
    for m, r in pairs.items():
        d = OUT[ds]["diff"][m]
        print(f"  diff {m:15s} vs {r:16s} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")

json.dump(OUT, open(os.path.join(ROOT, "revision", "final_analysis.json"), "w"), indent=1)
