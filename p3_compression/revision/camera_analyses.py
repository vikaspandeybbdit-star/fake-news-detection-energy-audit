"""Final-files analyses computed from existing result files (no new training or measurement):
 (1) energy per correct prediction, (2) complete training-pipeline cost and break-even volume,
 (3) paired-bootstrap non-inferiority margins, (4) EBT versus BERT-base on ambiguous examples (accuracy, confidence, ECE).
Writes camera_analyses.json."""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "final_analysis.py"), encoding="utf-8").read()
exec(compile(src.split("def point_mcc")[0], "final_analysis_head", "exec"))  # loaders: pred(), stat(), E, PULL, FIN, DS, SEEDS, GRID

FT = json.load(open(os.path.join(PULL, "FINAL_TABLES.json")))
R3 = json.load(open(os.path.join(HERE, "round3_tables.json")))
EO = json.load(open(os.path.join(HERE, "round3_ebt_ood.json")))
out = {}

# ---------- (1) energy per correct prediction (relative to BERT-base) and absolute mJ per correct prediction
JIT = []
for k in ("a", "b"):
    e = pd.read_csv(os.path.join(PULL, f"energy-unified-{k}", "out", "energy_unified.csv"))
    JIT.append(e[e.n_iters == 200].set_index("config").j_per_iter)
JIT = pd.concat(JIT, axis=1).mean(axis=1)
BS = 16
FTKEY = {"bert_base": "bert_base", "bert_prune25": "bert_prune25", "bert_prune50": "bert_prune50", "bert_prune75": "bert_prune75",
         "distil_prune0": "distil_prune0", "distil_prune25": "distil_prune25", "distil_prune50": "distil_prune50", "distil_prune75": "distil_prune75",
         "tinybert": "suite_tinybert", "bert_len64": "suite_bert_len64", "ebt": "ebt_T3", "snn": "spikebert", "dense_preln": "dense_preln"}
epc = {}
for m, ft in FTKEY.items():
    epc[m] = {}
    for ds in DS:
        acc = FT[ft][ds]["acc"]
        mj = JIT[CFGNAME[m]] / BS * 1000.0 / acc
        accb = FT["bert_base"][ds]["acc"]
        epc[m][ds] = {"acc": acc, "mj_per_correct": mj, "rel_per_correct": (E[m] / acc) / (1.0 / accb)}
out["energy_per_correct"] = epc
out["jpi_bert_base_mj_per_example"] = JIT[CFGNAME["bert_base"]] / BS * 1000.0

# ---------- (2) complete pipeline cost
tr = R3["train_rel"]
bt = EO["bert_train"]  # kJ and seconds of a complete BERT-base fine-tuning run per dataset
mj_bert = JIT[CFGNAME["bert_base"]] / BS * 1000.0  # mJ per example, BERT-base inference
pipe = {}
for ds in DS:
    teacher = bt[ds]["kj"]
    distil = tr["DistilBERT distillation (with teacher)"] * teacher            # same step count as the teacher fine-tune
    tiny = tr["TinyBERT-4L distillation (with teacher)"] * teacher
    recov = tr["BERT-base prune 75% (recovery)"] * teacher / 3.0               # one recovery epoch (3-epoch fine-tune = 3 units)
    cfgs = {
        "BERT-base fine-tuning (teacher)": (teacher, 1.0),
        "DistilBERT distillation": (teacher + distil, E["distil_prune0"]),
        "DistilBERT + 75% pruning": (teacher + distil + recov, E["distil_prune75"]),
        "TinyBERT-4L distillation": (teacher + tiny, E["tinybert"]),
        "BERT-base 64 tokens (no extra stage)": (tr["BERT-base, 64 tokens"] * teacher, E["bert_len64"]),
    }
    rows = {}
    for name, (kj, relE) in cfgs.items():
        saving_mj = mj_bert * (1.0 - relE)                                       # mJ saved per prediction vs BERT-base
        extra_kj = kj - teacher
        rows[name] = {"train_kj": kj, "extra_kj_vs_teacher_only": extra_kj, "rel_inference_energy": relE,
                      "break_even_predictions_million": (extra_kj * 1e6 / saving_mj / 1e6) if (saving_mj > 0 and extra_kj > 0) else None}
    pipe[ds] = rows
out["pipeline"] = pipe

# ---------- (3) non-inferiority of compressed configs against BERT-base (mean over seeds of paired MCC difference)
NB = 4000
rg = np.random.default_rng(1)
non = {}
for ds in DS:
    first = pred("suite_bert_base", ds, 42)
    n = len(first)
    idx = rg.integers(0, n, size=(NB, n))
    W = np.zeros((NB, n))
    for i in range(NB):
        W[i] = np.bincount(idx[i], minlength=n)
    base = {s: stat(W, pred("bert_base", ds, s)) for s in SEEDS}
    base_suite = {s: stat(W, pred("suite_bert_base", ds, s)) for s in SEEDS}  # paired reference of TinyBERT, 64 tokens, INT8 (as in Table 5)
    non[ds] = {}
    for m in ["bert_prune25", "bert_prune50", "bert_prune75", "distil_prune0", "distil_prune25", "distil_prune50", "distil_prune75", "tinybert", "bert_len64", "quant_int8"]:
        ref = base_suite if m in ("tinybert", "bert_len64", "quant_int8") else base
        diffs, ok = [], True
        for s in SEEDS:
            p = pred(m, ds, s)
            if p is None or len(p) != n:
                ok = False
                break
            diffs.append(stat(W, p) - ref[s])
        if not ok:
            continue
        d = np.mean(diffs, axis=0)
        non[ds][m] = {"mean_diff": float(d.mean()), "lo95_one_sided": float(np.percentile(d, 5)), "lo95_two_sided": float(np.percentile(d, 2.5)),
                      "hi95_two_sided": float(np.percentile(d, 97.5))}
out["noninferiority"] = non

# ---------- (4) EBT versus BERT-base on ambiguous examples (examples where BERT-base is near its decision boundary)


def ece(p, y, bins=10):
    conf = np.maximum(p, 1 - p)
    pred_ = (p >= 0.5).astype(int)
    corr = (pred_ == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


ebt = {}
for ds in ("liar", "fakenewsnet"):
    rows = {"hard": [], "easy": [], "all": []}
    for s in SEEDS:
        b = pred("suite_bert_base", ds, s)
        e = pred("ebt", ds, s)
        if b is None or e is None or "p_true" not in b or "p_true" not in e or len(b) != len(e):
            continue
        y = b["label"].to_numpy().astype(int)
        pb, pe = b["p_true"].to_numpy(), e["p_true"].to_numpy()
        hard = np.abs(pb - 0.5) < 0.2            # BERT-base probability within 0.2 of the boundary
        for name, msk in (("hard", hard), ("easy", ~hard), ("all", np.ones(len(y), bool))):
            rows[name].append({"seed": s, "n": int(msk.sum()),
                               "acc_bert": float(((pb[msk] >= 0.5) == y[msk]).mean()), "acc_ebt": float(((pe[msk] >= 0.5) == y[msk]).mean()),
                               "conf_bert": float(np.maximum(pb[msk], 1 - pb[msk]).mean()), "conf_ebt": float(np.maximum(pe[msk], 1 - pe[msk]).mean()),
                               "ece_bert": ece(pb[msk], y[msk]), "ece_ebt": ece(pe[msk], y[msk])})
    ebt[ds] = {k: {f: float(np.mean([r[f] for r in v])) for f in v[0] if f != "seed"} for k, v in rows.items() if v}
out["ebt_ambiguous"] = ebt

json.dump(out, open(os.path.join(HERE, "camera_analyses.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("pipeline",)}, indent=1)[:2500])
print(json.dumps(out["noninferiority"], indent=1)[:3000])
print(json.dumps(out["ebt_ambiguous"], indent=1))
print({m: {ds: round(v["rel_per_correct"], 3) for ds, v in d.items()} for m, d in epc.items()})
