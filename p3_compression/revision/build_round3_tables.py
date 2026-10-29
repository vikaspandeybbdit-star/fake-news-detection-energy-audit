"""LaTeX table bodies and summary numbers for the round-3 experiments (training energy, real-input energy)."""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULL = os.path.join(ROOT, "results_rev_altarch_pull")
BS = chr(92)
PC = BS + "%"
out = {}

# ---- training energy
T = {}
for k in "ab":
    d = pd.read_csv(os.path.join(PULL, f"train-energy-{k}", "out", "train_energy.csv"))
    assert (d.status == "ok").all()
    T[k] = d[d.n_steps == 50].set_index("config")
ref = "BERT-base fine-tuning"
rel = ((T["a"].j_per_step / T["a"].j_per_step[ref]) + (T["b"].j_per_step / T["b"].j_per_step[ref])) / 2
EPOCHS = {"BERT-base fine-tuning": 3, "BERT-base, 64 tokens": 3, "BERT-base prune 75% (recovery)": 1, "DistilBERT distillation (with teacher)": 3,
          "TinyBERT-4L distillation (with teacher)": 3, "EBT T=3 training": 3, "SpikeBERT T=16 training": 15, "Dense control training": 15}
LAB = {"BERT-base fine-tuning": "BERT-base fine-tuning", "BERT-base, 64 tokens": "BERT-base, 64 tokens", "BERT-base prune 75% (recovery)": "BERT-base, prune 75" + PC + " (recovery epoch)",
       "DistilBERT distillation (with teacher)": "DistilBERT distillation (teacher included)", "TinyBERT-4L distillation (with teacher)": "TinyBERT-4L distillation (teacher included)",
       "EBT T=3 training": "EBT, $T=3$", "SpikeBERT T=16 training": "SpikeBERT, $T=16$", "Dense control training": "Dense control"}
rows = []
for cfg in rel.sort_values().index:
    a = T["a"].loc[cfg]
    b = T["b"].loc[cfg]
    steps_liar = 640  # 10,240 LIAR training examples at batch size 16
    run_kj = a.j_per_step * steps_liar * EPOCHS[cfg] / 1000
    rows.append(f"{LAB[cfg]} & {int(a.tokens)} & {a.precision} & {a.peak_MB / 1024:.1f} & {a.s_per_step * 1000:.0f} & {a.j_per_step:.1f} & {b.j_per_step:.1f} & "
                f"{a.j_per_step / 16 * 1000:.0f} & {BS}textbf{{{rel[cfg]:.2f}}} {BS}{BS}")
out["train"] = "\n".join(rows)
out["train_rel"] = {k: float(v) for k, v in rel.items()}
out["train_peak_gb"] = {k: float(T["a"].loc[k].peak_MB / 1024) for k in rel.index}

# ---- real-input energy
R = {}
for k in "ab":
    d = pd.read_csv(os.path.join(PULL, f"realenergy-{k}", "out", "energy_real.csv"))
    r0 = d[(d.model == "BERT-base") & (d.policy == "pad256")].set_index("dataset").j_per_1000_ex
    d["rel"] = d.apply(lambda r: r.j_per_1000_ex / r0[r.dataset], axis=1)
    R[k] = d.set_index(["dataset", "model", "policy"]).rel
rm = ((R["a"] + R["b"]) / 2).unstack("policy")
L = pd.read_csv(os.path.join(PULL, "realenergy-a", "out", "token_lengths.csv")).set_index("dataset")
NAMES = {"liar": "LIAR", "fakenewsnet": "FakeNewsNet", "isot": "ISOT"}
rows = []
for ds in ("liar", "fakenewsnet", "isot"):
    for m in ("BERT-base", "DistilBERT", "TinyBERT-4L"):
        v = rm.loc[(ds, m)]
        rows.append(f"{NAMES[ds]} & {m} & {v['pad256']:.2f} & {v['trunc64']:.2f} & {v['dyn']:.2f} & {v['sorted']:.2f} {BS}{BS}")
out["real"] = "\n".join(rows)
out["real_rel"] = {f"{a}|{b}": {k: float(x) for k, x in rm.loc[(a, b)].items()} for a, b in rm.index}
out["lengths"] = {ds: dict(mean=float(L.loc[ds, "mean_len"]), median=float(L.loc[ds, "median_len"]), p95=float(L.loc[ds, "p95_len"]),
                           over64=float(L.loc[ds, "share_over_64"]), over256=float(L.loc[ds, "share_over_256"])) for ds in L.index}
json.dump(out, open(os.path.join(ROOT, "revision", "round3_tables.json"), "w"), indent=1)
print(out["train"])
print()
print(out["real"])
print(out["lengths"])
