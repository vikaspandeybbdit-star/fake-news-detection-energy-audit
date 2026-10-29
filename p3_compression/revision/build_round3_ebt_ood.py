"""Round-3 aggregation: EBT stochastic-draw variability and training energy (LIAR, FakeNewsNet), COVID-19 external transfer (3 seeds),
BERT-base real training energy. Writes round3_ebt_ood.json."""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULL = os.path.join(ROOT, "results_rev_altarch_pull")
out = {}

# ---- EBT draws and training energy
ebt = {}
for ds in ("liar", "fakenewsnet"):
    rec = []
    for s in (42, 43, 44):
        dd = glob.glob(os.path.join(PULL, f"ebtdraw-{ds}-s{s}", "out", "ebt", ds, f"s{s}", "draws.csv"))
        mm = glob.glob(os.path.join(PULL, f"ebtdraw-{ds}-s{s}", "out", "ebt", ds, f"s{s}", "metrics.csv"))
        if not dd or not mm:
            continue
        d = pd.read_csv(dd[0])
        n = d[pd.to_numeric(d.draw, errors="coerce").notna()]
        m = pd.read_csv(mm[0])
        t = m[m.split == "test"].iloc[0]
        rec.append(dict(seed=s, draws_mcc_mean=float(n.mcc.mean()), draws_mcc_sd=float(n.mcc.std(ddof=0)), draws_mcc_min=float(n.mcc.min()), draws_mcc_max=float(n.mcc.max()),
                        zero_mcc=float(d[d.draw == "zero_init"].mcc.iloc[0]), vote_mcc=float(d[d.draw == "majority_vote"].mcc.iloc[0]),
                        flip=float(d.example_flip_rate.iloc[0]), train_kj=float(t.train_energy_j) / 1000, train_s=float(t.train_time_s)))
    ebt[ds] = rec
out["ebt"] = ebt

# ---- COVID transfer and BERT-base real training energy
rows, cost = [], []
for d in glob.glob(os.path.join(PULL, "oodcovid-*")):
    for f in glob.glob(os.path.join(d, "out", "ood", "cross_*.csv")):
        seed = 44 if f.endswith("_s44.csv") else 43 if f.endswith("_s43.csv") else 42
        for _, r in pd.read_csv(f).iterrows():
            rows.append(dict(train=r["train"], test=r["test"], seed=seed, acc=r.accuracy, mcc=r.mcc))
    for f in glob.glob(os.path.join(d, "out", "ood", "run", "*", "s*", "train_cost.csv")):
        c = pd.read_csv(f).iloc[0]
        cost.append(dict(dataset=c.dataset.replace("cross_", ""), seed=int(c.seed), kj=float(c.train_energy_j) / 1000, s=float(c.train_s)))
D = pd.DataFrame(rows)
cross = {}
for (tr, te), g in D.groupby(["train", "test"]):
    cross[f"{tr}|{te}"] = dict(n=len(g), mcc=float(g.mcc.mean()), sd=float(g.mcc.std(ddof=1)), acc=float(g.acc.mean()))
out["cross"] = cross
C = pd.DataFrame(cost)
out["bert_train"] = {ds: dict(kj=float(g.kj.mean()), s=float(g.s.mean())) for ds, g in C.groupby("dataset")}
json.dump(out, open(os.path.join(ROOT, "revision", "round3_ebt_ood.json"), "w"), indent=1)
for ds, rec in ebt.items():
    for r in rec:
        print(ds, r)
print({k: v for k, v in out["bert_train"].items()})
BS = chr(92)
names = {"liar": "LIAR", "isot": "ISOT", "fakenewsnet": "FakeNewsNet"}
for tr in ("liar", "isot", "fakenewsnet"):
    print(names[tr], " & ".join(f"{cross[f'{tr}|{te}']['mcc']:.3f} $" + BS + f"pm$ {cross[f'{tr}|{te}']['sd']:.3f}" for te in ("liar", "isot", "fakenewsnet", "covid")))
