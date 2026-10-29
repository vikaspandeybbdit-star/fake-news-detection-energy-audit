"""Test-split, 3-seed rollup of every model in the paper, joined with single-session relative energy.
Writes results_rev_altarch_pull/FINAL_TABLES.json and FINAL_TABLES.md.
Sources: grid rev-* kernels (results_rev_final), suite-* (TinyBERT, INT8, BERT at 64 tokens, paired BERT-base),
altarch-ebt-*, altarch-snn-*, densemulti-* / altarch-snn-dense-* (dense control), energy-unified-{a,b}."""
import csv
import glob
import json
import os
import statistics as st

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PULL = os.path.join(ROOT, "results_rev_altarch_pull")
GRID = os.path.join(ROOT, "results_rev_final")
DS = ["liar", "isot", "fakenewsnet"]
SEEDS = ["42", "43", "44"]


def rows(pattern):
    out = []
    for p in glob.glob(pattern, recursive=True):
        with open(p, newline="", encoding="utf-8") as f:
            out += list(csv.DictReader(f))
    return out


def test_hf(rs, model):
    d = {}
    for r in rs:
        if r["model"] == model and r["split"] == "test" and r["precision_mode"] in ("amp", "cpu_int8"):
            d[(r["dataset"], r["seed"])] = (float(r["accuracy"]), float(r["mcc"]))
    return d


grid = rows(os.path.join(GRID, "rev-*", "**", "metrics.csv"))
suite = rows(os.path.join(PULL, "suite-*", "**", "metrics.csv"))
res = {}  # model -> dataset -> list of (acc, mcc) over seeds
for name in ["bert_base", "bert_prune25", "bert_prune50", "bert_prune75", "distil_prune0", "distil_prune25", "distil_prune50", "distil_prune75"]:
    res[name] = test_hf(grid, name)
for name in ["bert_base", "tinybert", "quant_int8", "bert_len64"]:
    res["suite_" + name] = test_hf(suite, name)

# EBT T=3 (main track), SpikeBERT, dense control
def single(pattern, filt):
    d = {}
    for r in rows(pattern):
        if r["split"] == "test" and filt(r):
            d[(r["dataset"], r["seed"])] = (float(r["accuracy"]), float(r["mcc"]))
    return d


res["ebt_T3"] = single(os.path.join(PULL, "altarch-ebt-*-s4?", "**", "metrics.csv"), lambda r: r.get("mcmc_num_steps") == "3")
res["spikebert"] = single(os.path.join(PULL, "altarch-snn-*-s4*", "**", "spiking", "**", "metrics.csv"), lambda r: True)
res["dense_preln"] = single(os.path.join(PULL, "*", "**", "dense_preln", "**", "metrics.csv"), lambda r: True)
res["dense_preln"].update(single(os.path.join(PULL, "altarch-snn-dense-preln-*", "**", "metrics.csv"), lambda r: True))
res["dense_warm"] = single(os.path.join(PULL, "*", "**", "dense_warm", "**", "metrics.csv"), lambda r: True)
res["dense_warm"].update(single(os.path.join(PULL, "altarch-snn-dense-warm-*", "**", "metrics.csv"), lambda r: True))

summary = {}
md = ["| model | dataset | n seeds | accuracy (mean +- SD) | MCC (mean +- SD) |", "|---|---|---|---|---|"]
for m, d in res.items():
    for ds in DS:
        v = [d[(ds, s)] for s in SEEDS if (ds, s) in d]
        if not v:
            continue
        a, c = [x[0] for x in v], [x[1] for x in v]
        sd = lambda z: st.stdev(z) if len(z) > 1 else 0.0
        summary.setdefault(m, {})[ds] = dict(n=len(v), acc=st.mean(a), acc_sd=sd(a), mcc=st.mean(c), mcc_sd=sd(c), mccs=c)
        md.append(f"| {m} | {ds} | {len(v)} | {100*st.mean(a):.2f} +- {100*sd(a):.2f} | {st.mean(c):.3f} +- {sd(c):.3f} |")

# single-session relative energy (mean of two replicates, 200-iteration window)
rel = {}
for k in ("a", "b"):
    e = pd.read_csv(os.path.join(PULL, f"energy-unified-{k}", "out", "energy_unified.csv"))
    e = e[e.n_iters == 200].set_index("config").j_per_iter
    rel[k] = e / e["BERT-base prune 0%"]
relm = pd.DataFrame(rel).mean(axis=1).to_dict()
summary["_relative_energy_vs_bert_base"] = relm
md += ["", "| config | energy relative to BERT-base (mean of 2 sessions) |", "|---|---|"]
md += [f"| {k} | {v:.3f} |" for k, v in sorted(relm.items(), key=lambda kv: kv[1])]

json.dump(summary, open(os.path.join(PULL, "FINAL_TABLES.json"), "w"), indent=1)
open(os.path.join(PULL, "FINAL_TABLES.md"), "w").write("\n".join(md))
print("\n".join(md))
