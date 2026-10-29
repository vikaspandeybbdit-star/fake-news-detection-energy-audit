"""Pareto-frontier stability across seeds (IEEE Access R2#2).
Test-split MCC per configuration from the 3-seed grid reruns (results_rev_final), energy ordering from the
paper's grid (energies differ in scale between sessions but the ordering of the 8 configurations is the same on
all three datasets). A configuration is on the frontier if no other configuration has lower-or-equal energy and
higher-or-equal MCC (one strict)."""
import csv
import glob
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_rev_final")
ENERGY = {  # J, paper grid (single session)
    "liar": {"bert_base": 486.92, "bert_prune25": 478.50, "bert_prune50": 404.58, "bert_prune75": 381.42,
             "distil_prune0": 246.24, "distil_prune25": 225.17, "distil_prune50": 200.07, "distil_prune75": 177.46},
    "isot": {"bert_base": 491.82, "bert_prune25": 458.93, "bert_prune50": 387.95, "bert_prune75": 368.70,
             "distil_prune0": 237.01, "distil_prune25": 215.09, "distil_prune50": 190.30, "distil_prune75": 166.66},
    "fakenewsnet": {"bert_base": 487.00, "bert_prune25": 467.59, "bert_prune50": 414.60, "bert_prune75": 368.79,
                    "distil_prune0": 239.11, "distil_prune25": 218.45, "distil_prune50": 193.81, "distil_prune75": 171.58},
}
DIR = {"liar": ["rev-liar"], "isot": ["rev-isota", "rev-isotb"], "fakenewsnet": ["rev-fnn"]}
CFG = list(ENERGY["liar"])


def load(ds, seed):
    for d in DIR[ds]:
        for f in glob.glob(os.path.join(ROOT, f"{d}-s{seed}", "**", "metrics.csv"), recursive=True):
            rows = [r for r in csv.DictReader(open(f)) if r["split"] == "test" and r["precision_mode"] == "fp16"]
            got = {r["model"]: float(r["mcc"]) for r in rows if r["model"] in ENERGY[ds]}
            if len(got) == 8:
                return got
    return None


def frontier(mcc, ds):
    keys = list(mcc)
    out = set()
    for a in keys:
        dom = any(ENERGY[ds][b] <= ENERGY[ds][a] and mcc[b] >= mcc[a] and (ENERGY[ds][b] < ENERGY[ds][a] or mcc[b] > mcc[a])
                  for b in keys if b != a)
        if not dom:
            out.add(a)
    return out


for ds in ENERGY:
    per = {s: load(ds, s) for s in (42, 43, 44)}
    print(f"\n== {ds}: seeds with all 8 configs:", [s for s, v in per.items() if v])
    ok = {s: v for s, v in per.items() if v}
    fr = {s: frontier(v, ds) for s, v in ok.items()}
    meanm = {c: st.mean(v[c] for v in ok.values()) for c in CFG}
    frm = frontier(meanm, ds)
    print(f"{'config':16s} {'energy':>7s} {'mean MCC':>9s} {'SD':>6s}  per-seed MCC            frontier seeds")
    for c in CFG:
        vals = [ok[s][c] for s in ok]
        sd = st.stdev(vals) if len(vals) > 1 else 0
        k = sum(c in fr[s] for s in ok)
        print(f"{c:16s} {ENERGY[ds][c]:7.1f} {meanm[c]:9.4f} {sd:6.4f}  {[round(x, 3) for x in vals]}  {k}/{len(ok)}{'  [mean-MCC frontier]' if c in frm else ''}")

# ---- LaTeX rows for the paper table
BS = chr(92)
NAMES = {"bert_base": "No distill, prune 0%", "bert_prune25": "No distill, prune 25%", "bert_prune50": "No distill, prune 50%",
         "bert_prune75": "No distill, prune 75%", "distil_prune0": "Distill, prune 0%", "distil_prune25": "Distill, prune 25%",
         "distil_prune50": "Distill, prune 50%", "distil_prune75": "Distill, prune 75%"}
cells = {}
for ds in ENERGY:
    ok = {s: load(ds, s) for s in (42, 43, 44)}
    fr = {s: frontier(v, ds) for s, v in ok.items()}
    for c in CFG:
        vals = [ok[s][c] for s in ok]
        cells[(c, ds)] = f"{st.mean(vals):.3f} {BS}pm$ {st.stdev(vals):.3f} ({sum(c in fr[s] for s in ok)}/3)".replace(BS + "pm$", "$" + BS + "pm$")
print()
for c in CFG:
    print(NAMES[c].replace("%", BS + "%") + " & " + " & ".join(cells[(c, ds)].replace("%", BS + "%") for ds in ("liar", "isot", "fakenewsnet")) + " " + BS + BS)
