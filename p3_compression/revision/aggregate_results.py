"""Walks results_rev_altarch_pull/, parses every metrics.csv, dedupes relay/fast
duplicate copies of the same (kind, dataset, seed[, variant/steps]) run, and
writes a clean per-run test-split summary plus mean+-SD rollups to
results_rev_altarch_pull/SUMMARY.md.
"""
import csv
import glob
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
PULL = os.path.join(os.path.dirname(HERE), "results_rev_altarch_pull")

rows = []
for mpath in glob.glob(os.path.join(PULL, "**", "metrics.csv"), recursive=True):
    folder = mpath.split(os.sep)[len(PULL.rstrip(os.sep).split(os.sep))]
    with open(mpath, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((folder, r))

def pick(items):
    # all copies of the same (dataset, seed, ...) key are deterministic reruns;
    # just take the first, prefer a non-smoke source if available.
    non_smoke = [x for x in items if x[0] != "altarch-smoke"]
    return (non_smoke or items)[0][1]

def f(x):
    return float(x)

DATASETS = ["liar", "isot", "fakenewsnet"]
out = []

# ---------------- EBT: 3-seed main track (T=3) ----------------
ebt_main = {}
for folder, r in rows:
    if "mcmc_num_steps" not in r or "variant" in r or "model" in r:
        continue
    if r["split"] != "test" or r["mcmc_num_steps"] != "3":
        continue
    ebt_main.setdefault((r["dataset"], r["seed"]), []).append((folder, r))

out.append("## EBT, multi-seed test-split (T=3, main track)\n")
out.append("| Dataset | seed42 | seed43 | seed44 | mean | SD |")
out.append("|---|---|---|---|---|---|")
ebt_rollup = {}
for ds in DATASETS:
    accs = []
    for seed in ("42", "43", "44"):
        key = (ds, seed)
        if key in ebt_main:
            accs.append(f(pick(ebt_main[key])["accuracy"]))
    mean_, sd_ = st.mean(accs), (st.stdev(accs) if len(accs) > 1 else 0.0)
    ebt_rollup[ds] = (mean_, sd_)
    cells = [f"{a:.4f}" for a in accs]
    while len(cells) < 3:
        cells.append("-")
    out.append(f"| {ds} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_:.4f} | {sd_:.4f} |")

out.append("\n### EBT energy / latency / throughput (mean over 5 repeats, per run) + energy-descent check\n")
out.append("| Dataset | Seed | Energy (J) mean±SD | Latency (s/iter) | Throughput (ex/s) | Energy descends monotonically |")
out.append("|---|---|---|---|---|---|")
for ds in DATASETS:
    for seed in ("42", "43", "44"):
        key = (ds, seed)
        if key not in ebt_main:
            continue
        r = pick(ebt_main[key])
        out.append(f"| {ds} | {seed} | {f(r['profile_energy_j_mean']):.2f} ± {f(r['profile_energy_j_std']):.2f} "
                    f"| {f(r['profile_latency_s_per_iter_mean']):.4f} | {f(r['profile_throughput_examples_per_s']):.1f} "
                    f"| {r['energy_descends_monotonically']} |")

# ---------------- EBT refinement-step ablation (LIAR, seed 42) ----------------
out.append("\n## EBT refinement-step ablation (LIAR, seed 42)\n")
out.append("| T | Accuracy | F1 | MCC | Energy (J) mean±SD |")
out.append("|---|---|---|---|---|")
for t, key in (("1", ("liar", "42", "1")), ("2", ("liar", "42", "2")), ("3 (main)", ("liar", "42", "3"))):
    items = [(folder, r) for folder, r in rows
              if "mcmc_num_steps" in r and "variant" not in r and "model" not in r
              and r["split"] == "test" and (r["dataset"], r["seed"], r["mcmc_num_steps"]) == key]
    if not items:
        continue
    r = pick(items)
    out.append(f"| {t} | {f(r['accuracy']):.4f} | {f(r['f1']):.4f} | {f(r['mcc']):.4f} "
                f"| {f(r['profile_energy_j_mean']):.2f} ± {f(r['profile_energy_j_std']):.2f} |")

# ---------------- SpikeBERT: spiking, 2-seed track ----------------
snn_spiking = {}
for folder, r in rows:
    if r.get("variant") != "spiking" or r["split"] != "test":
        continue
    snn_spiking.setdefault((r["dataset"], r["seed"]), []).append((folder, r))

out.append("\n## SpikeBERT (spiking), multi-seed test-split\n")
out.append("| Dataset | seed42 | seed43 | mean | SD |")
out.append("|---|---|---|---|---|")
for ds in DATASETS:
    accs = []
    for seed in ("42", "43"):
        key = (ds, seed)
        if key in snn_spiking:
            accs.append(f(pick(snn_spiking[key])["accuracy"]))
    mean_, sd_ = st.mean(accs), (st.stdev(accs) if len(accs) > 1 else 0.0)
    cells = [f"{a:.4f}" for a in accs]
    while len(cells) < 2:
        cells.append("-")
    out.append(f"| {ds} | {cells[0]} | {cells[1]} | {mean_:.4f} | {sd_:.4f} |")

# ---------------- SpikeBERT dense-from-scratch control ----------------
snn_dense = {}
for folder, r in rows:
    if r.get("variant") != "dense" or r["split"] != "test":
        continue
    snn_dense[r["dataset"]] = pick([(folder, r)])

out.append("\n## SpikeBERT dense-from-scratch control (seed 42, test-split)\n")
out.append("| Dataset | Accuracy | F1 | MCC | Collapsed to majority class? |")
out.append("|---|---|---|---|---|")
for ds in DATASETS:
    if ds not in snn_dense:
        continue
    r = snn_dense[ds]
    collapsed = "YES" if f(r["mcc"]) == 0.0 else "no"
    out.append(f"| {ds} | {f(r['accuracy']):.4f} | {f(r['f1']):.4f} | {f(r['mcc']):.4f} | {collapsed} |")

# spiking seed42 for same-row comparison
out.append("\n### Spiking vs. dense-control, same seed (42), test-split — isolates the confound\n")
out.append("| Dataset | Spiking acc | Spiking MCC | Dense-control acc | Dense-control MCC |")
out.append("|---|---|---|---|---|")
for ds in DATASETS:
    sp = pick(snn_spiking[(ds, "42")]) if (ds, "42") in snn_spiking else None
    de = snn_dense.get(ds)
    if sp and de:
        out.append(f"| {ds} | {f(sp['accuracy']):.4f} | {f(sp['mcc']):.4f} | {f(de['accuracy']):.4f} | {f(de['mcc']):.4f} |")

# ---------------- TinyBERT second baseline ----------------
tb = {}
for folder, r in rows:
    if "model" not in r or r["split"] != "test" or r.get("precision_mode") != "amp":
        continue
    tb[(r["model"], r["dataset"])] = pick([(folder, r)])

out.append("\n## TinyBERT second baseline vs. its own teacher (seed 42, test-split, same harness as DistilBERT)\n")
out.append("| Dataset | BERT-base (teacher) acc | TinyBERT acc | TinyBERT F1 | TinyBERT MCC | TinyBERT params |")
out.append("|---|---|---|---|---|---|")
for ds in DATASETS:
    bb = tb.get(("bert_base", ds))
    t = tb.get(("tinybert", ds))
    if bb and t:
        out.append(f"| {ds} | {f(bb['accuracy']):.4f} | {f(t['accuracy']):.4f} | {f(t['f1']):.4f} "
                    f"| {f(t['mcc']):.4f} | {t['n_params']} |")

text = "\n".join(out)
print(text)
with open(os.path.join(PULL, "SUMMARY.md"), "w", encoding="utf-8") as fh:
    fh.write(text + "\n")
