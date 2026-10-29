import csv
import glob
import os
import statistics as st

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_rev_altarch_pull")
rows = []
for p in glob.glob(os.path.join(ROOT, "**", "metrics.csv"), recursive=True):
    folder = os.path.normpath(p).replace(os.path.normpath(ROOT), "").strip(os.sep).split(os.sep)[0]
    if folder == "altarch-smoke":
        continue
    for r in csv.DictReader(open(p)):
        if r.get("variant") == "spiking" and not r.get("tag") and r["split"] == "test":
            rows.append((folder, r))
out = {}
for ds in ("liar", "isot", "fakenewsnet"):
    per = {}
    for seed in ("42", "43", "44"):
        c = [r for f, r in rows if r["dataset"] == ds and r["seed"] == seed]
        per[seed] = c[0]
    acc = [float(per[s]["accuracy"]) for s in per]
    mcc = [float(per[s]["mcc"]) for s in per]
    en = [float(per[s]["profile_energy_j_mean"]) for s in per]
    esd = [float(per[s]["profile_energy_j_std"]) for s in per]
    lat = [float(per[s]["profile_latency_s_per_iter_mean"]) for s in per]
    thr = [float(per[s]["profile_throughput_examples_per_s"]) for s in per]
    rsd = [100 * a / b for a, b in zip(esd, en)]
    print(ds, "acc", [round(x, 4) for x in acc], "mean %.4f sd %.4f" % (st.mean(acc), st.stdev(acc)))
    print("   mcc", [round(x, 4) for x in mcc], "mean %.4f sd %.4f" % (st.mean(mcc), st.stdev(mcc)))
    print("   energy", [round(x, 1) for x in en], "RSD%", [round(x, 2) for x in rsd], "lat", [round(x, 3) for x in lat], "thr", [round(x, 1) for x in thr])
