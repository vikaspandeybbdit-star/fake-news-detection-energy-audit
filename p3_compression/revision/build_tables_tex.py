"""Emits LaTeX table bodies for the consistency rebuild into tables_new.tex (one named block per table)."""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULL = os.path.join(ROOT, "results_rev_altarch_pull")
A = json.load(open(os.path.join(ROOT, "revision", "final_analysis.json")))
FT = json.load(open(os.path.join(PULL, "FINAL_TABLES.json")))
BS = chr(92)
PC = BS + "%"
out = {}

ea = pd.read_csv(os.path.join(PULL, "energy-unified-a", "out", "energy_unified.csv"))
eb = pd.read_csv(os.path.join(PULL, "energy-unified-b", "out", "energy_unified.csv"))


def piece(df, n):
    return df[df.n_iters == n].set_index("config")


a200, b200, a20, b20 = piece(ea, 200), piece(eb, 200), piece(ea, 20), piece(eb, 20)
REF = "BERT-base prune 0%"
rel200 = (a200.j_per_iter / a200.j_per_iter[REF] + b200.j_per_iter / b200.j_per_iter[REF]) / 2
rel20 = (a20.j_per_iter / a20.j_per_iter[REF] + b20.j_per_iter / b20.j_per_iter[REF]) / 2

# ---- Table: single-session energy, latency, throughput, memory
ORDER = [("BERT-base", "BERT-base prune 0%", 256, "FP16"), ("BERT-base, prune 25" + PC, "BERT-base prune 25%", 256, "FP16"),
         ("BERT-base, prune 50" + PC, "BERT-base prune 50%", 256, "FP16"), ("BERT-base, prune 75" + PC, "BERT-base prune 75%", 256, "FP16"),
         ("DistilBERT", "DistilBERT prune 0%", 256, "FP16"), ("DistilBERT, prune 25" + PC, "DistilBERT prune 25%", 256, "FP16"),
         ("DistilBERT, prune 50" + PC, "DistilBERT prune 50%", 256, "FP16"), ("DistilBERT, prune 75" + PC, "DistilBERT prune 75%", 256, "FP16"),
         ("TinyBERT-4L", "TinyBERT-4L", 256, "FP16"), ("BERT-base, 64 tokens", "BERT-base 64 tokens", 64, "FP16"),
         ("EBT, $T=1$", "EBT T=1 FP16", 256, "FP16"), ("EBT, $T=2$", "EBT T=2 FP16", 256, "FP16"), ("EBT, $T=3$", "EBT T=3 FP16", 256, "FP16"),
         ("EBT, $T=3$", "EBT T=3 FP32", 256, "FP32"), ("SpikeBERT, $T=16$", "SpikeBERT T=16 FP32", 64, "FP32"), ("Dense control", "Dense control FP32", 64, "FP32")]
rows = []
for lab, key, L, prec in ORDER:
    r = a200.loc[key]
    rows.append(f"{lab} & {L} & {prec} & {r.params_M:.1f} & {r.peak_MB:.0f} & {r.s_per_iter * 1000:.0f} & {r.throughput_ex_s:.0f} & "
                f"{a200.j_per_iter[key]:.2f} & {b200.j_per_iter[key]:.2f} & {rel20[key]:.2f} & \\textbf{{{rel200[key]:.2f}}} " + BS + BS)
out["energy"] = "\n".join(rows)

# ---- Table: main accuracy-energy table (test MCC mean +- SD, 3 seeds; frontier seeds in parentheses)
MAIN = [("BERT-base", "bert_base"), ("BERT-base, prune 25" + PC, "bert_prune25"), ("BERT-base, prune 50" + PC, "bert_prune50"),
        ("BERT-base, prune 75" + PC, "bert_prune75"), ("DistilBERT", "distil_prune0"), ("DistilBERT, prune 25" + PC, "distil_prune25"),
        ("DistilBERT, prune 50" + PC, "distil_prune50"), ("DistilBERT, prune 75" + PC, "distil_prune75"), ("TinyBERT-4L", "tinybert"),
        ("BERT-base, 64 tokens", "bert_len64")]
rows = []
for lab, m in MAIN:
    cells = []
    for ds in ("liar", "isot", "fakenewsnet"):
        d = A[ds]
        cells.append(f"{d['mcc_mean'][m]:.3f} $" + BS + f"pm$ {d['mcc_sd'][m]:.3f} ({d['frontier_seeds'][m]}/3)")
    rows.append(f"{lab} & {A['rel_energy'][m]:.2f} & " + " & ".join(cells) + " " + BS + BS)
out["main"] = "\n".join(rows)

# ---- Table: paired bootstrap differences
DIFF = [("DistilBERT, prune 75" + PC, "distil_prune75"), ("DistilBERT", "distil_prune0"), ("BERT-base, prune 75" + PC, "bert_prune75"),
        ("EBT", "ebt"), ("SpikeBERT", "snn"), ("Dense control (pre-LN)", "dense_preln"), ("TinyBERT-4L", "tinybert"),
        ("BERT-base, 64 tokens", "bert_len64"), ("INT8 quantization (CPU)", "quant_int8")]
rows = []
for lab, m in DIFF:
    cells = []
    for ds in ("liar", "isot", "fakenewsnet"):
        d = A[ds]["diff"][m]
        cells.append(f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]".replace("-", "$-$"))
    ref = {"bert_base": "BERT-base", "suite_bert_base": "paired BERT-base"}[A["liar"]["diff_ref"][m]]
    rows.append(f"{lab} & {ref} & " + " & ".join(cells) + " " + BS + BS)
out["diff"] = "\n".join(rows)

# ---- Table: EBT
rows = []
for ds, name in (("liar", "LIAR"), ("isot", "ISOT"), ("fakenewsnet", "FakeNewsNet")):
    e = FT["ebt_T3"][ds]
    rows.append(f"{name} & {100 * e['acc']:.1f}" + PC + f" $" + BS + f"pm$ {100 * e['acc_sd']:.1f} & {e['mcc']:.3f} $" + BS + f"pm$ {e['mcc_sd']:.3f} " + BS + BS)
out["ebt"] = "\n".join(rows)

# ---- Table: SpikeBERT and dense control
rows = []
for lab, key in (("SpikeBERT", "spikebert"), ("Dense control, warmup", "dense_warm"), ("Dense control, pre-LN", "dense_preln")):
    cells = []
    for ds in ("liar", "isot", "fakenewsnet"):
        e = FT[key][ds]
        cells.append(f"{100 * e['acc']:.1f}" + PC + f" / {e['mcc']:.3f} $" + BS + f"pm$ {e['mcc_sd']:.3f}")
    rows.append(f"{lab} & " + " & ".join(cells) + " " + BS + BS)
out["snn"] = "\n".join(rows)

# ---- Table: published comparison (test-split 3-seed mean accuracy)
rows = []
PUB = {"liar": 59.6, "fakenewsnet": 80.0, "isot": 100.0}
for ds, name in (("liar", "LIAR"), ("fakenewsnet", "FakeNewsNet"), ("isot", "ISOT")):
    b, d = FT["bert_base"][ds]["acc"] * 100, FT["distil_prune0"][ds]["acc"] * 100
    rows.append(f"{name} & {b:.1f}" + PC + f" & {d:.1f}" + PC + f" & {PUB[ds]:.1f}" + PC + " " + BS + BS)
out["pub"] = "\n".join(rows)

# ---- Table: baseline
rows = []
for ds, name, n in (("liar", "LIAR", 1267), ("isot", "ISOT", 4490), ("fakenewsnet", "FakeNewsNet", 2320)):
    e = FT["bert_base"][ds]
    rows.append(f"{name} & {n:,} & {100 * e['acc']:.1f}" + PC + f" $" + BS + f"pm$ {100 * e['acc_sd']:.1f} & {e['mcc']:.3f} $" + BS + f"pm$ {e['mcc_sd']:.3f} " + BS + BS)
out["baseline"] = "\n".join(rows)

open(os.path.join(ROOT, "revision", "tables_new.json"), "w").write(json.dumps(out, indent=1))
for k, v in out.items():
    print("==", k)
    print(v)
