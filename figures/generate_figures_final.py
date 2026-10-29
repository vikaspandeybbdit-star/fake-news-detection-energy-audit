"""Figures for the consistency rebuild: every energy axis is energy relative to BERT-base (256 tokens, FP16) measured in one
session (mean of two replicate sessions, 200-iteration window); every accuracy axis is the test-split MCC over three seeds.
Reuses fig5 and fig7 from generate_figures.py unchanged."""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import generate_figures as g
from generate_figures import BLACK, BLUE, GREEN, GREY, ORANGE, PURPLE, SKY, VERMILION, savefig

A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p3_compression", "revision", "final_analysis.json")))
E = A["rel_energy"]
RC = A["rel_by_config"]
DS = [("liar", "LIAR"), ("isot", "ISOT"), ("fakenewsnet", "FakeNewsNet")]
FRONT = ["bert_base", "bert_prune25", "bert_prune50", "bert_prune75", "distil_prune0", "distil_prune25", "distil_prune50", "distil_prune75", "tinybert", "bert_len64"]
LABEL = {"bert_base": "BERT-base", "bert_prune25": "BERT, prune 25%", "bert_prune50": "BERT, prune 50%", "bert_prune75": "BERT, prune 75%",
         "distil_prune0": "DistilBERT", "distil_prune25": "DistilBERT, prune 25%", "distil_prune50": "DistilBERT, prune 50%",
         "distil_prune75": "DistilBERT, prune 75%", "tinybert": "TinyBERT-4L", "bert_len64": "BERT-base, 64 tokens"}


def fig1():
    shares = {"LIAR": (36.5, 53.6, 4.2, 5.6), "ISOT": (36.5, 53.2, 4.5, 5.8), "FakeNewsNet": (36.7, 53.8, 4.2, 5.4)}
    labels = ["Self-attention", "Feed-forward", "Embedding", "Other/overhead"]
    colors = [BLUE, SKY, ORANGE, g.PURPLE]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    names = list(shares)
    bottoms = [0, 0, 0]
    for k, (lab, c) in enumerate(zip(labels, colors)):
        vals = [shares[n][k] for n in names]
        ax.bar(range(3), vals, bottom=bottoms, color=c, edgecolor="white", linewidth=0.6, label=lab, width=0.55)
        for i, v in enumerate(vals):
            if v > 8:
                ax.text(i, bottoms[i] + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="white" if c == BLUE else BLACK)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(3))
    ax.set_xticklabels(names)
    ax.set_ylabel("Share of inference time (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14), frameon=False)
    savefig(fig, "fig1_energy_breakdown")


def fig2():
    items = [(LABEL[m], E[g_name], c) for m, g_name, c in []]
    order = [("BERT-base, 64 tokens", "BERT-base 64 tokens", SKY), ("TinyBERT-4L", "TinyBERT-4L", GREEN), ("DistilBERT, prune 75%", "DistilBERT prune 75%", GREEN),
             ("DistilBERT", "DistilBERT prune 0%", GREEN), ("Dense control (64 tok, FP32)", "Dense control FP32", GREY), ("BERT, prune 75%", "BERT-base prune 75%", BLUE),
             ("BERT, prune 50%", "BERT-base prune 50%", BLUE), ("BERT, prune 25%", "BERT-base prune 25%", BLUE), ("BERT-base (reference)", "BERT-base prune 0%", BLACK),
             ("EBT, T=1 (FP16)", "EBT T=1 FP16", ORANGE), ("EBT, T=2 (FP16)", "EBT T=2 FP16", ORANGE), ("EBT, T=3 (FP16)", "EBT T=3 FP16", ORANGE),
             ("SpikeBERT (64 tok, FP32)", "SpikeBERT T=16 FP32", PURPLE), ("EBT, T=3 (FP32)", "EBT T=3 FP32", VERMILION)]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    y = list(range(len(order)))[::-1]
    vals = [E_[1] if False else None for E_ in order]
    for yi, (lab, key, col) in zip(y, order):
        v = RC[key]
        ax.barh(yi, v, color=col, height=0.62)
        ax.text(v * 1.04, yi, f"{v:.2f}", va="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.axvline(1.0, color=BLACK, lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([o[0] for o in order], fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(0.15, 16)
    ax.set_xlabel("Inference energy relative to BERT-base (256 tokens, FP16); log scale")
    savefig(fig, "fig2_compression_summary")


def _front(m, ds):
    return A[ds]["frontier_boot"][m]


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4))
    for ax, (ds, name) in zip(axes, DS):
        col = g.DATASET_COLOR[name]
        mean = {m: A[ds]["mcc_mean"][m] for m in FRONT}
        sd = {m: A[ds]["mcc_sd"][m] for m in FRONT}
        fr = set(A[ds]["frontier_mean"])
        pts = sorted([(E[m], mean[m]) for m in fr])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=1.3, zorder=2)
        for m in FRONT:
            on = m in fr
            ax.errorbar(E[m], mean[m], yerr=sd[m], fmt="D" if on else "o", ms=6.5 if on else 5, color=col if on else GREY,
                        ecolor=col if on else GREY, elinewidth=0.8, capsize=2, markeredgecolor="white", zorder=3)
            if m == "bert_len64":
                ax.annotate("64 tokens", (E[m], mean[m]), textcoords="offset points", xytext=(8, 4), fontsize=7.5)
            if m == "bert_base":
                ax.annotate("BERT-base", (E[m], mean[m]), textcoords="offset points", xytext=(-30, -14), fontsize=7.5)
        ax.set_title(name)
        ax.set_xlabel("Relative energy")
    axes[0].set_ylabel("Test MCC (mean $\\pm$ SD, 3 seeds)")
    handles = [Line2D([0], [0], marker="D", color=BLUE, linestyle="-", markersize=7, label="Pareto frontier of the seed-mean MCC"),
               Line2D([0], [0], marker="o", color=GREY, linestyle="None", markersize=6, label="Dominated configuration")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), frameon=False)
    fig.tight_layout()
    savefig(fig, "fig3_pareto_frontiers")


def fig4():
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    keys = [("BERT-base", "BERT-base prune 0%", BLACK), ("EBT $T$=1", "EBT T=1 FP16", ORANGE), ("EBT $T$=2", "EBT T=2 FP16", ORANGE),
            ("EBT $T$=3", "EBT T=3 FP16", ORANGE), ("EBT $T$=3\nFP32", "EBT T=3 FP32", VERMILION)]
    for i, (lab, k, c) in enumerate(keys):
        ax.bar(i, RC[k], color=c, width=0.6)
        ax.text(i, RC[k] + 0.2, f"{RC[k]:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[0] for k in keys], fontsize=8.5)
    ax.set_ylabel("Energy relative to BERT-base")
    ax.set_ylim(0, 11.2)
    savefig(fig, "fig4_ebt_energy")


def fig5():
    import numpy as np
    C = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p3_compression", "revision", "snn_curves.json")))
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    for ax, (ds, name) in zip(axes, DS):
        col = g.DATASET_COLOR[name]
        runs = [C[f"{ds}_{s}"] for s in (42, 43, 44)]
        for r in runs:
            ax.plot(range(len(r)), r, "-", color=col, lw=0.9, alpha=0.45)
        m = np.mean(np.array(runs), axis=0)
        ax.plot(range(len(m)), m, "-o", color=col, lw=1.8, markersize=3.5)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Validation MCC")
    fig.tight_layout()
    savefig(fig, "fig5_snn_training_curves")


def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4))
    for ax, (ds, name) in zip(axes, DS):
        col = g.DATASET_COLOR[name]
        fr = set(A[ds]["frontier_mean"])
        pts = sorted([(E[m], A[ds]["mcc_mean"][m]) for m in fr])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-D", color=col, lw=1.3, ms=6, markeredgecolor="white", zorder=2)
        for key, x, mk, c, sz in (("ebt", E["ebt"], "*", ORANGE, 13), ("snn", E["snn"], "P", PURPLE, 9), ("dense_preln", E["dense_preln"], "s", GREY, 7)):
            ax.errorbar([x], [A[ds]["mcc_mean"][key]], yerr=[A[ds]["mcc_sd"][key]], fmt=mk, ms=sz, color=c, ecolor=c, elinewidth=1.0, capsize=3,
                        markeredgecolor=BLACK, markeredgewidth=0.6, zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(0.15, 14)
        ax.set_title(name)
        ax.set_xlabel("Relative energy (log scale)")
    axes[0].set_ylabel("Test MCC (mean $\\pm$ SD, 3 seeds)")
    handles = [Line2D([0], [0], marker="D", color=BLUE, linestyle="-", markersize=7, label="Pareto frontier (BERT family)"),
               Line2D([0], [0], marker="*", color=ORANGE, linestyle="None", markersize=13, label="EBT ($T$=3, FP16)"),
               Line2D([0], [0], marker="P", color=PURPLE, linestyle="None", markersize=9, label="SpikeBERT"),
               Line2D([0], [0], marker="s", color=GREY, linestyle="None", markersize=7, label="Dense control (pre-LN)")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09), frameon=False)
    fig.tight_layout()
    savefig(fig, "fig6_frontier_comparison")


def fig8():
    T = [1, 2, 3]
    mcc = [0.111, 0.242, 0.247]
    energy = [RC["EBT T=1 FP16"], RC["EBT T=2 FP16"], RC["EBT T=3 FP16"]]
    fig, ax1 = plt.subplots(figsize=(5.0, 3.6))
    ax1.bar([t - 0.12 for t in T], energy, width=0.24, color=GREY, edgecolor="white")
    ax1.set_xlabel("Refinement steps $T$")
    ax1.set_ylabel("Energy relative to BERT-base")
    ax1.set_xticks(T)
    ax2 = ax1.twinx()
    ax2.plot(T, mcc, "-o", color=BLUE, markersize=6, lw=1.4)
    ax2.set_ylabel("Test MCC (LIAR, seed 42)", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax1.legend(handles=[g.mpatches.Patch(color=GREY, label="Relative energy"), Line2D([0], [0], marker="o", color=BLUE, linestyle="-", markersize=6, label="MCC")],
               loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig8_ebt_refinement_ablation")


def fig9():
    import statistics as st
    ft = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "p3_compression", "results_rev_altarch_pull", "FINAL_TABLES.json")))
    datasets = [("liar", "LIAR"), ("isot", "ISOT"), ("fakenewsnet", "FakeNewsNet")]
    series = [("SpikeBERT (3 seeds)", [ft["spikebert"][d]["mcc"] for d, _ in datasets], [ft["spikebert"][d]["mcc_sd"] for d, _ in datasets], PURPLE),
              ("Dense, SpikeBERT recipe (seed 42)", [0.0, 0.0, 0.0], [0, 0, 0], GREY),
              ("Dense, warmup + lower rate (3 seeds)", [ft["dense_warm"][d]["mcc"] for d, _ in datasets], [ft["dense_warm"][d]["mcc_sd"] for d, _ in datasets], SKY),
              ("Dense, pre-LN (3 seeds)", [ft["dense_preln"][d]["mcc"] for d, _ in datasets], [ft["dense_preln"][d]["mcc_sd"] for d, _ in datasets], GREEN)]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = list(range(len(datasets)))
    w = 0.2
    for k, (lab, vals, sds, col) in enumerate(series):
        ax.bar([i + (k - 1.5) * w for i in x], vals, yerr=sds, capsize=2, width=w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in datasets])
    ax.set_ylabel("Test MCC (mean $\\pm$ SD)")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7.5)
    savefig(fig, "fig9_snn_dense_control")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); g.fig7(); fig8(); fig9()
    print("figures regenerated")
