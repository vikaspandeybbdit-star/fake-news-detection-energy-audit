"""
Figures for JournalPaper1_alternative_paradigms_draft.md.
All numeric data sourced from RESULTS_AND_HYPERPARAMETERS_BACKUP.md
(sections cited inline). Colorblind-safe (Okabe-Ito) qualitative palette,
color + marker-shape redundancy so figures survive grayscale printing too.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUT = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"      # LIAR
GREEN = "#009E73"     # ISOT
VERMILION = "#D55E00" # FakeNewsNet
ORANGE = "#E69F00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
BLACK = "#1A1A1A"
GREY = "#999999"

DATASET_COLOR = {"LIAR": BLUE, "ISOT": GREEN, "FakeNewsNet": VERMILION}


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}_preview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: component-wise energy breakdown (backup Sec 1.2 / paper Sec 4.1)
# ---------------------------------------------------------------------------
def fig1():
    datasets = ["LIAR", "ISOT", "FakeNewsNet"]
    self_attn = [168.50, 171.46, 161.28]
    feed_fwd = [247.36, 249.93, 236.42]
    embed = [19.53, 21.35, 18.46]
    other = [25.92, 27.29, 23.73]
    totals = [461.3, 470.0, 439.4]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    x = range(len(datasets))
    colors = [BLUE, SKY, ORANGE, PURPLE]
    labels = ["Self-attention", "Feed-forward", "Embedding", "Other/overhead"]
    bottoms = [0, 0, 0]
    for vals, c, lab in zip([self_attn, feed_fwd, embed, other], colors, labels):
        ax.bar(x, vals, bottom=bottoms, color=c, edgecolor="white", linewidth=0.6, label=lab, width=0.55)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for xi, t in zip(x, totals):
        ax.text(xi, t + 8, f"{t:.1f} J", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Energy (J), 20 inference iterations")
    ax.set_ylim(0, 520)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.14), frameon=False)
    savefig(fig, "fig1_energy_breakdown")


# ---------------------------------------------------------------------------
# Fig 2: compression technique summary (paper Sec 4.2, matches backup Sec 3-4)
# ---------------------------------------------------------------------------
def fig2():
    datasets = ["LIAR", "ISOT", "FakeNewsNet"]
    quant_mcc = {"LIAR": 0.116, "ISOT": 1.000, "FakeNewsNet": 0.589}
    pruning = {"LIAR": (0.271, 395.1), "ISOT": (1.000, 403.0), "FakeNewsNet": (0.615, 392.8)}
    distill = {"LIAR": (0.251, 236.9), "ISOT": (1.000, 245.7), "FakeNewsNet": (0.624, 236.8)}

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1), sharey=False)
    for ax, ds in zip(axes, datasets):
        p_mcc, p_e = pruning[ds]
        d_mcc, d_e = distill[ds]
        ax.plot([p_e, d_e], [p_mcc, d_mcc], "--", color=GREY, lw=1, zorder=1)
        ax.scatter([p_e], [p_mcc], marker="s", s=70, color=BLUE, edgecolor="white",
                   zorder=3, label="Pruning 50%+recovery")
        ax.scatter([d_e], [d_mcc], marker="o", s=70, color=VERMILION, edgecolor="white",
                   zorder=3, label="Distillation")
        ax.scatter([min(p_e, d_e) - 40], [quant_mcc[ds]], marker="^", s=70, color=ORANGE,
                   edgecolor="white", zorder=3, label="Quantization (CPU)")
        ax.set_title(ds)
        ax.set_xlabel("Energy (J)")
        ax.annotate(f"Quant. (CPU): MCC={quant_mcc[ds]:.3f}\n(energy not GPU-comparable)",
                    xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7.3,
                    style="italic", color=GREY)
    axes[0].set_ylabel("MCC")
    handles = [
        Line2D([0], [0], marker="^", color=ORANGE, linestyle="None", markersize=8, label="Quantization (CPU)"),
        Line2D([0], [0], marker="s", color=BLUE, linestyle="None", markersize=8, label="Pruning 50%+recovery"),
        Line2D([0], [0], marker="o", color=VERMILION, linestyle="None", markersize=8, label="Distillation"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    savefig(fig, "fig2_compression_summary")


# ---------------------------------------------------------------------------
# Fig 3: Pareto frontiers, full 8-config grid per dataset (backup Sec 13)
# ---------------------------------------------------------------------------
GRID = {
    "LIAR": {
        (0, 0): (0.2766, 486.92), (0, 25): (0.2810, 478.50), (0, 50): (0.2709, 404.58), (0, 75): (0.2595, 381.42),
        (1, 0): (0.2686, 246.24), (1, 25): (0.2441, 225.17), (1, 50): (0.2406, 200.07), (1, 75): (0.2552, 177.46),
    },
    "ISOT": {
        (0, 0): (0.9996, 491.82), (0, 25): (0.9991, 458.93), (0, 50): (1.0000, 387.95), (0, 75): (1.0000, 368.70),
        (1, 0): (0.9996, 237.01), (1, 25): (0.9996, 215.09), (1, 50): (0.9996, 190.30), (1, 75): (0.9996, 166.66),
    },
    "FakeNewsNet": {
        (0, 0): (0.6322, 487.00), (0, 25): (0.6037, 467.59), (0, 50): (0.6060, 414.60), (0, 75): (0.6007, 368.79),
        (1, 0): (0.6258, 239.11), (1, 25): (0.6124, 218.45), (1, 50): (0.6013, 193.81), (1, 75): (0.5950, 171.58),
    },
}
FRONTIER = {
    "LIAR": [(1, 75), (1, 0), (0, 50), (0, 25)],
    "ISOT": [(1, 75), (0, 75)],
    "FakeNewsNet": [(1, 75), (1, 50), (1, 25), (1, 0), (0, 0)],
}


def _pareto_panel(ax, ds):
    color = DATASET_COLOR[ds]
    grid = GRID[ds]
    frontier_keys = set(FRONTIER[ds])
    dom_mcc = [v[0] for k, v in grid.items() if k not in frontier_keys]
    dom_e = [v[1] for k, v in grid.items() if k not in frontier_keys]
    ax.scatter(dom_e, dom_mcc, marker="o", s=34, color=GREY, alpha=0.7, zorder=2, label="Dominated config")
    fr = sorted([grid[k] for k in FRONTIER[ds]], key=lambda t: t[1])
    fe = [t[1] for t in fr]
    fm = [t[0] for t in fr]
    ax.plot(fe, fm, "-", color=color, lw=1.4, zorder=3)
    ax.scatter(fe, fm, marker="D", s=55, color=color, edgecolor="white", linewidth=0.8, zorder=4,
               label="Pareto frontier")
    ax.set_title(ds)
    ax.set_xlabel("Energy (J)")


def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
    for ax, ds in zip(axes, ["LIAR", "ISOT", "FakeNewsNet"]):
        _pareto_panel(ax, ds)
    axes[0].set_ylabel("MCC")
    handles = [
        Line2D([0], [0], marker="D", color=BLUE, linestyle="-", markersize=7, label="Pareto frontier"),
        Line2D([0], [0], marker="o", color=GREY, linestyle="None", markersize=6, label="Dominated configuration"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout()
    savefig(fig, "fig3_pareto_frontiers")


# ---------------------------------------------------------------------------
# Fig 4: EBT energy vs baseline (paper Sec 4.4)
# ---------------------------------------------------------------------------
def fig4():
    datasets = ["LIAR", "ISOT", "FakeNewsNet"]
    baseline = [461.3, 470.0, 439.4]
    ebt_fp16 = [943.5, 945.4, 994.4]
    ebt_fp32 = [1009.3, 1024.4, 1040.2]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = range(len(datasets))
    w = 0.26
    ax.bar([i - w for i in x], baseline, width=w, color=BLACK, label="Baseline (FP16)")
    ax.bar(list(x), ebt_fp16, width=w, color=BLUE, label="EBT (FP16)")
    ax.bar([i + w for i in x], ebt_fp32, width=w, color=VERMILION, label="EBT (FP32)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Energy (J), 20 inference iterations")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.14), frameon=False)
    savefig(fig, "fig4_ebt_energy")


# ---------------------------------------------------------------------------
# Fig 5: SpikeBERT training curves (backup Sec 6.4 LIAR full curve; Sec 4.5
# paper table FakeNewsNet peak/final only -- disclosed limitation)
# ---------------------------------------------------------------------------
def fig5():
    liar_epochs = list(range(15))
    liar_mcc = [0.1211, 0.1918, 0.2503, 0.2432, 0.2531, 0.2359, 0.2067, 0.1961,
                0.1932, 0.1954, 0.1949, 0.2050, 0.1971, 0.2017, 0.1910]
    fnn_epochs = [2, 14]
    fnn_mcc = [0.559, 0.5124]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ax = axes[0]
    ax.plot(liar_epochs, liar_mcc, "-o", color=BLUE, markersize=4, lw=1.3)
    ax.axvline(2, color=GREY, ls=":", lw=0.9)
    ax.set_ylim(0.10, 0.29)
    ax.annotate("peak (epoch 2)", xy=(2, 0.2503), xytext=(5.5, 0.278),
                fontsize=7.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    ax.set_title("LIAR (full 15-epoch curve)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MCC")

    ax = axes[1]
    ax.plot(fnn_epochs, fnn_mcc, "--s", color=VERMILION, markersize=8, lw=1.3)
    ax.annotate(f"{fnn_mcc[0]:.4f}", xy=(fnn_epochs[0], fnn_mcc[0]), xytext=(fnn_epochs[0] + 0.7, fnn_mcc[0] + 0.002), fontsize=8)
    ax.annotate(f"{fnn_mcc[1]:.4f}", xy=(fnn_epochs[1], fnn_mcc[1]), xytext=(fnn_epochs[1] - 2.4, fnn_mcc[1] + 0.004), fontsize=8)
    ax.set_title("FakeNewsNet (peak and final epoch only;\nno intermediate epochs in source)")
    ax.set_xlabel("Epoch")
    ax.set_xlim(-0.5, 15)
    ax.set_ylim(0.505, 0.572)

    fig.tight_layout()
    savefig(fig, "fig5_snn_training_curves")


# ---------------------------------------------------------------------------
# Fig 6: EBT / SpikeBERT positioned against the combined-framework frontier
# (paper Sec 4.6)
# ---------------------------------------------------------------------------
# Updated to the multi-seed mean MCC from the IEEE Access revision round (3 seeds for EBT,
# 2 for SpikeBERT, held-out test split; see RESULTS_AND_HYPERPARAMETERS_BACKUP.md Sec 17),
# with SD as an error bar on the MCC axis. Energy stays the original single-session figure
# (not directly comparable across the two profiling sessions -- paper Sec 5).
EBT_POINT = {"LIAR": (0.2237, 943.5, 0.0478), "ISOT": (0.9947, 945.4, 0.0043), "FakeNewsNet": (0.5706, 994.4, 0.0188)}
SNN_POINT = {"LIAR": (0.1715, 584.13, 0.0033), "ISOT": (0.9973, 1202.92, 0.0019), "FakeNewsNet": (0.5289, 594.0, 0.0133)}


def fig6():
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.3))
    for ax, ds in zip(axes, ["LIAR", "ISOT", "FakeNewsNet"]):
        color = DATASET_COLOR[ds]
        grid = GRID[ds]
        fr = sorted([grid[k] for k in FRONTIER[ds]], key=lambda t: t[1])
        fe = [t[1] for t in fr]
        fm = [t[0] for t in fr]
        ax.plot(fe, fm, "-", color=color, lw=1.4, zorder=2)
        ax.scatter(fe, fm, marker="D", s=50, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        em, ee, esd = EBT_POINT[ds]
        sm, se, ssd = SNN_POINT[ds]
        ax.errorbar([ee], [em], yerr=[esd], fmt="*", ms=14, color=ORANGE, ecolor=ORANGE,
                    elinewidth=1.1, capsize=3, markeredgecolor=BLACK, markeredgewidth=0.6, zorder=4)
        ax.errorbar([se], [sm], yerr=[ssd], fmt="P", ms=9, color=PURPLE, ecolor=PURPLE,
                    elinewidth=1.1, capsize=3, markeredgecolor=BLACK, markeredgewidth=0.6, zorder=4)
        ax.set_title(ds)
        ax.set_xlabel("Energy (J)")
    axes[0].set_ylabel("MCC")
    handles = [
        Line2D([0], [0], marker="D", color=BLUE, linestyle="-", markersize=7, label="Combined-framework frontier"),
        Line2D([0], [0], marker="*", color=ORANGE, linestyle="None", markersize=13, label="EBT"),
        Line2D([0], [0], marker="P", color=PURPLE, linestyle="None", markersize=9, label="SpikeBERT (GPU-simulated)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    savefig(fig, "fig6_frontier_comparison")


# ---------------------------------------------------------------------------
# Fig 7: architecture schematic (illustrative only)
# ---------------------------------------------------------------------------
def _box(ax, xy, w, h, text, fc="white", ec=BLACK):
    x, y = xy
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                                    linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.3)


def _arrow(ax, p1, p2):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", lw=1.1, color=BLACK))


def fig7():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.8)
    ax.axis("off")

    ax.text(0.1, 9.5, "(a) Energy-Based Transformer (EBT) adaptation", fontsize=10.5, weight="bold")
    y0 = 7.6
    h = 1.5
    xs = [0.2, 2.05, 3.9, 5.9, 8.1]
    ws = [1.55, 1.55, 1.7, 1.7, 1.5]
    labels_a = ["Context\nencoding\n(BERT)", "Concatenate\nw/ candidate\nlabel embed.",
                "Shallow\nbidirectional\ntransformer\n$\\rightarrow$ energy scalar",
                "Gradient-descent\nrefinement\n(3 steps, $\\partial E/\\partial$label)",
                "Final label\ndistribution"]
    colors_a = ["white", "white", "white", SKY, "white"]
    for x, w, lab, c in zip(xs, ws, labels_a, colors_a):
        _box(ax, (x, y0), w, h, lab, fc=c)
    for i in range(len(xs) - 1):
        _arrow(ax, (xs[i] + ws[i], y0 + h / 2), (xs[i + 1], y0 + h / 2))
    ax.annotate("", xy=(xs[3] + ws[3] * 0.75, y0 + h + 0.05), xytext=(xs[3] + ws[3] * 0.25, y0 + h + 0.05),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=BLUE,
                                 connectionstyle="arc3,rad=-0.6"))
    ax.text(xs[3] + ws[3] / 2, y0 + h + 1.05, "$\\times$3 refinement steps", fontsize=8, color=BLUE, ha="center")

    ax.text(0.1, 4.3, "(b) SpikeBERT (spiking transformer) adaptation", fontsize=10.5, weight="bold")
    y1 = 2.4
    xs_b = [0.2, 2.05, 3.9, 6.0]
    ws_b = [1.55, 1.6, 1.9, 1.55]
    labels_b = ["Token\nembeddings\n(from scratch)", "Spike encoding\n(T=16 timesteps)",
                "LIF-neuron spiking\nself-attention\nblocks ($\\times$6)", "Classification\nhead"]
    colors_b = ["white", "white", ORANGE, "white"]
    for x, w, lab, c in zip(xs_b, ws_b, labels_b, colors_b):
        _box(ax, (x, y1), w, h, lab, fc=c)
    for i in range(len(xs_b) - 1):
        _arrow(ax, (xs_b[i] + ws_b[i], y1 + h / 2), (xs_b[i + 1], y1 + h / 2))
    ax.text(3.9 + 1.9 / 2, y1 - 0.55, "No pretrained BERT weights reused", fontsize=8, style="italic", color=GREY,
            ha="center")

    fig.tight_layout()
    savefig(fig, "fig7_architecture_schematic")


# ---------------------------------------------------------------------------
# Fig 8: EBT refinement-step ablation, LIAR seed 42 (IEEE Access revision round)
# ---------------------------------------------------------------------------
def fig8():
    T = [1, 2, 3]
    acc = [58.0, 63.5, 63.8]
    mcc = [0.111, 0.242, 0.247]
    energy = [86.7, 121.1, 160.7]
    energy_err = [2.8, 5.6, 4.8]

    fig, ax1 = plt.subplots(figsize=(5.0, 3.6))
    ax1.bar([t - 0.12 for t in T], energy, yerr=energy_err, width=0.24, color=GREY,
            edgecolor="white", capsize=3, label="Energy (J)")
    ax1.set_xlabel("Refinement steps $T$")
    ax1.set_ylabel("Energy (J)")
    ax1.set_xticks(T)

    ax2 = ax1.twinx()
    ax2.plot(T, mcc, "-o", color=BLUE, markersize=6, lw=1.4, label="MCC")
    ax2.set_ylabel("MCC", color=BLUE)
    ax2.tick_params(axis="y", labelcolor=BLUE)

    handles = [
        mpatches.Patch(color=GREY, label="Energy (J), $\\pm$SD (5 repeats)"),
        Line2D([0], [0], marker="o", color=BLUE, linestyle="-", markersize=6, label="MCC"),
    ]
    ax1.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig8_ebt_refinement_ablation")


# ---------------------------------------------------------------------------
# Fig 9: SpikeBERT (spiking) vs. dense-from-scratch control, seed 42
# (IEEE Access revision round)
# ---------------------------------------------------------------------------
def fig9():
    datasets = ["LIAR", "ISOT", "FakeNewsNet"]
    series = [("SpikeBERT (spiking)", [0.173, 0.999, 0.520], PURPLE),
              ("Dense, SpikeBERT recipe (lr 6e-4)", [0.000, 0.000, 0.000], GREY),
              ("Dense, warmup + lr 1e-4", [0.117, 0.996, 0.492], SKY),
              ("Dense, pre-LN + warmup", [0.170, 0.997, 0.488], GREEN)]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = list(range(len(datasets)))
    w = 0.2
    for k, (lab, vals, col) in enumerate(series):
        ax.bar([i + (k - 1.5) * w for i in x], vals, width=w, color=col, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("MCC (test split, seed 42)")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=7.5)
    savefig(fig, "fig9_snn_dense_control")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print("All 9 figures regenerated with color palette.")
