"""Grid search over combine.py's (prune_ratio x distill) config space, computing the
Pareto frontier of accuracy/MCC vs. energy. This is the actual "propose a framework"
deliverable (see combine.py's docstring for the plan citation and honest novelty
framing) -- the frontier itself is the proposed output, not a single chosen winner;
which frontier point to recommend for deployment is a paper-writing decision (state the
tradeoff, don't pick one to imply objectivity a single number doesn't have).

Grid (8 cells per dataset): prune_ratio in {0%, 25%, 50%, 75%} x distill in {no, yes}.
0%/no-distill is the plain baseline; 50%/no-distill and 0%/distill duplicate cells this
project already has numbers for (prune.py's own 50% run, distill.py's own run) -- run
fresh through this pipeline anyway for one consistent measurement methodology and CSV
schema, rather than reconciling three different scripts' output formats. If quota is
tight when this is actually run, skipping those two redundant cells and splicing in the
existing `results/{dataset}/prune_50%_recovered_metrics.csv` /
`distill_metrics.csv` numbers is a legitimate shortcut -- not done by default here for
simplicity and measurement consistency.

STATUS: written but not yet run on Kaggle -- see combine.py's own STATUS note. Smoke-
test with --smoke_test (tiny data, 1 epoch per cell, all 8 cells) before spending quota
on the full grid.

Usage: python search.py --dataset liar
       python search.py --dataset liar --smoke_test
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from combine import run_config  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

GRID = [
    (distill, prune_ratio)
    for distill in (False, True)
    for prune_ratio in (0.0, 0.25, 0.5, 0.75)
]


def is_dominated(a: dict, b: dict) -> bool:
    """True if config b dominates config a: b has >= accuracy AND <= energy, with at
    least one strictly better -- the standard two-objective Pareto dominance check."""
    acc_better_or_eq = b["mcc"] >= a["mcc"]
    energy_better_or_eq = b["energy_j_total"] <= a["energy_j_total"]
    strictly_better = b["mcc"] > a["mcc"] or b["energy_j_total"] < a["energy_j_total"]
    return acc_better_or_eq and energy_better_or_eq and strictly_better


def pareto_frontier(results: list[dict]) -> list[dict]:
    return [r for r in results if not any(is_dominated(r, other) for other in results if other is not r)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="liar", choices=["liar", "isot", "fakenewsnet"])
    parser.add_argument("--smoke_test", action="store_true",
                         help="Tiny run per cell to check the whole grid doesn't crash before spending quota.")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    for distill, prune_ratio in GRID:
        result = run_config(args.dataset, prune_ratio, distill, device, smoke_test=args.smoke_test)
        results.append(result)

    if args.smoke_test:
        print(f"Smoke test finished without crashing: {len(results)} configs ran end-to-end.")
        return

    results_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    fieldnames = ["config", "dataset", "distill", "prune_ratio", "accuracy", "precision",
                  "recall", "f1", "mcc", "energy_j_total", "energy_method"]
    with open(os.path.join(results_dir, "search_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote {results_dir}/search_results.csv ({len(results)} configs)")

    frontier = pareto_frontier(results)
    frontier.sort(key=lambda r: r["energy_j_total"])
    with open(os.path.join(results_dir, "pareto_frontier.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frontier)
    print(f"Wrote {results_dir}/pareto_frontier.csv ({len(frontier)} of {len(results)} configs on the frontier)")

    for r in frontier:
        print(f"  {r['config']}: MCC={r['mcc']:.4f}, energy={r['energy_j_total']:.1f}J")


if __name__ == "__main__":
    main()
