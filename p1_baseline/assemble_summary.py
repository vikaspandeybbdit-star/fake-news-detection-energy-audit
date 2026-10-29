"""M5: assemble per-dataset energy_table.csv files (results/<dataset>/energy_table.csv)
into one cross-dataset summary CSV + markdown table for the P1 paper's results section.
"""

import csv
import os

from config import DATASETS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    rows = []
    for dataset in DATASETS:
        path = os.path.join(RESULTS_DIR, dataset, "energy_table.csv")
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["dataset"] = dataset
                rows.append(row)

    out_csv = os.path.join(RESULTS_DIR, "cross_dataset_summary.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "component", "time_share", "energy_j",
                                                "energy_method", "batch_size", "seq_len", "n_iters"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")

    out_md = os.path.join(RESULTS_DIR, "cross_dataset_summary.md")
    with open(out_md, "w") as f:
        f.write("| Dataset | Component | Time share | Energy (J) | Method |\n")
        f.write("|---|---|---:|---:|---|\n")
        for r in rows:
            f.write(f"| {r['dataset']} | {r['component']} | {float(r['time_share']):.1%} | "
                     f"{float(r['energy_j']):.2f} | {r['energy_method']} |\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
