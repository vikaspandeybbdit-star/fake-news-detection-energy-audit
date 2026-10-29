"""Verify all three datasets landed in the common {text,label,dataset} schema and report
row counts. Each download_*.py script already writes this schema directly, so there's no
merging to do here — just a sanity check before training starts.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATASETS  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")


def main():
    for dataset in DATASETS:
        for split in ("train", "val", "test"):
            path = os.path.join(DATA_DIR, f"{dataset}_{split}.parquet")
            if not os.path.exists(path):
                print(f"MISSING: {path} — run download_{dataset}.py first")
                continue
            df = pd.read_parquet(path)
            assert set(df.columns) >= {"text", "label", "dataset"}, f"{path} missing required columns"
            assert df["label"].isin([0, 1]).all(), f"{path} has non-binary labels"
            print(f"{dataset}_{split}: {len(df)} rows, label balance {df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
