"""Download ISOT (Ahmed, Traore, Saad 2018) via Kaggle mirror, build our own stratified
80/10/10 split (no official split exists in the source paper), save as parquet.

Label convention (consistent with download_liar.py): 1 = true/real, 0 = fake.

Not using kagglehub.dataset_download() unconditionally: inside a Kaggle *kernel*,
kagglehub's kaggle_cache_resolver tries to dynamically attach the dataset via Kaggle's
internal API, which refuses in non-interactive/batch kernel runs ("New Datasets cannot
be attached in non-interactive sessions"). Fix: pre-attach this dataset as a
dataset_source in kernel-metadata.json (mounted read-only, same as our own p1-baseline
dataset) and read it directly from /kaggle/input. Falls back to kagglehub for local runs
outside Kaggle, where that resolver isn't active.
"""

import glob
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SEED  # noqa: E402

KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"
KAGGLE_SLUG = "fake-and-real-news-dataset"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")


def _resolve_source_dir() -> str:
    matches = glob.glob(f"/kaggle/input/**/{KAGGLE_SLUG}", recursive=True)
    if matches:
        return matches[0]
    import kagglehub  # local/non-Kaggle fallback only

    return kagglehub.dataset_download(KAGGLE_DATASET)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = _resolve_source_dir()

    true_df = pd.read_csv(os.path.join(path, "True.csv"))
    fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))
    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df["dataset"] = "isot"
    df = df[["text", "label", "dataset"]]

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_path = os.path.join(OUT_DIR, f"isot_{name}.parquet")
        split_df.to_parquet(out_path, index=False)
        print(f"isot_{name}: {len(split_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
