"""Download FakeNewsNet via a Kaggle mirror (mohamedgreshamahdi/fakenewsnet) that ships the
real politifact/gossipcop-scale CSVs (politifact_fake.csv, politifact_real.csv,
gossipcop_fake.csv, gossipcop_real.csv; ~23k rows total) — avoids the official
KaiDMML/FakeNewsNet repo's live Twitter/article scraper, which has documented failure
modes (open issues #12, #16, #70).

NOTE: an earlier version of this script pointed at mdepak/fakenewsnet, which turned out to
be a *different*, much smaller dataset (the old BuzzFeed/PolitiFact news-content set,
~337 rows total, filenames like "PolitiFact_fake_news_content.csv") despite the
misleadingly identical "fakenewsnet" slug — that produced a real, chance-level (50%
accuracy, MCC 0.0) training run, not a code bug. mohamedgreshamahdi/fakenewsnet ships both
that legacy set AND the real large-scale lowercase politifact_*/gossipcop_* files; only
the lowercase four are used here, matched by exact filename, not a loose "fake"/"real"
substring (which would re-pull in the legacy files too).

Scope (per sign-off): title-only classification, politifact + gossipcop combined. This is
narrower than the original multimodal/propagation-aware benchmark — name that limitation
in the P1 paper text. Label convention: 1 = real, 0 = fake.

Not using kagglehub.dataset_download() unconditionally: inside a Kaggle *kernel*,
kagglehub's kaggle_cache_resolver tries to dynamically attach the dataset via Kaggle's
internal API, which refuses in non-interactive/batch kernel runs. Fix: pre-attach this
dataset as a dataset_source in kernel-metadata.json and read it from /kaggle/input
directly (same fix as download_isot.py). Falls back to kagglehub for local runs.
"""

import glob
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SEED  # noqa: E402

KAGGLE_DATASET = "mohamedgreshamahdi/fakenewsnet"
KAGGLE_SLUG = "fakenewsnet"
TARGET_FILES = ("politifact_fake.csv", "politifact_real.csv", "gossipcop_fake.csv", "gossipcop_real.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")


def _resolve_source_dir() -> str:
    # Owner-qualified glob, not just the bare slug: "fakenewsnet" also matches our own
    # p1_baseline/results/fakenewsnet/ output folder (same name, different thing) when
    # that's mounted alongside this dataset — bit us once already (real bug, not
    # hypothetical). Owner qualification rules that false match out.
    owner = KAGGLE_DATASET.split("/")[0]
    matches = glob.glob(f"/kaggle/input/**/{owner}/{KAGGLE_SLUG}", recursive=True)
    if matches:
        return matches[0]
    import kagglehub  # local/non-Kaggle fallback only

    return kagglehub.dataset_download(KAGGLE_DATASET)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = _resolve_source_dir()

    frames = []
    for filename in TARGET_FILES:
        matches = glob.glob(os.path.join(path, "**", filename), recursive=True)
        assert matches, f"{filename} not found under {path}"
        label = 1 if "real" in filename else 0
        df = pd.read_csv(matches[0])
        df = df[["title"]].rename(columns={"title": "text"})
        df["label"] = label
        frames.append(df)

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text"])
    df["dataset"] = "fakenewsnet"

    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED)

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out_path = os.path.join(OUT_DIR, f"fakenewsnet_{name}.parquet")
        split_df.to_parquet(out_path, index=False)
        print(f"fakenewsnet_{name}: {len(split_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
