"""Download LIAR (Wang 2017) directly from the original UCSB source, collapse to binary,
save the official train/valid/test split as parquet.

Not using HF `datasets load_dataset("liar")`: the Hub copy still ships a loading *script*,
and current `datasets` versions have dropped script support entirely ("Dataset scripts are
no longer supported"). Going straight to the source avoids that library-version churn.

Binary mapping (standard convention used across the comparative literature cited in the
research plan): {pants-fire, false, barely-true} -> 0 (false); {half-true, mostly-true, true} -> 1 (true).
"""

import csv
import io
import os
import urllib.request
import zipfile

import pandas as pd

URL = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"

# Original release ships no header row; column order per the dataset's own README.
COLUMNS = [
    "id", "label", "statement", "subject", "speaker", "job_title", "state_info",
    "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_on_fire_counts", "context",
]

FALSE_LABELS = {"pants-fire", "false", "barely-true"}
TRUE_LABELS = {"half-true", "mostly-true", "true"}

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")


def _parse_tsv(raw_bytes: bytes) -> pd.DataFrame:
    reader = csv.reader(io.StringIO(raw_bytes.decode("utf-8")), delimiter="\t")
    rows = [row for row in reader if row]
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["label"] = df["label"].map(lambda name: 1 if name in TRUE_LABELS else 0)
    df["text"] = df["statement"]
    df["dataset"] = "liar"
    return df[["text", "label", "dataset"]]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        zip_bytes = resp.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for split_name, member in (("train", "train.tsv"), ("val", "valid.tsv"), ("test", "test.tsv")):
            with zf.open(member) as f:
                df = _parse_tsv(f.read())
            out_path = os.path.join(OUT_DIR, f"liar_{split_name}.parquet")
            df.to_parquet(out_path, index=False)
            print(f"liar_{split_name}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
