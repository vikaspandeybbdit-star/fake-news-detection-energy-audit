"""Generalization checks for the IEEE Access revision (R1#7, R2#4), BERT-base, seed 42, same fixed training protocol
(3 epochs, best-validation-F1 epoch kept, FP16), test-split evaluation.

  --mode cross --train D     train on dataset D's train split, evaluate on the test splits of liar/isot/fakenewsnet
  --mode temporal            ISOT with a temporal split by publication date (train < 2017-07, val 2017-07..08, test >= 2017-09)
  --mode ablate              ISOT random split with source markers removed (Reuters dateline, 'featured image', agency credits)
Writes /kaggle/working/out/ood/<name>.csv (+ per-example predictions)."""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_seed  # noqa: E402
from run_seed import BERT, DATA_DIR, DEV, MAX_SEQ_LEN, BATCH_SIZE, Run  # noqa: E402

OUT = "/kaggle/working/out/ood" if os.path.isdir("/kaggle") else os.path.join(HERE, "..", "results_ood")
os.makedirs(OUT, exist_ok=True)


class OODRun(Run):
    def __init__(self, a, splits):
        super().__init__(a)
        self.splits = splits

    def data(self, tok, split):
        ds = Dataset.from_pandas(self.splits[split][["text", "label"]].reset_index(drop=True))
        if self.a.smoke:
            ds = ds.select(range(min(16, len(ds))))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="max_length"), batched=True)


def predict(model, tok, df):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(df), BATCH_SIZE):
            enc = tok(df["text"].iloc[i:i + BATCH_SIZE].tolist(), truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
                      return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                probs.append(model(**enc).logits.float().softmax(-1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def metrics_row(name, y, p1, **extra):
    pred = (p1 >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(name=name, n=len(y), pos_rate=float(np.mean(y)), accuracy=accuracy_score(y, pred), f1=f1, mcc=matthews_corrcoef(y, pred),
                tn=tn, fp=fp, fn=fn, tp=tp, **extra)


def fit(name, splits, smoke):
    a = argparse.Namespace(dataset=name, seed=42, out_dir=os.path.join(OUT, "run"), work_dir="/kaggle/temp" if os.path.isdir("/kaggle") else os.path.join(HERE, "_work_ood"), smoke=smoke)
    run = OODRun(a, splits)
    set_seed(42)
    tok = AutoTokenizer.from_pretrained(BERT)
    model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=2)
    ln = model.bert.embeddings.LayerNorm.weight
    assert not torch.allclose(ln, torch.ones_like(ln)), "pretrained LayerNorm not loaded"
    model = run.train(model.to(DEV), tok, name, 3)
    return model, tok


def load(ds):
    return {s: pd.read_parquet(os.path.join(DATA_DIR, f"{ds}_{s}.parquet")) for s in ("train", "val", "test")}


MARKERS = [r"\(reuters\)", r"reuters", r"featured image", r"21st century wire", r"getty images", r"image via", r"screengrab", r"screen capture"]


def clean(text):
    t = re.sub(r"^[^()]{0,60}\(Reuters\)\s*-?\s*", "", str(text), flags=re.I)
    for m in MARKERS:
        t = re.sub(m, " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cross", "temporal", "ablate"])
    ap.add_argument("--train", default="liar")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rows = []
    if a.mode == "cross":
        splits = load(a.train)
        model, tok = fit(f"cross_{a.train}", splits, a.smoke)
        for tgt in ("liar", "isot", "fakenewsnet"):
            te = load(tgt)["test"]
            te = te.iloc[:16] if a.smoke else te
            p1 = predict(model, tok, te)
            rows.append(metrics_row(f"train_{a.train}__test_{tgt}", te["label"].to_numpy(), p1, train=a.train, test=tgt))
            pd.DataFrame({"label": te["label"].to_numpy(), "p_true": p1}).to_csv(os.path.join(OUT, f"preds_{a.train}_to_{tgt}.csv"), index=False)
            print(rows[-1], flush=True)
        tag = f"cross_{a.train}"
    elif a.mode == "temporal":
        src = glob.glob("/kaggle/input/**/fake-and-real-news-dataset", recursive=True)
        path = src[0] if src else "."
        t = pd.read_csv(os.path.join(path, "True.csv")); t["label"] = 1
        f = pd.read_csv(os.path.join(path, "Fake.csv")); f["label"] = 0
        df = pd.concat([t, f], ignore_index=True)
        n_before = len(df)
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="mixed", errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        print("rows dropped for unparseable dates:", n_before - len(df), "of", n_before, flush=True)
        assert df["label"].nunique() == 2, "date parsing removed a whole class"
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
        print("date range by class:\n", df.groupby("label")["date"].agg(["min", "max", "count"]), flush=True)
        splits = {"train": df[df.date < "2017-07-01"], "val": df[(df.date >= "2017-07-01") & (df.date < "2017-09-01")], "test": df[df.date >= "2017-09-01"]}
        for k, v in splits.items():
            print(k, len(v), v["label"].value_counts().to_dict(), flush=True)
        model, tok = fit("temporal_isot", splits, a.smoke)
        te = splits["test"].iloc[:16] if a.smoke else splits["test"]
        p1 = predict(model, tok, te)
        rows.append(metrics_row("isot_temporal_test", te["label"].to_numpy(), p1, train_n=len(splits["train"]), val_n=len(splits["val"])))
        pd.DataFrame({"label": te["label"].to_numpy(), "p_true": p1}).to_csv(os.path.join(OUT, "preds_isot_temporal.csv"), index=False)
        print(rows[-1], flush=True)
        tag = "temporal"
    else:
        splits = load("isot")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        cl = {k: v.assign(text=v["text"].map(clean)) for k, v in splits.items()}
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        lr = LogisticRegression(max_iter=500).fit(vec.fit_transform(cl["train"]["text"]), cl["train"]["label"])
        p = lr.predict(vec.transform(cl["test"]["text"]))
        rows.append(dict(name="tfidf_logreg_cleaned_test", n=len(p), accuracy=accuracy_score(cl["test"]["label"], p), mcc=matthews_corrcoef(cl["test"]["label"], p)))
        print(rows[-1], flush=True)
        model, tok = fit("ablate_isot", cl, a.smoke)
        te = cl["test"].iloc[:16] if a.smoke else cl["test"]
        p1 = predict(model, tok, te)
        rows.append(metrics_row("bert_cleaned_isot_test", te["label"].to_numpy(), p1))
        pd.DataFrame({"label": te["label"].to_numpy(), "p_true": p1}).to_csv(os.path.join(OUT, "preds_isot_ablated.csv"), index=False)
        print(rows[-1], flush=True)
        tag = "ablate"
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    print("done", tag)


main()
