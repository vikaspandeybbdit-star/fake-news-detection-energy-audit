"""Majority-class and TF-IDF logistic-regression references: in-domain test metrics and transfer to the other datasets and the COVID-19 corpus."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p1_baseline", "results", "data")


def load(ds):
    return {s: pd.read_parquet(os.path.join(DATA_DIR, f"{ds}_{s}.parquet")) for s in ("train", "val", "test")}


def load_covid():
    from datasets import load_dataset
    d = load_dataset("nanyy1025/covid_fake_news")["test"].to_pandas()
    return pd.DataFrame({"text": d["tweet"].astype(str), "label": (d["label"].str.lower() == "real").astype(int)})

DS = ["liar", "isot", "fakenewsnet"]
data = {d: load(d) for d in DS}
data["covid"] = {"test": load_covid()}
out = {"majority": {}, "tfidf": {}}
for tr in DS:
    t = data[tr]["train"]
    maj = int(t["label"].mean() >= 0.5)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(t["text"].astype(str))
    clf = LogisticRegression(max_iter=2000, C=4.0).fit(X, t["label"])
    for te in DS + ["covid"]:
        e = data[te]["test"]
        y = e["label"].values
        pm = np.full(len(y), maj)
        pt = clf.predict(vec.transform(e["text"].astype(str)))
        out["majority"][f"{tr}|{te}"] = {"acc": float(accuracy_score(y, pm)), "mcc": float(matthews_corrcoef(y, pm))}
        out["tfidf"][f"{tr}|{te}"] = {"acc": float(accuracy_score(y, pt)), "mcc": float(matthews_corrcoef(y, pt))}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lexical_baselines.json"), "w"), indent=1)
for k in ("majority", "tfidf"):
    print(k)
    for tr in DS:
        print(" ", tr, {te: (round(out[k][f"{tr}|{te}"]["acc"], 3), round(out[k][f"{tr}|{te}"]["mcc"], 3)) for te in DS + ["covid"]})
