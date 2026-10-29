import os
"""Cheap ISOT leakage / artifact check (IEEE Access R2#4). CPU only.
1. Exact-duplicate text overlap between train/val/test.
2. Wire-service boilerplate frequency by class.
3. Trivial 'source-marker only' classifier (a single regex) vs. a TF-IDF logistic regression,
   to measure how separable ISOT is from surface features alone.
"""
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "p1_baseline", "results", "data") + "/"
tr, va, te = (pd.read_parquet(f"{D}isot_{s}.parquet") for s in ("train", "val", "test"))
print("columns:", list(tr.columns), "sizes:", len(tr), len(va), len(te))
tcol = "text" if "text" in tr.columns else tr.columns[0]
lcol = "label"
print("label balance train:", tr[lcol].value_counts().to_dict())

norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
sets = {n: set(norm(t) for t in d[tcol]) for n, d in (("train", tr), ("val", va), ("test", te))}
print("\n[1] exact duplicates across splits")
print("  train&test:", len(sets["train"] & sets["test"]), "of", len(sets["test"]), "unique test texts")
print("  train&val :", len(sets["train"] & sets["val"]), "of", len(sets["val"]))
print("  within-train duplicate rate:", 1 - len(sets["train"]) / len(tr))

print("\n[2] boilerplate marker frequency by class (label 1 = which class?)")
pat = {"(Reuters)": r"\(reuters\)", "Reuters (any)": r"reuters", "featured image": r"featured image",
       "21st century wire": r"21st century wire", "Getty/Image via": r"getty images|image via"}
for name, rx in pat.items():
    m = tr[tcol].str.lower().str.contains(rx, regex=True)
    print(f"  {name:20s}", tr.groupby(lcol)[tcol].apply(lambda s, rx=rx: s.str.lower().str.contains(rx, regex=True).mean()).round(3).to_dict())

print("\n[3] separability from surface features alone (test split)")
rx = r"\(reuters\)"
for lab_true in (0, 1):
    pred = te[tcol].str.lower().str.contains(rx, regex=True).astype(int)
    pred = pred if lab_true == 1 else 1 - pred
    print(f"  single-regex '(Reuters)' rule (marker => label {lab_true}): acc={accuracy_score(te[lcol], pred):.4f} mcc={matthews_corrcoef(te[lcol], pred):.4f}")
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
Xtr = vec.fit_transform(tr[tcol]); Xte = vec.transform(te[tcol])
clf = LogisticRegression(max_iter=500).fit(Xtr, tr[lcol])
p = clf.predict(Xte)
print(f"  TF-IDF + LogReg: acc={accuracy_score(te[lcol], p):.4f} mcc={matthews_corrcoef(te[lcol], p):.4f}")
head = te[tcol].str.slice(0, 40)
vec2 = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)).fit(tr[tcol].str.slice(0, 40))
clf2 = LogisticRegression(max_iter=500).fit(vec2.transform(tr[tcol].str.slice(0, 40)), tr[lcol])
p2 = clf2.predict(vec2.transform(head))
print(f"  first-40-chars-only LogReg: acc={accuracy_score(te[lcol], p2):.4f} mcc={matthews_corrcoef(te[lcol], p2):.4f}")
