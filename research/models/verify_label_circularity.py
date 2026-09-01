#!/usr/bin/env python3
"""
Re-derive the Layer-1 headline on Rahman-only rows (clean INSECURE ground truth),
plus trivial baselines, reusing train_rf.py's exact loading/imputation/params.

Answers two register items:
  - §2.1  is OE2's 0.85 bar below a trivial baseline?
  - §2.2  does the headline survive without rule-labelled rows?
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "models"))

from train_rf import (  # noqa: E402
    ALL_FEATURES,
    CHECKOV_FILL,
    impute_with_medians,
)

CSV = REPO / "research" / "data" / "tabular" / "rf_dataset.csv"
SEED = 42
RF_PARAMS = dict(
    n_estimators=500, max_depth=None, min_samples_leaf=2, max_features="sqrt",
    class_weight="balanced", n_jobs=-1, random_state=SEED, oob_score=True,
)


def load(subset: set[str] | None):
    """Replicates train_rf.load_dataset, optionally filtered by source."""
    with CSV.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if subset is None or r.get("source", "") in subset]

    for row in rows:  # same Checkov backfill as the real pipeline
        for ext_col, ckv_col in CHECKOV_FILL.items():
            if row.get(ext_col, "") == "" and row.get(ckv_col, "") != "":
                row[ext_col] = row[ckv_col]

    X = np.full((len(rows), len(ALL_FEATURES)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(ALL_FEATURES):
            v = row.get(col, "")
            if v != "":
                try:
                    X[i, j] = float(v)
                except (ValueError, TypeError):
                    pass
    y = np.array([int(r.get("label", 0) or 0) for r in rows], dtype=np.int64)
    return X, y, rows


def rf_macro_f1(X, y, label):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    Xtr, Xte, _ = impute_with_medians(Xtr, Xte, ALL_FEATURES)
    clf = RandomForestClassifier(**RF_PARAMS).fit(Xtr, ytr)
    f1 = f1_score(yte, clf.predict(Xte), average="macro")
    print(f"  RF macro-F1 ({label}, n={len(y)}, test={len(yte)}): {f1:.4f}")
    return f1


def trivial(X, y, label):
    """Best single-feature rule + OR-of-all-flags, both as macro-F1."""
    Xi, _, _ = impute_with_medians(X, X, ALL_FEATURES)
    best = (None, -1.0)
    for j, name in enumerate(ALL_FEATURES):
        col = Xi[:, j]
        if not np.all(np.isin(col, [0.0, 1.0])):
            continue  # only binary flags qualify as a "one-rule" baseline
        f1 = f1_score(y, (col == 1).astype(int), average="macro")
        if f1 > best[1]:
            best = (name, f1)
    print(f"  best 1-rule ({label}): {best[0]} = {best[1]:.4f}")

    flags = [j for j, n in enumerate(ALL_FEATURES)
             if np.all(np.isin(Xi[:, j], [0.0, 1.0]))]
    orpred = (Xi[:, flags].sum(axis=1) > 0).astype(int)
    print(f"  OR(flags)   ({label}): {f1_score(y, orpred, average='macro'):.4f}")
    return best


print("=" * 64)
print("FULL CORPUS (2,827 rows — includes rule-labelled attack_repos)")
print("=" * 64)
Xf, yf, rf_rows = load(None)
rf_macro_f1(Xf, yf, "full")
trivial(Xf, yf, "full")

print()
print("=" * 64)
print("RAHMAN-ONLY (github + gitlab — genuine INSECURE ground truth)")
print("=" * 64)
Xr, yr, rr_rows = load({"github", "gitlab"})
rf_macro_f1(Xr, yr, "rahman")
trivial(Xr, yr, "rahman")
