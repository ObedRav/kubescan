"""
train_rf.py
============
Train Layer 1 Random Forest classifier on rf_dataset.csv.

Features (24 columns):
  - 15 binary Rahman misconfiguration flags (excl. zero-variance cols)
  - 3 derived: cap_misuse, all_secrets, total_misconfigs
  - 6 extended: NO_RUN_AS_NON_ROOT … UNTRUSTED_REGISTRY
    (filled from checkov equivalents where available; imputed otherwise)

Target: label (0=secure, 1=misconfigured)

Also trains a 3-class severity model (label 0/1/2) as secondary output.

Outputs:
  models/checkpoints/rf_model.skops       – trained binary RF (skops format)
  models/checkpoints/rf_severity.skops    – trained 3-class RF
  models/checkpoints/rf_results.json      – metrics + feature importances
  models/checkpoints/rf_cv_results.json   – 5-fold CV results

Usage:
  python models/train_rf.py
  python models/train_rf.py --no-cv       # skip cross-validation
  python models/train_rf.py --seed 123
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from skops.io import dump as skops_dump

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# 15 Rahman binary flags (dropping 0-variance and near-constant cols)
RAHMAN_FEATURES = [
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET", "DOCKERSOCK_PATH",
    "CAP_SYS_ADMIN", "CAP_SYS_MODULE", "WITHIN_MANIFEST_SECRET",
    "SEC_CONT_OVER_PRIVIL", "ALLOW_PRIVI", "INSECURE_HTTP",
    "NO_SECU_CONTEXT", "HOST_ALIAS", "NO_DEFAULT_NSPACE",
    "NO_RESO", "NO_ROLLING_UPDATE",
]

DERIVED_FEATURES = ["cap_misuse", "all_secrets", "total_misconfigs"]

EXTENDED_FEATURES = [
    "NO_RUN_AS_NON_ROOT", "NO_READ_ONLY_ROOT_FS", "IMAGE_USES_LATEST",
    "SA_AUTOMOUNT_TOKEN", "USES_DEFAULT_SA", "UNTRUSTED_REGISTRY",
    "HOSTPATH_MOUNT",   # non-docker-sock hostPath volume (host FS escape)
]

# Checkov equivalents for extended features (used as fill-in for has_yaml=1 rows)
CHECKOV_FILL = {
    "NO_RUN_AS_NON_ROOT":   "checkov_no_run_as_nonroot",
    "NO_READ_ONLY_ROOT_FS": "checkov_no_readonly_rootfs",
    "IMAGE_USES_LATEST":    "checkov_image_latest",
    "SA_AUTOMOUNT_TOKEN":   "checkov_sa_automount",
    "USES_DEFAULT_SA":      "checkov_default_sa",
    "UNTRUSTED_REGISTRY":   "checkov_untrusted_registry",
    # HOSTPATH_MOUNT: no direct Checkov equivalent — read from column directly
}

ALL_FEATURES = RAHMAN_FEATURES + DERIVED_FEATURES + EXTENDED_FEATURES
# Total: 15 + 3 + 7 = 25 features


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------

def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Load rf_dataset.csv and return (X, y_binary, y_severity, feature_names).
    Extended features are filled from checkov equivalents, then median-imputed.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"  Loaded {len(rows)} rows")

    # ------------------------------------------------------------------
    # Step 1: Fill extended features from checkov equivalents
    #         (for rows where has_yaml=1)
    # ------------------------------------------------------------------
    for row in rows:
        for ext_col, ckv_col in CHECKOV_FILL.items():
            if row.get(ext_col, "") == "" and row.get(ckv_col, "") != "":
                row[ext_col] = row[ckv_col]

    # ------------------------------------------------------------------
    # Step 2: Build raw matrix (NaN for still-empty cells)
    # ------------------------------------------------------------------
    X_raw = np.full((len(rows), len(ALL_FEATURES)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(ALL_FEATURES):
            v = row.get(col, "")
            if v != "":
                try:
                    X_raw[i, j] = float(v)
                except (ValueError, TypeError):
                    pass

    # ------------------------------------------------------------------
    # Step 3: Median imputation per column
    # ------------------------------------------------------------------
    col_medians = np.nanmedian(X_raw, axis=0)
    # Conservative overrides (from dataset_config.json)
    for j, col in enumerate(ALL_FEATURES):
        if col == "SA_AUTOMOUNT_TOKEN":
            col_medians[j] = 1.0   # default K8s: automount on
        elif col == "UNTRUSTED_REGISTRY":
            col_medians[j] = 0.0   # assume trusted unless known otherwise

    inds = np.where(np.isnan(X_raw))
    X_raw[inds] = np.take(col_medians, inds[1])

    # ------------------------------------------------------------------
    # Step 4: Build label arrays
    # ------------------------------------------------------------------
    y_binary = np.array([int(r.get("label", 0) or 0) for r in rows], dtype=np.int64)

    y_severity = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        sv = row.get("severity_class", "")
        if sv != "":
            try:
                y_severity[i] = int(float(sv))
            except (ValueError, TypeError):
                y_severity[i] = y_binary[i]
        else:
            y_severity[i] = y_binary[i]

    # Log imputation stats
    filled_from_checkov = sum(
        1 for row in rows
        for ext_col in EXTENDED_FEATURES
        if ext_col in CHECKOV_FILL
        and row.get(ext_col, "") != "" and row.get(CHECKOV_FILL[ext_col], "") != ""
    )
    print(f"  Extended features: filled from Checkov for ~{filled_from_checkov//6} rows; "
          f"remainder imputed with column median")
    print(f"  label=0: {sum(y_binary==0)}, label=1: {sum(y_binary==1)}")
    print(f"  severity 0/1/2: {sum(y_severity==0)}/{sum(y_severity==1)}/{sum(y_severity==2)}")

    repo_names = [r.get("repo_name", "unknown") for r in rows]
    return X_raw, y_binary, y_severity, ALL_FEATURES, repo_names


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None, num_classes: int = 2) -> dict:
    labels = list(range(num_classes))
    metrics = {
        "accuracy":   float(accuracy_score(y_true, y_pred)),
        "macro_f1":   float(f1_score(y_true, y_pred, labels=labels,
                                     average="macro", zero_division=0)),
        "precision":  float(precision_score(y_true, y_pred, labels=labels,
                                            average="macro", zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, labels=labels,
                                         average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels,
                                  average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if y_prob is not None and num_classes == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        except Exception:
            pass
    return metrics


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    rf_params: dict,
    n_splits: int = 5,
    seed: int = 42,
    num_classes: int = 2,
) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_pred, y_prob, num_classes)
        fold_results.append(metrics)
        print(f"    fold {fold_idx}: macro-F1={metrics['macro_f1']:.4f}  "
              f"acc={metrics['accuracy']:.4f}", end="")
        if "roc_auc" in metrics:
            print(f"  AUC={metrics['roc_auc']:.4f}", end="")
        print()

    agg = {}
    for key in ["macro_f1", "accuracy", "precision", "recall"]:
        vals = [r[key] for r in fold_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"]  = float(np.std(vals))
    if "roc_auc" in fold_results[0]:
        auc_vals = [r["roc_auc"] for r in fold_results]
        agg["roc_auc_mean"] = float(np.mean(auc_vals))
        agg["roc_auc_std"]  = float(np.std(auc_vals))

    agg["folds"] = fold_results
    return agg


# ---------------------------------------------------------------------------
# Leave-One-Cluster-Out CV (grouped by repo_name)
# ---------------------------------------------------------------------------

def run_loco_cv(
    X: np.ndarray,
    y: np.ndarray,
    repo_names: list[str],
    rf_params: dict,
    num_classes: int = 2,
    min_val_size: int = 5,
) -> dict:
    """
    Leave-one-repo-out cross-validation.

    Holds out each unique repo_name in turn as the validation set,
    trains on all other repos. Repos with fewer than min_val_size rows
    are skipped (too small to give meaningful fold metrics).

    More conservative than stratified k-fold because it cannot accidentally
    train on manifests from the same repo as the test manifests.
    """
    from collections import defaultdict as _dd
    groups: dict[str, list[int]] = _dd(list)
    for i, rn in enumerate(repo_names):
        groups[rn].append(i)

    fold_results = []
    skipped = []
    for repo, val_idx in sorted(groups.items()):
        if len(val_idx) < min_val_size:
            skipped.append(repo)
            continue
        train_idx = [i for i, rn in enumerate(repo_names) if rn != repo]
        if len(train_idx) < 10:
            skipped.append(repo)
            continue

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Skip folds where val set has only one class
        if len(set(y_val)) < 2:
            skipped.append(repo)
            continue

        clf = RandomForestClassifier(**rf_params)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)
        metrics = compute_metrics(y_val, y_pred, y_prob, num_classes)
        metrics["repo"] = repo
        metrics["val_size"] = len(val_idx)
        fold_results.append(metrics)

    if not fold_results:
        return {"error": "No valid LOCO folds (all repos too small or single-class)"}

    agg = {}
    for key in ["macro_f1", "accuracy"]:
        vals = [r[key] for r in fold_results]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"]  = float(np.std(vals))
        agg[f"{key}_min"]  = float(np.min(vals))
    if "roc_auc" in fold_results[0]:
        auc_vals = [r["roc_auc"] for r in fold_results if "roc_auc" in r]
        if auc_vals:
            agg["roc_auc_mean"] = float(np.mean(auc_vals))
            agg["roc_auc_std"]  = float(np.std(auc_vals))

    agg["n_folds"]  = len(fold_results)
    agg["skipped"]  = skipped
    agg["folds"]    = fold_results
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent
    default_csv  = project_root / "data" / "tabular" / "rf_dataset.csv"
    default_out  = project_root / "models" / "checkpoints"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rf-dataset", type=Path, default=default_csv)
    parser.add_argument("--out-dir",    type=Path, default=default_out)
    parser.add_argument("--test-size",  type=float, default=0.20)
    parser.add_argument("--no-cv",      action="store_true")
    parser.add_argument("--loco-cv",    action="store_true",
                        help="Also run leave-one-repo-out cross-validation")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--ablation",   action="store_true",
                        help="Ablation: remove total_misconfigs to test for circular features")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.rf_dataset}")
    X, y_binary, y_severity, feature_names, repo_names = load_dataset(args.rf_dataset)
    print(f"  X shape: {X.shape}")

    # Ablation: drop total_misconfigs feature
    if args.ablation:
        if "total_misconfigs" in feature_names:
            drop_idx = feature_names.index("total_misconfigs")
            X = np.delete(X, drop_idx, axis=1)
            feature_names = [f for f in feature_names if f != "total_misconfigs"]
            print(f"  [ABLATION] Removed total_misconfigs (was index {drop_idx}). "
                  f"New shape: {X.shape}")
        else:
            print("  [ABLATION] total_misconfigs not found — skipping")

    # RF hyperparameters (tuned for class imbalance + feature correlations)
    rf_params = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": args.seed,
        "oob_score": True,
    }

    # ------------------------------------------------------------------
    # BINARY CLASSIFICATION (label 0/1)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("BINARY CLASSIFICATION  (secure vs. misconfigured)")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=args.test_size,
        stratify=y_binary, random_state=args.seed
    )
    print(f"  Train: {len(X_train)} ({sum(y_train==1)} misconfigured)  "
          f"Test: {len(X_test)} ({sum(y_test==1)} misconfigured)")

    rf_binary = RandomForestClassifier(**rf_params)
    rf_binary.fit(X_train, y_train)

    y_pred = rf_binary.predict(X_test)
    y_prob = rf_binary.predict_proba(X_test)
    metrics_bin = compute_metrics(y_test, y_pred, y_prob, num_classes=2)

    print("\n  Test results:")
    print(f"    Macro-F1  : {metrics_bin['macro_f1']:.4f}  (target: > 0.85)")
    print(f"    Accuracy  : {metrics_bin['accuracy']:.4f}")
    print(f"    AUC-ROC   : {metrics_bin.get('roc_auc', 'n/a'):.4f}")
    print(f"    OOB score : {rf_binary.oob_score_:.4f}")
    print("\n    Confusion matrix (rows=true, cols=pred):")
    for row in metrics_bin["confusion_matrix"]:
        print(f"      {row}")
    print(f"\n    Per-class F1:  secure={metrics_bin['per_class_f1'][0]:.4f}  "
          f"misconfigured={metrics_bin['per_class_f1'][1]:.4f}")
    print("\n    Full classification report:")
    print(classification_report(y_test, y_pred, target_names=["secure", "misconfigured"],
                                 zero_division=0))

    # Feature importances
    importances = rf_binary.feature_importances_
    top_idx = np.argsort(importances)[::-1]
    print("\n  Top 10 feature importances:")
    for i in top_idx[:10]:
        print(f"    {feature_names[i]:30s}: {importances[i]:.4f}")

    # Cross-validation
    cv_results_bin = {}
    if not args.no_cv:
        print("\n  5-fold cross-validation (binary):")
        cv_results_bin = run_cv(X, y_binary, rf_params, n_splits=5,
                                 seed=args.seed, num_classes=2)
        print("\n  CV summary:")
        print(f"    Macro-F1  : {cv_results_bin['macro_f1_mean']:.4f} ± {cv_results_bin['macro_f1_std']:.4f}")
        print(f"    Accuracy  : {cv_results_bin['accuracy_mean']:.4f} ± {cv_results_bin['accuracy_std']:.4f}")
        if "roc_auc_mean" in cv_results_bin:
            print(f"    AUC-ROC   : {cv_results_bin['roc_auc_mean']:.4f} ± {cv_results_bin['roc_auc_std']:.4f}")

    # Leave-one-repo-out cross-validation
    loco_results_bin = {}
    if args.loco_cv:
        print("\n  Leave-one-repo-out CV (binary):")
        loco_results_bin = run_loco_cv(X, y_binary, repo_names, rf_params, num_classes=2)
        if "error" not in loco_results_bin:
            print(f"    Folds evaluated: {loco_results_bin['n_folds']}  "
                  f"(skipped: {len(loco_results_bin['skipped'])})")
            print(f"    Macro-F1 : {loco_results_bin['macro_f1_mean']:.4f} ± "
                  f"{loco_results_bin['macro_f1_std']:.4f}  "
                  f"(min={loco_results_bin['macro_f1_min']:.4f})")
            print(f"    Accuracy : {loco_results_bin['accuracy_mean']:.4f} ± "
                  f"{loco_results_bin['accuracy_std']:.4f}")
            if "roc_auc_mean" in loco_results_bin:
                print(f"    AUC-ROC  : {loco_results_bin['roc_auc_mean']:.4f} ± "
                      f"{loco_results_bin['roc_auc_std']:.4f}")
            print(f"\n    Note: stratified 5-fold F1={cv_results_bin.get('macro_f1_mean', 0):.4f} vs "
                  f"LOCO F1={loco_results_bin['macro_f1_mean']:.4f} — gap indicates sampling bias")
        else:
            print(f"    {loco_results_bin['error']}")

    # ------------------------------------------------------------------
    # SEVERITY CLASSIFICATION (0/1/2)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SEVERITY CLASSIFICATION  (clean / low-medium / high-critical)")
    print(f"{'='*60}")

    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, y_severity, test_size=args.test_size,
        stratify=y_severity, random_state=args.seed
    )
    print(f"  Train: {len(X_tr2)}  Test: {len(X_te2)}")

    rf_sev = RandomForestClassifier(**rf_params)
    rf_sev.fit(X_tr2, y_tr2)
    y_pred2 = rf_sev.predict(X_te2)
    metrics_sev = compute_metrics(y_te2, y_pred2, num_classes=3)

    print("\n  Test results:")
    print(f"    Macro-F1  : {metrics_sev['macro_f1']:.4f}")
    print(f"    Accuracy  : {metrics_sev['accuracy']:.4f}")
    print(f"    OOB score : {rf_sev.oob_score_:.4f}")
    print(f"\n    Per-class F1: "
          f"clean={metrics_sev['per_class_f1'][0]:.4f}  "
          f"low_med={metrics_sev['per_class_f1'][1]:.4f}  "
          f"high_crit={metrics_sev['per_class_f1'][2]:.4f}")
    print("\n    Full classification report:")
    print(classification_report(y_te2, y_pred2,
                                 target_names=["clean", "low_medium", "high_critical"],
                                 zero_division=0))

    cv_results_sev = {}
    if not args.no_cv:
        print("\n  5-fold cross-validation (severity):")
        cv_results_sev = run_cv(X, y_severity, rf_params, n_splits=5,
                                 seed=args.seed, num_classes=3)
        print("\n  CV summary:")
        print(f"    Macro-F1  : {cv_results_sev['macro_f1_mean']:.4f} ± {cv_results_sev['macro_f1_std']:.4f}")
        print(f"    Accuracy  : {cv_results_sev['accuracy_mean']:.4f} ± {cv_results_sev['accuracy_std']:.4f}")

    # ------------------------------------------------------------------
    # Save models and results
    # ------------------------------------------------------------------
    # skops: type-validated serialization (no arbitrary code execution on load),
    # consumed by kubescan.model.rf_classifier
    skops_dump(rf_binary, args.out_dir / "rf_model.skops")
    skops_dump(rf_sev,    args.out_dir / "rf_severity.skops")

    results = {
        "feature_names": feature_names,
        "rf_params":     rf_params,
        "binary": {
            "test_metrics":   metrics_bin,
            "cv_metrics":     cv_results_bin,
            "loco_cv_metrics": loco_results_bin,
            "feature_importances": {
                feature_names[i]: float(importances[i])
                for i in top_idx
            },
            "oob_score": float(rf_binary.oob_score_),
        },
        "severity": {
            "test_metrics": metrics_sev,
            "cv_metrics":   cv_results_sev,
            "oob_score":    float(rf_sev.oob_score_),
        },
        "target": {
            "binary_f1_target": 0.85,
            "binary_f1_achieved": metrics_bin["macro_f1"],
            "target_met": metrics_bin["macro_f1"] >= 0.85,
        },
    }

    results_path = args.out_dir / "rf_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("  Models saved:")
    print(f"    {args.out_dir / 'rf_model.skops'}")
    print(f"    {args.out_dir / 'rf_severity.skops'}")
    print(f"  Results: {results_path}")
    target_met = "✓ TARGET MET" if metrics_bin["macro_f1"] >= 0.85 else "✗ below target (0.85)"
    print(f"\n  Binary F1 = {metrics_bin['macro_f1']:.4f}  [{target_met}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
