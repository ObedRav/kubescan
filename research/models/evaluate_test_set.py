"""
evaluate_test_set.py
====================
Evaluate the full 3-layer ensemble on the held-out test set.

Fixes addressed:
  Fix 2 — First formal evaluation on test.txt (never run before).
  Fix 3 — Reports P@5 alongside explicit P@5_ceiling (= min(n_chains,5)/5).
  Fix 5 — Reduces GNN variance by averaging softmax probs across all 5 fold
           models (gnn_fold_0.pt … gnn_fold_4.pt) instead of using gnn_best.pt
           alone.  Fold diversity acts as an implicit multi-model ensemble.

Ensemble scoring (uses weights from ga_weights.json):
    score(C) = w_rf * mean_rf_risk(C)
             + w_gnn * mean_gnn_chain_prob(C)   ← averaged across 5 folds
             + w_escape * escape_signal(C)      ← binary, same as GA tuning + CLI

escape_signal is the BINARY escape indicator (1.0 if any node has an escape
flag), matching run_ga_ensemble.py and kubescan.model.ga_ensemble exactly —
the evaluated quantity is the deployed quantity. escape_fraction is still
reported for context but never used for scoring.

Also reports:
  - Component ablation on the test set (escape-only / RF-only / GNN-only /
    equal weights / GA-best) so the marginal value of each layer is explicit.
  - 95% bootstrap confidence intervals (resampling test graphs) for P@5,
    macro-F1 and FPR_clean — mandatory context for a 15-graph test set.

Output:
    models/checkpoints/test_results.json
    (printed report to stdout)

Usage:
    python models/evaluate_test.py
    python models/evaluate_test.py --weights models/checkpoints/ga_weights.json
    python models/evaluate_test.py --show-rankings
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "04_build_datasets"))

try:
    import torch
    import torch.nn.functional as F
    from gnn_dataset import load_split
    from torch_geometric.loader import DataLoader
    HAS_TORCH = True
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nInstall: pip install torch torch-geometric")

try:
    from train_gnn import KubeGAT
except ImportError as e:
    sys.exit(f"Cannot import KubeGAT: {e}")

# Canonical escape-flag indices from the kubescan package (single source of truth)
from kubescan.model.ga_ensemble import ESCAPE_FLAG_INDICES
from kubescan.utils.device_utils import dataloader_kwargs, resolve_device

LABEL_MAP = {0: "clean", 1: "isolated", 2: "attack_chain"}


# ---------------------------------------------------------------------------
# GNN fold ensemble inference
# ---------------------------------------------------------------------------

def ensemble_predict(
    dataset,
    fold_models: list,
    device: torch.device,
) -> tuple[list[int], np.ndarray, list[float], list[float], list[float]]:
    """
    Average softmax probabilities across all fold models (Fix 5: reduce variance).

    Returns (true_labels, mean_probs[n,3], mean_rf_risks,
             escape_signals, escape_fracs).
    escape_signals is binary (scoring); escape_fracs is the per-cluster
    fraction (display only).
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False, **dataloader_kwargs(device))

    # Collect per-model softmax probs
    all_model_probs = []   # shape: [n_models][n_graphs][3]
    true_labels: list[int]    = []
    rf_risks:    list[float]  = []
    esc_signals: list[float]  = []
    esc_fracs:   list[float]  = []

    for model_idx, model in enumerate(fold_models):
        model.eval()
        model_probs = []

        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                probs = F.softmax(out, dim=-1).cpu().numpy()

                for g in range(batch.num_graphs):
                    model_probs.append(probs[g])

                    # Feature-based signals are model-independent: collect once
                    if model_idx == 0:
                        mask  = (batch.batch == g).cpu()
                        feats = batch.x[mask].cpu().numpy()
                        rf_risks.append(float(feats[:, -1].mean()))
                        esc_flags = feats[:, ESCAPE_FLAG_INDICES]
                        any_esc   = (esc_flags.max(axis=1) > 0)
                        esc_signals.append(1.0 if any_esc.any() else 0.0)
                        esc_fracs.append(float(any_esc.mean()))
                        true_labels.append(int(batch.y[g].item()))

        all_model_probs.append(model_probs)

    # Average probabilities across fold models
    mean_probs = np.array(all_model_probs).mean(axis=0)   # [n_graphs, 3]
    return true_labels, mean_probs, rf_risks, esc_signals, esc_fracs


# ---------------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------------

def precision_at_k(ranked_true: list[int], k: int, positive_label: int = 2) -> float:
    top_k = ranked_true[:k]
    return sum(1 for y in top_k if y == positive_label) / k if k else 0.0


def rank_metrics(
    scores: list[float],
    labels: list[int],
    k: int = 5,
) -> tuple[float, float]:
    """Return (P@k, FPR_clean@k) for one scoring of the test set."""
    ranked_idx  = sorted(range(len(labels)), key=lambda i: scores[i], reverse=True)
    ranked_true = [labels[i] for i in ranked_idx]
    k_eff = min(k, len(labels))
    p_k   = precision_at_k(ranked_true, k_eff)
    fpr   = sum(1 for y in ranked_true[:k_eff] if y == 0) / k_eff
    return p_k, fpr


def bootstrap_cis(
    scores: list[float],
    labels: list[int],
    preds: list[int],
    n_boot: int = 10_000,
    seed: int = 42,
    k: int = 5,
) -> dict:
    """
    95% percentile bootstrap CIs over test graphs for P@k, FPR_clean and
    macro-F1. With 15 graphs the intervals are wide by construction — that is
    precisely the information they convey.
    """
    rng = np.random.default_rng(seed)
    n   = len(labels)
    s   = np.asarray(scores)
    y   = np.asarray(labels)
    yhat = np.asarray(preds)

    p5s, fprs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p5, fpr = rank_metrics(s[idx].tolist(), y[idx].tolist(), k=k)
        p5s.append(p5)
        fprs.append(fpr)
        f1s.append(f1_score(y[idx], yhat[idx], average="macro", zero_division=0))

    def ci(vals: list[float]) -> tuple[float, float]:
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    return {
        "n_bootstrap":     n_boot,
        "p_at_k_ci95":     ci(p5s),
        "fpr_clean_ci95":  ci(fprs),
        "macro_f1_ci95":   ci(f1s),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    graphs_dir  = PROJECT_ROOT / "data" / "graphs"
    splits_dir  = PROJECT_ROOT / "data" / "splits"
    checkpoints = PROJECT_ROOT / "models" / "checkpoints"
    default_weights = checkpoints / "ga_weights.json"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--weights",       type=Path, default=default_weights,
                        help="Path to ga_weights.json")
    parser.add_argument("--hidden",        type=int,   default=64)
    parser.add_argument("--heads",         type=int,   default=4)
    parser.add_argument("--layers",        type=int,   default=3)
    parser.add_argument("--show-rankings", action="store_true",
                        help="Print per-cluster ranked scores")
    args = parser.parse_args()

    device = resolve_device()

    # ------------------------------------------------------------------
    # Load ensemble weights
    # ------------------------------------------------------------------
    if not args.weights.exists():
        sys.exit(f"ga_weights.json not found at {args.weights}. Run layer3_ga.py first.")

    with open(args.weights) as f:
        weights = json.load(f)

    w_rf     = weights["w_rf"]
    w_gnn    = weights["w_gnn"]
    w_escape = weights.get("w_escape", 0.0)
    print(f"Ensemble weights: w_rf={w_rf:.4f}  w_gnn={w_gnn:.4f}  w_escape={w_escape:.4f}")
    print(f"  (source: {weights.get('mode','?')} mode, "
          f"optimised on {weights.get('oof_n_graphs','?')} graphs, "
          f"{weights.get('oof_n_chains','?')} chains)")

    # ------------------------------------------------------------------
    # Load test set
    # ------------------------------------------------------------------
    test_file = splits_dir / "test.txt"
    if not test_file.exists():
        sys.exit(f"test.txt not found at {test_file}. Run create_splits.py first.")

    test_dataset = load_split(graphs_dir, test_file)
    print(f"\nTest set: {len(test_dataset)} graphs")

    # Report label distribution
    test_ids   = [t.strip() for t in test_file.read_text().splitlines() if t.strip()]
    test_labels_dist = Counter(test_dataset[i].y.item() for i in range(len(test_dataset)))
    n_chains   = test_labels_dist.get(2, 0)
    p5_ceiling = min(n_chains, 5) / 5
    print(f"  Distribution: clean={test_labels_dist.get(0,0)}  "
          f"isolated={test_labels_dist.get(1,0)}  "
          f"attack_chain={n_chains}")
    print(f"  P@5 ceiling: {p5_ceiling:.2f}  ({n_chains} chains / 5 → "
          f"{'ceiling=1.00' if p5_ceiling >= 1.0 else f'ceiling={p5_ceiling:.2f} (max achievable with {n_chains} chains)'})")

    # ------------------------------------------------------------------
    # Load all 5 fold models  (Fix 5: variance reduction via fold ensemble)
    # ------------------------------------------------------------------
    fold_models = []
    in_channels = test_dataset[0].x.shape[1]
    for fold_idx in range(5):
        model_path = checkpoints / f"gnn_fold_{fold_idx}.pt"
        if not model_path.exists():
            print(f"  [!] Missing gnn_fold_{fold_idx}.pt — skipping")
            continue
        model = KubeGAT(
            in_channels=in_channels,
            hidden=args.hidden,
            heads=args.heads,
            num_layers=args.layers,
            num_classes=3,
            dropout=0.3,
        ).to(device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        fold_models.append(model)

    if not fold_models:
        sys.exit("No fold models found. Run train_gnn.py first.")

    print(f"\nUsing {len(fold_models)}-model fold ensemble (gnn_fold_0 … gnn_fold_{len(fold_models)-1})")

    # ------------------------------------------------------------------
    # Inference — fold ensemble
    # ------------------------------------------------------------------
    true_labels, mean_probs, rf_risks, escape_signals, escape_fracs = ensemble_predict(
        test_dataset, fold_models, device
    )
    gnn_chain_probs = [float(p[2]) for p in mean_probs]
    n_test = len(true_labels)

    # ------------------------------------------------------------------
    # Ensemble scoring & ranking — binary escape signal (same as GA + CLI)
    # ------------------------------------------------------------------
    def score_with(wr: float, wg: float, we: float) -> list[float]:
        return [
            wr * rf_risks[i] + wg * gnn_chain_probs[i] + we * escape_signals[i]
            for i in range(n_test)
        ]

    ensemble_scores = score_with(w_rf, w_gnn, w_escape)

    ranked_idx  = sorted(range(n_test), key=lambda i: ensemble_scores[i], reverse=True)
    ranked_true = [true_labels[i] for i in ranked_idx]

    p_at_1 = precision_at_k(ranked_true, 1)
    p_at_3 = precision_at_k(ranked_true, 3)
    p_at_5 = precision_at_k(ranked_true, 5)

    fpr_clean = sum(1 for i in ranked_idx[:5] if true_labels[i] == 0) / 5

    # ------------------------------------------------------------------
    # Component ablation — what does each layer contribute on its own?
    # ------------------------------------------------------------------
    ablation_configs = [
        ("GA-best",      w_rf,  w_gnn, w_escape),
        ("Equal (1/3)",  1 / 3, 1 / 3, 1 / 3),
        ("RF-only",      1.0,   0.0,   0.0),
        ("GNN-only",     0.0,   1.0,   0.0),
        ("Escape-only",  0.0,   0.0,   1.0),
    ]
    ablation_results = []
    for name, wr, wg, we in ablation_configs:
        sc = score_with(wr, wg, we)
        ranked = [true_labels[i] for i in
                  sorted(range(n_test), key=lambda i, _sc=sc: _sc[i], reverse=True)]
        ablation_results.append({
            "config":    name,
            "w_rf":      round(wr, 4),
            "w_gnn":     round(wg, 4),
            "w_escape":  round(we, 4),
            "p_at_1":    precision_at_k(ranked, 1),
            "p_at_3":    precision_at_k(ranked, 3),
            "p_at_5":    precision_at_k(ranked, 5),
            "fpr_clean": sum(1 for y in ranked[:5] if y == 0) / 5,
        })
    p_at_5_gnn_only = next(r["p_at_5"] for r in ablation_results if r["config"] == "GNN-only")

    # ------------------------------------------------------------------
    # Classification metrics (fold-ensemble argmax)
    # ------------------------------------------------------------------
    preds    = mean_probs.argmax(axis=1).tolist()
    true_arr = np.array(true_labels)

    macro_f1 = float(f1_score(true_arr, preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(true_arr, preds, average=None,
                             labels=[0,1,2], zero_division=0).tolist()
    acc = float(accuracy_score(true_arr, preds))
    cm  = confusion_matrix(true_arr, preds, labels=[0,1,2]).tolist()

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals (95%, resampling test graphs)
    # ------------------------------------------------------------------
    cis = bootstrap_cis(ensemble_scores, true_labels, preds)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    SEP = "=" * 62
    print(f"\n{SEP}")
    print("FINAL TEST SET EVALUATION")
    print("(test clusters excluded from CV folds, GA tuning and augmentation)")
    print(SEP)
    print("\n  Ensemble ranking metrics:")
    print(f"    P@1        : {p_at_1:.2f}")
    print(f"    P@3        : {p_at_3:.2f}")
    print(f"    P@5        : {p_at_5:.2f}  (ceiling={p5_ceiling:.2f})  "
          f"95% CI [{cis['p_at_k_ci95'][0]:.2f}, {cis['p_at_k_ci95'][1]:.2f}]")
    print(f"    FPR_clean  : {fpr_clean:.2f}  "
          f"95% CI [{cis['fpr_clean_ci95'][0]:.2f}, {cis['fpr_clean_ci95'][1]:.2f}]")

    print("\n  Component ablation (test set):")
    print(f"    {'Config':14s} {'P@1':>5} {'P@3':>5} {'P@5':>5} {'FPR_clean':>10}")
    for r in ablation_results:
        print(f"    {r['config']:14s} {r['p_at_1']:>5.2f} {r['p_at_3']:>5.2f} "
              f"{r['p_at_5']:>5.2f} {r['fpr_clean']:>10.2f}")

    print("\n  Classification metrics (fold ensemble argmax):")
    print(f"    Macro-F1   : {macro_f1:.4f}  "
          f"95% CI [{cis['macro_f1_ci95'][0]:.2f}, {cis['macro_f1_ci95'][1]:.2f}]")
    print(f"    Accuracy   : {acc:.4f}")
    print("    Per-class F1:")
    for cls, name in LABEL_MAP.items():
        f1_val = per_class_f1[cls] if cls < len(per_class_f1) else 0.0
        print(f"      {name:15s}: {f1_val:.4f}")

    print("\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':15s} {'clean':>8} {'isolated':>10} {'attack':>8}")
    for i, row in enumerate(cm):
        print(f"  {LABEL_MAP[i]:15s} {row[0]:>8} {row[1]:>10} {row[2]:>8}")

    print(f"\n  GNN-only P@5 (no RF/escape):  {p_at_5_gnn_only:.2f}")
    print(f"  Ensemble P@5 (3-way):          {p_at_5:.2f}")

    if args.show_rankings:
        print("\n  Ranked clusters by ensemble score (top first):")
        print(f"  {'Rank':>4}  {'Cluster':40s}  {'True':>10}  {'EnsScore':>9}  "
              f"{'GNN_prob':>8}  {'RF_risk':>7}  {'Esc_frac':>8}")
        print(f"  {'-'*90}")
        for rank, idx in enumerate(ranked_idx):
            cid  = test_ids[idx] if idx < len(test_ids) else f"graph_{idx}"
            true = LABEL_MAP.get(true_labels[idx], str(true_labels[idx]))
            flag = " ← attack_chain" if true_labels[idx] == 2 else ""
            print(f"  {rank+1:>4}  {cid:40s}  {true:>10}  "
                  f"{ensemble_scores[idx]:>9.4f}  {gnn_chain_probs[idx]:>8.4f}  "
                  f"{rf_risks[idx]:>7.4f}  {escape_fracs[idx]:>8.4f}{flag}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "test_set": {
            "n_graphs":   len(true_labels),
            "n_chains":   n_chains,
            "p5_ceiling": p5_ceiling,
            "distribution": dict(test_labels_dist),
        },
        "ensemble": {
            "weights": {"w_rf": w_rf, "w_gnn": w_gnn, "w_escape": w_escape},
            "n_fold_models": len(fold_models),
        },
        "ranking_metrics": {
            "precision_at_1": p_at_1,
            "precision_at_3": p_at_3,
            "precision_at_5": p_at_5,
            "precision_at_5_ceiling": p5_ceiling,
            "hits_ceiling": p_at_5 >= p5_ceiling,
            "fpr_clean": fpr_clean,
        },
        "confidence_intervals_95": cis,
        "ablation": ablation_results,
        "classification_metrics": {
            "macro_f1":    macro_f1,
            "accuracy":    acc,
            "per_class_f1": {LABEL_MAP[i]: per_class_f1[i] for i in range(3)},
            "confusion_matrix": cm,
        },
        "gnn_only_p5": p_at_5_gnn_only,
        "ranked_clusters": [
            {
                "rank":           rank + 1,
                "cluster":        test_ids[idx] if idx < len(test_ids) else f"graph_{idx}",
                "true_label":     LABEL_MAP.get(true_labels[idx], str(true_labels[idx])),
                "ensemble_score": round(ensemble_scores[idx], 6),
                "gnn_chain_prob": round(gnn_chain_probs[idx], 6),
                "rf_mean_risk":   round(rf_risks[idx], 6),
                "escape_signal":  escape_signals[idx],
                "escape_fraction": round(escape_fracs[idx], 6),
            }
            for rank, idx in enumerate(ranked_idx)
        ],
    }

    out_path = checkpoints / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {out_path}")
    print(SEP)


if __name__ == "__main__":
    main()
