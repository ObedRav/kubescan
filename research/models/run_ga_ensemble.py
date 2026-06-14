"""
run_ga_ensemble.py
==================
Layer 3 — Genetic Algorithm for ensemble weight optimization.

Combines RF risk scores (node-level), GNN chain probabilities (cluster-level),
and escape-flag fraction (direct from node features) by finding optimal weights
[w_rf, w_gnn, w_esc] that maximise Precision@5 while minimising false positives.

Ensemble score for cluster C:
    score(C) = w_rf * mean_rf_risk(C)
             + w_gnn * gnn_chain_prob(C)
             + w_esc * escape_signal(C)

where:
  mean_rf_risk     = average RF risk_score across all nodes in C (node feat index -1)
  gnn_chain_prob   = softmax prob for class 2 (attack_chain) from GNN
  escape_signal    = 1.0 if ANY node has ≥1 escape flag set, else 0.0 (binary)
                     (ESCAPE_FLAG_INDICES = [0,1,2,3,4,5,7,24] in node features)
                     Binary signal avoids dilution in large clusters where a single
                     escape-capable manifest is still a critical risk.

Two validation modes:
  --val (default): use val.txt + gnn_best.pt  (15 graphs, 4 chains, ceiling 0.80)
  --oof           : use all fold_X_val.txt + gnn_fold_X.pt out-of-fold predictions
                    (96 original graphs, 25 chains, ceiling 1.00)
                    Recommended — gives a much more reliable weight estimate.

Optimisation objective:
    F = α * Precision@5 + β * (1 - FPR_clean)

Output:
    models/checkpoints/ga_weights.json      — best weights found
    models/checkpoints/ga_results.json      — full GA run statistics

Usage:
    python models/layer3_ga.py
    python models/layer3_ga.py --oof                         # recommended
    python models/layer3_ga.py --oof --generations 200 --pop-size 60
    python models/layer3_ga.py --alpha 0.7 --beta 0.3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

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

from provenance import provenance

# Escape-flag indices in the 26-dim node feature vector — canonical
# definition derived in the kubescan package (single source of truth).
from kubescan.model.ga_ensemble import ESCAPE_FLAG_INDICES
from kubescan.utils.device_utils import dataloader_kwargs, resolve_device

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model(in_channels, hidden, heads, num_layers, device):
    model = KubeGAT(
        in_channels=in_channels,
        hidden=hidden,
        heads=heads,
        num_layers=num_layers,
        num_classes=3,
        dropout=0.3,
    )
    model.to(device)
    return model


def _infer_dataset(model, dataset, device):
    """Run inference and return (true_labels, gnn_chain_probs, rf_risks, escape_signals)."""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, **dataloader_kwargs(device))
    true_labels, gnn_probs, rf_risks, esc_signals = [], [], [], []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = F.softmax(out, dim=-1).cpu().numpy()

            for g in range(batch.num_graphs):
                mask  = (batch.batch == g).cpu()
                feats = batch.x[mask].cpu().numpy()

                rf_mean_risk = float(feats[:, -1].mean())

                esc_flags  = feats[:, ESCAPE_FLAG_INDICES]
                esc_signal = 1.0 if (esc_flags.max(axis=1) > 0).any() else 0.0

                true_labels.append(int(batch.y[g].item()))
                gnn_probs.append(float(probs[g, 2]))
                rf_risks.append(rf_mean_risk)
                esc_signals.append(esc_signal)

    return true_labels, gnn_probs, rf_risks, esc_signals


# ---------------------------------------------------------------------------
# Data loading — val.txt mode (original behaviour)
# ---------------------------------------------------------------------------

def load_val_predictions(
    graphs_dir: Path,
    splits_dir: Path,
    model_path: Path,
    device: torch.device,
    hidden: int = 64,
    heads: int = 4,
    num_layers: int = 3,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Load predictions from gnn_best.pt on val.txt (15 graphs)."""
    val_dataset = load_split(graphs_dir, splits_dir / "val.txt")
    in_channels = val_dataset[0].x.shape[1]

    model = _build_model(in_channels, hidden, heads, num_layers, device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    labels, probs, risks, escs = _infer_dataset(model, val_dataset, device)
    return labels, probs, risks, escs


# ---------------------------------------------------------------------------
# Data loading — OOF mode (Fix 6)
# ---------------------------------------------------------------------------

def load_oof_predictions(
    graphs_dir: Path,
    splits_dir: Path,
    checkpoints: Path,
    device: torch.device,
    num_folds: int = 5,
    hidden: int = 64,
    heads: int = 4,
    num_layers: int = 3,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """
    Out-of-fold predictions across all 5 folds.

    For each fold k, loads gnn_fold_k.pt and runs inference on fold_k_val.txt.
    Because each fold model never saw its val clusters during training, these
    are genuinely held-out predictions — much more reliable for weight tuning
    than a single 15-graph validation set.

    Returns predictions for all 96 original clusters (25 attack chains).
    """
    all_labels, all_probs, all_risks, all_escs = [], [], [], []

    for fold_idx in range(num_folds):
        val_file   = splits_dir / f"fold_{fold_idx}_val.txt"
        model_path = checkpoints / f"gnn_fold_{fold_idx}.pt"

        if not val_file.exists():
            print(f"  [!] Missing {val_file} — skipping fold {fold_idx}")
            continue
        if not model_path.exists():
            print(f"  [!] Missing {model_path} — skipping fold {fold_idx}")
            continue

        val_dataset = load_split(graphs_dir, val_file)
        in_channels = val_dataset[0].x.shape[1]

        model = _build_model(in_channels, hidden, heads, num_layers, device)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )

        labels, probs, risks, escs = _infer_dataset(model, val_dataset, device)
        all_labels.extend(labels)
        all_probs.extend(probs)
        all_risks.extend(risks)
        all_escs.extend(escs)

        from collections import Counter
        dist = Counter(labels)
        n_chains = dist.get(2, 0)
        print(f"  fold {fold_idx}: {len(labels)} clusters  "
              f"(chains={n_chains}, clean={dist.get(0,0)}, isolated={dist.get(1,0)})")

    return all_labels, all_probs, all_risks, all_escs


# ---------------------------------------------------------------------------
# Objective function  (Fix 4: now includes escape_signals + w_escape)
# ---------------------------------------------------------------------------

def compute_objective(
    true_labels:  list[int],
    gnn_probs:    list[float],
    rf_risks:     list[float],
    w_rf:         float,
    w_gnn:        float,
    escape_signals: list[float] | None = None,
    w_escape:     float = 0.0,
    k:            int   = 5,
    alpha:        float = 0.7,
    beta:         float = 0.3,
    chain_label:  int   = 2,
    clean_label:  int   = 0,
) -> dict:
    """
    Compute the GA objective for weights (w_rf, w_gnn, w_escape).
    Weights are normalised internally so they sum to 1.
    """
    # Normalise weights to sum to 1 (handles floating-point drift)
    total = w_rf + w_gnn + w_escape
    if total <= 0:
        total = 1.0
    w_rf /= total
    w_gnn /= total
    w_escape /= total

    n = len(true_labels)
    if escape_signals is not None and w_escape > 0:
        ensemble = [
            w_rf * rf_risks[i] + w_gnn * gnn_probs[i] + w_escape * escape_signals[i]
            for i in range(n)
        ]
    else:
        ensemble = [w_rf * rf_risks[i] + w_gnn * gnn_probs[i] for i in range(n)]

    ranked_idx = sorted(range(n), key=lambda i: ensemble[i], reverse=True)
    top_k_idx  = ranked_idx[:k]

    p_at_k    = sum(1 for i in top_k_idx if true_labels[i] == chain_label) / k
    fpr_clean = sum(1 for i in top_k_idx if true_labels[i] == clean_label) / k
    objective = alpha * p_at_k + beta * (1.0 - fpr_clean)

    return {
        "score":            objective,
        "p_at_k":           p_at_k,
        "fpr_clean":        fpr_clean,
        "w_rf":             w_rf,
        "w_gnn":            w_gnn,
        "w_escape":         w_escape,
        "ensemble_scores":  ensemble,
    }


# ---------------------------------------------------------------------------
# Grid search  (Fix 4: 2D grid over w_gnn × w_esc; Fix 6: larger dataset)
# ---------------------------------------------------------------------------

def grid_search(
    true_labels:  list[int],
    gnn_probs:    list[float],
    rf_risks:     list[float],
    escape_signals: list[float],
    n_steps:      int   = 20,
    k:            int   = 5,
    alpha:        float = 0.7,
    beta:         float = 0.3,
) -> dict:
    """
    2D exhaustive grid search over (w_gnn, w_esc) with w_rf = 1 - w_gnn - w_esc.
    n_steps=20 gives 231 evaluations — fast and deterministic.
    """
    best = None
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            w_gnn   = i / n_steps
            w_escape = j / n_steps
            w_rf    = 1.0 - w_gnn - w_escape
            result  = compute_objective(
                true_labels, gnn_probs, rf_risks,
                w_rf=w_rf, w_gnn=w_gnn,
                escape_signals=escape_signals, w_escape=w_escape,
                k=k, alpha=alpha, beta=beta,
            )
            if best is None or result["score"] > best["score"]:
                best = result
    return best


# ---------------------------------------------------------------------------
# Genetic Algorithm  (2D chromosome: [w_gnn, w_esc])
# ---------------------------------------------------------------------------

def run_ga(
    true_labels:  list[int],
    gnn_probs:    list[float],
    rf_risks:     list[float],
    escape_signals: list[float],
    pop_size:     int   = 60,
    generations:  int   = 150,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.7,
    elite_frac:   float = 0.1,
    k:            int   = 5,
    alpha:        float = 0.7,
    beta:         float = 0.3,
    seed:         int   = 42,
) -> dict:
    """
    2D GA: chromosome = [w_gnn, w_esc] ∈ [0,1]².
    w_rf = max(0, 1 − w_gnn − w_esc); all three are then normalised to sum=1.
    """
    rng    = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    n_elite = max(1, int(pop_size * elite_frac))

    # Initialise: sample simplex uniformly (Dirichlet)
    population = []
    for _ in range(pop_size):
        raw = np_rng.dirichlet([1, 1, 1])  # [w_rf, w_gnn, w_esc]
        population.append((float(raw[1]), float(raw[2])))

    best_ever = None
    history   = []

    for gen in range(generations):
        fitness = []
        for w_gnn, w_esc in population:
            w_rf = max(0.0, 1.0 - w_gnn - w_esc)
            result = compute_objective(
                true_labels, gnn_probs, rf_risks,
                w_rf=w_rf, w_gnn=w_gnn,
                escape_signals=escape_signals, w_escape=w_esc,
                k=k, alpha=alpha, beta=beta,
            )
            fitness.append((result["score"], w_gnn, w_esc, result))

        fitness.sort(key=lambda x: x[0], reverse=True)

        if best_ever is None or fitness[0][0] > best_ever["score"]:
            best_ever = fitness[0][3]

        gen_best = fitness[0][0]
        gen_mean = np.mean([f[0] for f in fitness])
        history.append({
            "gen": gen, "best": gen_best, "mean": gen_mean,
            "best_w_gnn": fitness[0][1], "best_w_esc": fitness[0][2],
        })

        if gen % 20 == 0 or gen == generations - 1:
            bw = fitness[0][3]
            print(f"  gen {gen:3d}: best={gen_best:.4f}  mean={gen_mean:.4f}  "
                  f"w_rf={bw['w_rf']:.3f}  w_gnn={bw['w_gnn']:.3f}  "
                  f"w_esc={bw['w_escape']:.3f}  "
                  f"P@{k}={bw['p_at_k']:.2f}  FPR_clean={bw['fpr_clean']:.2f}")

        # Elitism
        elite = [(f[1], f[2]) for f in fitness[:n_elite]]
        new_pop = list(elite)

        # Tournament + crossover + mutation
        def tournament(k_t=3, _fitness=fitness):
            cands = rng.sample(_fitness, min(k_t, len(_fitness)))
            best_c = max(cands, key=lambda x: x[0])
            return (best_c[1], best_c[2])

        while len(new_pop) < pop_size:
            pa = tournament()
            pb = tournament()

            if rng.random() < crossover_rate:
                child_gnn = (pa[0] + pb[0]) / 2.0
                child_esc = (pa[1] + pb[1]) / 2.0
            else:
                child_gnn, child_esc = pa

            if rng.random() < mutation_rate:
                child_gnn += float(np_rng.normal(0, 0.08))
                child_esc += float(np_rng.normal(0, 0.08))

            # Clamp and ensure w_rf >= 0
            child_gnn = max(0.0, min(1.0, child_gnn))
            child_esc = max(0.0, min(1.0 - child_gnn, child_esc))
            new_pop.append((child_gnn, child_esc))

        population = new_pop

    return {
        "best":     best_ever,
        "history":  history,
        "pop_size": pop_size,
        "generations": generations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    graphs_dir  = PROJECT_ROOT / "data" / "graphs"
    splits_dir  = PROJECT_ROOT / "data" / "splits"
    checkpoints = PROJECT_ROOT / "models" / "checkpoints"
    model_path  = checkpoints / "gnn_best.pt"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--oof",          action="store_true",
                        help="Use out-of-fold predictions from all 5 fold models (recommended)")
    parser.add_argument("--model",        type=Path, default=model_path,
                        help="Path to GNN model (used only without --oof)")
    parser.add_argument("--generations",  type=int,   default=150)
    parser.add_argument("--pop-size",     type=int,   default=60)
    parser.add_argument("--k",            type=int,   default=5)
    parser.add_argument("--alpha",        type=float, default=0.7)
    parser.add_argument("--beta",         type=float, default=0.3)
    parser.add_argument("--hidden",       type=int,   default=64)
    parser.add_argument("--heads",        type=int,   default=4)
    parser.add_argument("--layers",       type=int,   default=3)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    device = resolve_device()

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    if args.oof:
        print("Loading out-of-fold predictions (all 5 fold models)...")
        true_labels, gnn_probs, rf_risks, escape_signals = load_oof_predictions(
            graphs_dir, splits_dir, checkpoints, device,
            hidden=args.hidden, heads=args.heads, num_layers=args.layers,
        )
    else:
        print("Loading GNN predictions on val.txt...")
        true_labels, gnn_probs, rf_risks, escape_signals = load_val_predictions(
            graphs_dir, splits_dir, args.model, device,
            hidden=args.hidden, heads=args.heads, num_layers=args.layers,
        )

    from collections import Counter
    dist = Counter(true_labels)
    n_chains = dist.get(2, 0)
    p5_ceiling = min(n_chains, args.k) / args.k
    print(f"\nDataset: {len(true_labels)} graphs  "
          f"(chains={n_chains}, clean={dist.get(0,0)}, isolated={dist.get(1,0)})")
    print(f"P@{args.k} ceiling: {p5_ceiling:.2f}  "
          f"({'OOF mode' if args.oof else 'val.txt mode'})")
    print(f"Escape signal stats (binary): "
          f"clusters_with_escape={sum(1 for e in escape_signals if e > 0)}/{len(escape_signals)}")

    # ------------------------------------------------------------------
    # 2D Grid search baseline
    # ------------------------------------------------------------------
    print(f"\n2D Grid search baseline (n_steps=20, {21*22//2} evaluations):")
    grid_best = grid_search(
        true_labels, gnn_probs, rf_risks, escape_signals,
        n_steps=20, k=args.k, alpha=args.alpha, beta=args.beta,
    )
    print(f"  Best: w_rf={grid_best['w_rf']:.3f}  w_gnn={grid_best['w_gnn']:.3f}  "
          f"w_esc={grid_best['w_escape']:.3f}  "
          f"score={grid_best['score']:.4f}  "
          f"P@{args.k}={grid_best['p_at_k']:.2f}  "
          f"FPR_clean={grid_best['fpr_clean']:.2f}")

    # ------------------------------------------------------------------
    # GA optimisation
    # ------------------------------------------------------------------
    print(f"\nGenetic Algorithm: {args.pop_size} pop × {args.generations} gen (2D: w_gnn, w_esc)")
    print(f"  Objective: {args.alpha:.1f}*P@{args.k} + {args.beta:.1f}*(1-FPR_clean)")
    ga_result = run_ga(
        true_labels, gnn_probs, rf_risks, escape_signals,
        pop_size=args.pop_size,
        generations=args.generations,
        k=args.k,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
    )

    best = ga_result["best"]
    print(f"\n{'='*60}")
    print("GA RESULT")
    print(f"{'='*60}")
    print(f"  Best weights: w_rf={best['w_rf']:.4f}  "
          f"w_gnn={best['w_gnn']:.4f}  w_esc={best['w_escape']:.4f}")
    print(f"  Objective   : {best['score']:.4f}")
    print(f"  P@{args.k}        : {best['p_at_k']:.4f}")
    print(f"  FPR_clean   : {best['fpr_clean']:.4f}")
    print(f"  P@{args.k} ceiling: {p5_ceiling:.2f}")

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print(f"\nComparison at P@{args.k}:")
    configs = [
        ("GNN-only",         1.0, 0.0, 0.0),
        ("RF-only",          0.0, 1.0, 0.0),
        ("Escape-only",      0.0, 0.0, 1.0),
        ("Equal (1/3 each)", 1/3, 1/3, 1/3),
        ("Grid-best",        grid_best["w_gnn"], grid_best["w_rf"], grid_best["w_escape"]),
        ("GA-best",          best["w_gnn"],       best["w_rf"],      best["w_escape"]),
    ]
    for label, w_gnn, w_rf, w_esc in configs:
        r = compute_objective(
            true_labels, gnn_probs, rf_risks,
            w_rf=w_rf, w_gnn=w_gnn,
            escape_signals=escape_signals, w_escape=w_esc,
            k=args.k, alpha=args.alpha, beta=args.beta,
        )
        print(f"  {label:22s}: P@{args.k}={r['p_at_k']:.2f}  "
              f"FPR_clean={r['fpr_clean']:.2f}  score={r['score']:.4f}  "
              f"w_rf={r['w_rf']:.2f}  w_gnn={r['w_gnn']:.2f}  w_esc={r['w_escape']:.2f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    weights_out = checkpoints / "ga_weights.json"
    results_out = checkpoints / "ga_results.json"

    weights = {
        "w_rf":      best["w_rf"],
        "w_gnn":     best["w_gnn"],
        "w_escape":  best["w_escape"],
        "p_at_k":    best["p_at_k"],
        "fpr_clean": best["fpr_clean"],
        "objective": best["score"],
        "k":         args.k,
        "alpha":     args.alpha,
        "beta":      args.beta,
        "p_at_k_ceiling": p5_ceiling,
        "mode":      "oof" if args.oof else "val",
        "oof_n_graphs":  len(true_labels) if args.oof else None,
        "oof_n_chains":  n_chains if args.oof else None,
        "grid_best": {
            "w_rf":    grid_best["w_rf"],
            "w_gnn":   grid_best["w_gnn"],
            "w_escape": grid_best["w_escape"],
            "p_at_k":  grid_best["p_at_k"],
        },
        "note": (
            "Ensemble score = w_rf*mean_rf_risk + w_gnn*gnn_chain_prob + w_escape*escape_signal. "
            "escape_signal = 1.0 if ANY node has an ESCAPE_FLAG set, else 0.0 (binary). "
            "ESCAPE_FLAG_INDICES derived from kubescan FEATURE_COLS in 26-dim node feature vector."
        ),
        "_provenance": provenance(
            seed=args.seed, mode="oof" if args.oof else "val",
            generations=args.generations, pop_size=args.pop_size,
        ),
    }
    with open(weights_out, "w") as f:
        json.dump(weights, f, indent=2)

    ga_result_serializable = {
        "best_weights": weights,
        "generations":  args.generations,
        "pop_size":     args.pop_size,
        "history":      ga_result["history"],
    }
    with open(results_out, "w") as f:
        json.dump(ga_result_serializable, f, indent=2)

    print(f"\n  Saved: {weights_out}")
    print(f"  Saved: {results_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
