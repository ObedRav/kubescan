"""
make_fixtures.py
================
Generate tiny, deterministic checkpoints so the CLI integration tests run in
CI without the full (large, iCloud-synced) trained checkpoints.

The fixtures are NOT meant to be accurate — they are dimensionally valid and
just good enough that the pipeline executes and a handful of structural
invariants hold:
  * the RF is fit so escape-laden manifests outrank clean ones,
  * the GAT folds are valid KubeGAT modules (random init, eval mode),
  * ga_weights.json uses equal thirds.

Regenerate with:
  python kubescan/tests/fixtures/make_fixtures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from skops.io import dump as skops_dump

from kubescan.model.gat_encoder import GATConfig, KubeGAT

OUT = Path(__file__).parent / "checkpoints"
SEED = 0


def make_rf() -> RandomForestClassifier:
    """Tiny RF that learns 'more flags set → misconfigured'."""
    rng = np.random.default_rng(SEED)
    n_feat = 25
    # Clean rows: mostly zeros. Misconfigured rows: several flags set.
    clean   = (rng.random((40, n_feat)) < 0.05).astype(np.float32)
    misconf = (rng.random((40, n_feat)) < 0.6).astype(np.float32)
    X = np.vstack([clean, misconf])
    y = np.array([0] * 40 + [1] * 40)
    rf = RandomForestClassifier(
        n_estimators=16, max_depth=4, random_state=SEED, class_weight="balanced"
    )
    rf.fit(X, y)
    return rf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    skops_dump(make_rf(), OUT / "rf_model.skops")

    # Two GAT folds at the production dimensions (random init is fine for a
    # smoke fixture; predictions need only be valid probabilities).
    for fold in range(2):
        torch.manual_seed(SEED + fold)
        model = KubeGAT(
            in_channels=GATConfig.in_channels,
            hidden=GATConfig.hidden_dim,
            heads=GATConfig.num_heads,
            num_layers=GATConfig.num_layers,
        )
        torch.save(model.state_dict(), OUT / f"gnn_fold_{fold}.pt")

    with open(OUT / "ga_weights.json", "w", encoding="utf-8") as f:
        json.dump(
            {"w_rf": 1 / 3, "w_gnn": 1 / 3, "w_escape": 1 / 3,
             "_note": "fixture weights — equal thirds, not trained"},
            f, indent=2,
        )

    print(f"Wrote fixtures to {OUT}/ (rf_model.skops, gnn_fold_0..1.pt, ga_weights.json)")


if __name__ == "__main__":
    main()
