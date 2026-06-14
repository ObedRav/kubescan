"""
ga_ensemble.py
==============
Ensemble scorer.

Combines RF risk scores, GNN chain probabilities, and escape fraction using
the weights produced by research/models/run_ga_ensemble.py (OOF mode).

Score formula:
    score(C) = w_rf * mean_rf_risk(C)
             + w_gnn * gnn_chain_prob(C)
             + w_escape * escape_fraction(C)
"""
from __future__ import annotations

__all__ = [
    "ESCAPE_FLAG_INDICES",
    "LABEL_NAMES",
    "LATERAL_FLAG_INDICES",
    "EnsembleScorer",
    "compute_escape_fraction",
    "compute_escape_signal",
    "run_gnn_ensemble",
]

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from ..exceptions import ModelLoadError
from ..utils.graph_builder import ESCAPE_FLAGS, LATERAL_FLAGS
from ..utils.yaml_parser import FEATURE_COLS

if TYPE_CHECKING:
    from .gat_encoder import KubeGAT

logger = logging.getLogger(__name__)

# Derived from FEATURE_COLS + flag sets — single source of truth, no magic
# indices. Research scripts must import these instead of hardcoding them.
ESCAPE_FLAG_INDICES: Final[list[int]] = [
    i for i, col in enumerate(FEATURE_COLS) if col in ESCAPE_FLAGS
]
LATERAL_FLAG_INDICES: Final[list[int]] = [
    i for i, col in enumerate(FEATURE_COLS) if col in LATERAL_FLAGS
]

LABEL_NAMES: Final[dict[int, str]] = {
    0: "CLEAN",
    1: "ISOLATED_MISCONFIG",
    2: "ATTACK_CHAIN",
}

# Fixed decision thresholds on the ensemble score (thesis §4: not re-optimised by the GA)
_SCORE_HIGH_THRESHOLD:     Final[float] = 0.60
_SCORE_MODERATE_THRESHOLD: Final[float] = 0.30


class EnsembleScorer:
    """
    Load GA-optimised weights and score a cluster given model predictions.

    Parameters
    ----------
    weights_path : path to ga_weights.json
    """

    def __init__(self, weights_path: Path) -> None:
        with open(weights_path) as f:
            w = json.load(f)
        total = w["w_rf"] + w["w_gnn"] + w.get("w_escape", 0.0)
        if total <= 0.0:
            raise ValueError(
                f"Ensemble weights sum to {total}; ga_weights.json may be corrupt."
            )
        self.w_rf     = w["w_rf"]            / total
        self.w_gnn    = w["w_gnn"]           / total
        self.w_escape = w.get("w_escape", 0.0) / total
        logger.debug(
            "Ensemble weights: w_rf=%.4f w_gnn=%.4f w_escape=%.4f",
            self.w_rf, self.w_gnn, self.w_escape,
        )

    @classmethod
    def from_checkpoints(cls, checkpoints_dir: Path) -> EnsembleScorer:
        path = Path(checkpoints_dir) / "ga_weights.json"
        if not path.exists():
            raise ModelLoadError(
                path,
                "ga_weights.json not found — run research/models/run_ga_ensemble.py --oof first.",
            )
        return cls(path)

    def score(
        self,
        mean_rf_risk:   float,
        chain_prob:     float,
        escape_signal:  float,
    ) -> float:
        """
        Return ensemble score in [0, 1].

        Parameters
        ----------
        escape_signal : binary signal — 1.0 if any escape-capable manifest
                        exists in the cluster, 0.0 otherwise.
                        Use compute_escape_signal(), not compute_escape_fraction().
        """
        return (
            self.w_rf     * mean_rf_risk
            + self.w_gnn    * chain_prob
            + self.w_escape * escape_signal
        )

    def predict_label(self, ensemble_score: float) -> int:
        """
        Cluster verdict from the ensemble score (fixed thresholds, thesis §4):
          2 (ATTACK_CHAIN)       if ensemble_score >= 0.60
          1 (ISOLATED_MISCONFIG) if ensemble_score >= 0.30
          0 (CLEAN)              otherwise

        Note: with a non-zero w_escape, any escape-capable manifest already
        contributes w_escape to the score via the binary escape signal, so
        escape-bearing clusters land at ISOLATED or above by construction.
        """
        if ensemble_score >= _SCORE_HIGH_THRESHOLD:
            return 2
        if ensemble_score >= _SCORE_MODERATE_THRESHOLD:
            return 1
        return 0


def run_gnn_ensemble(
    pyg_data:    Data,
    fold_models: list[KubeGAT],
    device:      torch.device,
) -> tuple[float, float, float]:
    """
    Average softmax probabilities across all fold models.
    Returns (chain_prob, clean_prob, isolated_prob).
    """
    x          = pyg_data.x.to(device)
    edge_index = pyg_data.edge_index.to(device)
    edge_attr  = pyg_data.edge_attr.to(device)
    # global_mean/max_pool needs a batch vector; all nodes belong to graph 0
    batch      = torch.zeros(x.size(0), dtype=torch.long, device=device)

    all_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for model in fold_models:
            model.eval()
            out   = model(x, edge_index, edge_attr, batch)
            probs = F.softmax(out, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
    mean_probs = np.mean(all_probs, axis=0)
    return float(mean_probs[2]), float(mean_probs[0]), float(mean_probs[1])


def compute_escape_fraction(node_features_list: list[np.ndarray]) -> float:
    """
    Fraction of nodes with at least one escape flag set.

    Used for display and reporting only. Use compute_escape_signal() for scoring.

    Parameters
    ----------
    node_features_list : list of per-node feature vectors (each length 25 or 26).
    """
    if not node_features_list:
        return 0.0
    matrix = np.stack(node_features_list)          # [N, D]
    esc    = matrix[:, ESCAPE_FLAG_INDICES]         # [N, len(ESCAPE_FLAGS)]
    return float((esc.max(axis=1) > 0).mean())


def compute_escape_signal(node_features_list: list[np.ndarray]) -> float:
    """
    Binary signal: 1.0 if any node has at least one escape flag set, 0.0 otherwise.

    Unlike compute_escape_fraction(), this is not diluted by cluster size — a single
    escape-capable manifest in a 1000-node cluster still returns 1.0.
    Pass this to EnsembleScorer.score() as escape_signal.

    Parameters
    ----------
    node_features_list : list of per-node feature vectors (each length 25 or 26).
    """
    if not node_features_list:
        return 0.0
    matrix = np.stack(node_features_list)          # [N, D]
    esc    = matrix[:, ESCAPE_FLAG_INDICES]         # [N, len(ESCAPE_FLAGS)]
    return 1.0 if (esc.max(axis=1) > 0).any() else 0.0
