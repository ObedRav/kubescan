"""
test_ga_ensemble.py
===================
Unit tests for kubescan/model/ga_ensemble.py.
One assertion per test; name encodes condition + expected result.
"""
from __future__ import annotations

__all__: list[str] = []

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from kubescan.model.ga_ensemble import (
    ESCAPE_FLAG_INDICES,
    EnsembleScorer,
    compute_escape_fraction,
    compute_escape_signal,
    run_gnn_ensemble,
)
from kubescan.model.gat_encoder import KubeGAT


def _scorer(
    tmp_path: Path, w_rf: float = 1.0, w_gnn: float = 1.0, w_esc: float = 1.0
) -> EnsembleScorer:
    p = tmp_path / "ga_weights.json"
    p.write_text(json.dumps({"w_rf": w_rf, "w_gnn": w_gnn, "w_escape": w_esc}))
    return EnsembleScorer(p)


def test_escape_fraction_all_zeros_returns_zero() -> None:
    feats = [np.zeros(25, dtype=np.float32)]
    assert compute_escape_fraction(feats) == 0.0


def test_escape_fraction_with_host_pid_flag_returns_one() -> None:
    feats = [np.zeros(25, dtype=np.float32)]
    feats[0][ESCAPE_FLAG_INDICES[0]] = 1.0
    assert compute_escape_fraction(feats) == 1.0


def test_escape_signal_all_zeros_returns_zero() -> None:
    feats = [np.zeros(25, dtype=np.float32)]
    assert compute_escape_signal(feats) == 0.0


def test_escape_signal_one_escape_node_in_large_cluster_returns_one() -> None:
    # 99 clean nodes + 1 escape node → signal must be 1.0 regardless of fraction
    feats = [np.zeros(25, dtype=np.float32) for _ in range(100)]
    feats[-1][ESCAPE_FLAG_INDICES[0]] = 1.0
    assert compute_escape_signal(feats) == 1.0


def test_scorer_normalises_weights_to_sum_one(tmp_path: Path) -> None:
    s = _scorer(tmp_path, w_rf=2.0, w_gnn=1.0, w_esc=1.0)
    assert abs(s.w_rf + s.w_gnn + s.w_escape - 1.0) < 1e-9


def test_score_is_weighted_sum_of_components(tmp_path: Path) -> None:
    s = _scorer(tmp_path)  # equal weights → 1/3 each
    assert abs(s.score(0.3, 0.6, 1.0) - (0.3 + 0.6 + 1.0) / 3) < 1e-9


def test_predict_label_high_score_is_attack_chain(tmp_path: Path) -> None:
    assert _scorer(tmp_path).predict_label(0.60) == 2


def test_predict_label_moderate_score_is_isolated(tmp_path: Path) -> None:
    assert _scorer(tmp_path).predict_label(0.30) == 1


def test_predict_label_low_score_is_clean(tmp_path: Path) -> None:
    assert _scorer(tmp_path).predict_label(0.29) == 0


def test_predict_label_escape_weight_alone_reaches_isolated(tmp_path: Path) -> None:
    # binary escape signal times w_escape (~1/3) must already cross the 0.30 bar
    s = _scorer(tmp_path)
    assert s.predict_label(s.score(0.0, 0.0, 1.0)) == 1


# ---------------------------------------------------------------------------
# run_gnn_ensemble — direct-tensor forward path (no DataLoader)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_gat() -> KubeGAT:
    """Minimal KubeGAT matching NODE_FEATURE_DIM=26; small dims for speed."""
    torch.manual_seed(0)
    model = KubeGAT(
        in_channels=26,
        hidden=4,
        heads=1,
        num_layers=1,
        num_classes=3,
        dropout=0.0,
        num_edge_types=5,
        edge_emb_dim=4,
    )
    model.eval()
    return model


@pytest.fixture()
def tiny_graph() -> Data:
    """Three-node cluster graph with four directed edges and 26-dim features."""
    torch.manual_seed(0)
    return Data(
        x=torch.randn(3, 26),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.zeros(4, dtype=torch.long),
        y=torch.tensor([2]),
    )


def test_run_gnn_ensemble_returns_three_floats(tiny_gat: KubeGAT, tiny_graph: Data) -> None:
    result = run_gnn_ensemble(tiny_graph, [tiny_gat], torch.device("cpu"))
    assert len(result) == 3 and all(isinstance(v, float) for v in result)


def test_run_gnn_ensemble_probs_sum_to_one(tiny_gat: KubeGAT, tiny_graph: Data) -> None:
    chain_p, clean_p, iso_p = run_gnn_ensemble(tiny_graph, [tiny_gat], torch.device("cpu"))
    assert abs(chain_p + clean_p + iso_p - 1.0) < 1e-5


def test_run_gnn_ensemble_all_probs_in_unit_interval(tiny_gat: KubeGAT, tiny_graph: Data) -> None:
    for prob in run_gnn_ensemble(tiny_graph, [tiny_gat], torch.device("cpu")):
        assert 0.0 <= prob <= 1.0


def test_run_gnn_ensemble_two_identical_models_matches_single_model(
    tiny_gat: KubeGAT, tiny_graph: Data
) -> None:
    clone = copy.deepcopy(tiny_gat)
    single = run_gnn_ensemble(tiny_graph, [tiny_gat], torch.device("cpu"))
    averaged = run_gnn_ensemble(tiny_graph, [tiny_gat, clone], torch.device("cpu"))
    assert abs(single[0] - averaged[0]) < 1e-6
