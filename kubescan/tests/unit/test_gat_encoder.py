"""
test_gat_encoder.py
===================
Unit tests for kubescan/model/gat_encoder.py.
"""
from __future__ import annotations

__all__: list[str] = []

from pathlib import Path

import pytest
import torch

from kubescan.exceptions import ModelLoadError
from kubescan.model.gat_encoder import NUM_FOLDS, GATConfig, KubeGAT, load_fold_ensemble

# ---------------------------------------------------------------------------
# GATConfig defaults
# ---------------------------------------------------------------------------

def test_gatconfig_default_in_channels() -> None:
    assert GATConfig.in_channels == 26


def test_gatconfig_default_num_classes() -> None:
    assert GATConfig.num_classes == 3


def test_gatconfig_default_num_edge_types() -> None:
    assert GATConfig.num_edge_types == 5


# ---------------------------------------------------------------------------
# KubeGAT forward
# ---------------------------------------------------------------------------

def test_kubegat_forward_output_shape() -> None:
    model = KubeGAT()
    model.eval()
    n_nodes = 4
    n_edges = 6
    x          = torch.zeros(n_nodes, GATConfig.in_channels)
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 2, 3, 0, 3, 1]], dtype=torch.long)
    edge_attr  = torch.zeros(n_edges, 1, dtype=torch.long)
    batch      = torch.zeros(n_nodes, dtype=torch.long)
    with torch.inference_mode():
        out = model(x, edge_index, edge_attr, batch)
    assert out.shape == (1, GATConfig.num_classes)


def test_kubegat_forward_single_node() -> None:
    model = KubeGAT()
    model.eval()
    x          = torch.zeros(1, GATConfig.in_channels)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_attr  = torch.zeros(0, 1, dtype=torch.long)
    batch      = torch.zeros(1, dtype=torch.long)
    with torch.inference_mode():
        out = model(x, edge_index, edge_attr, batch)
    assert out.shape == (1, GATConfig.num_classes)


# ---------------------------------------------------------------------------
# load_fold_ensemble
# ---------------------------------------------------------------------------

def test_load_fold_ensemble_returns_all_folds(fixture_checkpoints: Path) -> None:
    models = load_fold_ensemble(fixture_checkpoints)
    assert len(models) == NUM_FOLDS


def test_load_fold_ensemble_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match="gnn_fold_"):
        load_fold_ensemble(tmp_path)


def test_load_fold_ensemble_models_in_eval_mode(fixture_checkpoints: Path) -> None:
    models = load_fold_ensemble(fixture_checkpoints)
    assert all(not m.training for m in models)


def test_load_fold_ensemble_uses_config_json(fixture_checkpoints: Path, tmp_path: Path) -> None:
    """If gnn_config.json is missing, defaults are used without error."""
    import shutil
    shutil.copytree(fixture_checkpoints, tmp_path / "ckpts")
    (tmp_path / "ckpts" / "gnn_config.json").unlink(missing_ok=True)
    models = load_fold_ensemble(tmp_path / "ckpts")
    assert len(models) == NUM_FOLDS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_checkpoints() -> Path:
    return Path(__file__).parent.parent / "fixtures" / "checkpoints"
