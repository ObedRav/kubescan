"""
gat_encoder.py
==============
KubeGAT architecture + checkpoint loading.

This is the SINGLE definition of the architecture: research/models/train_gnn.py
imports KubeGAT from here, so training and inference can never drift apart.

Edge types (directory proximity, privilege reach, SA lateral, namespace, RBAC)
are categorical — they pass through an nn.Embedding before reaching GATConv,
instead of being cast to a float scalar where type 4 would spuriously count as
"more edge" than type 1.
"""
from __future__ import annotations

__all__ = ["NUM_FOLDS", "GATConfig", "KubeGAT", "load_fold_ensemble"]

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool

from ..exceptions import ModelLoadError
from ..utils.device_utils import resolve_device

logger = logging.getLogger(__name__)

NUM_FOLDS: Final[int] = 5


@dataclass(frozen=True)
class GATConfig:
    """Hyperparameters for KubeGAT — must match the trained checkpoints exactly."""
    in_channels:    int   = 26
    hidden_dim:     int   = 64
    num_heads:      int   = 4
    num_layers:     int   = 3
    num_classes:    int   = 3
    dropout:        float = 0.3
    num_edge_types: int   = 5   # EdgeType enum in utils.graph_builder
    edge_emb_dim:   int   = 8   # embedding fed to GATConv as edge_dim


class KubeGAT(nn.Module):
    """
    Graph Attention Network for Kubernetes attack-chain classification.

    Architecture: edge-type embedding → L GAT layers → global mean+max pooling
    → 2-layer MLP. Trained to classify cluster graphs as
    clean / isolated / attack_chain.
    """

    def __init__(
        self,
        in_channels: int   = GATConfig.in_channels,
        hidden:      int   = GATConfig.hidden_dim,
        heads:       int   = GATConfig.num_heads,
        num_layers:  int   = GATConfig.num_layers,
        num_classes: int   = GATConfig.num_classes,
        dropout:     float = GATConfig.dropout,
        num_edge_types: int = GATConfig.num_edge_types,
        edge_emb_dim:   int = GATConfig.edge_emb_dim,
    ) -> None:
        super().__init__()
        self.dropout    = dropout
        self.num_layers = num_layers

        self.edge_emb = nn.Embedding(num_edge_types, edge_emb_dim)

        self.input_proj = nn.Linear(in_channels, hidden * heads)
        self.input_norm = nn.LayerNorm(hidden * heads)

        self.gat_layers: nn.ModuleList = nn.ModuleList()
        self.norms:       nn.ModuleList = nn.ModuleList()
        for i in range(num_layers):
            in_dim  = hidden * heads
            out_dim = hidden
            concat  = (i < num_layers - 1)
            self.gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads if concat else 1,
                    dropout=dropout,
                    edge_dim=edge_emb_dim,
                    concat=concat,
                )
            )
            out_size = out_dim * heads if concat else out_dim
            self.norms.append(nn.LayerNorm(out_size))

        pool_dim = hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(
        self,
        x:          Tensor,
        edge_index: Tensor,
        edge_attr:  Tensor,
        batch:      Tensor,
    ) -> Tensor:
        # Categorical edge type → learned embedding ([E, 1] int → [E, emb_dim])
        edge_feat = self.edge_emb(edge_attr.view(-1).long())

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for gat, norm in zip(self.gat_layers, self.norms, strict=True):
            residual = x
            x = gat(x, edge_index, edge_attr=edge_feat)
            x = norm(x)
            x = F.elu(x)
            if residual.shape == x.shape:
                x = x + residual
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        return self.classifier(torch.cat([x_mean, x_max], dim=-1))


def _load_gat_config(checkpoints_dir: Path) -> GATConfig:
    """
    Load GATConfig from gnn_config.json if present (written by train_gnn.py).
    Falls back to GATConfig() defaults for backward compatibility with checkpoints
    that pre-date config serialisation.
    """
    config_path = Path(checkpoints_dir) / "gnn_config.json"
    if not config_path.exists():
        logger.debug("gnn_config.json not found in %s — using GATConfig defaults", checkpoints_dir)
        return GATConfig()
    with config_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    cfg_fields = {
        "in_channels", "hidden_dim", "num_heads", "num_layers",
        "num_classes", "dropout", "num_edge_types", "edge_emb_dim",
    }
    kwargs = {k: v for k, v in raw.items() if k in cfg_fields}
    logger.debug("Loaded GATConfig from %s: %s", config_path.name, kwargs)
    return GATConfig(**kwargs)


def load_fold_ensemble(
    checkpoints_dir: Path,
    device:      torch.device | None = None,
) -> list[KubeGAT]:
    """
    Load all NUM_FOLDS fold models (gnn_fold_0.pt … gnn_fold_4.pt).
    Returns a list of KubeGAT instances in eval() mode.
    Averaging predictions across folds reduces variance (implicit ensemble).

    Architecture hyperparameters are read from gnn_config.json (written by
    train_gnn.py). If the JSON is absent, GATConfig defaults are used so that
    checkpoints from earlier training runs still load.
    """
    if device is None:
        device = resolve_device()

    cfg = _load_gat_config(checkpoints_dir)

    models: list[KubeGAT] = []
    for fold_idx in range(NUM_FOLDS):
        path = Path(checkpoints_dir) / f"gnn_fold_{fold_idx}.pt"
        if not path.exists():
            logger.debug("Fold checkpoint not found, skipping: %s", path)
            continue
        model = KubeGAT(
            in_channels=cfg.in_channels,
            hidden=cfg.hidden_dim,
            heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
            dropout=cfg.dropout,
            num_edge_types=cfg.num_edge_types,
            edge_emb_dim=cfg.edge_emb_dim,
        ).to(device)
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        model.eval()
        models.append(model)
        logger.debug("Loaded fold model: %s", path.name)

    if not models:
        raise ModelLoadError(
            Path(checkpoints_dir),
            "No gnn_fold_*.pt files found. Run research/models/train_gnn.py first.",
        )
    if len(models) < NUM_FOLDS:
        logger.warning(
            "Only %d of %d GNN fold checkpoints found in %s — "
            "ensemble quality is degraded; re-run train_gnn.py to regenerate all folds",
            len(models), NUM_FOLDS, checkpoints_dir,
        )
    logger.info("Loaded %d GNN fold models from %s", len(models), checkpoints_dir)
    return models
