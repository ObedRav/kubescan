"""
kubescan — Kubernetes attack-chain risk scanner.

Loads trained GNN + Random Forest ensemble checkpoints and scores any
directory of Kubernetes YAML manifests for attack-chain risk.

Quickstart:
    kubescan scan ./my-cluster-manifests/
    kubescan scan ./configs/ --format json
    kubescan scan ./configs/ --checkpoints-dir /path/to/checkpoints
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
