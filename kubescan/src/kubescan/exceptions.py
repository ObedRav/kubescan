"""
exceptions.py
=============
Typed exception hierarchy for kubescan.

All kubescan errors inherit from KubescanError.
Catch KubescanError at the CLI boundary; catch specific subclasses in tests.
"""
from __future__ import annotations

__all__ = [
    "CheckpointNotFoundError",
    "GraphBuildError",
    "KubescanError",
    "ManifestParseError",
    "ModelLoadError",
]

from pathlib import Path


class KubescanError(Exception):
    """Base for all kubescan errors — catch this at the CLI boundary."""


class CheckpointNotFoundError(KubescanError):
    def __init__(self, checkpoints_dir: Path) -> None:
        super().__init__(
            f"Checkpoints directory not found: {checkpoints_dir}\n"
            "Options:\n"
            "  1. Pass --checkpoints-dir /path/to/research/models/checkpoints\n"
            "  2. Set KUBESCAN_CHECKPOINTS env var\n"
            "  3. Run from TFE root (symlink kubescan/checkpoints/trained/ is auto-detected)"
        )
        self.checkpoints_dir = checkpoints_dir


class ManifestParseError(KubescanError):
    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Cannot parse {path}: {reason}")
        self.path = path


class GraphBuildError(KubescanError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Graph construction failed: {reason}")


class ModelLoadError(KubescanError):
    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Cannot load model from {path}: {reason}")
        self.path = path
