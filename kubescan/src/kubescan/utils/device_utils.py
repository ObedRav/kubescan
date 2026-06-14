from __future__ import annotations

__all__ = ["resolve_device"]

import torch


def resolve_device() -> torch.device:
    """Return the best available compute device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
