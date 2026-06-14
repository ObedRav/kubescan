from __future__ import annotations

__all__ = ["dataloader_kwargs", "resolve_device"]

import os
from typing import Final

import torch

_MAX_DATALOADER_WORKERS: Final[int] = 4


def resolve_device() -> torch.device:
    """Return the best available compute device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dataloader_kwargs(device: torch.device) -> dict[str, object]:
    """Device-appropriate DataLoader kwargs: async prefetch workers + CUDA pin_memory."""
    num_workers = min(_MAX_DATALOADER_WORKERS, os.cpu_count() or 1)
    return {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
