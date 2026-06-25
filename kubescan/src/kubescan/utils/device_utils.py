from __future__ import annotations

__all__ = ["dataloader_kwargs", "resolve_device"]

import os
from typing import TYPE_CHECKING, Final

from ..exceptions import KubescanError

if TYPE_CHECKING:
    import torch

_MAX_DATALOADER_WORKERS: Final[int] = 4


class KubescanDependencyError(KubescanError):
    def __init__(self, package: str) -> None:
        super().__init__(
            f"Optional dependency '{package}' is not installed. "
            f"Install it with: pip install {package}"
        )
        self.package = package


def resolve_device() -> torch.device:
    """Return the best available compute device: CUDA → MPS → CPU."""
    try:
        import torch
    except ImportError as exc:
        raise KubescanDependencyError("torch") from exc
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
