"""
seed_utils.py
=============
Global random-seed initialisation for reproducible training runs.
"""
from __future__ import annotations

__all__ = ["set_global_seed"]

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set random seed for random, numpy, and torch (CUDA + MPS included)."""
    import torch  # lazy — kubescan package does not require torch at import time
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        # CUDA scatter/gather kernels (used by GATConv message passing) pick
        # non-deterministic algorithms by default regardless of seed — only
        # matters on CUDA, MPS/CPU runs were already effectively deterministic.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("seed=%d", seed)
