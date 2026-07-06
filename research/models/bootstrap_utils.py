"""
bootstrap_utils.py
===================
Shared bootstrap-resampling primitive, used by both run_ga_ensemble.py
(--select-method bootstrap) and evaluate_test_set.py (bootstrap_cis) so the
"draw n indices with replacement" mechanic lives in exactly one place.
"""
from __future__ import annotations

import numpy as np

__all__ = ["resample_indices"]


def resample_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n indices in [0, n) with replacement (one bootstrap resample)."""
    return rng.integers(0, n, size=n)
