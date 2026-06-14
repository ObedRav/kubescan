"""
test_device_utils.py
====================
Unit tests for kubescan/utils/device_utils.py.
One assertion per test; name encodes condition + expected result.
"""
from __future__ import annotations

__all__: list[str] = []

import torch
import pytest

from kubescan.utils.device_utils import _MAX_DATALOADER_WORKERS, dataloader_kwargs, resolve_device


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------

def test_resolve_device_returns_torch_device() -> None:
    result = resolve_device()
    assert isinstance(result, torch.device)


def test_resolve_device_type_is_valid_string() -> None:
    result = resolve_device()
    assert result.type in {"cuda", "mps", "cpu"}


# ---------------------------------------------------------------------------
# dataloader_kwargs
# ---------------------------------------------------------------------------

def test_dataloader_kwargs_has_num_workers_key() -> None:
    kwargs = dataloader_kwargs(torch.device("cpu"))
    assert "num_workers" in kwargs


def test_dataloader_kwargs_has_pin_memory_key() -> None:
    kwargs = dataloader_kwargs(torch.device("cpu"))
    assert "pin_memory" in kwargs


def test_dataloader_kwargs_has_persistent_workers_key() -> None:
    kwargs = dataloader_kwargs(torch.device("cpu"))
    assert "persistent_workers" in kwargs


def test_dataloader_kwargs_cpu_pin_memory_is_false() -> None:
    kwargs = dataloader_kwargs(torch.device("cpu"))
    assert kwargs["pin_memory"] is False


def test_dataloader_kwargs_mps_pin_memory_is_false() -> None:
    kwargs = dataloader_kwargs(torch.device("mps"))
    assert kwargs["pin_memory"] is False


def test_dataloader_kwargs_cuda_pin_memory_is_true() -> None:
    kwargs = dataloader_kwargs(torch.device("cuda"))
    assert kwargs["pin_memory"] is True


def test_dataloader_kwargs_num_workers_is_at_least_one() -> None:
    kwargs = dataloader_kwargs(torch.device("cpu"))
    assert kwargs["num_workers"] >= 1


def test_dataloader_kwargs_num_workers_does_not_exceed_cap() -> None:
    kwargs = dataloader_kwargs(torch.device("cpu"))
    assert kwargs["num_workers"] <= _MAX_DATALOADER_WORKERS


def test_dataloader_kwargs_persistent_workers_consistent_with_num_workers() -> None:
    kwargs = dataloader_kwargs(torch.device("cpu"))
    assert kwargs["persistent_workers"] == (kwargs["num_workers"] > 0)
