"""
provenance.py
=============
Train-time provenance stamping.

Every checkpoint-result JSON written during training carries a `_provenance`
block — git commit, seeds, package versions, timestamp — so any reported number
can be traced to the exact state that produced it without relying on a separate
post-hoc snapshot step.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_TFE_ROOT = Path(__file__).resolve().parent.parent.parent


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_TFE_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=_TFE_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def provenance(seed: int | None = None, **extra: object) -> dict:
    """
    Build a provenance block to embed under "_provenance" in result JSONs.

    Parameters
    ----------
    seed  : the training seed, if applicable.
    extra : any additional run-defining fields (e.g. conv type, layers).
    """
    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "torch_geometric", "sklearn", "numpy", "networkx", "skops"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"

    block: dict[str, object] = {
        "created_utc":   datetime.now(timezone.utc).isoformat(),
        "git_commit":    _git_commit(),
        "git_dirty":     _git_dirty(),
        "platform":      f"{platform.system()} {platform.machine()}",
        "versions":      versions,
    }
    if seed is not None:
        block["seed"] = seed
    block.update(extra)
    return block
