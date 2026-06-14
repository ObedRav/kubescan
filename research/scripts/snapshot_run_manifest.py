"""
snapshot_run_manifest.py
========================
Fingerprint the current experiment state into run_manifest.json.

Captures everything needed to tie reported thesis numbers to an exact,
verifiable state: dataset checksums, split checksums, package versions,
checkpoint checksums, and the headline metrics from the results JSONs.

Run after every training cycle:
  python scripts/snapshot_run_manifest.py
"""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent          # research/
TFE_ROOT     = PROJECT_ROOT.parent
OUT_PATH     = PROJECT_ROOT / "models" / "checkpoints" / "run_manifest.json"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=TFE_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def pkg_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for mod in ("torch", "torch_geometric", "sklearn", "numpy", "networkx", "skops"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "not installed"
    return versions


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    data   = PROJECT_ROOT / "data"
    ckpt   = PROJECT_ROOT / "models" / "checkpoints"
    splits = data / "splits"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform":    f"{platform.system()} {platform.machine()}",
        "git_commit":  git_commit(),
        "versions":    pkg_versions(),
        "data_checksums": {
            "rf_dataset.csv":     sha256(data / "tabular" / "rf_dataset.csv"),
            "graph_manifest.csv": sha256(data / "graphs" / "graph_manifest.csv"),
            "graphs_cache.npz":   sha256(data / "graphs" / "graphs_cache.npz"),
            "splits_config.json": sha256(splits / "splits_config.json"),
            **{f.name: sha256(f) for f in sorted(splits.glob("*.txt"))},
        },
        "checkpoint_checksums": {
            f.name: sha256(f)
            for f in sorted(ckpt.glob("*"))
            if f.suffix in (".pt", ".skops", ".json") and f.name != "run_manifest.json"
        },
        "headline_metrics": {
            "gnn_cv":   load_json(ckpt / "cv_results.json"),
            "ga":       load_json(ckpt / "ga_weights.json"),
            "test":     (load_json(ckpt / "test_results.json") or {}).get("ranking_metrics"),
            "test_cis": (load_json(ckpt / "test_results.json") or {}).get("confidence_intervals_95"),
        },
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
