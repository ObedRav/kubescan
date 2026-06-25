"""
build_graph_cache.py
====================
Consolidate all per-cluster .npz graph files into a single graphs_cache.npz.

Why: the training pipeline opens hundreds of small .npz files per split, and
per-file open cost (filesystem scanning, I/O throttling of background jobs)
dominates load time. One consolidated file turns ~2,000 opens into 1.

The cache stores every array under the key "<safe_name>::<field>". It is
invalidated and rebuilt by re-running this script whenever graphs change
(e.g. after build_graphs.py or augment_graphs.py).

Usage:
  python scripts/04_build_datasets/build_graph_cache.py
"""

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # research/
GRAPHS_DIR   = PROJECT_ROOT / "data" / "graphs"
CACHE_PATH   = GRAPHS_DIR / "graphs_cache.npz"

# Pipeline ordering constraint: each .npz "x" tensor has shape [N, 26] where
# column 25 is the RF risk_score appended by build_graphs.py.  This cache must
# therefore be rebuilt AFTER train_rf.py and build_graphs.py have run so that
# x[:, 25] reflects the current RF model, not a stale one.

# Parallel reads matter on iCloud-synced volumes: evicted (dataless) files are
# materialised on first read, and the CloudDocs daemon serves concurrent
# requests in parallel. Reading also snapshots the bytes into memory before
# disk-pressure eviction can claw the file back.
N_WORKERS = 16


def _read_one(row: dict) -> tuple[str, dict[str, np.ndarray] | None]:
    safe = row["safe_name"]
    npz_path = GRAPHS_DIR / f"{safe}.npz"
    if not npz_path.exists():
        return safe, None
    for attempt in range(3):
        try:
            d = np.load(npz_path, allow_pickle=False)
            return safe, {k: d[k].copy() for k in d.files}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    return safe, None


def main() -> None:
    manifest = GRAPHS_DIR / "graph_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    arrays: dict[str, np.ndarray] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for safe, fields in pool.map(_read_one, rows):
            done += 1
            if fields is None:
                print(f"  [skip] {safe}.npz not found")
                continue
            for key, val in fields.items():
                arrays[f"{safe}::{key}"] = val
            if done % 50 == 0:
                print(f"  {done}/{len(rows)} graphs  ({time.time() - t0:.0f}s)")

    # Atomic write: save to .tmp then rename so a crash never leaves a partial file.
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    np.savez_compressed(tmp_path, **arrays)
    # Post-write validation: verify the cache is readable before committing.
    _verify_cache(tmp_path)
    tmp_path.rename(CACHE_PATH)

    n_graphs = len({k.split("::")[0] for k in arrays})
    print(f"Wrote {CACHE_PATH.name}: {n_graphs} graphs, "
          f"{len(arrays)} arrays, {CACHE_PATH.stat().st_size / 1e6:.1f} MB "
          f"in {time.time() - t0:.0f}s")


def _verify_cache(path: Path) -> None:
    """Assert cache is readable; raise RuntimeError if not."""
    try:
        cache = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"Cache validation failed for {path}: {exc}") from exc
    n_graphs = len({k.split("::")[0] for k in cache.files})
    if n_graphs == 0:
        raise RuntimeError(f"Cache at {path} contains no graphs")


if __name__ == "__main__":
    main()
