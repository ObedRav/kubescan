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
    with open(manifest, newline="", encoding="utf-8") as f:
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

    np.savez_compressed(CACHE_PATH, **arrays)
    n_graphs = len({k.split("::")[0] for k in arrays})
    print(f"Wrote {CACHE_PATH.name}: {n_graphs} graphs, "
          f"{len(arrays)} arrays, {CACHE_PATH.stat().st_size / 1e6:.1f} MB "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
