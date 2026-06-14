"""
augment_graphs.py
==================
Graph augmentation to address the class imbalance in attack-chain graphs.

The GNN dataset has 13 attack-chain graphs vs 25 clean + 44 isolated.
This script generates augmented variants of chain (label=2) graphs using:
  1. Edge dropout   — remove % of non-critical edges (types 0=proximity, 3=namespace)
  2. Feature masking — zero out % of low-importance node features
  3. Subgraph sampling — extract connected subgraphs from large clusters (>50 nodes)

Critical edges (types 1=privilege_reach, 2=sa_lateral) are NEVER dropped — they
define the chain and must be preserved in every augmented variant.

Output:
  dataset/graphs/<name>_aug_<k>.npz        – augmented graph files
  dataset/graphs/graph_manifest.csv        – updated manifest with augmented entries

Usage:
  python scripts/augment_graphs.py                      # default: 15 variants per chain
  python scripts/augment_graphs.py --variants 10
  python scripts/augment_graphs.py --dry-run            # show counts without writing
"""

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np

# Edge type codes (must match graph_builder.py)
EDGE_PROXIMITY  = 0   # droppable
EDGE_PRIV_REACH = 1   # KEEP — defines chain
EDGE_SA_LATERAL = 2   # KEEP — defines chain
EDGE_SEMANTIC_NS = 3  # droppable

# Feature indices to mask (low-importance per rf_findings.md)
# DO NOT mask HOSTPATH_MOUNT (index 24) or risk_score (index 25) — escape signal
MASKABLE_FEATURES = [
    9,   # SECCOMP_UNCONFINED (always 0 in dataset — masking = no-op but harmless)
    10,  # VALID_TAINT_SECRET (always 0)
    13,  # NO_NETWORK_POLICY (always 1 — masking = no-op)
    14,  # HOST_ALIAS
    17,  # NO_ROLLING_UPDATE
    20,  # IMAGE_USES_LATEST
]

# Node-feature indices defining the attack-chain label rule — canonical
# definitions derived in the kubescan package (single source of truth)
from kubescan.model.ga_ensemble import (
    ESCAPE_FLAG_INDICES as ESCAPE_IDX,
)
from kubescan.model.ga_ensemble import (
    LATERAL_FLAG_INDICES as LATERAL_IDX,
)


def satisfies_chain_rule(x: np.ndarray) -> bool:
    """
    Re-check the label-2 rule on a (possibly subsampled) node feature matrix:
    >= 2 escape-capable nodes, or >= 1 escape node plus >= 1 lateral node
    (privilege-reach edges connect escape nodes to all others, so the
    escape -> lateral path exists whenever both node kinds are present).
    Augmented variants that inherit label 2 must still satisfy this rule.
    """
    esc = x[:, ESCAPE_IDX].max(axis=1) > 0
    lat = x[:, LATERAL_IDX].max(axis=1) > 0
    return int(esc.sum()) >= 2 or (bool(esc.any()) and bool(lat.any()))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=False)
    return {k: d[k].copy() for k in d.files}


def save_npz(path: Path, arrays: dict[str, np.ndarray]):
    np.savez_compressed(path, **arrays)


def augment_edge_dropout(
    arrays: dict[str, np.ndarray],
    drop_rate: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Randomly remove edges of types 0 (proximity) and 3 (semantic_ns).
    Preserve ALL type-1 (privilege_reach) and type-2 (sa_lateral) edges.
    """
    edge_index = arrays["edge_index"]   # [2, E]
    edge_attr  = arrays["edge_attr"]    # [E, 1]

    if edge_index.shape[1] == 0:
        return arrays

    types = edge_attr.ravel()
    droppable = np.where((types == EDGE_PROXIMITY) | (types == EDGE_SEMANTIC_NS))[0]
    n_drop = int(len(droppable) * drop_rate)
    drop_indices = rng.choice(droppable, size=n_drop, replace=False) if n_drop > 0 else np.array([], dtype=np.int64)

    keep_mask = np.ones(edge_index.shape[1], dtype=bool)
    keep_mask[drop_indices] = False

    result = arrays.copy()
    result["edge_index"] = edge_index[:, keep_mask]
    result["edge_attr"]  = edge_attr[keep_mask]
    return result


def augment_feature_mask(
    arrays: dict[str, np.ndarray],
    mask_rate: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Randomly zero-out a fraction of maskable node feature dimensions.
    """
    x = arrays["x"].copy()   # [N, 25]
    n_nodes = x.shape[0]

    for feat_idx in MASKABLE_FEATURES:
        # For each node independently, mask this feature with probability mask_rate
        mask = rng.random(n_nodes) < mask_rate
        x[mask, feat_idx] = 0.0

    result = arrays.copy()
    result["x"] = x
    return result


def sample_subgraph(
    arrays: dict[str, np.ndarray],
    target_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    """
    Sample a connected subgraph of approximately target_size nodes.
    Starts BFS from a chain-relevant node (highest risk_score or escape flag).
    Returns None if the graph is smaller than target_size.
    """
    n_nodes = arrays["x"].shape[0]
    if n_nodes <= target_size:
        return None

    edge_index = arrays["edge_index"]   # [2, E]
    edge_attr  = arrays["edge_attr"]    # [E, 1]

    # Build adjacency (undirected) for BFS
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for col in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, col]), int(edge_index[1, col])
        adj[src].append(dst)
        adj[dst].append(src)

    # Start from the node with highest risk_score (most security-relevant)
    risk_scores = arrays["risk_scores"]
    start = int(np.argmax(risk_scores))

    # BFS to collect connected subgraph of target_size nodes
    visited = {start}
    queue = [start]
    head = 0
    while len(visited) < target_size and head < len(queue):
        node = queue[head]
        head += 1
        neighbors = adj[node]
        rng.shuffle(neighbors)
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= target_size:
                    break

    if len(visited) < 2:
        return None

    subgraph_nodes = sorted(visited)
    node_remap = {old: new for new, old in enumerate(subgraph_nodes)}

    # Re-index node features
    new_x           = arrays["x"][subgraph_nodes]
    new_node_labels = arrays["node_labels"][subgraph_nodes]
    new_risk_scores = arrays["risk_scores"][subgraph_nodes]

    # Filter and re-index edges within subgraph
    kept_edges = []
    kept_attrs = []
    for col in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, col]), int(edge_index[1, col])
        if src in node_remap and dst in node_remap:
            kept_edges.append([node_remap[src], node_remap[dst]])
            kept_attrs.append(edge_attr[col])

    if kept_edges:
        new_edge_index = np.array(kept_edges, dtype=np.int64).T  # [2, E']
        new_edge_attr  = np.array(kept_attrs, dtype=np.int64)
    else:
        new_edge_index = np.zeros((2, 0), dtype=np.int64)
        new_edge_attr  = np.zeros((0, 1), dtype=np.int64)

    # Preserve graph label — subgraph inherits parent label
    return {
        "x":           new_x,
        "edge_index":  new_edge_index,
        "edge_attr":   new_edge_attr,
        "y":           arrays["y"].copy(),
        "node_labels": new_node_labels,
        "risk_scores": new_risk_scores,
    }


def generate_variants(
    arrays: dict[str, np.ndarray],
    n_variants: int,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    """
    Generate n_variants augmented copies of a graph using a mix of strategies.
    """
    rng = np.random.default_rng(seed)
    n_nodes = arrays["x"].shape[0]
    variants = []

    for k in range(n_variants):
        aug = {key: val.copy() for key, val in arrays.items()}

        # Strategy mix based on variant index
        # Variants 0-4:  edge dropout only (mild)
        # Variants 5-9:  edge dropout + feature masking
        # Variants 10+:  subgraph (large clusters) or edge dropout + masking (small)

        drop_rate = rng.uniform(0.10, 0.35)
        mask_rate = rng.uniform(0.10, 0.25)

        if k >= 10 and n_nodes > 50:
            # Subgraph sampling — only keep the subgraph if it still satisfies
            # the chain rule it inherits its label from; otherwise fall back to
            # edge-dropout/masking on the full graph (no label noise).
            sub_size = int(rng.uniform(0.4, 0.7) * n_nodes)
            sub_size = max(sub_size, 5)
            sub = sample_subgraph(aug, sub_size, rng)
            if sub is not None and satisfies_chain_rule(sub["x"]):
                aug = sub

        # Always apply edge dropout
        aug = augment_edge_dropout(aug, drop_rate, rng)

        if k >= 5:
            aug = augment_feature_mask(aug, mask_rate, rng)

        variants.append(aug)

    return variants


def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent.parent  # research/ (scripts live in scripts/03_augment/)
    default_graphs   = project_root / "data" / "graphs"
    default_manifest = default_graphs / "graph_manifest.csv"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graphs-dir",  type=Path, default=default_graphs)
    parser.add_argument("--variants",    type=int,  default=15,
                        help="Augmented variants per chain graph (default: 15)")
    parser.add_argument("--target-label",type=int,  default=2,
                        help="Only augment graphs with this label (default: 2=attack_chain)")
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    # Load manifest — keep only ORIGINAL rows as augmentation sources so the
    # script is idempotent (re-running never augments previous augmentations).
    with open(default_manifest, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    original_rows = [r for r in manifest_rows if "_aug_" not in r["cluster"]]

    chain_rows = [r for r in original_rows if int(r["label"]) == args.target_label]
    print(f"Found {len(chain_rows)} original graphs with label={args.target_label}")
    print(f"Generating {args.variants} variants each → +{len(chain_rows) * args.variants} graphs")

    if args.dry_run:
        print("[dry-run] No files written.")
        return

    # Remove stale augmented graphs from previous runs (idempotent regeneration)
    stale = sorted(args.graphs_dir.glob("*_aug_*.npz"))
    for path in stale:
        path.unlink()
    if stale:
        print(f"Removed {len(stale)} stale augmented .npz files")

    new_manifest_rows = []
    total_written = 0

    for row in chain_rows:
        safe_name = row["safe_name"]
        npz_path  = args.graphs_dir / f"{safe_name}.npz"
        if not npz_path.exists():
            print(f"  [skip] {safe_name}.npz not found")
            continue

        arrays = load_npz(npz_path)
        # crc32 (not built-in hash()) so per-graph seeds are stable across
        # processes — Python string hashing is randomised per run.
        name_seed = zlib.crc32(safe_name.encode("utf-8")) % 10000
        variants = generate_variants(arrays, args.variants, seed=args.seed + name_seed)

        for k, aug_arrays in enumerate(variants):
            aug_name = f"{safe_name}_aug_{k:02d}"
            aug_path = args.graphs_dir / f"{aug_name}.npz"
            save_npz(aug_path, aug_arrays)

            new_manifest_rows.append({
                "cluster":    f"{row['cluster']}_aug_{k:02d}",
                "safe_name":  aug_name,
                "n_nodes":    aug_arrays["x"].shape[0],
                "n_edges":    aug_arrays["edge_index"].shape[1],
                "label":      row["label"],
                "label_name": row["label_name"],
                "escape_cnt": row["escape_cnt"],   # inherited
                "sa_cnt":     row["sa_cnt"],        # inherited
                "misc_cnt":   row["misc_cnt"],      # inherited
                "npz_file":   f"{aug_name}.npz",
            })
            total_written += 1

        print(f"  {row['cluster']}: wrote {len(variants)} variants")

    # Rewrite manifest as originals + freshly generated rows (idempotent)
    updated_manifest = original_rows + new_manifest_rows
    with open(default_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(updated_manifest)

    # Label distribution after augmentation
    label_counts = {}
    for r in updated_manifest:
        lbl = int(r["label"])
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    print(f"\nDone. Wrote {total_written} augmented graphs.")
    print(f"Updated manifest: {default_manifest}")
    print("\nNew label distribution:")
    labels = {0: "clean", 1: "isolated", 2: "attack_chain"}
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl} ({labels.get(lbl, lbl):15s}): {cnt}")
    print(f"  Total graphs: {sum(label_counts.values())}")


if __name__ == "__main__":
    main()
