"""
gnn_dataset.py
==============
PyTorch Geometric dataset wrapper for the K8s cluster attack-chain graphs.

Loads .npz files from dataset/graphs/ and serves them as PyG Data objects.
Also provides utility functions for batch loading and normalization.

Usage:
  from gnn_dataset import KubeClusterDataset, load_split

  # Full dataset
  dataset = KubeClusterDataset(graphs_dir="dataset/graphs")
  print(f"Graphs: {len(dataset)}, features: {dataset.num_node_features}")

  # Train split
  train_set = load_split("dataset/graphs", "dataset/splits/train.txt")

  # DataLoader
  from torch_geometric.loader import DataLoader
  loader = DataLoader(train_set, batch_size=16, shuffle=True)

Requirements:
  pip install torch torch-geometric
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    HAS_TORCH = True
except ImportError as exc:
    HAS_TORCH = False
    raise ImportError(
        "PyTorch and PyTorch Geometric are required:\n"
        "  pip install torch\n"
        "  pip install torch-geometric"
    ) from exc


# ---------------------------------------------------------------------------
# Label names (matches graph_builder.py)
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "clean", 1: "isolated_misconfig", 2: "attack_chain"}
NODE_FEATURE_DIM = 26

FEATURE_NAMES = [
    # indices 0-17: Rahman binary flags
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET", "DOCKERSOCK_PATH",
    "CAP_SYS_ADMIN", "CAP_SYS_MODULE", "WITHIN_MANIFEST_SECRET",
    "SEC_CONT_OVER_PRIVIL", "ALLOW_PRIVI", "SECCOMP_UNCONFINED",
    "VALID_TAINT_SECRET", "INSECURE_HTTP", "NO_SECU_CONTEXT",
    "NO_NETWORK_POLICY", "HOST_ALIAS", "NO_DEFAULT_NSPACE",
    "NO_RESO", "NO_ROLLING_UPDATE",
    # indices 18-24: extended features
    "NO_RUN_AS_NON_ROOT", "NO_READ_ONLY_ROOT_FS", "IMAGE_USES_LATEST",
    "SA_AUTOMOUNT_TOKEN", "USES_DEFAULT_SA", "UNTRUSTED_REGISTRY",
    "HOSTPATH_MOUNT",
    # index 25: risk score
    "risk_score",
]


# ---------------------------------------------------------------------------
# Low-level: npz → PyG Data
# ---------------------------------------------------------------------------

def arrays_to_data(d: dict, cluster_name: str | None = None) -> Data:
    """Build a PyG Data object from a dict of numpy arrays (npz field names)."""
    data = Data(
        x          = torch.FloatTensor(d["x"]),
        edge_index = torch.LongTensor(d["edge_index"]),
        edge_attr  = torch.LongTensor(d["edge_attr"]),
        y          = torch.LongTensor(d["y"]),
        node_y     = torch.LongTensor(d["node_labels"]),
        risk       = torch.FloatTensor(d["risk_scores"]),
    )
    if cluster_name is not None:
        data.cluster = cluster_name
    return data


def npz_to_data(npz_path: Path, cluster_name: str | None = None) -> Data:
    """
    Load a single .npz graph file and return a PyG Data object.

    Fields set on Data:
      x           FloatTensor [N, 25]  – node feature matrix
      edge_index  LongTensor  [2, E]   – COO edge list
      edge_attr   LongTensor  [E, 1]   – edge type codes
      y           LongTensor  [1]      – graph-level label (0/1/2)
      node_y      LongTensor  [N]      – per-node binary label
      risk        FloatTensor [N]      – per-node risk score
      cluster     str                  – cluster name (if provided)
    """
    d = np.load(npz_path, allow_pickle=False)
    return arrays_to_data({k: d[k] for k in d.files}, cluster_name=cluster_name)


# ---------------------------------------------------------------------------
# KubeClusterDataset — InMemoryDataset subclass
# ---------------------------------------------------------------------------

class KubeClusterDataset(InMemoryDataset):
    """
    In-memory PyG dataset of Kubernetes cluster graphs.

    Parameters
    ----------
    graphs_dir : str | Path
        Directory containing .npz graph files and graph_manifest.csv.
    cluster_names : list[str] | None
        If provided, load only these clusters (for split-based loading).
    transform : Callable | None
        PyG transform applied on-the-fly (e.g., NormalizeFeatures()).
    pre_transform : Callable | None
        Transform applied once at load time.
    binary : bool
        If True, collapse label 1 and 2 → 1 (binary: clean vs. any-misconfig).
        Default: False (3-class: clean / isolated / chain).
    """

    def __init__(
        self,
        graphs_dir: str | Path,
        cluster_names: list[str] | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        binary: bool = False,
    ):
        self.graphs_dir    = Path(graphs_dir)
        self.cluster_names = cluster_names  # subset filter
        self.binary        = binary
        # InMemoryDataset calls download() and process() in __init__
        # We skip those by using root=None and loading directly
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self._load_graphs()

    # InMemoryDataset requires these; we skip them
    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def download(self): pass
    def process(self): pass

    def _load_graphs(self):
        """Load all matching graphs into memory (consolidated cache if present)."""
        import csv as csv_mod
        manifest_path = self.graphs_dir / "graph_manifest.csv"

        # Build name → safe_name mapping from manifest
        name_to_safe: dict[str, str] = {}
        if manifest_path.exists():
            with open(manifest_path, newline="", encoding="utf-8") as f:
                for row in csv_mod.DictReader(f):
                    name_to_safe[row["cluster"]] = row["safe_name"]

        # Determine which clusters to load
        if self.cluster_names is not None:
            targets = self.cluster_names
        else:
            targets = list(name_to_safe.keys())

        # Consolidated cache (graphs_cache.npz, built by build_graph_cache.py):
        # one file open instead of one per graph — per-file open latency
        # dominates split loading otherwise.
        cache_path = self.graphs_dir / "graphs_cache.npz"
        cache = np.load(cache_path, allow_pickle=False) if cache_path.exists() else None

        data_list = []
        for cluster in targets:
            safe = name_to_safe.get(cluster, cluster)
            if cache is not None and f"{safe}::x" in cache.files:
                data = arrays_to_data(
                    {k: cache[f"{safe}::{k}"] for k in
                     ("x", "edge_index", "edge_attr", "y", "node_labels", "risk_scores")},
                    cluster_name=cluster,
                )
            else:
                npz_path = self.graphs_dir / f"{safe}.npz"
                if not npz_path.exists():
                    continue
                data = npz_to_data(npz_path, cluster_name=cluster)
            if self.binary:
                # Collapse: 0 → 0 (clean),  1/2 → 1 (any misconfig)
                data.y = (data.y > 0).long()
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)
        self._cluster_list = [d.cluster for d in data_list]

    def __len__(self) -> int:
        return self.slices["y"].numel() - 1

    @property
    def num_node_features(self) -> int:
        return NODE_FEATURE_DIM

    @property
    def num_classes(self) -> int:
        return 2 if self.binary else 3

    def cluster_names_list(self) -> list[str]:
        return self._cluster_list


# ---------------------------------------------------------------------------
# Convenience loader using split .txt files
# ---------------------------------------------------------------------------

def load_split(
    graphs_dir: str | Path,
    split_file: str | Path,
    transform: Callable | None = None,
    binary: bool = False,
) -> KubeClusterDataset:
    """
    Load a subset of graphs specified by a split .txt file (one cluster per line).

    Example:
        train_set = load_split("dataset/graphs", "dataset/splits/train.txt")
    """
    split_path = Path(split_file)
    with open(split_path, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return KubeClusterDataset(
        graphs_dir=graphs_dir,
        cluster_names=names,
        transform=transform,
        binary=binary,
    )


# ---------------------------------------------------------------------------
# Feature statistics (for normalization)
# ---------------------------------------------------------------------------

def compute_feature_stats(dataset: KubeClusterDataset) -> dict:
    """
    Compute mean and std of node features across all graphs.
    Use these to normalise before training.
    """
    all_x = []
    for i in range(len(dataset)):
        data = dataset[i]
        all_x.append(data.x)

    all_x_cat = torch.cat(all_x, dim=0)  # [total_nodes, 25]
    mean = all_x_cat.mean(dim=0)
    std  = all_x_cat.std(dim=0).clamp(min=1e-6)

    return {
        "mean": mean,
        "std":  std,
        "feature_names": FEATURE_NAMES,
        "n_total_nodes": all_x_cat.shape[0],
    }


# ---------------------------------------------------------------------------
# Quick diagnostics (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    graphs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).parent.parent / "graphs"
    splits_dir = graphs_dir.parent / "splits"

    print(f"Loading full dataset from {graphs_dir}...")
    full = KubeClusterDataset(graphs_dir)
    print(f"  Graphs: {len(full)}")
    print(f"  Node features: {full.num_node_features}")
    print(f"  Classes: {full.num_classes}")

    labels = torch.cat([full[i].y for i in range(len(full))])
    from collections import Counter
    print(f"  Label distribution: {dict(Counter(labels.tolist()))}")

    if (splits_dir / "train.txt").exists():
        train = load_split(graphs_dir, splits_dir / "train.txt")
        val   = load_split(graphs_dir, splits_dir / "val.txt")
        test  = load_split(graphs_dir, splits_dir / "test.txt")
        print(f"\nSplit sizes — train:{len(train)}, val:{len(val)}, test:{len(test)}")

    stats = compute_feature_stats(full)
    print(f"\nFeature stats (total nodes: {stats['n_total_nodes']}):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:30s}: mean={stats['mean'][i]:.3f}, std={stats['std'][i]:.3f}")
