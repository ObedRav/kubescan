"""
build_graphs.py
================
Convert K8s manifest clusters (grouped by repo_name) into graph objects
for GNN Layer 2 attack-chain prediction.

Graph model
-----------
  One graph per cluster (repo_name).
  Each node = one K8s manifest row from rf_dataset.csv.
  Node features (NODE_FEATURE_DIM = 25):
      indices 0-17  : 18 Rahman binary misconfiguration flags
      indices 18-23 : 6 extended security features (NO_RUN_AS_NON_ROOT …)
      index 24      : risk_score (float, 0-1)

  Edges (directed):
      type 0 – directory_proximity  : manifests in same YAML sub-directory
      type 1 – privilege_reach      : privileged pod → all others in cluster
      type 2 – sa_lateral           : SA-exposed pod → co-located pods
      type 3 – semantic_namespace   : same K8s namespace (from parsed YAML)

  Graph-level attack-chain label (3 classes):
      0 – clean          : all nodes label=0
      1 – isolated       : ≥1 misconfigured node, no compounding chain
      2 – attack_chain   : escape-capable node reachable from / to an
                           SA/lateral-movement node, or ≥2 escape nodes

Output
------
  dataset/graphs/<cluster_name>.npz   – node/edge arrays (numpy)
  dataset/graphs/<cluster_name>.json  – human-readable graph
  dataset/graphs/graph_manifest.csv   – index (name, n_nodes, n_edges, label, …)
  dataset/graphs/graphs_all.pt        – combined list of dicts (if torch present)

Usage
-----
  python scripts/graph_builder.py
  python scripts/graph_builder.py --min-nodes 2 --dry-run
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Feature schema — must match rf_dataset.csv column order
# ---------------------------------------------------------------------------

RAHMAN_FLAGS = [
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET", "DOCKERSOCK_PATH",
    "CAP_SYS_ADMIN", "CAP_SYS_MODULE", "WITHIN_MANIFEST_SECRET",
    "SEC_CONT_OVER_PRIVIL", "ALLOW_PRIVI", "SECCOMP_UNCONFINED",
    "VALID_TAINT_SECRET", "INSECURE_HTTP", "NO_SECU_CONTEXT",
    "NO_NETWORK_POLICY", "HOST_ALIAS", "NO_DEFAULT_NSPACE",
    "NO_RESO", "NO_ROLLING_UPDATE",
]

EXTENDED_FLAGS = [
    "NO_RUN_AS_NON_ROOT", "NO_READ_ONLY_ROOT_FS", "IMAGE_USES_LATEST",
    "SA_AUTOMOUNT_TOKEN", "USES_DEFAULT_SA", "UNTRUSTED_REGISTRY",
    "HOSTPATH_MOUNT",   # non-docker-sock hostPath volume (host FS escape)
]

ALL_FEATURE_COLS = RAHMAN_FLAGS + EXTENDED_FLAGS  # 25 binary features
# index 25 = risk_score  →  NODE_FEATURE_DIM = 26
NODE_FEATURE_DIM = 26

# Edge type codes
EDGE_DIR_PROXIMITY  = 0   # same directory
EDGE_PRIV_REACH     = 1   # privileged pod → other
EDGE_SA_LATERAL     = 2   # SA-exposed pod → co-located pod
EDGE_SEMANTIC_NS    = 3   # same K8s namespace (from YAML)
EDGE_RBAC_PRIV      = 4   # pod whose SA has an elevated RoleBinding → others

# Flags that grant pod-escape / host-level access
ESCAPE_FLAGS = {
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET",
    "DOCKERSOCK_PATH", "CAP_SYS_ADMIN", "CAP_SYS_MODULE",
    "SEC_CONT_OVER_PRIVIL",
    "HOSTPATH_MOUNT",   # hostPath vol (non-docker-sock) — host FS access
}
# Flags that enable lateral movement / credential theft
LATERAL_FLAGS = {
    "SA_AUTOMOUNT_TOKEN", "USES_DEFAULT_SA",
    "WITHIN_MANIFEST_SECRET", "ALLOW_PRIVI",
}


# ---------------------------------------------------------------------------
# YAML semantic parser
# ---------------------------------------------------------------------------

WORKLOAD_KINDS = {
    "Pod", "Deployment", "DaemonSet", "StatefulSet", "ReplicaSet",
    "ReplicationController", "Job", "CronJob",
}

def _safe_load_all(path: Path) -> list[dict]:
    """Load all YAML documents from a file; skip non-dict items silently."""
    docs = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        for doc in yaml.safe_load_all(raw):
            if isinstance(doc, dict):
                docs.append(doc)
    except Exception:
        pass
    return docs


def _get_pod_spec(doc: dict) -> dict | None:
    """Return the PodSpec for any workload kind."""
    kind = doc.get("kind", "")
    spec = doc.get("spec") or {}
    if not isinstance(spec, dict):
        return None
    if kind == "Pod":
        return spec
    if kind == "CronJob":
        jt = (spec.get("jobTemplate") or {})
        jt_spec = (jt.get("spec") or {})
        tmpl = (jt_spec.get("template") or {})
        return (tmpl.get("spec") or {}) if isinstance(tmpl, dict) else {}
    template = spec.get("template") or {}
    if not isinstance(template, dict):
        return None
    return template.get("spec") or {}


def parse_yaml_semantics(local_path: str) -> dict:
    """
    Parse a downloaded K8s YAML file and return semantic metadata:
      namespace, serviceAccountName, rbac_subjects, rbac_roleref
    Returns {} if the file cannot be parsed.
    """
    path = Path(local_path)
    if not path.exists():
        return {}

    result = {
        "namespace": None,
        "service_account": None,
        "rbac_subjects": [],   # SA names this RoleBinding grants
        "rbac_roleref": None,  # Role/ClusterRole name
        "is_workload": False,
        "hostpath_mount": False,  # any non-docker-sock hostPath volume
    }

    docs = _safe_load_all(path)
    for doc in docs:
        kind = doc.get("kind", "")
        meta = doc.get("metadata", {}) or {}

        ns = meta.get("namespace")
        if ns and result["namespace"] is None:
            result["namespace"] = ns

        if kind in WORKLOAD_KINDS:
            result["is_workload"] = True
            pod_spec = _get_pod_spec(doc) or {}
            sa = pod_spec.get("serviceAccountName")
            if sa and result["service_account"] is None:
                result["service_account"] = sa
            # Detect non-docker-sock hostPath mounts (escape vector)
            for vol in (pod_spec.get("volumes") or []):
                if not isinstance(vol, dict):
                    continue
                hp = vol.get("hostPath")
                if not hp:
                    continue
                path_val = str(hp.get("path", ""))
                if "/var/run/docker.sock" not in path_val and "/docker.sock" not in path_val:
                    result["hostpath_mount"] = True

        if kind in ("RoleBinding", "ClusterRoleBinding"):
            roleref = doc.get("roleRef", {})
            result["rbac_roleref"] = roleref.get("name")
            subjects = doc.get("subjects") or []
            for s in subjects:
                if isinstance(s, dict) and s.get("kind") == "ServiceAccount":
                    name = s.get("name")
                    if name:
                        result["rbac_subjects"].append(name)

    return result


# ---------------------------------------------------------------------------
# Node feature vector builder
# ---------------------------------------------------------------------------

def row_to_feature_vector(row: dict) -> np.ndarray:
    """
    Build a 26-dim numpy float32 feature vector from a rf_dataset row.
    Indices 0-24: 25 binary flags (18 Rahman + 7 extended incl. HOSTPATH_MOUNT).
    Index 25: risk_score (float, 0-1).
    Missing / empty extended features are imputed as 0.
    """
    vec = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
    for i, col in enumerate(ALL_FEATURE_COLS):
        val = row.get(col, "")
        try:
            vec[i] = float(val) if val != "" else 0.0
        except (ValueError, TypeError):
            vec[i] = 0.0
    # risk_score at index 25
    try:
        vec[25] = float(row.get("risk_score", 0) or 0)
    except (ValueError, TypeError):
        vec[25] = 0.0
    return vec


# ---------------------------------------------------------------------------
# Path-based directory key
# ---------------------------------------------------------------------------

def dir_key(yaml_path: str, depth: int = 2) -> str:
    """
    Extract a "directory fingerprint" from a yaml_path.
    Two manifests sharing the same fingerprint are in the same sub-tree.
    depth=2 means: last 2 path components before the filename.
    """
    parts = Path(yaml_path).parts
    if len(parts) >= depth + 1:
        return "/".join(parts[-(depth + 1):-1])
    return str(Path(yaml_path).parent)


# ---------------------------------------------------------------------------
# Single-cluster graph builder
# ---------------------------------------------------------------------------

def build_cluster_graph(
    cluster_name: str,
    rows: list[dict],
    manifest_ok: dict[str, str],  # yaml_url → local_path (status=ok)
    key_to_url: dict[tuple, str],  # (repo, relpath) → yaml_url
) -> dict | None:
    """
    Build a directed NetworkX graph for one cluster.
    Returns a dict with graph, label, stats, or None if rows is empty.
    """
    if not rows:
        return None

    G = nx.DiGraph()

    # ------------------------------------------------------------------
    # 1. Add nodes
    # ------------------------------------------------------------------
    node_data: list[dict] = []
    for idx, row in enumerate(rows):
        feats = row_to_feature_vector(row)
        flags = {col: int(float(row.get(col, 0) or 0)) for col in ALL_FEATURE_COLS}

        G.add_node(idx, **{
            "manifest_id":   row.get("manifest_id", ""),
            "yaml_path":     row.get("yaml_path", ""),
            "label":         int(row.get("label", 0) or 0),
            "risk_score":    float(row.get("risk_score", 0) or 0),
            "severity_class":int(str(row.get("severity_class", 0) or 0)),
            "has_yaml":      int(row.get("has_yaml", 0) or 0),
            "features":      feats,
            **flags,
        })
        node_data.append({**flags,
                          "label": int(row.get("label", 0) or 0),
                          "risk_score": float(row.get("risk_score", 0) or 0),
                          "yaml_path": row.get("yaml_path", ""),
                          "has_yaml": int(row.get("has_yaml", 0) or 0)})

    n = len(rows)

    # ------------------------------------------------------------------
    # 2. YAML semantic parsing — runs BEFORE edge construction so that
    #    HOSTPATH_MOUNT can influence escape_nodes and privilege_reach edges.
    #    Also collects RBAC subjects and pod SA names for edge type 4.
    # ------------------------------------------------------------------
    ns_groups: dict[str, list[int]] = defaultdict(list)
    pod_sa:    dict[int, str] = {}          # idx → serviceAccountName
    elevated_sas: set[str]   = set()        # SA names with elevated RoleBindings

    for idx, row in enumerate(rows):
        local = _resolve_local(row.get("yaml_path", ""), key_to_url, manifest_ok)
        if not local:
            ns_groups["_default"].append(idx)
            continue
        sem = parse_yaml_semantics(local)
        ns = sem.get("namespace") or "_default"
        ns_groups[ns].append(idx)

        # HOSTPATH_MOUNT: from YAML, override CSV value for escape detection
        if sem.get("hostpath_mount"):
            node_data[idx]["HOSTPATH_MOUNT"] = 1

        # Collect SA name used by this pod
        sa = sem.get("service_account")
        if sa:
            pod_sa[idx] = sa

        # Collect elevated SA names from any RoleBindings in this file
        for subj in (sem.get("rbac_subjects") or []):
            if subj:
                elevated_sas.add(subj)

    # ------------------------------------------------------------------
    # 3. Edge type 0 — directory proximity  (undirected → bidirectional)
    # ------------------------------------------------------------------
    dir_groups: dict[str, list[int]] = defaultdict(list)
    for idx, nd in enumerate(node_data):
        dk = dir_key(nd["yaml_path"])
        dir_groups[dk].append(idx)

    for members in dir_groups.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                G.add_edge(a, b, edge_type=EDGE_DIR_PROXIMITY)
                G.add_edge(b, a, edge_type=EDGE_DIR_PROXIMITY)

    # ------------------------------------------------------------------
    # 4. Edge type 1 — privilege-reach  (directed: escape_node → all others)
    #    Now includes HOSTPATH_MOUNT escape nodes (detected from YAML above)
    # ------------------------------------------------------------------
    escape_nodes = [
        idx for idx, nd in enumerate(node_data)
        if any(nd.get(f, 0) for f in ESCAPE_FLAGS)
    ]
    for src in escape_nodes:
        for dst in range(n):
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type=EDGE_PRIV_REACH)

    # ------------------------------------------------------------------
    # 5. Edge type 2 — SA lateral movement  (directed: sa_node → all others)
    # ------------------------------------------------------------------
    sa_nodes = [
        idx for idx, nd in enumerate(node_data)
        if any(nd.get(f, 0) for f in LATERAL_FLAGS)
    ]
    for src in sa_nodes:
        for dst in range(n):
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type=EDGE_SA_LATERAL)

    # ------------------------------------------------------------------
    # 6. Edge type 3 — semantic namespace  (bidirectional, ns_groups from step 2)
    # ------------------------------------------------------------------
    for members in ns_groups.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if not G.has_edge(a, b):
                    G.add_edge(a, b, edge_type=EDGE_SEMANTIC_NS)
                if not G.has_edge(b, a):
                    G.add_edge(b, a, edge_type=EDGE_SEMANTIC_NS)

    # ------------------------------------------------------------------
    # 6b. Edge type 4 — RBAC privilege escalation
    #     Pods whose ServiceAccount has an elevated RoleBinding (SA name
    #     appears as a subject in any RoleBinding/ClusterRoleBinding parsed
    #     across this cluster) get directed edges to all other pods.
    #     This captures privilege escalation via RBAC that isn't reflected
    #     in individual pod flags (e.g., cluster-admin SA binding).
    # ------------------------------------------------------------------
    if elevated_sas:
        rbac_priv_nodes = [idx for idx, sa in pod_sa.items() if sa in elevated_sas]
        for src in rbac_priv_nodes:
            for dst in range(n):
                if dst != src and not G.has_edge(src, dst):
                    G.add_edge(src, dst, edge_type=EDGE_RBAC_PRIV)

    # ------------------------------------------------------------------
    # 7. Graph-level attack-chain label
    # ------------------------------------------------------------------
    label = _compute_graph_label(G, node_data, n)

    return {
        "cluster":    cluster_name,
        "graph":      G,
        "node_data":  node_data,
        "label":      label,
        "n_nodes":    n,
        "n_edges":    G.number_of_edges(),
        "escape_cnt": len(escape_nodes),
        "sa_cnt":     len(sa_nodes),
        "misc_cnt":   sum(1 for nd in node_data if nd["label"] == 1),
    }


def _compute_graph_label(
    G: nx.DiGraph,
    node_data: list[dict],
    n: int,
) -> int:
    """
    Assign cluster-level attack-chain severity.

    0 – clean       : all nodes secure
    1 – isolated    : some misconfigured nodes, no compounding chain
    2 – chain       : at least two attack stages connected by a path:
                      (escape_node → … → lateral_node) or
                      ≥ 2 escape-capable nodes in the same cluster
    """
    all_secure = all(nd["label"] == 0 for nd in node_data)
    if all_secure:
        return 0

    escape_idxs = [i for i, nd in enumerate(node_data)
                   if any(nd.get(f, 0) for f in ESCAPE_FLAGS)]
    lateral_idxs = [i for i, nd in enumerate(node_data)
                    if any(nd.get(f, 0) for f in LATERAL_FLAGS)]

    # ≥2 escape nodes = compound escalation possible even without lateral path
    if len(escape_idxs) >= 2:
        return 2

    # escape → lateral path (or reverse)
    if escape_idxs and lateral_idxs:
        for src in escape_idxs:
            for dst in lateral_idxs:
                if src == dst:
                    continue
                if nx.has_path(G, src, dst) or nx.has_path(G, dst, src):
                    return 2

    return 1


# ---------------------------------------------------------------------------
# Path resolution (re-used from scan_with_tools logic)
# ---------------------------------------------------------------------------

def _extract_repo_relpath(path: str) -> tuple[str, str] | None:
    """Extract (repo_name, relative_path) from a rahman dataset yaml_path."""
    patterns = [
        r"GITHUB_REPOS(?:_NODEPLOY)?/([^/]+)/(.+)",
        r"GITLAB(?:_K8S_REPOS_RAW_UNFILTERED|_REPOS)/([^/]+)/(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, path)
        if m:
            return m.group(1), m.group(2)
    return None


def _resolve_local(
    yaml_path: str,
    key_to_url: dict,
    manifest_ok: dict,
) -> str | None:
    """Resolve rf_dataset yaml_path → downloaded local file path."""
    result = _extract_repo_relpath(yaml_path)
    if not result:
        # Special sources (badpods, kubernetes_goat) store real local paths
        if Path(yaml_path).exists():
            return yaml_path
        return None
    url = key_to_url.get(result)
    if not url:
        return None
    return manifest_ok.get(url)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def graph_to_arrays(result: dict) -> dict[str, np.ndarray]:
    """
    Convert a cluster graph result to numpy arrays for saving.

    Arrays:
      x           [N, 25]  node features
      edge_index  [2, E]   edge (src, dst) pairs
      edge_attr   [E, 1]   edge type code
      y           [1]      graph label
      node_labels [N]      per-node binary label (for node-level tasks)
      risk_scores [N]      per-node risk score
    """
    G: nx.DiGraph = result["graph"]
    n = result["n_nodes"]
    node_data = result["node_data"]

    x = np.stack([G.nodes[i]["features"] for i in range(n)]).astype(np.float32)

    edges = list(G.edges(data=True))
    if edges:
        src = np.array([e[0] for e in edges], dtype=np.int64)
        dst = np.array([e[1] for e in edges], dtype=np.int64)
        edge_index = np.stack([src, dst])
        edge_attr = np.array(
            [e[2].get("edge_type", 0) for e in edges], dtype=np.int64
        ).reshape(-1, 1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, 1), dtype=np.int64)

    y = np.array([result["label"]], dtype=np.int64)
    node_labels = np.array([nd["label"] for nd in node_data], dtype=np.int64)
    risk_scores = np.array([nd["risk_score"] for nd in node_data], dtype=np.float32)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": y,
        "node_labels": node_labels,
        "risk_scores": risk_scores,
    }


def graph_to_json(result: dict) -> dict:
    """Serialise a cluster graph result to a JSON-compatible dict."""
    G: nx.DiGraph = result["graph"]
    node_data = result["node_data"]

    nodes_out = []
    for i, nd in enumerate(node_data):
        nodes_out.append({
            "id": i,
            "yaml_path": nd["yaml_path"],
            "label": nd["label"],
            "risk_score": round(nd["risk_score"], 4),
            "has_yaml": nd["has_yaml"],
            "features": {col: nd.get(col, 0) for col in ALL_FEATURE_COLS},
        })

    edges_out = []
    edge_type_names = {
        EDGE_DIR_PROXIMITY: "directory_proximity",
        EDGE_PRIV_REACH:    "privilege_reach",
        EDGE_SA_LATERAL:    "sa_lateral",
        EDGE_SEMANTIC_NS:   "semantic_namespace",
    }
    for src, dst, data in G.edges(data=True):
        edges_out.append({
            "src": src,
            "dst": dst,
            "type": edge_type_names.get(data.get("edge_type", 0), "unknown"),
        })

    return {
        "cluster": result["cluster"],
        "label": result["label"],
        "n_nodes": result["n_nodes"],
        "n_edges": result["n_edges"],
        "escape_cnt": result["escape_cnt"],
        "sa_cnt": result["sa_cnt"],
        "misc_cnt": result["misc_cnt"],
        "nodes": nodes_out,
        "edges": edges_out,
    }


# ---------------------------------------------------------------------------
# Build lookup tables from download manifest + URLS.csv
# ---------------------------------------------------------------------------

def build_lookups(
    urls_csv: Path,
    manifest_csv: Path,
) -> tuple[dict[tuple, str], dict[str, str]]:
    """
    Returns:
      key_to_url  : (repo_name, relpath) → yaml_url
      manifest_ok : yaml_url → local_path  (status=ok rows only)
    """
    key_to_url: dict[tuple, str] = {}
    for csv_path in [urls_csv, urls_csv.parent / "GITLAB-URLS.csv"]:
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                orig = row.get("YAML_PATH", "").strip()
                url  = row.get("YAML_URL",  "").strip()
                # URLS.csv uses GITHUB_K8S_REPOS_RAW_UNFILTERED or GITLAB pattern
                for pat in [
                    r"GITHUB_K8S_REPOS_RAW_UNFILTERED/([^/]+)/(.+)",
                    r"GITLAB_K8S_REPOS_RAW_UNFILTERED/([^/]+)/(.+)",
                ]:
                    m = re.search(pat, orig)
                    if m and url:
                        key_to_url[(m.group(1), m.group(2))] = url
                        break

    manifest_ok: dict[str, str] = {}
    if manifest_csv.exists():
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url   = row.get("yaml_url",   "").strip()
                local = row.get("local_path",  "").strip()
                if url and local and row.get("status") == "ok":
                    manifest_ok[url] = local

    return key_to_url, manifest_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent

    default_rf       = project_root / "data" / "tabular" / "rf_dataset.csv"
    default_out      = project_root / "data" / "graphs"
    default_manifest = project_root / "data" / "raw" / "rahman" / "download_manifest.csv"
    default_urls     = project_root / "original-dataset" / "rahman" / "DATASET" / "GITHUB-URLS.csv"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rf-dataset",   type=Path, default=default_rf)
    parser.add_argument("--out-dir",      type=Path, default=default_out)
    parser.add_argument("--manifest-csv", type=Path, default=default_manifest)
    parser.add_argument("--urls-csv",     type=Path, default=default_urls)
    parser.add_argument("--min-nodes",    type=int,  default=1,
                        help="Skip clusters with fewer than N nodes (default: 1)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Build graphs but do not write files")
    parser.add_argument("--json",         action="store_true",
                        help="Also write per-graph JSON files (larger output)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"Loading {args.rf_dataset}...")
    with open(args.rf_dataset, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    print(f"  {len(all_rows)} rows")

    # ------------------------------------------------------------------
    # Build path lookup tables
    # ------------------------------------------------------------------
    print("Building path lookup tables...")
    key_to_url, manifest_ok = build_lookups(args.urls_csv, args.manifest_csv)
    print(f"  URL keys: {len(key_to_url)},  downloaded files: {len(manifest_ok)}")

    # ------------------------------------------------------------------
    # Group rows by cluster
    # ------------------------------------------------------------------
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        clusters[row["repo_name"]].append(row)

    print(f"  {len(clusters)} unique clusters")

    # ------------------------------------------------------------------
    # Build one graph per cluster
    # ------------------------------------------------------------------
    results = []
    skipped = 0

    for cluster_name, rows in sorted(clusters.items()):
        if len(rows) < args.min_nodes:
            skipped += 1
            continue

        result = build_cluster_graph(cluster_name, rows, manifest_ok, key_to_url)
        if result is None:
            skipped += 1
            continue

        results.append(result)

    print(f"\nBuilt {len(results)} cluster graphs  ({skipped} skipped, min_nodes={args.min_nodes})")

    # ------------------------------------------------------------------
    # Summary / label distribution
    # ------------------------------------------------------------------
    label_counts = [0, 0, 0]
    for r in results:
        label_counts[r["label"]] += 1

    print("\nGraph-level label distribution:")
    labels_str = ["clean", "isolated", "chain"]
    for i, (lbl, cnt) in enumerate(zip(labels_str, label_counts)):
        pct = 100 * cnt / max(len(results), 1)
        print(f"  {i} ({lbl:8s}): {cnt:3d}  ({pct:.1f}%)")

    print("\nNode count stats:")
    sizes = [r["n_nodes"] for r in results]
    print(f"  min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")
    print("\nEdge count stats:")
    edges = [r["n_edges"] for r in results]
    print(f"  min={min(edges)}, max={max(edges)}, mean={sum(edges)/len(edges):.1f}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ------------------------------------------------------------------
    # Write per-cluster .npz  (and optionally .json)
    # ------------------------------------------------------------------
    print(f"\nWriting graphs to {args.out_dir}/...")
    manifest_rows = []

    for result in results:
        safe_name = re.sub(r"[^\w\-]", "_", result["cluster"])
        arrays = graph_to_arrays(result)

        npz_path = args.out_dir / f"{safe_name}.npz"
        np.savez_compressed(npz_path, **arrays)

        if args.json:
            json_path = args.out_dir / f"{safe_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(graph_to_json(result), f, indent=2)

        manifest_rows.append({
            "cluster":    result["cluster"],
            "safe_name":  safe_name,
            "n_nodes":    result["n_nodes"],
            "n_edges":    result["n_edges"],
            "label":      result["label"],
            "label_name": labels_str[result["label"]],
            "escape_cnt": result["escape_cnt"],
            "sa_cnt":     result["sa_cnt"],
            "misc_cnt":   result["misc_cnt"],
            "npz_file":   f"{safe_name}.npz",
        })

    # ------------------------------------------------------------------
    # Write graph_manifest.csv
    # ------------------------------------------------------------------
    manifest_path = args.out_dir / "graph_manifest.csv"
    fieldnames = list(manifest_rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"  graph_manifest.csv  ({len(manifest_rows)} rows)")

    # ------------------------------------------------------------------
    # Optionally save all graphs as a combined .pt (if torch available)
    # ------------------------------------------------------------------
    if HAS_TORCH:
        combined = []
        for result in results:
            arrays = graph_to_arrays(result)
            combined.append({
                "cluster":     result["cluster"],
                "x":           torch.FloatTensor(arrays["x"]),
                "edge_index":  torch.LongTensor(arrays["edge_index"]),
                "edge_attr":   torch.LongTensor(arrays["edge_attr"]),
                "y":           torch.LongTensor(arrays["y"]),
                "node_labels": torch.LongTensor(arrays["node_labels"]),
                "risk_scores": torch.FloatTensor(arrays["risk_scores"]),
            })
        pt_path = args.out_dir / "graphs_all.pt"
        torch.save(combined, pt_path)
        print(f"  graphs_all.pt  ({len(combined)} graphs, torch format)")
    else:
        print("  [torch not installed — skipping graphs_all.pt; install with: pip install torch]")

    print(f"\nDone. Output directory: {args.out_dir}")
    print(f"  Feature dimensionality : {NODE_FEATURE_DIM}")
    print(f"  Total graphs           : {len(results)}")
    print("  To load a graph in Python:")
    print("    import numpy as np")
    print("    d = np.load('dataset/graphs/<name>.npz')")
    print(f"    # x={NODE_FEATURE_DIM}-dim node features, edge_index, edge_attr, y (label)")


if __name__ == "__main__":
    main()
