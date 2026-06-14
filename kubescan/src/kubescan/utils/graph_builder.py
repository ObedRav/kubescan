"""
graph_builder.py
================
In-memory cluster graph builder for inference.

Mirrors research/scripts/02_extract/build_graphs.py but works from local file
paths directly (no CSV/URL lookup infrastructure needed). Only graph construction
and PyG conversion logic lives here — no training artifacts.
"""
from __future__ import annotations

__all__ = [
    "EDGE_DIR_PROXIMITY",
    "EDGE_PRIV_REACH",
    "EDGE_RBAC_PRIV",
    "EDGE_SA_LATERAL",
    "EDGE_SEMANTIC_NS",
    "ESCAPE_FLAGS",
    "LATERAL_FLAGS",
    "NODE_FEATURE_DIM",
    "EdgeType",
    "build_cluster_graph",
    "graph_to_pyg",
]

import logging
from enum import IntEnum
from pathlib import Path
from typing import Final, TypeAlias

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from .yaml_parser import (
    FEATURE_COLS,
    WORKLOAD_KINDS,
    _get_pod_spec,
    _safe_dict,
    _safe_load_all,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

# Node feature layout (must match research training):
#   Indices 0-24: binary flags (FEATURE_COLS order)
#   Index 25:     risk_score (from RF)
NODE_FEATURE_DIM: Final[int] = 26
RISK_SCORE_INDEX: Final[int] = NODE_FEATURE_DIM - 1

DEFAULT_NAMESPACE:    Final[str] = "_default"
DEFAULT_DIR_KEY_DEPTH: Final[int] = 2
_DOCKER_SOCK_PATH:    Final[str] = "docker.sock"

# Role names that always grant privileged access regardless of rules
_PRIVILEGED_ROLE_NAMES: Final[frozenset[str]] = frozenset({
    "cluster-admin", "admin", "edit",
})

# Verbs that grant privilege escalation regardless of resource
_PRIVILEGE_ESCALATION_VERBS: Final[frozenset[str]] = frozenset({
    "*", "escalate", "bind", "impersonate",
})


class EdgeType(IntEnum):
    """Typed edge categories for the cluster attack graph."""
    DIR_PROXIMITY = 0   # manifests in the same directory
    PRIV_REACH    = 1   # escape-capable node → all others
    SA_LATERAL    = 2   # lateral-movement-capable node → all others
    SEMANTIC_NS   = 3   # manifests in the same Kubernetes namespace
    RBAC_PRIV     = 4   # pod bound to a privileged ServiceAccount via RBAC


# Backward-compatible aliases (used by cli.py and tests)
EDGE_DIR_PROXIMITY = EdgeType.DIR_PROXIMITY
EDGE_PRIV_REACH    = EdgeType.PRIV_REACH
EDGE_SA_LATERAL    = EdgeType.SA_LATERAL
EDGE_SEMANTIC_NS   = EdgeType.SEMANTIC_NS
EDGE_RBAC_PRIV     = EdgeType.RBAC_PRIV

ESCAPE_FLAGS: Final[frozenset[str]] = frozenset({
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET",
    "DOCKERSOCK_PATH", "CAP_SYS_ADMIN", "CAP_SYS_MODULE",
    "SEC_CONT_OVER_PRIVIL", "HOSTPATH_MOUNT",
})
LATERAL_FLAGS: Final[frozenset[str]] = frozenset({
    "SA_AUTOMOUNT_TOKEN", "USES_DEFAULT_SA",
    "WITHIN_MANIFEST_SECRET", "ALLOW_PRIVI",
})

# Type alias for the dict returned by build_cluster_graph
GraphResult: TypeAlias = dict[str, object]


# ---------------------------------------------------------------------------
# RBAC privilege detection helpers
# ---------------------------------------------------------------------------

def _is_privileged_role(rules: list[object]) -> bool:
    """
    Return True if any rule in a Role/ClusterRole grants privileged access.

    A role is privileged when it has:
    - wildcard verbs (*) on any resource, OR
    - escalation verbs (escalate, bind, impersonate) on any resource.
    """
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        verbs = set(rule.get("verbs") or [])
        if verbs & _PRIVILEGE_ESCALATION_VERBS:
            return True
    return False


def _collect_privileged_roles(yaml_paths: list[Path]) -> frozenset[str]:
    """
    First pass over all manifest files: collect names of privileged roles.

    A role is privileged if its name is in _PRIVILEGED_ROLE_NAMES or if its
    rules grant privilege-escalation verbs.  ClusterRoles are globally
    privileged; Roles are treated the same way (conservative assumption).
    """
    privileged: set[str] = set()
    for path in yaml_paths:
        for doc in _safe_load_all(path):
            kind = doc.get("kind", "")
            if kind not in ("Role", "ClusterRole"):
                continue
            meta = _safe_dict(doc.get("metadata"))
            name = meta.get("name", "")
            if not name:
                continue
            if name in _PRIVILEGED_ROLE_NAMES:
                privileged.add(name)
                continue
            rules = doc.get("rules") or []
            if _is_privileged_role(rules):
                privileged.add(name)
    return frozenset(privileged)


# ---------------------------------------------------------------------------
# YAML semantic parser (namespace, SA, hostPath — for edge construction only)
# ---------------------------------------------------------------------------

def _parse_yaml_semantics(path: Path, privileged_roles: frozenset[str]) -> dict[str, object]:
    result: dict[str, object] = {
        "namespace":       None,
        "service_account": None,
        "rbac_subjects":   [],
        "hostpath_mount":  False,
    }
    for doc in _safe_load_all(path):
        kind = doc.get("kind", "")
        meta = _safe_dict(doc.get("metadata"))
        ns   = meta.get("namespace")
        if ns and result["namespace"] is None:
            result["namespace"] = ns
        if kind in WORKLOAD_KINDS:
            pod_spec = _get_pod_spec(doc) or {}
            sa = pod_spec.get("serviceAccountName")
            if sa and result["service_account"] is None:
                result["service_account"] = sa
            for vol in (pod_spec.get("volumes") or []):
                if not isinstance(vol, dict):
                    continue
                hp = _safe_dict(vol.get("hostPath"))
                if hp:
                    path_val = str(hp.get("path", ""))
                    if _DOCKER_SOCK_PATH not in path_val:
                        result["hostpath_mount"] = True
        if kind in ("RoleBinding", "ClusterRoleBinding"):
            role_ref = _safe_dict(doc.get("roleRef")).get("name", "")
            # Only emit RBAC subjects when the binding targets a privileged role
            if role_ref in privileged_roles:
                subjects: list[str] = []
                for s in (doc.get("subjects") or []):
                    if isinstance(s, dict) and s.get("kind") == "ServiceAccount":
                        name = s.get("name")
                        if name:
                            subjects.append(name)
                existing = result.get("rbac_subjects") or []
                result["rbac_subjects"] = list(existing) + subjects
    return result


def _dir_key(yaml_path: str, depth: int = DEFAULT_DIR_KEY_DEPTH) -> str:
    parts = Path(yaml_path).parts
    if len(parts) >= depth + 1:
        return "/".join(parts[-(depth + 1):-1])
    return str(Path(yaml_path).parent)


# ---------------------------------------------------------------------------
# Graph construction helpers (one edge type each)
# ---------------------------------------------------------------------------

def _add_undirected_edges(
    G:         nx.DiGraph,
    members:   list[int],
    edge_type: EdgeType,
) -> None:
    """Add bidirectional edges between all pairs in members."""
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            G.add_edge(a, b, edge_type=edge_type)
            G.add_edge(b, a, edge_type=edge_type)


def _build_nodes(
    feats_list:  list[dict[str, object]],
    risk_scores: list[float],
    yaml_paths:  list[Path],
) -> tuple[nx.DiGraph, list[dict[str, object]]]:
    """Create graph nodes from feature dicts and risk scores."""
    G:         nx.DiGraph           = nx.DiGraph()
    node_data: list[dict[str, object]] = []

    for idx, (feats, risk, path) in enumerate(zip(feats_list, risk_scores, yaml_paths, strict=True)):
        vec   = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)
        flags: dict[str, int] = {}
        for i, col in enumerate(FEATURE_COLS):
            val      = float(feats.get(col, 0) or 0)
            vec[i]   = val
            flags[col] = int(val)
        vec[RISK_SCORE_INDEX] = float(risk)

        G.add_node(idx, features=vec, yaml_path=str(path), **flags)
        node_data.append({
            **flags,
            "label":      1 if any(flags.values()) else 0,
            "risk_score": risk,
            "yaml_path":  str(path),
            "file_name":  path.name,
        })

    return G, node_data


def _enrich_with_yaml_semantics(
    G:         nx.DiGraph,
    node_data: list[dict[str, object]],
    yaml_paths: list[Path],
) -> tuple[dict[str, list[int]], dict[int, str], set[str]]:
    """
    Parse each manifest for namespace / SA / RBAC / hostPath semantics.

    Performs a first pass over all files to identify privileged roles, then
    a second pass to collect namespace groups, SA assignments, and elevated SAs.
    Only SAs bound to genuinely privileged roles are added to elevated_sas.

    Returns (ns_groups, pod_sa, elevated_sas) for edge construction.
    """
    privileged_roles = _collect_privileged_roles(yaml_paths)
    logger.debug("Privileged roles detected: %s", privileged_roles or "(none)")

    ns_groups:    dict[str, list[int]] = {}
    pod_sa:       dict[int, str]       = {}
    elevated_sas: set[str]             = set()

    for idx, path in enumerate(yaml_paths):
        sem = _parse_yaml_semantics(path, privileged_roles)
        ns  = str(sem.get("namespace") or DEFAULT_NAMESPACE)
        ns_groups.setdefault(ns, []).append(idx)

        if sem.get("hostpath_mount"):
            node_data[idx]["HOSTPATH_MOUNT"] = 1
            G.nodes[idx]["HOSTPATH_MOUNT"]   = 1

        sa = sem.get("service_account")
        if sa:
            pod_sa[idx] = str(sa)

        for subj in (sem.get("rbac_subjects") or []):
            if subj:
                elevated_sas.add(str(subj))

    return ns_groups, pod_sa, elevated_sas


def _add_proximity_edges(
    G:         nx.DiGraph,
    node_data: list[dict[str, object]],
) -> None:
    dir_groups: dict[str, list[int]] = {}
    for idx, nd in enumerate(node_data):
        dk = _dir_key(str(nd["yaml_path"]))
        dir_groups.setdefault(dk, []).append(idx)
    for members in dir_groups.values():
        _add_undirected_edges(G, members, EdgeType.DIR_PROXIMITY)


def _add_privilege_edges(
    G:            nx.DiGraph,
    escape_nodes: list[int],
    n:            int,
) -> None:
    for src in escape_nodes:
        for dst in range(n):
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type=EdgeType.PRIV_REACH)


def _add_lateral_edges(
    G:        nx.DiGraph,
    sa_nodes: list[int],
    n:        int,
) -> None:
    for src in sa_nodes:
        for dst in range(n):
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type=EdgeType.SA_LATERAL)


def _add_semantic_edges(
    G:         nx.DiGraph,
    ns_groups: dict[str, list[int]],
) -> None:
    for members in ns_groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if not G.has_edge(a, b):
                    G.add_edge(a, b, edge_type=EdgeType.SEMANTIC_NS)
                if not G.has_edge(b, a):
                    G.add_edge(b, a, edge_type=EdgeType.SEMANTIC_NS)


def _add_rbac_edges(
    G:            nx.DiGraph,
    pod_sa:       dict[int, str],
    elevated_sas: set[str],
    n:            int,
) -> None:
    if not elevated_sas:
        return
    rbac_nodes = [idx for idx, sa in pod_sa.items() if sa in elevated_sas]
    for src in rbac_nodes:
        for dst in range(n):
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type=EdgeType.RBAC_PRIV)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_cluster_graph(
    feats_list:  list[dict[str, object]],
    risk_scores: list[float],
    yaml_paths:  list[Path],
) -> GraphResult:
    """
    Build a NetworkX DiGraph for one cluster.

    Parameters
    ----------
    feats_list  : per-manifest feature dicts from yaml_parser
    risk_scores : per-manifest RF risk scores (float in [0, 1])
    yaml_paths  : actual file paths (used for YAML semantic parsing)

    Returns
    -------
    GraphResult dict with keys: graph, node_data, escape_nodes, sa_nodes
    """
    n = len(feats_list)

    G, node_data = _build_nodes(feats_list, risk_scores, yaml_paths)
    ns_groups, pod_sa, elevated_sas = _enrich_with_yaml_semantics(G, node_data, yaml_paths)

    escape_nodes = [i for i, nd in enumerate(node_data) if any(nd.get(f, 0) for f in ESCAPE_FLAGS)]
    sa_nodes     = [i for i, nd in enumerate(node_data) if any(nd.get(f, 0) for f in LATERAL_FLAGS)]

    _add_proximity_edges(G, node_data)
    _add_privilege_edges(G, escape_nodes, n)
    _add_lateral_edges(G, sa_nodes, n)
    _add_semantic_edges(G, ns_groups)
    _add_rbac_edges(G, pod_sa, elevated_sas, n)

    logger.debug(
        "Built cluster graph: %d nodes, %d edges, %d escape nodes",
        G.number_of_nodes(), G.number_of_edges(), len(escape_nodes),
    )
    return {
        "graph":        G,
        "node_data":    node_data,
        "escape_nodes": escape_nodes,
        "sa_nodes":     sa_nodes,
    }


def graph_to_pyg(graph_result: GraphResult) -> Data:
    """Convert build_cluster_graph() output to a PyG Data object."""
    G         = graph_result["graph"]
    node_data = graph_result["node_data"]
    n         = len(node_data)

    x = np.stack([G.nodes[i]["features"] for i in range(n)]).astype(np.float32)

    edges = list(G.edges(data=True))
    if edges:
        src        = np.array([e[0] for e in edges], dtype=np.int64)
        dst        = np.array([e[1] for e in edges], dtype=np.int64)
        edge_index = np.stack([src, dst])
        edge_attr  = np.array(
            [e[2].get("edge_type", 0) for e in edges], dtype=np.int64
        ).reshape(-1, 1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, 1), dtype=np.int64)

    return Data(
        x          = torch.FloatTensor(x),
        edge_index = torch.LongTensor(edge_index),
        edge_attr  = torch.LongTensor(edge_attr),
        y          = torch.LongTensor([0]),
        batch      = torch.zeros(n, dtype=torch.long),
    )
