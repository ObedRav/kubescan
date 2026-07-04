"""
predict.py
==========
End-to-end inference pipeline (Fix 1).

Takes a directory of Kubernetes YAML manifests, runs the full 3-layer ensemble,
and returns a ranked risk report — chain probability, escape indicators,
and per-manifest risk scores — suitable for DevOps consumption.

Pipeline:
  1. YAML → feature extraction  (yaml_feature_extractor)
  2. Features → RF risk_score   (rf_model.pkl, per manifest)
  3. Manifests + risk scores → cluster graph  (graph_builder logic)
  4. Graph → GNN chain prob     (gnn_fold_0..4 ensemble, reduces variance)
  5. Ensemble score             (ga_weights.json: w_rf + w_gnn + w_escape)
  6. Report (JSON or text)

Usage:
  python models/predict.py --cluster-dir /path/to/yamls
  python models/predict.py --cluster-dir /path/to/yamls --format json
  python models/predict.py --cluster-dir /path/to/yamls --cluster-name my-cluster
  python models/predict.py --cluster-dir /path/to/yamls --show-nodes
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "02_extract"))

try:
    from kubescan.model.ga_ensemble import (
        ESCAPE_FLAG_INDICES,
        SCORE_HIGH_THRESHOLD,
        SCORE_MODERATE_THRESHOLD,
        run_gnn_ensemble,
    )
    from kubescan.model.gat_encoder import load_fold_ensemble
    from kubescan.model.rf_classifier import RFClassifier
    from kubescan.utils.device_utils import resolve_device
    from kubescan.utils.graph_builder import graph_to_pyg
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT.parent / "kubescan" / "src"))
    from kubescan.model.ga_ensemble import (
        ESCAPE_FLAG_INDICES,
        SCORE_HIGH_THRESHOLD,
        SCORE_MODERATE_THRESHOLD,
        run_gnn_ensemble,
    )
    from kubescan.model.gat_encoder import load_fold_ensemble
    from kubescan.model.rf_classifier import RFClassifier
    from kubescan.utils.device_utils import resolve_device
    from kubescan.utils.graph_builder import graph_to_pyg

try:
    from extract_yaml_features import extract_features_from_dir
except ImportError as e:
    sys.exit(f"Cannot import extract_yaml_features: {e}")

try:
    import networkx as nx
    from build_graphs import (
        ALL_FEATURE_COLS,
        EDGE_DIR_PROXIMITY,
        EDGE_PRIV_REACH,
        EDGE_RBAC_PRIV,
        EDGE_SA_LATERAL,
        EDGE_SEMANTIC_NS,
        ESCAPE_FLAGS,
        LATERAL_FLAGS,
        dir_key,
        parse_yaml_semantics,
    )
except ImportError as e:
    sys.exit(f"Cannot import graph_builder: {e}")


LABEL_NAMES = {0: "CLEAN", 1: "ISOLATED_MISCONFIG", 2: "ATTACK_CHAIN"}

# Fallback ensemble weights used when ga_weights.json is absent
_DEFAULT_W_RF:     float = 0.36
_DEFAULT_W_GNN:    float = 0.64
_DEFAULT_W_ESCAPE: float = 0.0
RISK_EMOJI  = {0: "✓", 1: "⚠", 2: "✗"}  # for text report


# ---------------------------------------------------------------------------
# Step 1: Extract YAML features
# ---------------------------------------------------------------------------

def extract_cluster_features(cluster_dir: Path) -> list[dict]:
    """Extract per-manifest features from all YAML files in cluster_dir."""
    results = extract_features_from_dir(cluster_dir)
    if not results:
        sys.exit(
            f"No workload resources found in {cluster_dir}.\n"
            "Ensure the directory contains Kubernetes manifest YAML files.",
        )
    return results


# ---------------------------------------------------------------------------
# Step 2: RF inference — risk_score per manifest (via RFClassifier)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 3: Build in-memory cluster graph
# ---------------------------------------------------------------------------

def _build_node_feature_vector(feats: dict, risk_score: float) -> np.ndarray:
    """
    Build 26-dim node feature vector matching graph_builder's NODE_FEATURE_DIM=26.
    Indices 0-24: ALL_FEATURE_COLS (18 Rahman + 7 Extended)
    Index 25: risk_score
    """
    vec = np.zeros(26, dtype=np.float32)
    for i, col in enumerate(ALL_FEATURE_COLS):  # 25 features
        vec[i] = float(feats.get(col, 0) or 0)
    vec[25] = float(risk_score)
    return vec


def build_graph(
    feats_list: list[dict],
    risk_scores: list[float],
    yaml_paths: list[Path],
) -> dict:
    """
    Build a NetworkX graph for the cluster from extracted features.
    Mirrors the logic in graph_builder.build_cluster_graph() but works
    entirely from local file paths without the CSV/URL infrastructure.
    """
    n = len(feats_list)
    G = nx.DiGraph()

    # Build node data
    node_data = []
    for idx, (feats, risk, path) in enumerate(zip(feats_list, risk_scores, yaml_paths)):
        feat_vec = _build_node_feature_vector(feats, risk)
        flags    = {col: int(float(feats.get(col, 0) or 0)) for col in ALL_FEATURE_COLS}
        label    = 1 if any(flags.values()) else 0

        G.add_node(idx, features=feat_vec, yaml_path=str(path), **flags)
        node_data.append({
            **flags,
            "label":      label,
            "risk_score": risk,
            "yaml_path":  str(path),
            "has_yaml":   1,
            "file_name":  path.name,
        })

    # YAML semantic enrichment (namespace, SA, HOSTPATH_MOUNT from YAML parsing)
    ns_groups  = {}
    pod_sa     = {}
    elevated_sas = set()

    for idx, path in enumerate(yaml_paths):
        sem = parse_yaml_semantics(str(path))
        ns  = sem.get("namespace") or "_default"
        ns_groups.setdefault(ns, []).append(idx)

        # YAML-detected hostPath overrides extractor (more reliable)
        if sem.get("hostpath_mount"):
            node_data[idx]["HOSTPATH_MOUNT"] = 1
            G.nodes[idx]["HOSTPATH_MOUNT"]   = 1

        sa = sem.get("service_account")
        if sa:
            pod_sa[idx] = sa
        for subj in (sem.get("rbac_subjects") or []):
            if subj:
                elevated_sas.add(subj)

    # Edge type 0: directory proximity
    dir_groups = {}
    for idx, nd in enumerate(node_data):
        dk = dir_key(nd["yaml_path"])
        dir_groups.setdefault(dk, []).append(idx)

    for members in dir_groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                G.add_edge(a, b, edge_type=EDGE_DIR_PROXIMITY)
                G.add_edge(b, a, edge_type=EDGE_DIR_PROXIMITY)

    # Edge type 1: privilege-reach (escape node → all others)
    escape_nodes = [
        idx for idx, nd in enumerate(node_data)
        if any(nd.get(f, 0) for f in ESCAPE_FLAGS)
    ]
    # PRIV_REACH always overwrites lower-priority edge types (e.g. DIR_PROXIMITY)
    # so that escape-capable nodes are never silently downgraded to mere neighbours.
    # Mirrors graph_builder._add_privilege_edges — no has_edge guard.
    for src in escape_nodes:
        for dst in range(n):
            if dst != src:
                G.add_edge(src, dst, edge_type=EDGE_PRIV_REACH)

    # Edge type 2: SA lateral movement
    sa_nodes = [
        idx for idx, nd in enumerate(node_data)
        if any(nd.get(f, 0) for f in LATERAL_FLAGS)
    ]
    for src in sa_nodes:
        for dst in range(n):
            if dst != src and not G.has_edge(src, dst):
                G.add_edge(src, dst, edge_type=EDGE_SA_LATERAL)

    # Edge type 3: semantic namespace
    for members in ns_groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if not G.has_edge(a, b):
                    G.add_edge(a, b, edge_type=EDGE_SEMANTIC_NS)
                if not G.has_edge(b, a):
                    G.add_edge(b, a, edge_type=EDGE_SEMANTIC_NS)

    # Edge type 4: RBAC privilege escalation
    if elevated_sas:
        rbac_nodes = [idx for idx, sa in pod_sa.items() if sa in elevated_sas]
        for src in rbac_nodes:
            for dst in range(n):
                if dst != src and not G.has_edge(src, dst):
                    G.add_edge(src, dst, edge_type=EDGE_RBAC_PRIV)

    return {"graph": G, "node_data": node_data, "escape_nodes": escape_nodes, "sa_nodes": sa_nodes}


# ---------------------------------------------------------------------------
# Step 4: GNN fold ensemble inference (run_gnn_ensemble from kubescan.model.ga_ensemble)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 5: Ensemble score
# ---------------------------------------------------------------------------

def compute_ensemble_score(
    mean_rf_risk:  float,
    chain_prob:    float,
    escape_signal: float,
    weights:       dict,
) -> float:
    w_rf     = weights.get("w_rf",     _DEFAULT_W_RF)
    w_gnn    = weights.get("w_gnn",    _DEFAULT_W_GNN)
    w_escape = weights.get("w_escape", _DEFAULT_W_ESCAPE)
    total    = w_rf + w_gnn + w_escape
    if total <= 0:
        total = 1.0
    return (w_rf * mean_rf_risk + w_gnn * chain_prob + w_escape * escape_signal) / total


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _flag_summary(nd: dict) -> list[str]:
    """Return list of security flag names that are set for this node."""
    return [col for col in ALL_FEATURE_COLS if nd.get(col, 0)]


def print_text_report(
    cluster_name:   str,
    cluster_dir:    Path,
    node_data:      list[dict],
    risk_scores:    list[float],
    yaml_paths:     list[Path],
    chain_prob:     float,
    clean_prob:     float,
    escape_fraction: float,
    mean_rf_risk:   float,
    ensemble_score: float,
    weights:        dict,
    escape_nodes:   list[int],
    sa_nodes:       list[int],
    show_nodes:     bool = False,
):
    # Cluster-level verdict — mirrors EnsembleScorer.predict_label thresholds
    if ensemble_score >= SCORE_HIGH_THRESHOLD:
        verdict = "ATTACK_CHAIN  ✗ HIGH RISK"
    elif ensemble_score >= SCORE_MODERATE_THRESHOLD:
        verdict = "ISOLATED / SUSPICIOUS  ⚠ REVIEW"
    else:
        verdict = "CLEAN  ✓ LOW RISK"

    SEP = "=" * 65
    print(f"\n{SEP}")
    print("  KUBERNETES SECURITY RISK REPORT")
    print(f"  Cluster : {cluster_name}")
    print(f"  Path    : {cluster_dir}")
    print(SEP)
    print(f"\n  VERDICT: {verdict}")
    print(f"\n  Ensemble score   : {ensemble_score:.4f}")
    print(f"  Chain probability: {chain_prob:.4f}  (GNN, {len(weights.get('mode', 'oof'))} fold ensemble)")
    print(f"  Clean probability: {clean_prob:.4f}")
    print(f"  Mean RF risk     : {mean_rf_risk:.4f}")
    print(f"  Escape fraction  : {escape_fraction:.4f}  "
          f"({len(escape_nodes)}/{len(node_data)} manifests have escape flags)")

    print(f"\n  Weights used: w_rf={weights.get('w_rf',0):.3f}  "
          f"w_gnn={weights.get('w_gnn',0):.3f}  "
          f"w_escape={weights.get('w_escape',0):.3f}")

    print(f"\n  Manifests analysed: {len(node_data)}")
    print(f"  Escape-capable   : {len(escape_nodes)}  "
          f"(TRUE_HOST_PID/IPC/NET, DOCKERSOCK, CAP_SYS_ADMIN/MODULE, SEC_CONT, HOSTPATH)")
    print(f"  Lateral-capable  : {len(sa_nodes)}  "
          f"(SA_AUTOMOUNT, USES_DEFAULT_SA, WITHIN_MANIFEST_SECRET, ALLOW_PRIVI)")

    if show_nodes:
        print(f"\n  {'Manifest':<45} {'Risk':>5}  {'Type':>8}  Flags")
        print(f"  {'-'*85}")
        for orig_idx, (nd, risk, path) in sorted(
            enumerate(zip(node_data, risk_scores, yaml_paths)),
            key=lambda item: item[1][1],
            reverse=True,
        ):
            flags  = _flag_summary(nd)
            is_esc = orig_idx in escape_nodes
            is_lat = orig_idx in sa_nodes
            tag    = ("ESC" if is_esc else "") + ("LAT" if is_lat else "")
            flag_str = ", ".join(flags[:5]) + ("…" if len(flags) > 5 else "")
            print(f"  {path.name:<45} {risk:>5.3f}  {tag:>8}  {flag_str}")

    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoints = PROJECT_ROOT / "models" / "checkpoints"
    default_weights = checkpoints / "ga_weights.json"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cluster-dir",  type=Path, required=True,
                        help="Directory containing Kubernetes YAML manifests")
    parser.add_argument("--cluster-name", type=str,  default=None,
                        help="Human-readable cluster name (defaults to directory name)")
    parser.add_argument("--weights",      type=Path, default=default_weights,
                        help="Path to ga_weights.json")
    parser.add_argument("--checkpoints-dir", type=Path, default=checkpoints,
                        help="Directory containing rf_model.skops and gnn_fold_*.pt")
    parser.add_argument("--format",       choices=["text", "json"], default="text")
    parser.add_argument("--show-nodes",   action="store_true",
                        help="Show per-manifest risk breakdown in text report")
    args = parser.parse_args()

    cluster_dir  = args.cluster_dir.resolve()
    cluster_name = args.cluster_name or cluster_dir.name
    device       = resolve_device()
    ckpt_dir     = args.checkpoints_dir

    if not cluster_dir.is_dir():
        sys.exit(f"Cluster directory not found: {cluster_dir}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    try:
        rf_model = RFClassifier.from_checkpoints(ckpt_dir)
    except Exception as exc:
        sys.exit(f"RF model load failed: {exc}. Run train_rf.py first.")

    if not args.weights.exists():
        sys.exit(f"Ensemble weights not found: {args.weights}. Run run_ga_ensemble.py first.")
    with args.weights.open(encoding="utf-8") as f:
        weights = json.load(f)

    try:
        fold_models = load_fold_ensemble(ckpt_dir, device=device)
    except Exception as exc:
        sys.exit(f"GNN fold models load failed: {exc}. Run train_gnn.py first.")

    # ------------------------------------------------------------------
    # Step 1: Extract features
    # ------------------------------------------------------------------
    if args.format == "text":
        print(f"Scanning {cluster_dir} ...")

    feats_list = extract_cluster_features(cluster_dir)

    # Map back to file paths (extractor returns dicts with 'yaml_path' or 'source_file')
    yaml_paths = []
    for feat_dict in feats_list:
        # yaml_feature_extractor stores the source file path in the dict
        p = feat_dict.get("yaml_path") or feat_dict.get("source_file") or ""
        yaml_paths.append(Path(p) if p else cluster_dir)

    if args.format == "text":
        print(f"  Found {len(feats_list)} manifest(s) with workload resources")

    # ------------------------------------------------------------------
    # Step 2: RF inference → risk_score (RFClassifier handles derived features)
    # ------------------------------------------------------------------
    risk_scores = rf_model.predict_risk_scores(feats_list)

    # ------------------------------------------------------------------
    # Step 3: Build graph
    # ------------------------------------------------------------------
    graph_result = build_graph(feats_list, risk_scores, yaml_paths)
    node_data    = graph_result["node_data"]
    escape_nodes = graph_result["escape_nodes"]
    sa_nodes     = graph_result["sa_nodes"]

    # ------------------------------------------------------------------
    # Step 4: GNN ensemble
    # ------------------------------------------------------------------
    pyg_data = graph_to_pyg(graph_result)
    chain_prob, clean_prob, _isolated_prob = run_gnn_ensemble(pyg_data, fold_models, device)

    # ------------------------------------------------------------------
    # Step 5: Ensemble score
    # ------------------------------------------------------------------
    mean_rf_risk = float(np.mean(risk_scores))

    escape_flags_matrix = np.array([
        [float(nd.get(ALL_FEATURE_COLS[i], 0)) for i in ESCAPE_FLAG_INDICES]
        for nd in node_data
    ]) if node_data else np.zeros((0, len(ESCAPE_FLAG_INDICES)))
    escape_fraction = float((escape_flags_matrix.max(axis=1) > 0).mean()) if node_data else 0.0
    escape_signal   = 1.0 if node_data and (escape_flags_matrix.max(axis=1) > 0).any() else 0.0

    ensemble_score = compute_ensemble_score(mean_rf_risk, chain_prob, escape_signal, weights)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if args.format == "json":
        output = {
            "cluster":          cluster_name,
            "cluster_dir":      str(cluster_dir),
            "verdict":          LABEL_NAMES[
                2 if ensemble_score >= SCORE_HIGH_THRESHOLD
                else (1 if ensemble_score >= SCORE_MODERATE_THRESHOLD else 0)
            ],
            "ensemble_score":   round(ensemble_score, 6),
            "chain_probability": round(chain_prob, 6),
            "clean_probability": round(clean_prob, 6),
            "mean_rf_risk":     round(mean_rf_risk, 6),
            "escape_fraction":  round(escape_fraction, 6),
            "n_manifests":      len(feats_list),
            "n_escape_capable": len(escape_nodes),
            "n_lateral_capable": len(sa_nodes),
            "weights":          {
                k: weights[k] for k in ("w_rf", "w_gnn", "w_escape") if k in weights
            },
            "manifests": [
                {
                    "file":           node_data[i]["file_name"],
                    "risk_score":     round(risk_scores[i], 6),
                    "escape_capable": i in escape_nodes,
                    "lateral_capable": i in sa_nodes,
                    "flags":          _flag_summary(node_data[i]),
                }
                for i in sorted(range(len(node_data)), key=lambda x: risk_scores[x], reverse=True)
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_text_report(
            cluster_name=cluster_name,
            cluster_dir=cluster_dir,
            node_data=node_data,
            risk_scores=risk_scores,
            yaml_paths=yaml_paths,
            chain_prob=chain_prob,
            clean_prob=clean_prob,
            escape_fraction=escape_fraction,
            mean_rf_risk=mean_rf_risk,
            ensemble_score=ensemble_score,
            weights=weights,
            escape_nodes=escape_nodes,
            sa_nodes=sa_nodes,
            show_nodes=args.show_nodes,
        )


if __name__ == "__main__":
    main()
