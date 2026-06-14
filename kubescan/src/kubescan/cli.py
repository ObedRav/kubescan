"""
cli.py
======
kubescan CLI — `kubescan scan <path>`

Entry point for the Kubernetes attack-chain risk scanner.
Loads trained checkpoints and scores a cluster's YAML manifests.

Usage:
    kubescan scan ./my-cluster/
    kubescan scan ./configs/ --format json
    kubescan scan ./configs/ --checkpoints-dir /path/to/research/models/checkpoints
    kubescan scan ./configs/ --show-nodes
    kubescan --verbose scan ./configs/
"""
from __future__ import annotations

__all__ = ["main"]

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

import click
import numpy as np
import torch
import yaml

from .exceptions import CheckpointNotFoundError, KubescanError
from .model.ga_ensemble import (
    LABEL_NAMES,
    EnsembleScorer,
    compute_escape_fraction,
    compute_escape_signal,
    run_gnn_ensemble,
)
from .model.gat_encoder import load_fold_ensemble
from .model.rf_classifier import RFClassifier
from .utils.device_utils import resolve_device
from .utils.graph_builder import build_cluster_graph, graph_to_pyg
from .utils.yaml_parser import FEATURE_COLS, extract_cluster_features

logger = logging.getLogger(__name__)

# Default checkpoints location — relative to this package file
_PKG_DIR:      Final[Path] = Path(__file__).parent
_DEFAULT_CKPT: Final[Path] = _PKG_DIR.parent.parent.parent / "checkpoints" / "trained"
_SEP_WIDTH:    Final[int]  = 66

_VERDICT_TEXT: Final[dict[int, str]] = {
    2: "ATTACK_CHAIN          ✗  HIGH RISK — review immediately",
    1: "ISOLATED / SUSPICIOUS ⚠  MODERATE RISK — manual review advised",
    0: "CLEAN                 ✓  LOW RISK",
}


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

def _resolve_checkpoints(checkpoints_dir: Path | None) -> Path:
    """Resolve checkpoints directory: CLI arg → env var → package default."""
    if checkpoints_dir:
        return checkpoints_dir
    env = os.environ.get("KUBESCAN_CHECKPOINTS")
    if env:
        return Path(env)
    if _DEFAULT_CKPT.exists():
        return _DEFAULT_CKPT
    raise CheckpointNotFoundError(_DEFAULT_CKPT)


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def _flag_list(node: dict[str, object]) -> list[str]:
    return [col for col in FEATURE_COLS if node.get(col, 0)]


def _format_verdict_line(label: int) -> str:
    return _VERDICT_TEXT[label]


def _format_node_row(
    nd:          dict[str, object],
    risk:        float,
    path:        Path,
    orig_idx:    int,
    escape_nodes: list[int],
    sa_nodes:    list[int],
) -> str:
    is_esc   = orig_idx in escape_nodes
    is_lat   = orig_idx in sa_nodes
    tag      = ("ESC" if is_esc else "") + ("LAT" if is_lat else "")
    flags    = _flag_list(nd)
    flag_str = ", ".join(flags[:4]) + ("…" if len(flags) > 4 else "")
    return f"  {path.name:<45} {risk:>5.3f}  {tag:>8}  {flag_str}"


def _print_node_table(
    node_data:    list[dict[str, object]],
    risk_scores:  list[float],
    yaml_paths:   list[Path],
    escape_nodes: list[int],
    sa_nodes:     list[int],
) -> None:
    click.echo(f"\n  {'Manifest':<45} {'Risk':>5}  {'Type':>8}  Top flags")
    click.echo(f"  {'-' * 88}")
    path_to_orig_idx = {str(yp): i for i, yp in enumerate(yaml_paths)}
    sorted_nodes = sorted(
        zip(node_data, risk_scores, yaml_paths, strict=True),
        key=lambda t: t[1],
        reverse=True,
    )
    for nd, risk, path in sorted_nodes:
        orig_idx = path_to_orig_idx[str(path)]
        click.echo(_format_node_row(nd, risk, path, orig_idx, escape_nodes, sa_nodes))


def _print_text_report(
    cluster_name:   str,
    cluster_dir:    Path,
    node_data:      list[dict[str, object]],
    risk_scores:    list[float],
    yaml_paths:     list[Path],
    chain_prob:     float,
    clean_prob:     float,
    escape_frac:    float,
    mean_rf_risk:   float,
    ensemble_score: float,
    scorer:         EnsembleScorer,
    escape_nodes:   list[int],
    sa_nodes:       list[int],
    show_nodes:     bool,
) -> None:
    label   = scorer.predict_label(ensemble_score)
    sep     = "=" * _SEP_WIDTH
    click.echo(f"\n{sep}")
    click.echo("  KUBESCAN  ·  Attack-Chain Risk Report")
    click.echo(f"  Cluster : {cluster_name}")
    click.echo(f"  Path    : {cluster_dir}")
    click.echo(sep)
    click.echo(f"\n  VERDICT  ·  {_format_verdict_line(label)}")
    click.echo(f"\n  Ensemble score    : {ensemble_score:.4f}")
    click.echo(f"  Chain probability : {chain_prob:.4f}   (5-fold GNN ensemble)")
    click.echo(f"  Clean probability : {clean_prob:.4f}")
    click.echo(f"  Mean RF risk      : {mean_rf_risk:.4f}")
    click.echo(
        f"  Escape fraction   : {escape_frac:.4f}   "
        f"({len(escape_nodes)}/{len(node_data)} manifests have escape flags)"
    )
    click.echo(f"  Lateral fraction  : {len(sa_nodes)}/{len(node_data)} manifests have lateral flags")
    click.echo(f"\n  Weights: w_rf={scorer.w_rf:.3f}  w_gnn={scorer.w_gnn:.3f}  w_escape={scorer.w_escape:.3f}")

    if show_nodes:
        _print_node_table(node_data, risk_scores, yaml_paths, escape_nodes, sa_nodes)

    click.echo(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# kubectl live-scan helpers
# ---------------------------------------------------------------------------

_NAMESPACED_RESOURCES: Final[str] = (
    "pods,deployments,daemonsets,statefulsets,replicasets,"
    "jobs,cronjobs,roles,rolebindings"
)
_CLUSTER_RESOURCES: Final[str] = "clusterroles,clusterrolebindings"


def _expand_yaml_to_files(yaml_text: str, out_dir: Path, prefix: str) -> None:
    """
    Parse kubectl YAML output and write each individual resource to its own file.
    Handles both plain documents and `List` wrappers produced by kubectl.
    """
    try:
        docs = list(yaml.safe_load_all(yaml_text))
    except yaml.YAMLError:
        return
    counter = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        items: list[object] = (
            doc.get("items", []) if doc.get("kind") == "List" else [doc]
        )
        for item in items:
            if not isinstance(item, dict) or not item.get("kind"):
                continue
            fpath = out_dir / f"{prefix}_{counter:04d}.yaml"
            fpath.write_text(yaml.dump(item, default_flow_style=False))
            counter += 1


def _fetch_live_manifests(
    namespace:      str | None,
    all_namespaces: bool,
    tmp_dir:        Path,
) -> None:
    """
    Shell out to kubectl to fetch live cluster state and write resources to tmp_dir.
    Fetches both namespace-scoped workloads/RBAC and cluster-scoped roles.
    """
    ns_args: list[str] = (
        ["--all-namespaces"] if all_namespaces
        else (["-n", namespace] if namespace else [])
    )
    commands: list[tuple[list[str], str]] = [
        (["kubectl", "get", _NAMESPACED_RESOURCES, "-o", "yaml", *ns_args], "ns"),
        (["kubectl", "get", _CLUSTER_RESOURCES, "-o", "yaml"], "cluster"),
    ]
    for cmd, label in commands:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
        except FileNotFoundError as exc:
            raise KubescanError(
                "kubectl not found — install kubectl and ensure it is in PATH"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise KubescanError(
                f"kubectl timed out after 30 s ({' '.join(cmd[:3])}) — "
                "check cluster connectivity"
            ) from exc
        if proc.returncode != 0:
            raise KubescanError(
                f"kubectl failed ({' '.join(cmd[:3])}): {proc.stderr.strip()}"
            )
        _expand_yaml_to_files(proc.stdout, tmp_dir, label)


# ---------------------------------------------------------------------------
# Shared inference pipeline
# ---------------------------------------------------------------------------

def _run_inference_pipeline(
    cluster_dir:   Path,
    cluster_name:  str,
    rf:            RFClassifier,
    scorer:        EnsembleScorer,
    gnn_fold:      list,
    device:        torch.device,
    output_format: str,
    show_nodes:    bool,
) -> None:
    """
    Extract features, build graph, run GNN ensemble, and emit report.
    Shared by the `scan` (file-based) and `live` (kubectl-based) commands.
    """
    feats_list = extract_cluster_features(cluster_dir)
    if not feats_list:
        click.echo(
            f"No Kubernetes workload resources found in {cluster_dir}.\n"
            "Ensure the directory contains Deployment/Pod/DaemonSet/etc. manifests.",
            err=True,
        )
        sys.exit(1)

    yaml_paths = [Path(str(f.get("yaml_path", cluster_dir))) for f in feats_list]

    if output_format == "text":
        click.echo(f"  {len(feats_list)} manifest(s) with workload resources found")

    risk_scores  = rf.predict_risk_scores(feats_list)
    graph_result = build_cluster_graph(feats_list, risk_scores, yaml_paths)
    node_data    = graph_result["node_data"]
    escape_nodes = graph_result["escape_nodes"]
    sa_nodes     = graph_result["sa_nodes"]

    pyg_data                         = graph_to_pyg(graph_result)
    chain_prob, clean_prob, _iso_prob = run_gnn_ensemble(pyg_data, gnn_fold, device)

    mean_rf_risk   = float(np.mean(risk_scores))
    node_feat_vecs = [
        np.array(
            [float(nd.get(FEATURE_COLS[i], 0)) for i in range(len(FEATURE_COLS))],
            dtype=np.float32,
        )
        for nd in node_data
    ]
    escape_frac    = compute_escape_fraction(node_feat_vecs)   # for display
    escape_signal  = compute_escape_signal(node_feat_vecs)     # for scoring
    ensemble_score = scorer.score(mean_rf_risk, chain_prob, escape_signal)

    if output_format == "json":
        label  = scorer.predict_label(ensemble_score)
        result = {
            "cluster":           cluster_name,
            "cluster_dir":       str(cluster_dir),
            "verdict":           LABEL_NAMES[label],
            "ensemble_score":    round(ensemble_score, 6),
            "chain_probability": round(chain_prob, 6),
            "clean_probability": round(clean_prob, 6),
            "mean_rf_risk":      round(mean_rf_risk, 6),
            "escape_fraction":   round(escape_frac, 6),
            "n_manifests":       len(feats_list),
            "n_escape_capable":  len(escape_nodes),
            "n_lateral_capable": len(sa_nodes),
            "weights": {
                "w_rf":     round(scorer.w_rf, 4),
                "w_gnn":    round(scorer.w_gnn, 4),
                "w_escape": round(scorer.w_escape, 4),
            },
            "manifests": [
                {
                    "file":            node_data[i]["file_name"],
                    "risk_score":      round(risk_scores[i], 6),
                    "escape_capable":  i in escape_nodes,
                    "lateral_capable": i in sa_nodes,
                    "flags":           _flag_list(node_data[i]),
                }
                for i in sorted(range(len(node_data)), key=lambda x: risk_scores[x], reverse=True)
            ],
        }
        click.echo(json.dumps(result, indent=2))
    else:
        _print_text_report(
            cluster_name=cluster_name,
            cluster_dir=cluster_dir,
            node_data=node_data,
            risk_scores=risk_scores,
            yaml_paths=yaml_paths,
            chain_prob=chain_prob,
            clean_prob=clean_prob,
            escape_frac=escape_frac,
            mean_rf_risk=mean_rf_risk,
            ensemble_score=ensemble_score,
            scorer=scorer,
            escape_nodes=escape_nodes,
            sa_nodes=sa_nodes,
            show_nodes=show_nodes,
        )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.version_option(package_name="kubescan")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Kubernetes attack-chain risk scanner using GNN + Random Forest ensemble."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@main.command()
@click.argument("cluster_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--checkpoints-dir", "-c", type=click.Path(path_type=Path), default=None,
              help="Directory with trained model checkpoints")
@click.option("--cluster-name", "-n", type=str, default=None,
              help="Human-readable cluster name (defaults to directory name)")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]),
              default="text", show_default=True)
@click.option("--show-nodes", is_flag=True,
              help="Show per-manifest risk breakdown (text format only)")
def scan(
    cluster_dir:     Path,
    checkpoints_dir: Path | None,
    cluster_name:    str | None,
    output_format:   str,
    show_nodes:      bool,
) -> None:
    """
    Scan a directory of Kubernetes YAML manifests for attack-chain risk.

    CLUSTER_DIR should contain .yaml / .yml manifest files (scanned recursively).

    Example:
        kubescan scan ./my-cluster/manifests/
        kubescan scan ./configs/ --format json --checkpoints-dir ./research/models/checkpoints
    """
    cluster_dir  = cluster_dir.resolve()
    cluster_name = cluster_name or cluster_dir.name
    device       = resolve_device()

    try:
        ckpt_dir = _resolve_checkpoints(checkpoints_dir)
    except KubescanError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        rf       = RFClassifier.from_checkpoints(ckpt_dir)
        scorer   = EnsembleScorer.from_checkpoints(ckpt_dir)
        gnn_fold = load_fold_ensemble(ckpt_dir, device=device)
    except KubescanError as exc:
        raise click.ClickException(str(exc)) from exc

    if output_format == "text":
        click.echo(f"Scanning {cluster_dir} ({len(gnn_fold)} GNN fold models loaded)…")
    logger.info("Scanning %s with %d GNN folds", cluster_dir, len(gnn_fold))

    _run_inference_pipeline(
        cluster_dir=cluster_dir,
        cluster_name=cluster_name,
        rf=rf,
        scorer=scorer,
        gnn_fold=gnn_fold,
        device=device,
        output_format=output_format,
        show_nodes=show_nodes,
    )


@main.command()
@click.option("--namespace", "-n", type=str, default=None,
              help="Kubernetes namespace to scan (omit for current namespace)")
@click.option("--all-namespaces", "-A", is_flag=True,
              help="Scan all namespaces (equivalent to kubectl -A)")
@click.option("--cluster-name", type=str, default=None,
              help="Human-readable cluster name for the report")
@click.option("--checkpoints-dir", "-c", type=click.Path(path_type=Path), default=None,
              help="Directory with trained model checkpoints")
@click.option("--format", "output_format", type=click.Choice(["text", "json"]),
              default="text", show_default=True)
@click.option("--show-nodes", is_flag=True,
              help="Show per-resource risk breakdown (text format only)")
def live(
    namespace:       str | None,
    all_namespaces:  bool,
    cluster_name:    str | None,
    checkpoints_dir: Path | None,
    output_format:   str,
    show_nodes:      bool,
) -> None:
    """
    Scan a live Kubernetes cluster via kubectl for attack-chain risk.

    Fetches the current state of workloads and RBAC resources directly from the
    cluster (requires kubectl configured with appropriate permissions).

    Example:
        kubescan live -n production
        kubescan live --all-namespaces --format json
        kubescan live -n kube-system --show-nodes
    """
    device = resolve_device()

    try:
        ckpt_dir = _resolve_checkpoints(checkpoints_dir)
    except KubescanError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        rf       = RFClassifier.from_checkpoints(ckpt_dir)
        scorer   = EnsembleScorer.from_checkpoints(ckpt_dir)
        gnn_fold = load_fold_ensemble(ckpt_dir, device=device)
    except KubescanError as exc:
        raise click.ClickException(str(exc)) from exc

    ns_label     = "all namespaces" if all_namespaces else (namespace or "current namespace")
    report_name  = cluster_name or ns_label

    if output_format == "text":
        click.echo(
            f"Fetching live cluster state ({ns_label}, {len(gnn_fold)} GNN fold models)…"
        )
    logger.info("Live scan: namespace=%s all=%s", namespace, all_namespaces)

    with tempfile.TemporaryDirectory(prefix="kubescan_live_") as tmp:
        tmp_dir = Path(tmp)
        try:
            _fetch_live_manifests(namespace, all_namespaces, tmp_dir)
        except KubescanError as exc:
            raise click.ClickException(str(exc)) from exc

        _run_inference_pipeline(
            cluster_dir=tmp_dir,
            cluster_name=report_name,
            rf=rf,
            scorer=scorer,
            gnn_fold=gnn_fold,
            device=device,
            output_format=output_format,
            show_nodes=show_nodes,
        )
