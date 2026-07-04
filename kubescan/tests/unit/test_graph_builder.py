"""
test_graph_builder.py
=====================
Unit tests for kubescan/utils/graph_builder.py.
One assertion per test; name encodes condition + expected result.
"""
from __future__ import annotations

__all__: list[str] = []

from pathlib import Path

from kubescan.utils.graph_builder import build_cluster_graph, graph_to_pyg
from kubescan.utils.yaml_parser import extract_cluster_features


def test_build_cluster_graph_node_count_matches_input(cluster_dir: Path) -> None:
    feats_list  = extract_cluster_features(cluster_dir)
    risk_scores = [0.1, 0.9]
    yaml_paths  = [Path(str(f["yaml_path"])) for f in feats_list]
    result = build_cluster_graph(feats_list, risk_scores, yaml_paths)
    assert result["graph"].number_of_nodes() == 2


def test_build_cluster_graph_node_data_length_matches_input(cluster_dir: Path) -> None:
    feats_list  = extract_cluster_features(cluster_dir)
    risk_scores = [0.1, 0.9]
    yaml_paths  = [Path(str(f["yaml_path"])) for f in feats_list]
    result = build_cluster_graph(feats_list, risk_scores, yaml_paths)
    assert len(result["node_data"]) == 2


def test_build_cluster_graph_attack_manifest_produces_escape_node(cluster_dir: Path) -> None:
    feats_list  = extract_cluster_features(cluster_dir)
    risk_scores = [0.1, 0.9]
    yaml_paths  = [Path(str(f["yaml_path"])) for f in feats_list]
    result = build_cluster_graph(feats_list, risk_scores, yaml_paths)
    assert len(result["escape_nodes"]) > 0


def test_graph_to_pyg_node_feature_shape(cluster_dir: Path) -> None:
    feats_list  = extract_cluster_features(cluster_dir)
    risk_scores = [0.1, 0.9]
    yaml_paths  = [Path(str(f["yaml_path"])) for f in feats_list]
    result = build_cluster_graph(feats_list, risk_scores, yaml_paths)
    pyg = graph_to_pyg(result)
    assert pyg.x.shape == (2, 26)


def test_graph_to_pyg_edge_index_has_two_rows(cluster_dir: Path) -> None:
    feats_list  = extract_cluster_features(cluster_dir)
    risk_scores = [0.1, 0.9]
    yaml_paths  = [Path(str(f["yaml_path"])) for f in feats_list]
    result = build_cluster_graph(feats_list, risk_scores, yaml_paths)
    pyg = graph_to_pyg(result)
    assert pyg.edge_index.shape[0] == 2
