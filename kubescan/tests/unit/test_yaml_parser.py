"""
test_yaml_parser.py
===================
Unit tests for kubescan/utils/yaml_parser.py.
One assertion per test; name encodes condition + expected result.
"""
from __future__ import annotations

__all__: list[str] = []

from pathlib import Path

from kubescan.utils.yaml_parser import extract_cluster_features, extract_features_from_file

# ---------------------------------------------------------------------------
# Clean manifest — all escape flags should be unset
# ---------------------------------------------------------------------------

def test_extract_features_clean_host_pid_unset(clean_yaml: Path) -> None:
    feats = extract_features_from_file(clean_yaml)
    assert feats is not None
    assert feats["TRUE_HOST_PID"] == 0


def test_extract_features_clean_host_net_unset(clean_yaml: Path) -> None:
    feats = extract_features_from_file(clean_yaml)
    assert feats is not None
    assert feats["TRUE_HOST_NET"] == 0


def test_extract_features_clean_cap_sys_admin_unset(clean_yaml: Path) -> None:
    feats = extract_features_from_file(clean_yaml)
    assert feats is not None
    assert feats["CAP_SYS_ADMIN"] == 0


def test_extract_features_clean_no_privileged_override(clean_yaml: Path) -> None:
    feats = extract_features_from_file(clean_yaml)
    assert feats is not None
    assert feats["SEC_CONT_OVER_PRIVIL"] == 0


def test_extract_features_clean_pinned_image_tag(clean_yaml: Path) -> None:
    feats = extract_features_from_file(clean_yaml)
    assert feats is not None
    assert feats["IMAGE_USES_LATEST"] == 0


# ---------------------------------------------------------------------------
# Attack manifest — escape flags must be detected
# ---------------------------------------------------------------------------

def test_extract_features_attack_host_pid_set(attack_yaml: Path) -> None:
    feats = extract_features_from_file(attack_yaml)
    assert feats is not None
    assert feats["TRUE_HOST_PID"] == 1


def test_extract_features_attack_host_net_set(attack_yaml: Path) -> None:
    feats = extract_features_from_file(attack_yaml)
    assert feats is not None
    assert feats["TRUE_HOST_NET"] == 1


def test_extract_features_attack_docker_sock_set(attack_yaml: Path) -> None:
    feats = extract_features_from_file(attack_yaml)
    assert feats is not None
    assert feats["DOCKERSOCK_PATH"] == 1


def test_extract_features_attack_privileged_override_set(attack_yaml: Path) -> None:
    feats = extract_features_from_file(attack_yaml)
    assert feats is not None
    assert feats["SEC_CONT_OVER_PRIVIL"] == 1


def test_extract_features_attack_cap_sys_admin_set(attack_yaml: Path) -> None:
    feats = extract_features_from_file(attack_yaml)
    assert feats is not None
    assert feats["CAP_SYS_ADMIN"] == 1


def test_extract_features_attack_latest_image_set(attack_yaml: Path) -> None:
    feats = extract_features_from_file(attack_yaml)
    assert feats is not None
    assert feats["IMAGE_USES_LATEST"] == 1


# ---------------------------------------------------------------------------
# Image reference parsing — registry trust and tag pinning
# ---------------------------------------------------------------------------

def _pod_with_image(tmp_path: Path, image: str) -> Path:
    p = tmp_path / "pod.yaml"
    p.write_text(
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: p\nspec:\n"
        f"  containers:\n    - name: c\n      image: {image}\n"
    )
    return p


def test_registry_spoof_suffix_domain_is_untrusted(tmp_path: Path) -> None:
    feats = extract_features_from_file(_pod_with_image(tmp_path, "gcr.io.evil.com/app:1.0"))
    assert feats is not None
    assert feats["UNTRUSTED_REGISTRY"] == 1


def test_registry_trusted_subdomain_is_trusted(tmp_path: Path) -> None:
    feats = extract_features_from_file(_pod_with_image(tmp_path, "eu.gcr.io/proj/app:1.0"))
    assert feats is not None
    assert feats["UNTRUSTED_REGISTRY"] == 0


def test_registry_dockerhub_username_is_trusted(tmp_path: Path) -> None:
    feats = extract_features_from_file(_pod_with_image(tmp_path, "someuser/app:1.0"))
    assert feats is not None
    assert feats["UNTRUSTED_REGISTRY"] == 0


def test_registry_unknown_host_with_port_is_untrusted(tmp_path: Path) -> None:
    feats = extract_features_from_file(_pod_with_image(tmp_path, "reg.local:5000/app:1.0"))
    assert feats is not None
    assert feats["UNTRUSTED_REGISTRY"] == 1


def test_image_untagged_behind_port_registry_uses_latest(tmp_path: Path) -> None:
    feats = extract_features_from_file(_pod_with_image(tmp_path, "reg.local:5000/app"))
    assert feats is not None
    assert feats["IMAGE_USES_LATEST"] == 1


def test_image_digest_pinned_not_latest(tmp_path: Path) -> None:
    feats = extract_features_from_file(
        _pod_with_image(tmp_path, "gcr.io/proj/app@sha256:deadbeef")
    )
    assert feats is not None
    assert feats["IMAGE_USES_LATEST"] == 0


# ---------------------------------------------------------------------------
# Non-workload resource → None
# ---------------------------------------------------------------------------

def test_extract_features_configmap_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "configmap.yaml"
    p.write_text("apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: cm\ndata:\n  key: val\n")
    assert extract_features_from_file(p) is None


# ---------------------------------------------------------------------------
# Directory extraction
# ---------------------------------------------------------------------------

def test_extract_cluster_features_returns_both_manifests(cluster_dir: Path) -> None:
    results = extract_cluster_features(cluster_dir)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# NO_NETWORK_POLICY — resolved at cluster level
# ---------------------------------------------------------------------------

_NETPOL_MANIFEST = (
    "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\n"
    "metadata:\n  name: default-deny\nspec:\n  podSelector: {}\n"
)


def test_cluster_without_netpol_flags_all_workloads(cluster_dir: Path) -> None:
    results = extract_cluster_features(cluster_dir)
    assert all(f["NO_NETWORK_POLICY"] == 1 for f in results)


def test_cluster_with_netpol_file_clears_flag_for_all(cluster_dir: Path) -> None:
    (cluster_dir / "netpol.yaml").write_text(_NETPOL_MANIFEST)
    results = extract_cluster_features(cluster_dir)
    assert all(f["NO_NETWORK_POLICY"] == 0 for f in results)


def test_single_file_without_netpol_sets_flag(attack_yaml: Path) -> None:
    feats = extract_features_from_file(attack_yaml)
    assert feats is not None
    assert feats["NO_NETWORK_POLICY"] == 1
