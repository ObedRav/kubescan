"""
yaml_parser.py
==============
YAML feature extraction for kubescan.

Extracts 25 binary security flags from Kubernetes workload manifests.
Returns None for non-workload resources (ConfigMap, Secret, etc.) — this is
intentional and documented behaviour, not a silent failure.
"""
from __future__ import annotations

__all__ = [
    "FEATURE_COLS",
    "TRUSTED_REGISTRIES",
    "WORKLOAD_KINDS",
    "YAML_GLOB_PATTERNS",
    "extract_cluster_features",
    "extract_features_from_file",
]

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML required: pip install pyyaml") from exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema constants — single source of truth for feature names and layout
# ---------------------------------------------------------------------------

FEATURE_COLS: Final[list[str]] = [
    # Rahman SLI-KUBE flags (18)
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET", "DOCKERSOCK_PATH",
    "CAP_SYS_ADMIN", "CAP_SYS_MODULE", "WITHIN_MANIFEST_SECRET",
    "SEC_CONT_OVER_PRIVIL", "ALLOW_PRIVI", "SECCOMP_UNCONFINED",
    "VALID_TAINT_SECRET", "INSECURE_HTTP", "NO_SECU_CONTEXT",
    "NO_NETWORK_POLICY", "HOST_ALIAS", "NO_DEFAULT_NSPACE",
    "NO_RESO", "NO_ROLLING_UPDATE",
    # Extended (7)
    "NO_RUN_AS_NON_ROOT", "NO_READ_ONLY_ROOT_FS", "IMAGE_USES_LATEST",
    "SA_AUTOMOUNT_TOKEN", "USES_DEFAULT_SA", "UNTRUSTED_REGISTRY",
    "HOSTPATH_MOUNT",
]

TRUSTED_REGISTRIES: Final[frozenset[str]] = frozenset({
    "gcr.io", "k8s.gcr.io", "registry.k8s.io", "quay.io",
    "docker.io", "ghcr.io", "mcr.microsoft.com", "public.ecr.aws",
})

WORKLOAD_KINDS: Final[frozenset[str]] = frozenset({
    "Pod", "Deployment", "DaemonSet", "StatefulSet", "ReplicaSet",
    "ReplicationController", "Job", "CronJob",
})

YAML_GLOB_PATTERNS: Final[tuple[str, ...]] = ("**/*.yaml", "**/*.yml")

# String constants for Kubernetes field names (avoids magic strings in logic)
_DOCKER_SOCK_PATH:  Final[str]            = "docker.sock"
_PROBE_KEYS:        Final[tuple[str, ...]] = ("livenessProbe", "readinessProbe")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_dict(obj: object) -> dict[str, object]:
    """Return obj if it is a dict, otherwise an empty dict."""
    return obj if isinstance(obj, dict) else {}


def _safe_load_all(path: Path) -> list[dict[str, object]]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        return [d for d in yaml.safe_load_all(raw) if isinstance(d, dict)]
    except Exception as exc:
        logger.warning("Skipping unparseable YAML file %s: %s", path, exc)
        return []


def _iter_containers(pod_spec: dict[str, object]) -> Iterator[dict[str, object]]:
    for key in ("initContainers", "containers"):
        for container in (pod_spec.get(key) or []):
            if isinstance(container, dict):
                yield container


def _image_uses_latest(image: str) -> bool:
    """
    True if the image reference is unpinned: empty, no tag, or tag ``latest``.
    Digest-pinned references (``…@sha256:…``) are never flagged.
    Registry ports (``host:5000/img``) are not mistaken for tags.
    """
    if not image:
        return True
    last_segment = image.rsplit("/", 1)[-1]
    if "@" in last_segment:
        return False
    if ":" not in last_segment:
        return True
    tag = last_segment.rsplit(":", 1)[-1]
    return tag.lower() in ("", "latest")


def _image_from_untrusted_registry(image: str) -> bool:
    """
    True if the image's registry host is not in TRUSTED_REGISTRIES.

    Docker reference semantics: the first path segment is a registry host only
    if it contains a dot or a port — otherwise it is a Docker Hub namespace
    (implicit docker.io, trusted). Matching is exact-host or subdomain
    (``eu.gcr.io`` matches ``gcr.io``); substring spoofs like
    ``gcr.io.evil.com`` are rejected.
    """
    if "/" not in image:
        return False
    first = image.split("/", 1)[0]
    if "." not in first and ":" not in first:
        return False
    registry_host = first.split(":", 1)[0]
    return not any(
        registry_host == t or registry_host.endswith("." + t)
        for t in TRUSTED_REGISTRIES
    )


def _get_pod_spec(doc: dict[str, object]) -> dict[str, object] | None:
    kind = doc.get("kind", "")
    spec = _safe_dict(doc.get("spec"))
    if kind == "Pod":
        return spec
    if kind == "CronJob":
        jt   = _safe_dict(spec.get("jobTemplate"))
        tmpl = _safe_dict(_safe_dict(jt.get("spec")).get("template"))
        return _safe_dict(tmpl.get("spec"))
    tmpl = spec.get("template")
    return _safe_dict(tmpl.get("spec")) if isinstance(tmpl, dict) else None


# ---------------------------------------------------------------------------
# Feature extraction helpers (one concern each)
# ---------------------------------------------------------------------------

def _extract_host_features(
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> None:
    """Set host-namespace flags (TRUE_HOST_PID, TRUE_HOST_IPC, TRUE_HOST_NET)."""
    if pod_spec.get("hostPID"):
        feats["TRUE_HOST_PID"] = 1
    if pod_spec.get("hostIPC"):
        feats["TRUE_HOST_IPC"] = 1
    if pod_spec.get("hostNetwork"):
        feats["TRUE_HOST_NET"] = 1


def _extract_volume_features(
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> None:
    """Set hostPath volume flags (DOCKERSOCK_PATH, HOSTPATH_MOUNT)."""
    for vol in (pod_spec.get("volumes") or []):
        if not isinstance(vol, dict):
            continue
        hp = _safe_dict(vol.get("hostPath"))
        if not hp:
            continue
        path_val = str(hp.get("path", ""))
        if _DOCKER_SOCK_PATH in path_val:
            feats["DOCKERSOCK_PATH"] = 1
        else:
            feats["HOSTPATH_MOUNT"] = 1


def _extract_container_features(
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> tuple[bool, bool, bool, bool]:
    """
    Inspect all containers and set per-container security flags.

    Returns
    -------
    (has_resources, has_security_ctx, has_run_as_root, has_writable_fs)
    """
    has_resources    = False
    has_security_ctx = False
    has_run_as_root  = False
    has_writable_fs  = False

    for ctr in _iter_containers(pod_spec):
        if ctr.get("resources"):
            has_resources = True

        sc = _safe_dict(ctr.get("securityContext"))
        if sc:
            has_security_ctx = True

        if sc.get("privileged"):
            feats["SEC_CONT_OVER_PRIVIL"] = 1
            feats["ALLOW_PRIVI"]          = 1
        if sc.get("allowPrivilegeEscalation"):
            feats["ALLOW_PRIVI"] = 1

        caps = _safe_dict(sc.get("capabilities"))
        adds = [str(c).upper() for c in (caps.get("add") or [])]
        if "SYS_ADMIN" in adds or "ALL" in adds:
            feats["CAP_SYS_ADMIN"] = 1
        if "SYS_MODULE" in adds or "ALL" in adds:
            feats["CAP_SYS_MODULE"] = 1

        if sc.get("runAsNonRoot") is False or sc.get("runAsUser") == 0:
            has_run_as_root = True
        if sc.get("readOnlyRootFilesystem") is False:
            has_writable_fs = True

        seccomp = _safe_dict(sc.get("seccompProfile"))
        if seccomp.get("type") == "Unconfined":
            feats["SECCOMP_UNCONFINED"] = 1

        if ctr.get("automountServiceAccountToken") is not False:
            if pod_spec.get("automountServiceAccountToken") is not False:
                feats["SA_AUTOMOUNT_TOKEN"] = 1

        image = str(ctr.get("image", ""))
        if _image_uses_latest(image):
            feats["IMAGE_USES_LATEST"] = 1
        if _image_from_untrusted_registry(image):
            feats["UNTRUSTED_REGISTRY"] = 1

        for probe_key in _PROBE_KEYS:
            probe = _safe_dict(ctr.get(probe_key))
            ha    = _safe_dict(probe.get("httpGet"))
            if ha.get("scheme", "").upper() == "HTTP":
                feats["INSECURE_HTTP"] = 1

        for env in (ctr.get("env") or []):
            if isinstance(env, dict) and _safe_dict(env.get("valueFrom")).get("secretKeyRef"):
                feats["WITHIN_MANIFEST_SECRET"] = 1

    return has_resources, has_security_ctx, has_run_as_root, has_writable_fs


def _extract_workload_metadata(
    doc:      dict[str, object],
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> None:
    """Set SA, namespace, and rolling-update flags from workload doc."""
    meta    = _safe_dict(doc.get("metadata"))
    ns      = str(meta.get("namespace") or "")
    sa_name = str(pod_spec.get("serviceAccountName") or "")

    if not sa_name or sa_name == "default":
        feats["USES_DEFAULT_SA"] = 1
    if not ns or ns == "default":
        feats["NO_DEFAULT_NSPACE"] = 1

    strategy = _safe_dict(_safe_dict(doc.get("spec")).get("strategy"))
    if strategy.get("type") == "Recreate":
        feats["NO_ROLLING_UPDATE"] = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_file(yaml_path: Path) -> tuple[dict[str, object] | None, bool]:
    """
    Extract features from one file.

    Returns (feats_or_None, file_declares_network_policy). The second element
    is reported even for non-workload files so extract_cluster_features can
    resolve NO_NETWORK_POLICY at cluster level — NetworkPolicies usually live
    in their own manifests, separate from the workloads they protect.
    """
    docs = _safe_load_all(yaml_path)
    if not docs:
        return None, False

    feats: dict[str, int]  = dict.fromkeys(FEATURE_COLS, 0)
    has_workload   = False
    has_net_policy = False

    for doc in docs:
        kind = str(doc.get("kind", ""))
        if kind == "NetworkPolicy":
            has_net_policy = True
            continue
        if kind not in WORKLOAD_KINDS:
            continue

        has_workload = True
        pod_spec     = _get_pod_spec(doc) or {}

        _extract_host_features(pod_spec, feats)
        _extract_volume_features(pod_spec, feats)
        _extract_workload_metadata(doc, pod_spec, feats)

        has_resources, has_security_ctx, has_run_as_root, has_writable_fs = (
            _extract_container_features(pod_spec, feats)
        )

        if not has_security_ctx:
            feats["NO_SECU_CONTEXT"] = 1
        if not has_resources:
            feats["NO_RESO"] = 1
        if has_run_as_root:
            feats["NO_RUN_AS_NON_ROOT"] = 1
        if has_writable_fs:
            feats["NO_READ_ONLY_ROOT_FS"] = 1

    if not has_workload:
        return None, has_net_policy

    feats["NO_NETWORK_POLICY"] = 0 if has_net_policy else 1

    result: dict[str, object] = {**feats, "yaml_path": str(yaml_path)}
    return result, has_net_policy


def extract_features_from_file(yaml_path: Path) -> dict[str, object] | None:
    """
    Extract security feature flags from a Kubernetes YAML file.

    Returns None if the file contains no workload resources — this is expected
    behaviour for ConfigMaps, Secrets, RBAC, etc. and is not an error.

    NO_NETWORK_POLICY here reflects only this file; when scanning a directory
    use extract_cluster_features, which resolves it across the whole cluster.
    """
    feats, _ = _extract_file(yaml_path)
    return feats


_EXTRACT_WORKERS: Final[int] = 8


def extract_cluster_features(cluster_dir: Path) -> list[dict[str, object]]:
    """
    Extract per-manifest features from all YAML/YML files in cluster_dir.
    Returns only files that contain workload resources.

    Files are parsed by a thread pool: extraction is dominated by file I/O
    (worst case on cloud-synced/evicted volumes), which releases the GIL.
    Result order is deterministic — it follows the sorted file list.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in YAML_GLOB_PATTERNS:
        for path in sorted(Path(cluster_dir).glob(pattern)):
            if path not in seen:
                seen.add(path)
                paths.append(path)

    if not paths:
        return []

    with ThreadPoolExecutor(max_workers=min(_EXTRACT_WORKERS, len(paths))) as pool:
        extracted = list(pool.map(_extract_file, paths))

    # Cluster-level NO_NETWORK_POLICY: a NetworkPolicy anywhere in the cluster
    # clears the flag for every workload (matches training-data semantics,
    # where the flag is resolved per repository, not per file).
    cluster_has_netpol = any(has_np for _, has_np in extracted)
    results = [feats for feats, _ in extracted if feats is not None]
    for feats in results:
        feats["NO_NETWORK_POLICY"] = 0 if cluster_has_netpol else 1
    return results
