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
import re
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

# Kinds that support rolling-update strategy (must match training script guard)
_ROLLING_UPDATE_KINDS: Final[frozenset[str]] = frozenset({
    "Deployment", "StatefulSet", "DaemonSet",
})

# String constants for Kubernetes field names (avoids magic strings in logic)
_DOCKER_SOCK_MAIN_PATH:      Final[str]            = "/var/run/docker.sock"
_DOCKER_SOCK_ALT_PATH:       Final[str]            = "/docker.sock"
_INSECURE_HTTP_PATTERN:      Final[str]            = "http://"
_HTTP_LOCALHOST_EXCLUSIONS:  Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1"})

# Credential detection patterns (mirror extract_yaml_features.py — single logic)
_SECRET_KEY_PATTERNS: Final[re.Pattern[str]] = re.compile(
    r"(password|passwd|pass|secret|token|credential|api[_\-]?key|private[_\-]?key|"
    r"access[_\-]?key|auth[_\-]?key|client[_\-]?secret|db[_\-]?pass|"
    r"redis[_\-]?pass|mysql[_\-]?pass|postgres[_\-]?pass)",
    re.IGNORECASE,
)
_PLACEHOLDER_PATTERNS: Final[re.Pattern[str]] = re.compile(
    r"^\s*$|^\$\{|^\{\{|^<|^CHANGE_ME$|^changeme$|^todo|^tbd|^xxx|^placeholder|^dummy|^fake|^test|^your",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_dict(obj: object) -> dict[str, object]:
    """Return obj if it is a dict, otherwise an empty dict."""
    return obj if isinstance(obj, dict) else {}


def _safe_load_all(path: Path) -> list[dict[str, object]]:
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
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


def _spec_has_insecure_http(value: object) -> bool:
    """
    Recursively scan a spec value for http:// URLs, excluding localhost references.
    Matches the recursive leaf-value scan used by the research training extractor.
    """
    if isinstance(value, str):
        lower = value.lower()
        return _INSECURE_HTTP_PATTERN in lower and not any(
            excl in lower for excl in _HTTP_LOCALHOST_EXCLUSIONS
        )
    if isinstance(value, dict):
        return any(_spec_has_insecure_http(v) for v in value.values())
    if isinstance(value, list):
        return any(_spec_has_insecure_http(item) for item in value)
    return False


# ---------------------------------------------------------------------------
# Feature extraction helpers (one concern each)
# ---------------------------------------------------------------------------

def _extract_host_features(
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> None:
    """Set host-namespace flags (TRUE_HOST_PID, TRUE_HOST_IPC, TRUE_HOST_NET, HOST_ALIAS)."""
    if pod_spec.get("hostPID") is True:
        feats["TRUE_HOST_PID"] = 1
    if pod_spec.get("hostIPC") is True:
        feats["TRUE_HOST_IPC"] = 1
    if pod_spec.get("hostNetwork") is True:
        feats["TRUE_HOST_NET"] = 1
    if pod_spec.get("hostAliases"):
        feats["HOST_ALIAS"] = 1


def _is_docker_sock(path_val: str) -> bool:
    return _DOCKER_SOCK_MAIN_PATH in path_val or _DOCKER_SOCK_ALT_PATH in path_val


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
        if _is_docker_sock(path_val):
            feats["DOCKERSOCK_PATH"] = 1
        else:
            feats["HOSTPATH_MOUNT"] = 1


def _extract_pod_security_context(
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> bool:
    """
    Check pod-level securityContext for root-user and seccomp settings.

    Returns pod_run_as_root. Container-level NO_READ_ONLY_ROOT_FS is checked
    exclusively in _extract_container_features (readOnlyRootFilesystem is not
    a valid pod-level field, so checking it at pod level always fires).
    """
    pod_sc = _safe_dict(pod_spec.get("securityContext"))
    pod_seccomp = _safe_dict(pod_sc.get("seccompProfile"))
    if pod_seccomp.get("type") == "Unconfined":
        feats["SECCOMP_UNCONFINED"] = 1
    # Pod runs as root only when neither runAsNonRoot nor a non-zero runAsUser is set.
    pod_run_as_user = pod_sc.get("runAsUser")
    pod_run_as_root = (
        pod_sc.get("runAsNonRoot") is not True
        and (pod_run_as_user is None or pod_run_as_user == 0)
    )
    return pod_run_as_root


def _extract_container_features(
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> tuple[bool, bool, bool, bool]:
    """
    Inspect all containers and set per-container security flags.

    Returns
    -------
    (any_missing_limits, any_missing_security_ctx, has_run_as_root, has_writable_fs)

    any_missing_limits       — True if ANY container lacks resource limits (matches training)
    any_missing_security_ctx — True if ANY container lacks a securityContext (matches training)
    has_run_as_root          — True if ANY container may run as root
    has_writable_fs          — True if ANY container has a writable root filesystem
    """
    any_missing_limits       = False
    any_missing_security_ctx = False
    has_run_as_root          = False
    has_writable_fs          = False

    pod_sc = _safe_dict(pod_spec.get("securityContext"))
    pod_run_as_non_root: bool = pod_sc.get("runAsNonRoot") is True
    pod_run_as_user = pod_sc.get("runAsUser")

    for ctr in _iter_containers(pod_spec):
        resources = _safe_dict(ctr.get("resources"))
        if not resources.get("limits"):
            any_missing_limits = True

        sc = _safe_dict(ctr.get("securityContext"))
        if not sc:
            any_missing_security_ctx = True

        if sc.get("privileged") is True:
            feats["SEC_CONT_OVER_PRIVIL"] = 1
            feats["ALLOW_PRIVI"]          = 1
        if sc.get("allowPrivilegeEscalation") is True:
            feats["ALLOW_PRIVI"] = 1

        caps = _safe_dict(sc.get("capabilities"))
        adds = [str(c).upper() for c in (caps.get("add") or [])]
        if "SYS_ADMIN" in adds or "ALL" in adds:
            feats["CAP_SYS_ADMIN"] = 1
        if "SYS_MODULE" in adds or "ALL" in adds:
            feats["CAP_SYS_MODULE"] = 1

        # Container may run as root only when neither the container nor the pod
        # enforces runAsNonRoot, and no non-zero runAsUser is set at either level.
        ctr_run_as_user = sc.get("runAsUser")
        uid = ctr_run_as_user if ctr_run_as_user is not None else pod_run_as_user
        if sc.get("runAsNonRoot") is not True and not pod_run_as_non_root:
            if uid is None or uid == 0:
                has_run_as_root = True
        if sc.get("readOnlyRootFilesystem") is not True:
            has_writable_fs = True

        seccomp = _safe_dict(sc.get("seccompProfile"))
        if seccomp.get("type") == "Unconfined":
            feats["SECCOMP_UNCONFINED"] = 1

        image = str(ctr.get("image", ""))
        if _image_uses_latest(image):
            feats["IMAGE_USES_LATEST"] = 1
        if _image_from_untrusted_registry(image):
            feats["UNTRUSTED_REGISTRY"] = 1

        for mount in (ctr.get("volumeMounts") or []):
            if isinstance(mount, dict) and _is_docker_sock(str(mount.get("mountPath", ""))):
                feats["DOCKERSOCK_PATH"] = 1

    return any_missing_limits, any_missing_security_ctx, has_run_as_root, has_writable_fs


def _extract_workload_metadata(
    doc:      dict[str, object],
    pod_spec: dict[str, object],
    feats:    dict[str, int],
) -> None:
    """Set SA, namespace, and rolling-update flags from workload doc."""
    meta    = _safe_dict(doc.get("metadata"))
    ns      = str(meta.get("namespace") or "")
    sa_name = str(pod_spec.get("serviceAccountName") or "")

    if not sa_name or sa_name.lower() == "default":
        feats["USES_DEFAULT_SA"] = 1
    if not ns or ns.lower() == "default":
        feats["NO_DEFAULT_NSPACE"] = 1

    # Pod-level SA automount default: True unless explicitly disabled.
    if pod_spec.get("automountServiceAccountToken") is not False:
        feats["SA_AUTOMOUNT_TOKEN"] = 1

    # Old-style seccomp annotation (pre-1.19 Kubernetes).
    annotations = _safe_dict(meta.get("annotations"))
    for ann_key, ann_val in annotations.items():
        if "seccomp" in str(ann_key).lower() and "unconfined" in str(ann_val).lower():
            feats["SECCOMP_UNCONFINED"] = 1

    # Only Deployment/StatefulSet/DaemonSet have rolling-update strategies —
    # flagging CronJob/Job/Pod would cause systematic train/inference skew.
    kind = str(doc.get("kind", ""))
    if kind in _ROLLING_UPDATE_KINDS:
        spec_dict    = _safe_dict(doc.get("spec"))
        strategy_raw = spec_dict.get("strategy")
        if strategy_raw is None:
            strategy_raw = spec_dict.get("updateStrategy")
        # Flag when strategy is absent (None), empty ({}), or Recreate type.
        if (strategy_raw is None or not strategy_raw
                or str(_safe_dict(strategy_raw).get("type", "")).lower() == "recreate"):
            feats["NO_ROLLING_UPDATE"] = 1


# ---------------------------------------------------------------------------
# Credential detection (FIN-001 — mirrors extract_yaml_features.py logic)
# ---------------------------------------------------------------------------

def _is_plausible_secret(value: object) -> bool:
    if not isinstance(value, str):
        return False
    v = value.strip()
    if len(v) < 3:
        return False
    if _PLACEHOLDER_PATTERNS.search(v):
        return False
    if v.startswith("$") or v.startswith("%("):
        return False
    return True


def _scan_resource_for_secrets(obj: object) -> bool:
    """Recursively scan a resource dict for hard-coded credentials."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _SECRET_KEY_PATTERNS.search(str(k)):
                if isinstance(v, str) and _is_plausible_secret(v):
                    return True
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            val_ = item.get("value", "")
                            if _is_plausible_secret(str(val_)):
                                return True
            if _scan_resource_for_secrets(v):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if _scan_resource_for_secrets(item):
                return True
    return False


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

        pod_run_as_root = _extract_pod_security_context(pod_spec, feats)
        any_missing_limits, any_missing_security_ctx, has_run_as_root, has_writable_fs = (
            _extract_container_features(pod_spec, feats)
        )

        if any_missing_security_ctx:
            feats["NO_SECU_CONTEXT"] = 1
        if any_missing_limits:
            feats["NO_RESO"] = 1
        if has_run_as_root or pod_run_as_root:
            feats["NO_RUN_AS_NON_ROOT"] = 1
        if has_writable_fs:
            feats["NO_READ_ONLY_ROOT_FS"] = 1
        if _spec_has_insecure_http(_safe_dict(doc.get("spec"))):
            feats["INSECURE_HTTP"] = 1
        if _scan_resource_for_secrets(doc):
            feats["WITHIN_MANIFEST_SECRET"] = 1

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
