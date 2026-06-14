"""
extract_yaml_features.py
=========================
Extract security feature flags from raw Kubernetes YAML files.

Mirrors the logic from Rahman et al.'s SLI-KUBE tool (scanner.py / parser.py) so that
features from new manifests (BadPods, Kubernetes Goat, downloaded YAMLs) are consistent
with the existing rf_dataset.csv columns. Extended with 6 additional features identified
as critical gaps during dataset validation.

Features produced — 18 SLI-KUBE-compatible + 6 extended:
  TRUE_HOST_PID, TRUE_HOST_IPC, TRUE_HOST_NET, DOCKERSOCK_PATH,
  CAP_SYS_ADMIN, CAP_SYS_MODULE, WITHIN_MANIFEST_SECRET,
  SEC_CONT_OVER_PRIVIL, ALLOW_PRIVI, SECCOMP_UNCONFINED, VALID_TAINT_SECRET,
  INSECURE_HTTP, NO_SECU_CONTEXT, NO_NETWORK_POLICY, HOST_ALIAS,
  NO_DEFAULT_NSPACE, NO_RESO, NO_ROLLING_UPDATE,
  [EXTENDED]
  NO_RUN_AS_NON_ROOT    — container can run as root (CIS 6.1.1 / CKV_K8S_30)
  NO_READ_ONLY_ROOT_FS  — writable root filesystem (CKV_K8S_22)
  IMAGE_USES_LATEST     — image tag is latest or missing (CKV_K8S_39)
  SA_AUTOMOUNT_TOKEN    — serviceAccountToken auto-mounted (CKV_K8S_35)
  USES_DEFAULT_SA       — using the default service account (CKV_K8S_36)
  UNTRUSTED_REGISTRY    — image not from a known trusted registry (CKV_K8S_15)

Usage (standalone):
  from yaml_feature_extractor import extract_features_from_file, FEATURE_COLS

Usage (batch):
  results = extract_features_from_dir("/path/to/yaml/dir")
"""

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML required: pip install pyyaml") from exc


# ---------------------------------------------------------------------------
# Feature column names (canonical order — matches rf_dataset.csv)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "TRUE_HOST_PID",
    "TRUE_HOST_IPC",
    "TRUE_HOST_NET",
    "DOCKERSOCK_PATH",
    "CAP_SYS_ADMIN",
    "CAP_SYS_MODULE",
    "WITHIN_MANIFEST_SECRET",
    "SEC_CONT_OVER_PRIVIL",
    "ALLOW_PRIVI",
    "SECCOMP_UNCONFINED",
    "VALID_TAINT_SECRET",      # Complex taint analysis — always 0 (not implemented)
    "INSECURE_HTTP",
    "NO_SECU_CONTEXT",
    "NO_NETWORK_POLICY",
    "HOST_ALIAS",
    "NO_DEFAULT_NSPACE",
    "NO_RESO",
    "NO_ROLLING_UPDATE",
    # --- Extended features (gap-fill) ---
    "NO_RUN_AS_NON_ROOT",
    "NO_READ_ONLY_ROOT_FS",
    "IMAGE_USES_LATEST",
    "SA_AUTOMOUNT_TOKEN",
    "USES_DEFAULT_SA",
    "UNTRUSTED_REGISTRY",
    "HOSTPATH_MOUNT",       # non-docker-sock hostPath volume (host FS escape)
]

# Features present in the original Rahman CSV (first 18) — used for backward compat
RAHMAN_FEATURE_COLS = [
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET", "DOCKERSOCK_PATH",
    "CAP_SYS_ADMIN", "CAP_SYS_MODULE", "WITHIN_MANIFEST_SECRET", "SEC_CONT_OVER_PRIVIL",
    "ALLOW_PRIVI", "SECCOMP_UNCONFINED", "VALID_TAINT_SECRET", "INSECURE_HTTP",
    "NO_SECU_CONTEXT", "NO_NETWORK_POLICY", "HOST_ALIAS", "NO_DEFAULT_NSPACE",
    "NO_RESO", "NO_ROLLING_UPDATE",
]

# Trusted registries — images from these are considered verified
TRUSTED_REGISTRIES = {
    "gcr.io", "k8s.gcr.io", "registry.k8s.io",
    "quay.io", "docker.io", "ghcr.io",
    "mcr.microsoft.com", "public.ecr.aws",
    "registry.hub.docker.com",
}

# Workload kinds that wrap a pod spec (path to pod spec differs per kind)
_WORKLOAD_WITH_TEMPLATE = {
    "deployment", "statefulset", "daemonset", "replicaset",
    "replicationcontroller", "job",
}
_CRONJOB_KIND = "cronjob"
_POD_KIND = "pod"

# Credential-related key name patterns (from SLI-KUBE constants)
_SECRET_KEY_PATTERNS = re.compile(
    r"(password|passwd|pass|secret|token|credential|api[_\-]?key|private[_\-]?key|"
    r"access[_\-]?key|auth[_\-]?key|client[_\-]?secret|db[_\-]?pass|"
    r"redis[_\-]?pass|mysql[_\-]?pass|postgres[_\-]?pass)",
    re.IGNORECASE,
)
# Values that look like real secrets (not empty, not env var references, not placeholders)
_PLACEHOLDER_PATTERNS = re.compile(
    r"^\s*$|^\$\{|^\{\{|^<|^CHANGE_ME$|^changeme$|^todo|^tbd|^xxx|^placeholder|^dummy|^fake|^test|^your",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_values_recursively(obj: Any):
    """Yield all leaf values in a nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _get_values_recursively(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _get_values_recursively(item)
    else:
        yield obj


def _get_vals_for_key(obj: Any, target_key: str, collector: list) -> None:
    """Collect all values whose key equals target_key (recursive)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                collector.append(v)
            else:
                _get_vals_for_key(v, target_key, collector)
    elif isinstance(obj, list):
        for item in obj:
            _get_vals_for_key(item, target_key, collector)


def _load_all_docs(path: Path) -> list[dict]:
    """Load all YAML documents from a file, skipping None/invalid ones."""
    docs = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for doc in yaml.safe_load_all(f):
                if doc and isinstance(doc, dict):
                    docs.append(doc)
    except Exception:
        pass
    return docs


def _get_pod_spec(resource: dict) -> dict | None:
    """
    Navigate to the PodSpec for any supported workload kind.
    Returns None if not a workload resource.
    """
    kind = str(resource.get("kind", "")).lower()
    spec = resource.get("spec") or {}

    if kind == _POD_KIND:
        return spec

    if kind in _WORKLOAD_WITH_TEMPLATE:
        template = spec.get("template") or {}
        return template.get("spec") or {}

    if kind == _CRONJOB_KIND:
        job_template = spec.get("jobTemplate") or {}
        job_spec = job_template.get("spec") or {}
        pod_template = job_spec.get("template") or {}
        return pod_template.get("spec") or {}

    return None  # Not a workload


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_host_flags(pod_spec: dict) -> tuple[int, int, int]:
    """Returns (hostPID, hostIPC, hostNetwork) as 0/1."""
    return (
        1 if pod_spec.get("hostPID") is True else 0,
        1 if pod_spec.get("hostIPC") is True else 0,
        1 if pod_spec.get("hostNetwork") is True else 0,
    )


def _check_docker_sock(pod_spec: dict) -> int:
    """Detect docker socket volume mount."""
    volumes = pod_spec.get("volumes") or []
    for vol in volumes:
        if not isinstance(vol, dict):
            continue
        host_path = vol.get("hostPath") or {}
        path_val = str(host_path.get("path", ""))
        if "/var/run/docker.sock" in path_val or "/docker.sock" in path_val:
            return 1

    # Also check volumeMounts in containers for completeness
    for ctr in _iter_containers(pod_spec):
        mounts = ctr.get("volumeMounts") or []
        for mount in mounts:
            mp = str(mount.get("mountPath", ""))
            if "/var/run/docker.sock" in mp or "/docker.sock" in mp:
                return 1
    return 0


def _check_hostpath_mount(pod_spec: dict) -> int:
    """
    Detect any non-docker-socket hostPath volume mount.
    Mounting the host filesystem grants read/write access to host data.
    The docker socket is already captured by DOCKERSOCK_PATH — this flag
    captures all other hostPath mounts (e.g. mounting / or /etc for escape).
    CKV_K8S_28 / CIS 5.2.3
    """
    volumes = pod_spec.get("volumes") or []
    for vol in volumes:
        if not isinstance(vol, dict):
            continue
        hp = vol.get("hostPath")
        if not hp:
            continue
        path_val = str(hp.get("path", ""))
        # Skip docker socket (already captured by DOCKERSOCK_PATH)
        if "/var/run/docker.sock" in path_val or "/docker.sock" in path_val:
            continue
        # Any other hostPath mount is flagged
        return 1
    return 0


def _iter_containers(pod_spec: dict):
    """Iterate over all containers and initContainers."""
    yield from pod_spec.get("containers") or []
    yield from pod_spec.get("initContainers") or []


def _check_capabilities(pod_spec: dict) -> tuple[int, int]:
    """Returns (CAP_SYS_ADMIN, CAP_SYS_MODULE) as 0/1.
    capabilities.add: ["ALL"] grants every capability including SYS_ADMIN and SYS_MODULE.
    """
    sys_admin = sys_module = 0
    for ctr in _iter_containers(pod_spec):
        sc = ctr.get("securityContext") or {}
        caps = sc.get("capabilities") or {}
        adds = [str(c).upper() for c in (caps.get("add") or [])]
        if "SYS_ADMIN" in adds or "ALL" in adds:
            sys_admin = 1
        if "SYS_MODULE" in adds or "ALL" in adds:
            sys_module = 1
    return sys_admin, sys_module


def _check_privileged(pod_spec: dict) -> int:
    """Detect privileged: true in any container."""
    for ctr in _iter_containers(pod_spec):
        sc = ctr.get("securityContext") or {}
        if sc.get("privileged") is True:
            return 1
    return 0


def _check_allow_privi_escalation(pod_spec: dict) -> int:
    """Detect allowPrivilegeEscalation: true (explicit or implicit via privileged: true).
    In K8s, privileged: true automatically implies allowPrivilegeEscalation cannot be false.
    """
    for ctr in _iter_containers(pod_spec):
        sc = ctr.get("securityContext") or {}
        if sc.get("allowPrivilegeEscalation") is True:
            return 1
        if sc.get("privileged") is True:
            return 1  # privileged mode implies full privilege escalation rights
    return 0


def _check_seccomp_unconfined(resource: dict, pod_spec: dict) -> int:
    """Detect explicitly unconfined seccomp profile."""
    # New-style: seccompProfile.type: Unconfined
    pod_sc = pod_spec.get("securityContext") or {}
    seccomp = pod_sc.get("seccompProfile") or {}
    if str(seccomp.get("type", "")).lower() == "unconfined":
        return 1
    for ctr in _iter_containers(pod_spec):
        sc = ctr.get("securityContext") or {}
        ctr_seccomp = sc.get("seccompProfile") or {}
        if str(ctr_seccomp.get("type", "")).lower() == "unconfined":
            return 1

    # Old-style: annotation
    annotations = (resource.get("metadata") or {}).get("annotations") or {}
    for ann_key, ann_val in annotations.items():
        if "seccomp" in ann_key.lower() and "unconfined" in str(ann_val).lower():
            return 1
    return 0


def _check_no_secu_context(pod_spec: dict) -> int:
    """Flag if any container is missing a securityContext entirely."""
    for ctr in _iter_containers(pod_spec):
        if not ctr.get("securityContext"):
            return 1
    return 0


def _check_no_resources(pod_spec: dict) -> int:
    """Flag if any container is missing resource limits."""
    for ctr in _iter_containers(pod_spec):
        resources = ctr.get("resources") or {}
        if not resources.get("limits"):
            return 1
    return 0


def _check_no_rolling_update(resource: dict) -> int:
    """Flag Deployments/StatefulSets/DaemonSets without RollingUpdate strategy."""
    kind = str(resource.get("kind", "")).lower()
    if kind not in ("deployment", "statefulset", "daemonset"):
        return 0
    spec = resource.get("spec") or {}
    strategy = spec.get("strategy") or spec.get("updateStrategy") or {}
    strategy_type = str(strategy.get("type", "")).lower()
    # If no strategy defined, or type is Recreate — flag it
    if not strategy or strategy_type == "recreate":
        return 1
    return 0


def _check_host_aliases(pod_spec: dict) -> int:
    """Detect hostAliases usage."""
    return 1 if pod_spec.get("hostAliases") else 0


def _check_no_default_namespace(resource: dict) -> int:
    """Flag if namespace is missing or explicitly 'default'."""
    metadata = resource.get("metadata") or {}
    ns = metadata.get("namespace", "")
    if not ns or str(ns).lower() == "default":
        return 1
    return 0


def _check_insecure_http(resource: dict) -> int:
    """Flag any http:// (not https://) URL in spec values."""
    spec = resource.get("spec")
    if not spec:
        return 0
    for val in _get_values_recursively(spec):
        if isinstance(val, str) and re.search(r"http://", val, re.IGNORECASE):
            # Exclude localhost references (common in probes, less of a risk)
            if "localhost" not in val and "127.0.0.1" not in val:
                return 1
    return 0


def _is_plausible_secret(value: str) -> bool:
    """Return True if a value looks like a real credential (not a placeholder)."""
    if not isinstance(value, str):
        return False
    value = value.strip()
    if len(value) < 3:
        return False
    if _PLACEHOLDER_PATTERNS.search(value):
        return False
    # Env var references like $(VAR) or $VAR are not real secrets
    if value.startswith("$") or value.startswith("%("):
        return False
    return True


# ---------------------------------------------------------------------------
# Extended check functions (gap-fill features)
# ---------------------------------------------------------------------------

def _check_no_run_as_non_root(pod_spec: dict) -> int:
    """
    Flag if any container could run as root.
    Checks both pod-level and container-level securityContext.
    CIS Benchmark 5.2.6 / CKV_K8S_30 / CKV_K8S_40
    """
    pod_sc = pod_spec.get("securityContext") or {}
    pod_non_root = pod_sc.get("runAsNonRoot")
    pod_run_as_user = pod_sc.get("runAsUser")

    for ctr in _iter_containers(pod_spec):
        sc = ctr.get("securityContext") or {}
        ctr_non_root = sc.get("runAsNonRoot")
        ctr_run_as_user = sc.get("runAsUser")

        # Container is safe if explicitly runAsNonRoot=True at either level
        if ctr_non_root is True or pod_non_root is True:
            continue
        # Or if runAsUser is a non-zero UID at either level
        uid = ctr_run_as_user if ctr_run_as_user is not None else pod_run_as_user
        if uid is not None and int(uid) != 0:
            continue
        # Otherwise: potentially runs as root
        return 1
    return 0


def _check_no_read_only_root_fs(pod_spec: dict) -> int:
    """
    Flag if any container has a writable root filesystem.
    CIS Benchmark 5.2.4 / CKV_K8S_22
    """
    for ctr in _iter_containers(pod_spec):
        sc = ctr.get("securityContext") or {}
        if sc.get("readOnlyRootFilesystem") is not True:
            return 1
    return 0


def _check_image_uses_latest(pod_spec: dict) -> int:
    """
    Flag if any container image uses :latest tag or has no tag.
    CKV_K8S_39 — unpinned images introduce supply chain risk.
    """
    for ctr in _iter_containers(pod_spec):
        image = str(ctr.get("image", ""))
        if not image:
            return 1
        # No tag at all, or explicit :latest
        if ":" not in image:
            return 1
        tag = image.rsplit(":", 1)[-1]
        if tag.lower() == "latest" or not tag:
            return 1
    return 0


def _check_sa_automount_token(resource: dict, pod_spec: dict) -> int:
    """
    Flag if service account token is auto-mounted.
    Default is True in K8s — only safe if explicitly disabled.
    CIS Benchmark 5.1.6 / CKV_K8S_35
    """
    # Pod-spec level override takes precedence
    pod_automount = pod_spec.get("automountServiceAccountToken")
    if pod_automount is False:
        return 0
    # If not explicitly disabled at pod level, it follows SA default (which is True)
    return 1


def _check_uses_default_sa(pod_spec: dict) -> int:
    """
    Flag if the pod uses the default service account (no explicit SA set).
    CIS Benchmark 5.1.5 / CKV_K8S_36
    """
    sa_name = str(pod_spec.get("serviceAccountName", "")).strip()
    if not sa_name or sa_name.lower() == "default":
        return 1
    return 0


def _check_untrusted_registry(pod_spec: dict) -> int:
    """
    Flag if any container image comes from a non-standard registry.
    Images with no registry prefix use Docker Hub (trusted).
    Single-word images (e.g. 'ubuntu') also use Docker Hub.
    CKV_K8S_15
    """
    for ctr in _iter_containers(pod_spec):
        image = str(ctr.get("image", "")).split(":")[0]  # strip tag
        if not image:
            continue
        parts = image.split("/")
        # Single name (e.g. 'ubuntu') or docker.io short name — trusted
        if len(parts) == 1:
            continue
        # Check if first part is a known trusted registry domain
        registry = parts[0]
        if "." not in registry and ":" not in registry:
            # Not a domain — it's a Docker Hub username (e.g. docker.io/library/nginx)
            continue
        # It IS a domain — check if trusted
        if not any(registry == t or registry.endswith("." + t) for t in TRUSTED_REGISTRIES):
            return 1
    return 0


def _check_within_manifest_secret(resource: dict) -> int:
    """
    Detect hard-coded credentials in env vars, ConfigMap data, or Secret data.
    Conservative: only flag if key name AND value both look like real credentials.
    """
    def _scan_dict(obj: Any) -> bool:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _SECRET_KEY_PATTERNS.search(str(k)):
                    if isinstance(v, str) and _is_plausible_secret(v):
                        return True
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                val_ = item.get("value", "")
                                if _is_plausible_secret(str(val_)):
                                    return True
                if _scan_dict(v):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if _scan_dict(item):
                    return True
        return False

    return 1 if _scan_dict(resource) else 0


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_features_from_resource(resource: dict) -> dict[str, int] | None:
    """
    Extract all 18 feature flags from a single Kubernetes resource dict.
    Returns None if the resource is not a workload (e.g. Service, ConfigMap).
    """
    pod_spec = _get_pod_spec(resource)
    if pod_spec is None:
        return None  # Not a workload resource

    host_pid, host_ipc, host_net = _check_host_flags(pod_spec)
    cap_admin, cap_module = _check_capabilities(pod_spec)

    return {
        "TRUE_HOST_PID":         host_pid,
        "TRUE_HOST_IPC":         host_ipc,
        "TRUE_HOST_NET":         host_net,
        "DOCKERSOCK_PATH":       _check_docker_sock(pod_spec),
        "CAP_SYS_ADMIN":         cap_admin,
        "CAP_SYS_MODULE":        cap_module,
        "WITHIN_MANIFEST_SECRET":_check_within_manifest_secret(resource),
        "SEC_CONT_OVER_PRIVIL":  _check_privileged(pod_spec),
        "ALLOW_PRIVI":           _check_allow_privi_escalation(pod_spec),
        "SECCOMP_UNCONFINED":    _check_seccomp_unconfined(resource, pod_spec),
        "VALID_TAINT_SECRET":    0,  # Requires cross-file def-use analysis — not implemented
        "INSECURE_HTTP":         _check_insecure_http(resource),
        "NO_SECU_CONTEXT":       _check_no_secu_context(pod_spec),
        "NO_NETWORK_POLICY":     1,  # Conservative: assume absent unless explicitly provided
        "HOST_ALIAS":            _check_host_aliases(pod_spec),
        "NO_DEFAULT_NSPACE":     _check_no_default_namespace(resource),
        "NO_RESO":               _check_no_resources(pod_spec),
        "NO_ROLLING_UPDATE":     _check_no_rolling_update(resource),
        # Extended features
        "NO_RUN_AS_NON_ROOT":    _check_no_run_as_non_root(pod_spec),
        "NO_READ_ONLY_ROOT_FS":  _check_no_read_only_root_fs(pod_spec),
        "IMAGE_USES_LATEST":     _check_image_uses_latest(pod_spec),
        "SA_AUTOMOUNT_TOKEN":    _check_sa_automount_token(resource, pod_spec),
        "USES_DEFAULT_SA":       _check_uses_default_sa(pod_spec),
        "UNTRUSTED_REGISTRY":    _check_untrusted_registry(pod_spec),
        "HOSTPATH_MOUNT":        _check_hostpath_mount(pod_spec),
    }


def extract_features_from_file(yaml_path: Path, assume_network_policy: bool = False) -> dict | None:
    """
    Extract features from a YAML file that may contain multiple K8s resources.

    Strategy: merge flags across all workload resources in the file using OR
    (any resource having the flag = file has the flag). This matches the
    per-file aggregation used in SLI-KUBE for multi-document YAMLs.

    Parameters
    ----------
    yaml_path            : path to the YAML file
    assume_network_policy: if True, set NO_NETWORK_POLICY=0 (namespace has a policy)

    Returns dict with all 18 feature columns + metadata, or None if no workloads found.
    """
    docs = _load_all_docs(yaml_path)
    if not docs:
        return None

    merged: dict[str, int] = dict.fromkeys(FEATURE_COLS, 0)
    found_workload = False
    resource_kinds = []

    has_network_policy = False
    for doc in docs:
        feats = extract_features_from_resource(doc)
        if feats is not None:
            found_workload = True
            for col in FEATURE_COLS:
                merged[col] = merged[col] | feats[col]
        kind = str(doc.get("kind", "unknown"))
        resource_kinds.append(kind)
        if kind == "NetworkPolicy":
            has_network_policy = True

    if not found_workload:
        return None

    if assume_network_policy or has_network_policy:
        merged["NO_NETWORK_POLICY"] = 0

    return {
        **merged,
        "_yaml_path":       str(yaml_path),
        "_resource_kinds":  ",".join(resource_kinds),
    }


def extract_features_from_dir(
    dir_path: Path,
    assume_network_policy: bool = False,
    recurse: bool = True,
) -> list[dict]:
    """
    Batch extract features from all YAML files in a directory.
    Returns list of feature dicts (files with no workload resources are skipped).
    """
    results = []
    pattern = "**/*.yaml" if recurse else "*.yaml"
    for yaml_path in sorted(Path(dir_path).glob(pattern)):
        feats = extract_features_from_file(yaml_path, assume_network_policy)
        if feats:
            results.append(feats)
    # Also handle .yml extension
    pattern_yml = "**/*.yml" if recurse else "*.yml"
    for yaml_path in sorted(Path(dir_path).glob(pattern_yml)):
        if yaml_path.suffix == ".yml":
            feats = extract_features_from_file(yaml_path, assume_network_policy)
            if feats:
                results.append(feats)
    return results


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yaml_feature_extractor.py <yaml_file_or_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        results = extract_features_from_dir(target)
        print(f"Processed {len(results)} YAML files")
        for r in results[:5]:
            print(json.dumps(r, indent=2))
    else:
        result = extract_features_from_file(target)
        print(json.dumps(result, indent=2) if result else "No workload resources found")
