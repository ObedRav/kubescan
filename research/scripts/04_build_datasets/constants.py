"""
constants.py
============
Shared constants for the RF dataset pipeline.

SEVERITY_WEIGHTS is the single source of truth for flag severity — used by
build_rf_dataset.py (risk_score normalisation) and enrich_rf_dataset.py
(severity_class thresholding). Keeping it here prevents the two files from
drifting apart silently.
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Severity weights
# Based on CIS Kubernetes Benchmark v1.8 + MITRE ATT&CK for Containers
# ---------------------------------------------------------------------------
SEVERITY_WEIGHTS: Final[dict[str, float]] = {
    # Critical — direct container breakout / secrets exposure
    "TRUE_HOST_PID":          3.0,   # T1611 Escape to Host via hostPID
    "TRUE_HOST_IPC":          3.0,   # T1611 Escape to Host via hostIPC
    "TRUE_HOST_NET":          3.0,   # T1611 host network namespace access
    "DOCKERSOCK_PATH":        3.0,   # T1611 Docker socket → full node control
    "CAP_SYS_ADMIN":          3.0,   # T1611 CAP_SYS_ADMIN → near-root
    "CAP_SYS_MODULE":         3.0,   # T1611 kernel module loading
    "WITHIN_MANIFEST_SECRET": 3.0,   # T1552 hard-coded credentials in manifest
    # High — privilege escalation paths
    "SEC_CONT_OVER_PRIVIL":   2.5,   # privileged: true (full node access)
    "ALLOW_PRIVI":            2.5,   # allowPrivilegeEscalation: true
    "SECCOMP_UNCONFINED":     2.0,   # unconfined seccomp → arbitrary syscalls
    "VALID_TAINT_SECRET":     2.0,   # taint-based secret exposure
    # Medium — common misconfigs with meaningful attack surface
    "INSECURE_HTTP":          1.5,   # T1040 plaintext traffic sniffing
    "NO_SECU_CONTEXT":        1.5,   # absent securityContext → defaults unsafe
    "NO_NETWORK_POLICY":      1.0,   # T1570 unrestricted lateral movement
    "HOST_ALIAS":             1.0,   # DNS spoofing risk via hostAliases
    # Low — operational / hardening gaps (real but not directly exploitable)
    "NO_DEFAULT_NSPACE":      0.5,   # default namespace leaks workloads
    "NO_RESO":                0.5,   # missing resource limits (DoS risk)
    "NO_ROLLING_UPDATE":      0.3,   # availability risk, not direct security
}

MAX_RISK: Final[float] = sum(SEVERITY_WEIGHTS.values())
