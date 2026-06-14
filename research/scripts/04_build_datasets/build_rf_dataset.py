"""
build_rf_dataset.py
====================
Build the Random Forest Layer 1 training dataset from Rahman et al. (ACM TOSEM 2023).

Sources
-------
  - GITHUB-FINAL-COUNT.csv : 1,590 GitHub manifests × 18 misconfiguration type counts
  - GITLAB-FINAL-COUNT.csv :   449 GitLab  manifests × 18 misconfiguration type counts
  - GITHUB_METRICS.csv     : extra metadata (INSECURE label, SIZE, AGE, COMMITS, DEVS)
  - GITLAB_METRICS.csv     : same for GitLab

Output
------
  dataset/tabular/rf_dataset.csv
    Each row = one Kubernetes manifest
    Columns:
      - manifest_id       : unique identifier
      - source            : "github" | "gitlab"
      - repo_name         : extracted cluster/repo identifier (used as cluster_id for GNN)
      - yaml_path         : original local path (for traceability)
      - <18 binary feature columns>  : binarised misconfiguration flags (0/1)
      - total_misconfigs  : count of distinct misconfiguration types present
      - cap_misuse        : CAP_SYS_ADMIN + CAP_SYS_MODULE (derived)
      - all_secrets       : WITHIN_MANIFEST_SECRET + VALID_TAINT_SECRET (derived)
      - risk_score        : severity-weighted score in [0, 1]
      - label             : binary target — 0 = Secure, 1 = Misconfigured
                            sourced from INSECURE column in *_METRICS.csv
      - size_bytes        : file size (from METRICS, where joined)
      - age_days          : manifest age in days (from METRICS)
      - commits           : commit count (from METRICS)
      - devs              : developer count (from METRICS)
      - is_deployable     : DEPLOY flag from METRICS

Usage
-----
  python scripts/build_rf_dataset.py

  Optional flags:
    --data-dir    Path to original-dataset/rahman/DATASET/  (default: auto-detected)
    --out-dir     Output directory                          (default: dataset/tabular/)
    --no-filter   Keep Helm charts and non-K8s manifests   (default: filter them out)
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Severity weights for risk_score computation
# Based on CIS Kubernetes Benchmark v1.8 + MITRE ATT&CK for Containers severity
# ---------------------------------------------------------------------------
SEVERITY_WEIGHTS: dict[str, float] = {
    # Critical — direct container breakout / secrets exposure
    "TRUE_HOST_PID":         3.0,  # T1611 Escape to Host via hostPID
    "TRUE_HOST_IPC":         3.0,  # T1611 Escape to Host via hostIPC
    "TRUE_HOST_NET":         3.0,  # T1611 host network namespace access
    "DOCKERSOCK_PATH":       3.0,  # T1611 Docker socket mount → full node control
    "CAP_SYS_ADMIN":         3.0,  # T1611 CAP_SYS_ADMIN → near-root privileges
    "CAP_SYS_MODULE":        3.0,  # T1611 kernel module loading
    "WITHIN_MANIFEST_SECRET":3.0,  # T1552 hard-coded credentials in manifest
    # High — privilege escalation paths
    "SEC_CONT_OVER_PRIVIL":  2.5,  # privileged: true  (full node access)
    "ALLOW_PRIVI":           2.5,  # allowPrivilegeEscalation: true
    "SECCOMP_UNCONFINED":    2.0,  # unconfined seccomp → arbitrary syscalls
    "VALID_TAINT_SECRET":    2.0,  # taint-based secret exposure
    # Medium — common misconfigs with meaningful attack surface
    "INSECURE_HTTP":         1.5,  # T1040 plaintext traffic sniffing
    "NO_SECU_CONTEXT":       1.5,  # absent securityContext → defaults unsafe
    "NO_NETWORK_POLICY":     1.0,  # T1570 unrestricted lateral movement
    "HOST_ALIAS":            1.0,  # DNS spoofing risk via hostAliases
    # Low — operational/hardening gaps (real but not directly exploitable)
    "NO_DEFAULT_NSPACE":     0.5,  # default namespace leaks workloads
    "NO_RESO":               0.5,  # missing resource limits (DoS risk)
    "NO_ROLLING_UPDATE":     0.3,  # availability risk, not direct security
}

# Maximum possible weighted sum (all flags set = 1) — used for normalisation
MAX_RISK = sum(SEVERITY_WEIGHTS.values())

# Misconfiguration columns in the order they appear in the source CSVs
MISC_COLS = list(SEVERITY_WEIGHTS.keys())

# Columns copied from *_METRICS.csv
METRICS_COLS = ["INSECURE", "DEPLOY", "SIZE", "AGE", "COMMITS", "DEVS", "MINORS"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_repo_name(yaml_path: str, source: str) -> str:
    """
    Extract a stable repo identifier from the absolute local path.

    GitHub paths: .../GITHUB_REPOS/<repo_dir>/...  or  .../GITHUB_REPOS_NODEPLOY/<repo_dir>/...
    GitLab paths: .../GITLAB_K8S_REPOS_RAW_UNFILTERED/<repo_dir>/...
    """
    patterns = [
        r"GITHUB_REPOS(?:_NODEPLOY)?/([^/]+)/",
        r"GITLAB_K8S_REPOS_RAW_UNFILTERED/([^/]+)/",
        r"K8S_REPOS/[^/]+/([^/]+)/",
    ]
    for pat in patterns:
        m = re.search(pat, yaml_path)
        if m:
            return m.group(1)
    # Fallback: second-to-last directory
    parts = Path(yaml_path).parts
    return parts[-2] if len(parts) >= 2 else "unknown"


def binarize(value: str) -> int:
    """Convert a count string to binary (0 or 1)."""
    try:
        return 1 if int(value) > 0 else 0
    except (ValueError, TypeError):
        return 0


def compute_risk_score(row_bin: dict[str, int]) -> float:
    """
    Compute severity-weighted risk score in [0, 1].
    Uses binarised flags (presence/absence) multiplied by severity weights.
    """
    weighted_sum = sum(
        SEVERITY_WEIGHTS[col] * row_bin[col]
        for col in MISC_COLS
        if col in row_bin
    )
    return round(min(1.0, weighted_sum / MAX_RISK), 4)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_metrics_lookup(path: Path) -> dict[str, dict]:
    """Return {YAML_FULL_PATH: metrics_row} for fast join."""
    rows = load_csv(path)
    return {r["YAML_FULL_PATH"]: r for r in rows}


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir: Path,
    out_dir: Path,
    filter_non_k8s: bool = True,
) -> Path:

    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    manifest_id = 0

    sources = [
        ("github", data_dir / "GITHUB-FINAL-COUNT.csv", data_dir / "GITHUB_METRICS.csv"),
        ("gitlab", data_dir / "GITLAB-FINAL-COUNT.csv", data_dir / "GITLAB_METRICS.csv"),
    ]

    stats: dict[str, dict] = {}

    for source, count_path, metrics_path in sources:
        print(f"\n[{source.upper()}] Loading {count_path.name}...")
        count_rows = load_csv(count_path)
        metrics_lut = load_metrics_lookup(metrics_path)

        total = len(count_rows)
        skipped_non_k8s = 0
        skipped_helm = 0
        missing_metrics = 0

        for row in count_rows:
            yaml_path = row["YAML_FULL_PATH"]

            # Optionally filter out non-Kubernetes manifests and Helm templates
            if filter_non_k8s:
                if row.get("K8S_STATUS", "True") != "True":
                    skipped_non_k8s += 1
                    continue
                if row.get("HELM_STATUS", "False") == "True":
                    skipped_helm += 1
                    continue

            # Binarise all misconfiguration columns
            bin_flags = {col: binarize(row.get(col, "0")) for col in MISC_COLS}

            # Derived aggregate features
            cap_misuse = bin_flags.get("CAP_SYS_ADMIN", 0) | bin_flags.get("CAP_SYS_MODULE", 0)
            all_secrets = bin_flags.get("WITHIN_MANIFEST_SECRET", 0) | bin_flags.get("VALID_TAINT_SECRET", 0)
            total_misconfigs = sum(bin_flags.values())
            risk_score = compute_risk_score(bin_flags)

            # Join with METRICS for label and extra features
            metrics = metrics_lut.get(yaml_path, {})
            if not metrics:
                missing_metrics += 1
                label = 1 if total_misconfigs > 0 else 0  # fallback label
                size_bytes = age_days = commits = devs = is_deployable = ""
                is_minor = ""
            else:
                label = int(metrics.get("INSECURE", "0"))
                size_bytes = metrics.get("SIZE", "")
                age_days = metrics.get("AGE", "")
                commits = metrics.get("COMMITS", "")
                devs = metrics.get("DEVS", "")
                is_deployable = metrics.get("DEPLOY", "")
                is_minor = metrics.get("MINORS", "")

            repo_name = extract_repo_name(yaml_path, source)

            record = {
                "manifest_id":   manifest_id,
                "source":        source,
                "repo_name":     repo_name,
                "yaml_path":     yaml_path,
                **bin_flags,
                "cap_misuse":        cap_misuse,
                "all_secrets":       all_secrets,
                "total_misconfigs":  total_misconfigs,
                "risk_score":        risk_score,
                "label":             label,
                # Extended features (filled by scan_with_tools.py after YAML download)
                "NO_RUN_AS_NON_ROOT":   "",
                "NO_READ_ONLY_ROOT_FS": "",
                "IMAGE_USES_LATEST":    "",
                "SA_AUTOMOUNT_TOKEN":   "",
                "USES_DEFAULT_SA":      "",
                "UNTRUSTED_REGISTRY":   "",
                # Enrichment columns (filled by enrich_rf_dataset.py)
                "severity_class":    "",
                "mitre_technique":   "",
                "attack_description":"",
                # Checkov/kube-linter columns (filled by scan_with_tools.py)
                "checkov_failed_count":       "",
                "checkov_passed_count":       "",
                "checkov_umi_score":          "",
                "kl_failed_count":            "",
                "checkov_no_run_as_nonroot":  "",
                "checkov_no_readonly_rootfs": "",
                "checkov_image_latest":       "",
                "checkov_sa_automount":       "",
                "checkov_default_sa":         "",
                "checkov_untrusted_registry": "",
                "checkov_no_seccomp":         "",
                "checkov_no_apparmor":        "",
                "checkov_allow_privi_esc":    "",
                "checkov_caps_not_dropped":   "",
                "has_yaml":                   "0",
                # Metadata
                "size_bytes":        size_bytes,
                "age_days":          age_days,
                "commits":           commits,
                "devs":              devs,
                "is_deployable":     is_deployable,
                "is_minor":          is_minor,
            }
            records.append(record)
            manifest_id += 1

        kept = total - skipped_non_k8s - skipped_helm
        stats[source] = {
            "total": total,
            "kept": kept,
            "skipped_non_k8s": skipped_non_k8s,
            "skipped_helm": skipped_helm,
            "missing_metrics": missing_metrics,
            "label_1": sum(1 for r in records[-kept:] if r["label"] == 1),
            "label_0": sum(1 for r in records[-kept:] if r["label"] == 0),
        }

    # Write output CSV
    out_path = out_dir / "rf_dataset.csv"
    if not records:
        print("ERROR: no records produced — check input paths.")
        sys.exit(1)

    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Print summary
    print("\n" + "=" * 60)
    print("RF DATASET BUILD SUMMARY")
    print("=" * 60)
    for src, s in stats.items():
        print(f"\n  [{src.upper()}]")
        print(f"    Input rows     : {s['total']}")
        print(f"    Kept           : {s['kept']}")
        print(f"    Skipped (non-K8s): {s['skipped_non_k8s']}")
        print(f"    Skipped (Helm) : {s['skipped_helm']}")
        print(f"    Missing metrics: {s['missing_metrics']}")
        print(f"    Label=1 (misconfigured): {s['label_1']}")
        print(f"    Label=0 (secure)       : {s['label_0']}")

    total_kept = len(records)
    total_label_1 = sum(1 for r in records if r["label"] == 1)
    total_label_0 = total_kept - total_label_1
    print(f"\n  TOTAL ROWS     : {total_kept}")
    print(f"  Label=1        : {total_label_1} ({100*total_label_1/total_kept:.1f}%)")
    print(f"  Label=0        : {total_label_0} ({100*total_label_0/total_kept:.1f}%)")

    unique_repos = len({r["repo_name"] for r in records})
    print(f"  Unique repos   : {unique_repos}  (→ cluster snapshots for GNN)")
    print(f"\n  Output: {out_path}")
    print("=" * 60)

    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent  # TFE/

    default_data_dir = project_root / "original-dataset" / "rahman" / "DATASET"
    default_out_dir  = project_root / "data" / "tabular"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir, help="Path to Rahman DATASET/ directory")
    parser.add_argument("--out-dir",  type=Path, default=default_out_dir,  help="Output directory for rf_dataset.csv")
    parser.add_argument("--no-filter", action="store_true", help="Keep Helm charts and non-K8s manifests")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data directory not found: {args.data_dir}")
        sys.exit(1)

    build_dataset(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        filter_non_k8s=not args.no_filter,
    )


if __name__ == "__main__":
    main()
