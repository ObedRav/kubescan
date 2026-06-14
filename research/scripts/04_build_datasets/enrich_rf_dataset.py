"""
enrich_rf_dataset.py
=====================
Enriches rf_dataset.csv with two improvements:

1. Severity stratification
   Adds a `severity_class` column to every existing row:
     0 = clean       — label=0 (no misconfigurations per Rahman INSECURE flag)
     1 = low_medium  — label=1 but only medium/low-severity flags present
                       (INSECURE_HTTP, NO_ROLLING_UPDATE, NO_NETWORK_POLICY,
                        NO_SECU_CONTEXT, NO_RESO, NO_DEFAULT_NSPACE, HOST_ALIAS)
     2 = high_critical — label=1 AND at least one critical/high flag present
                         (TRUE_HOST_PID, TRUE_HOST_IPC, TRUE_HOST_NET,
                          DOCKERSOCK_PATH, CAP_SYS_ADMIN, CAP_SYS_MODULE,
                          WITHIN_MANIFEST_SECRET, SEC_CONT_OVER_PRIVIL,
                          ALLOW_PRIVI, SECCOMP_UNCONFINED)

2. Append BadPods + Kubernetes Goat manifests
   These directly address the critical feature sparsity problem: most critical
   flags appear in < 1% of Rahman rows. BadPods and Kubernetes Goat provide
   intentionally misconfigured manifests with known, documented attack paths.

   BadPods (BishopFox/badPods) — 8 attack categories:
     everything-allowed  → T1611 (Escape to Host, full node access)
     priv-and-hostpid    → T1611
     priv                → T1611 (privileged container)
     hostpid             → T1611, T1613 (process discovery)
     hostipc             → T1611
     hostnetwork         → T1610 (network namespace access)
     hostpath            → T1611 (node filesystem read)
     nothing-allowed     → clean baseline (label=0)

   Kubernetes Goat (madhuakula/kubernetes-goat) — 14 scenarios covering RBAC
   abuse, container escape, secret exposure, and lateral movement paths.

Output: dataset/tabular/rf_dataset.csv (overwritten in-place with new column + rows)

Usage:
  python scripts/enrich_rf_dataset.py

  Optional:
    --badpods-dir      Path to badPods repo       (default: auto-detected)
    --goat-dir         Path to kubernetes-goat    (default: auto-detected)
    --rf-dataset       Path to rf_dataset.csv     (default: auto-detected)
    --dry-run          Print stats without writing
"""

import argparse
import csv
import sys
from pathlib import Path

# Allow importing from same scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "02_extract"))
from extract_yaml_features import (
    FEATURE_COLS,
    extract_features_from_file,
)

# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

CRITICAL_FLAGS = {
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET",
    "DOCKERSOCK_PATH", "CAP_SYS_ADMIN", "CAP_SYS_MODULE",
    "WITHIN_MANIFEST_SECRET", "SEC_CONT_OVER_PRIVIL",
    "ALLOW_PRIVI", "SECCOMP_UNCONFINED",
}

SEVERITY_WEIGHTS = {
    "TRUE_HOST_PID":         3.0,
    "TRUE_HOST_IPC":         3.0,
    "TRUE_HOST_NET":         3.0,
    "DOCKERSOCK_PATH":       3.0,
    "CAP_SYS_ADMIN":         3.0,
    "CAP_SYS_MODULE":        3.0,
    "WITHIN_MANIFEST_SECRET":3.0,
    "SEC_CONT_OVER_PRIVIL":  2.5,
    "ALLOW_PRIVI":           2.5,
    "SECCOMP_UNCONFINED":    2.0,
    "VALID_TAINT_SECRET":    2.0,
    "INSECURE_HTTP":         1.5,
    "NO_SECU_CONTEXT":       1.5,
    "NO_NETWORK_POLICY":     1.0,
    "HOST_ALIAS":            1.0,
    "NO_DEFAULT_NSPACE":     0.5,
    "NO_RESO":               0.5,
    "NO_ROLLING_UPDATE":     0.3,
}
MAX_RISK = sum(SEVERITY_WEIGHTS.values())


def compute_severity_class(row: dict) -> int:
    label = int(row.get("label", 0))
    if label == 0:
        return 0
    has_critical = any(int(row.get(f, 0)) == 1 for f in CRITICAL_FLAGS)
    return 2 if has_critical else 1


def compute_risk_score(flags: dict[str, int]) -> float:
    weighted = sum(SEVERITY_WEIGHTS.get(col, 0) * flags.get(col, 0) for col in FEATURE_COLS)
    return round(min(1.0, weighted / MAX_RISK), 4)


# ---------------------------------------------------------------------------
# BadPods category → attack metadata
# ---------------------------------------------------------------------------

# Maps directory name → (mitre_technique, description)
BADPODS_CATEGORY_META: dict[str, tuple[str, str]] = {
    "everything-allowed": (
        "T1611",
        "Full node access: privileged + hostPID + hostIPC + hostNetwork + hostPath=/",
    ),
    "priv-and-hostpid": (
        "T1611",
        "Privileged container + hostPID — process namespace escape to host",
    ),
    "priv": (
        "T1611",
        "Privileged container — full kernel access via securityContext.privileged=true",
    ),
    "hostpid": (
        "T1611,T1613",
        "hostPID=true — read host process list, potential ptrace-based escape",
    ),
    "hostipc": (
        "T1611",
        "hostIPC=true — shared memory access with host processes",
    ),
    "hostnetwork": (
        "T1610",
        "hostNetwork=true — full host network stack access, can sniff traffic",
    ),
    "hostpath": (
        "T1611",
        "hostPath volume mount to / — read/write node filesystem",
    ),
    "nothing-allowed": (
        "",
        "Clean baseline pod — no special permissions (used as negative control)",
    ),
}

# Maps Kubernetes Goat scenario dir name → (mitre_technique, description)
GOAT_SCENARIO_META: dict[str, tuple[str, str]] = {
    "insecure-rbac":           ("T1078,T1613", "ClusterRoleBinding to cluster-admin SA"),
    "system-monitor":          ("T1613,T1611", "DaemonSet with privileged + hostPID for node monitoring"),
    "hidden-in-layers":        ("T1525",       "Malicious content hidden in container image layers"),
    "poor-registry":           ("T1525",       "Unverified/poor registry — supply chain risk"),
    "build-code":              ("T1610",       "Build container with excessive privileges"),
    "cache-store":             ("T1552",       "Insecure credential storage in cache/env vars"),
    "hunger-check":            ("T1499",       "Resource exhaustion via missing limits"),
    "batch-check":             ("T1059",       "Batch job with command injection surface"),
    "health-check":            ("",            "Health check deployment — baseline"),
    "internal-proxy":          ("T1090",       "Internal proxy for lateral movement"),
    "metadata-db":             ("T1552",       "Metadata DB with potential credential exposure"),
    "docker-bench-security":   ("",            "Docker bench security scanner (non-attack)"),
    "kube-bench-security":     ("",            "CIS benchmark scanner job (non-attack)"),
    "kubernetes-goat-home":    ("",            "Kubernetes Goat home scenario (baseline)"),
    "kyverno-namespace-exec-block": ("",       "Kyverno admission controller policy"),
}


def badpods_category_from_path(yaml_path: Path) -> str:
    """Extract the BadPods attack category from the file path."""
    parts = yaml_path.parts
    manifests_idx = next((i for i, p in enumerate(parts) if p == "manifests"), None)
    if manifests_idx is not None and manifests_idx + 1 < len(parts):
        return parts[manifests_idx + 1]
    return "unknown"


def goat_scenario_from_path(yaml_path: Path) -> str:
    """Extract the Kubernetes Goat scenario name from the file path."""
    parts = yaml_path.parts
    scenarios_idx = next((i for i, p in enumerate(parts) if p == "scenarios"), None)
    if scenarios_idx is not None and scenarios_idx + 1 < len(parts):
        return parts[scenarios_idx + 1]
    return "unknown"


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def flags_to_row(
    manifest_id: int,
    source: str,
    repo_name: str,
    yaml_path: str,
    flags: dict[str, int],
    label: int,
    mitre_technique: str = "",
    attack_description: str = "",
    size_bytes: str = "",
    age_days: str = "",
    commits: str = "",
    devs: str = "",
    is_deployable: str = "1",
    is_minor: str = "0",
) -> dict:
    total_misconfigs = sum(flags[c] for c in FEATURE_COLS)
    risk_score = compute_risk_score(flags)
    has_critical = any(flags.get(f, 0) == 1 for f in CRITICAL_FLAGS)
    severity_class = (2 if has_critical else 1) if label == 1 else 0

    return {
        "manifest_id":       manifest_id,
        "source":            source,
        "repo_name":         repo_name,
        "yaml_path":         yaml_path,
        **{c: flags[c] for c in FEATURE_COLS},
        "cap_misuse":        flags.get("CAP_SYS_ADMIN", 0) | flags.get("CAP_SYS_MODULE", 0),
        "all_secrets":       flags.get("WITHIN_MANIFEST_SECRET", 0) | flags.get("VALID_TAINT_SECRET", 0),
        "total_misconfigs":  total_misconfigs,
        "risk_score":        risk_score,
        "label":             label,
        "severity_class":    severity_class,
        "mitre_technique":   mitre_technique,
        "attack_description":attack_description,
        "size_bytes":        size_bytes,
        "age_days":          age_days,
        "commits":           commits,
        "devs":              devs,
        "is_deployable":     is_deployable,
        "is_minor":          is_minor,
    }


# ---------------------------------------------------------------------------
# Process BadPods
# ---------------------------------------------------------------------------

def process_badpods(badpods_dir: Path, start_id: int) -> list[dict]:
    rows = []
    manifest_id = start_id
    manifests_dir = badpods_dir / "manifests"
    if not manifests_dir.exists():
        print(f"  [warn] BadPods manifests/ not found at {manifests_dir}")
        return rows

    for yaml_path in sorted(manifests_dir.glob("**/*.yaml")):
        category = badpods_category_from_path(yaml_path)
        mitre, desc = BADPODS_CATEGORY_META.get(category, ("", "Unknown BadPods category"))

        # "nothing-allowed" is the clean baseline
        is_nothing = category == "nothing-allowed"

        feats = extract_features_from_file(yaml_path, assume_network_policy=False)
        if feats is None:
            continue

        flags = {c: feats[c] for c in FEATURE_COLS}
        label = 0 if is_nothing else 1

        row = flags_to_row(
            manifest_id=manifest_id,
            source="badpods",
            repo_name=f"badpods_{category}",
            yaml_path=str(yaml_path),
            flags=flags,
            label=label,
            mitre_technique=mitre,
            attack_description=desc,
            is_deployable="1",
        )
        rows.append(row)
        manifest_id += 1

    return rows


# ---------------------------------------------------------------------------
# Process Kubernetes Goat
# ---------------------------------------------------------------------------

def process_kubernetes_goat(goat_dir: Path, start_id: int) -> list[dict]:
    rows = []
    manifest_id = start_id
    scenarios_dir = goat_dir / "scenarios"
    if not scenarios_dir.exists():
        print(f"  [warn] Kubernetes Goat scenarios/ not found at {scenarios_dir}")
        return rows

    # Also process infrastructure/ and platforms/ if they have manifests
    search_dirs = [scenarios_dir]
    for extra in ["infrastructure", "platforms"]:
        d = goat_dir / extra
        if d.exists():
            search_dirs.append(d)

    for search_dir in search_dirs:
        for yaml_path in sorted(search_dir.glob("**/*.yaml")):
            # Skip Helm chart metadata files that aren't workloads
            if yaml_path.name in ("Chart.yaml", "NOTES.txt"):
                continue

            scenario = goat_scenario_from_path(yaml_path) if search_dir == scenarios_dir else search_dir.name
            mitre, desc = GOAT_SCENARIO_META.get(scenario, ("", f"Kubernetes Goat: {scenario}"))

            feats = extract_features_from_file(yaml_path, assume_network_policy=False)
            if feats is None:
                continue

            flags = {c: feats[c] for c in FEATURE_COLS}
            total_critical = sum(flags.get(f, 0) for f in CRITICAL_FLAGS)
            total_any = sum(flags.values())

            # Labeling: clean scanners / bench jobs have no attack surface → label=0
            non_attack_scenarios = {
                "health-check", "docker-bench-security", "kube-bench-security",
                "kubernetes-goat-home", "kyverno-namespace-exec-block",
            }
            if scenario in non_attack_scenarios and total_critical == 0:
                label = 0
            else:
                label = 1 if total_any > 0 else 0

            row = flags_to_row(
                manifest_id=manifest_id,
                source="kubernetes_goat",
                repo_name=f"kubernetes_goat_{scenario}",
                yaml_path=str(yaml_path),
                flags=flags,
                label=label,
                mitre_technique=mitre,
                attack_description=desc,
                is_deployable="1",
            )
            rows.append(row)
            manifest_id += 1

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent

    default_rf      = project_root / "data" / "tabular" / "rf_dataset.csv"
    default_badpods = project_root / "data" / "raw" / "badpods"
    default_goat    = project_root / "data" / "raw" / "kubernetes_goat"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rf-dataset",  type=Path, default=default_rf)
    parser.add_argument("--badpods-dir", type=Path, default=default_badpods)
    parser.add_argument("--goat-dir",    type=Path, default=default_goat)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    # ---- Load existing dataset ----
    print(f"Loading {args.rf_dataset}...")
    with open(args.rf_dataset, newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))

    # Add new columns to existing rows (severity_class, mitre_technique, attack_description)
    for row in existing_rows:
        row["severity_class"]    = compute_severity_class(row)
        row["mitre_technique"]   = ""
        row["attack_description"]= ""

    print(f"  Existing rows: {len(existing_rows)}")
    sc_counts = {0: 0, 1: 0, 2: 0}
    for r in existing_rows:
        sc_counts[int(r["severity_class"])] += 1
    print(f"  severity_class: clean={sc_counts[0]}  low_medium={sc_counts[1]}  high_critical={sc_counts[2]}")

    # ---- Process BadPods ----
    print(f"\nProcessing BadPods from {args.badpods_dir}...")
    badpods_rows = process_badpods(args.badpods_dir, start_id=len(existing_rows))
    print(f"  BadPods rows extracted: {len(badpods_rows)}")
    bp_sc = {0: 0, 1: 0, 2: 0}
    bp_critical = {}
    for r in badpods_rows:
        bp_sc[int(r["severity_class"])] = bp_sc.get(int(r["severity_class"]), 0) + 1
        # Collect which critical flags are set
        for f in CRITICAL_FLAGS:
            if int(r.get(f, 0)) == 1:
                bp_critical[f] = bp_critical.get(f, 0) + 1
    print(f"  severity_class: clean={bp_sc[0]}  low_medium={bp_sc[1]}  high_critical={bp_sc[2]}")
    print(f"  Critical flags now present: {bp_critical}")

    # ---- Process Kubernetes Goat ----
    print(f"\nProcessing Kubernetes Goat from {args.goat_dir}...")
    goat_rows = process_kubernetes_goat(args.goat_dir, start_id=len(existing_rows) + len(badpods_rows))
    print(f"  Kubernetes Goat rows extracted: {len(goat_rows)}")
    gt_sc = {0: 0, 1: 0, 2: 0}
    for r in goat_rows:
        gt_sc[int(r["severity_class"])] = gt_sc.get(int(r["severity_class"]), 0) + 1
    print(f"  severity_class: clean={gt_sc[0]}  low_medium={gt_sc[1]}  high_critical={gt_sc[2]}")

    # ---- Combine ----
    all_rows = existing_rows + badpods_rows + goat_rows

    # Reassign manifest_ids sequentially
    for i, row in enumerate(all_rows):
        row["manifest_id"] = i

    total = len(all_rows)
    total_sc = {0: 0, 1: 0, 2: 0}
    for r in all_rows:
        total_sc[int(r["severity_class"])] = total_sc.get(int(r["severity_class"]), 0) + 1
    total_label1 = sum(1 for r in all_rows if int(r["label"]) == 1)

    print(f"\n{'='*60}")
    print("ENRICHMENT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows       : {total}")
    print(f"  Label=0 (secure) : {total - total_label1} ({100*(total-total_label1)/total:.1f}%)")
    print(f"  Label=1 (misc)   : {total_label1} ({100*total_label1/total:.1f}%)")
    print(f"  severity_class=0 (clean)        : {total_sc[0]}")
    print(f"  severity_class=1 (low_medium)   : {total_sc[1]}")
    print(f"  severity_class=2 (high_critical): {total_sc[2]}")

    # Final critical flag coverage
    print("\n  Critical feature coverage after enrichment:")
    for f in sorted(CRITICAL_FLAGS):
        cnt = sum(1 for r in all_rows if int(r.get(f, 0)) == 1)
        pct = 100 * cnt / total
        print(f"    {f:30s}: {cnt:4d} ({pct:.1f}%)")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ---- Write updated CSV ----
    fieldnames = list(all_rows[0].keys())
    with open(args.rf_dataset, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Saved: {args.rf_dataset}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
