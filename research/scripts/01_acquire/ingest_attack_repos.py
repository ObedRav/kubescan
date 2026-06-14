"""
ingest_attack_repos.py
======================
Ingests newly cloned attack/security demo repos into rf_dataset.csv.

Each repo is treated as one "cluster" for GNN graph construction.
Features are extracted via yaml_feature_extractor.py (same pipeline as Rahman data).

Repos ingested (from dataset/raw/attack_repos/):
  kubernetes-goof            snyk-labs: SA token theft + RBAC chains + escape pods
  infra-goof-k8s             snyk-labs/infrastructure-as-code-goof k8s/templates/
  k8s-escape                 KimberleyMsengezi: explicit privileged pod escape chain
  kubernetes-ctf             thedojoseries: OWASP CTF K8s scenarios
  kube-goat                  ksoclabs: ~10 deliberately vulnerable scenarios
  kustomizegoat              bridgecrewio: insecure Kustomize overlays
  k8s-security-lab           anshumaan-10: 10 exploit+fix YAML scenarios
  kube_security_lab          raesene: attacker manifests with privileged pods (label=2)
  minik8s-ctf                quarkslab: CTF challenges with privileged/hostPath pods (label=2)
  kubeaudit-fixtures         Shopify/kubeaudit: auditor test fixtures (26 escape resources, label=2)
  gatekeeper-library         OPA gatekeeper: PSP constraint test Pods (92 escape resources, label=2)
  datree-tests               datreeio: policy test manifests (8 escape resources, label=2)
  securekubernetes           securekubernetes demo: hostpath pod + Falco DaemonSet (label=2)
  kube-pod-escape            danielsagi: hostPath symlink escape + SA token (label=1)

Usage:
  python scripts/ingest_attack_repos.py
  python scripts/ingest_attack_repos.py --dry-run
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "02_extract"))
from extract_yaml_features import (
    FEATURE_COLS,
    extract_features_from_file,
)

# ---------------------------------------------------------------------------
# Cluster definitions: name → (repo_dir_relative_to_attack_repos, recurse)
# ---------------------------------------------------------------------------
ATTACK_REPOS_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "attack_repos"

CLUSTERS = [
    {
        "cluster":  "kubernetes-goof",
        "dir":      ATTACK_REPOS_DIR / "kubernetes-goof",
        "recurse":  True,
        "note":     "snyk-labs: SA token theft, RBAC chains, privileged pods",
    },
    {
        "cluster":  "infra-goof-k8s",
        "dir":      ATTACK_REPOS_DIR / "infrastructure-as-code-goof" / "k8s" / "templates",
        "recurse":  False,
        "note":     "snyk-labs iac-goof: 27 security template examples (21 with escape flags)",
    },
    {
        "cluster":  "k8s-escape",
        "dir":      ATTACK_REPOS_DIR / "Kubernetes-Container-Escape-Cluster-Breakout",
        "recurse":  True,
        "note":     "KimberleyMsengezi: privileged pod + hostPath escape chain",
    },
    {
        "cluster":  "kubernetes-ctf",
        "dir":      ATTACK_REPOS_DIR / "kubernetes-ctf",
        "recurse":  True,
        "note":     "thedojoseries: OWASP CTF K8s privilege escalation scenarios",
    },
    {
        "cluster":  "kube-goat",
        "dir":      ATTACK_REPOS_DIR / "kube-goat",
        "recurse":  True,
        "note":     "ksoclabs: deliberately vulnerable K8s cluster scenarios",
    },
    {
        "cluster":  "kustomizegoat",
        "dir":      ATTACK_REPOS_DIR / "kustomizegoat",
        "recurse":  True,
        "note":     "bridgecrewio: insecure Kustomize overlays",
    },
    {
        "cluster":  "k8s-security-lab",
        "dir":      ATTACK_REPOS_DIR / "k8s-security-lab",
        "recurse":  True,
        "note":     "anshumaan-10: 10 exploit+fix scenarios",
    },
    {
        "cluster":  "kube_security_lab",
        "dir":      ATTACK_REPOS_DIR / "kube_security_lab",
        "recurse":  True,
        "note":     "raesene: attacker manifests with privileged pods (hostPID/IPC/NET/privileged/hostPath)",
    },
    {
        "cluster":  "minik8s-ctf",
        "dir":      ATTACK_REPOS_DIR / "minik8s-ctf",
        "recurse":  True,
        "note":     "quarkslab: CTF challenges with privileged and hostPath pods",
    },
    {
        "cluster":  "kubeaudit-fixtures",
        "dir":      ATTACK_REPOS_DIR / "kubeaudit-fixtures",
        "recurse":  True,
        "note":     "Shopify/kubeaudit: auditor test fixtures with escape patterns",
    },
    {
        "cluster":  "gatekeeper-library",
        "dir":      ATTACK_REPOS_DIR / "gatekeeper-library" / "library" / "pod-security-policy",
        "recurse":  True,
        "note":     "OPA gatekeeper PSP constraint library: deny-case Pods with escape flags (privileged, hostPath, hostPID, etc.)",
    },
    {
        "cluster":  "datree-tests",
        "dir":      ATTACK_REPOS_DIR / "datree-tests",
        "recurse":  True,
        "note":     "datreeio: K8s policy validation test manifests including escape patterns",
    },
    {
        "cluster":  "securekubernetes",
        "dir":      ATTACK_REPOS_DIR / "securekubernetes",
        "recurse":  True,
        "note":     "securekubernetes demo: hostPath pod + Falco DaemonSet with host access",
    },
    {
        "cluster":  "kube-pod-escape",
        "dir":      ATTACK_REPOS_DIR / "kube-pod-escape",
        "recurse":  True,
        "note":     "danielsagi: hostPath /var/log symlink escape + SA token exfiltration",
    },
]

# ---------------------------------------------------------------------------
# Escape flag columns — used to compute per-row label (0=clean, 1=misconfig)
# ---------------------------------------------------------------------------
ESCAPE_COLS = {
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET",
    "DOCKERSOCK_PATH", "CAP_SYS_ADMIN", "CAP_SYS_MODULE",
    "SEC_CONT_OVER_PRIVIL", "ALLOW_PRIVI", "HOSTPATH_MOUNT",
}
MISCONFIG_COLS = set(FEATURE_COLS) - {"VALID_TAINT_SECRET"}  # all real flags


def compute_label(row: dict) -> int:
    """0=clean, 1=misconfig (any flag set)."""
    return 1 if any(int(row.get(c, 0)) for c in MISCONFIG_COLS) else 0


def compute_severity(row: dict) -> int:
    """0=clean, 1=low_medium, 2=high_critical."""
    escape_set = any(int(row.get(c, 0)) for c in ESCAPE_COLS)
    if escape_set:
        return 2
    misconfig = compute_label(row)
    return misconfig  # 0 or 1


def find_yamls(directory: Path, recurse: bool) -> list[Path]:
    yamls = []
    if recurse:
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".yaml") or f.endswith(".yml"):
                    yamls.append(Path(root) / f)
    else:
        yamls = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))
    return sorted(yamls)


def main():
    project_root = Path(__file__).parent.parent.parent
    rf_csv = project_root / "data" / "tabular" / "rf_dataset.csv"

    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load existing dataset
    with open(rf_csv, newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))
    fieldnames = list(existing_rows[0].keys())
    next_id = max(int(r.get("manifest_id", 0) or 0) for r in existing_rows) + 1

    # Check which clusters already exist
    existing_repos = {r["repo_name"] for r in existing_rows}

    new_rows = []
    cluster_stats = []

    for cluster_def in CLUSTERS:
        cluster_name = cluster_def["cluster"]
        cluster_dir  = cluster_def["dir"]

        if cluster_name in existing_repos:
            print(f"  [skip] {cluster_name} already in dataset")
            continue

        if not cluster_dir.exists():
            print(f"  [skip] {cluster_name}: directory not found: {cluster_dir}")
            continue

        yamls = find_yamls(cluster_dir, cluster_def["recurse"])
        if not yamls:
            print(f"  [skip] {cluster_name}: no YAML files found")
            continue

        cluster_rows = []
        n_skipped = 0
        for ypath in yamls:
            feats = extract_features_from_file(ypath)
            if feats is None:
                n_skipped += 1
                continue

            row = dict.fromkeys(fieldnames, "")
            row["manifest_id"] = str(next_id)
            row["source"]      = "attack_repos"
            row["repo_name"]   = cluster_name
            row["yaml_path"]   = str(ypath)

            # Feature columns
            for col in FEATURE_COLS:
                row[col] = str(feats.get(col, 0))

            # Derived columns
            cap_misuse = int(row.get("CAP_SYS_ADMIN","0") or 0) | int(row.get("CAP_SYS_MODULE","0") or 0)
            secrets    = int(row.get("WITHIN_MANIFEST_SECRET","0") or 0) | int(row.get("VALID_TAINT_SECRET","0") or 0)
            total_mc   = sum(int(row.get(c, 0) or 0) for c in FEATURE_COLS)
            label      = compute_label(row)
            sev        = compute_severity(row)

            row["cap_misuse"]     = str(cap_misuse)
            row["all_secrets"]    = str(secrets)
            row["total_misconfigs"] = str(total_mc)
            row["risk_score"]     = ""   # recomputed by graph_builder
            row["label"]          = str(label)
            row["severity_class"] = str(sev)
            row["mitre_technique"] = ""
            row["attack_description"] = cluster_def["note"]
            row["has_yaml"]       = "1"
            row["size_bytes"]     = str(ypath.stat().st_size)
            row["age_days"]       = "0"
            row["commits"]        = "0"
            row["devs"]           = "0"
            row["is_deployable"]  = "1"
            row["is_minor"]       = "0"

            # Checkov columns: leave empty (not scanned)
            for col in fieldnames:
                if col.startswith("checkov_") or col in ("checkov_failed_count","checkov_passed_count","checkov_umi_score","kl_failed_count"):
                    row[col] = row.get(col, "")

            cluster_rows.append(row)
            next_id += 1

        if not cluster_rows:
            print(f"  [skip] {cluster_name}: no workload resources found in {len(yamls)} YAMLs")
            continue

        escape_cnt = sum(1 for r in cluster_rows if any(int(r.get(c,0) or 0) for c in ESCAPE_COLS))
        print(f"  {cluster_name}: {len(cluster_rows)} resources ({n_skipped} skipped), "
              f"{escape_cnt} with escape flags — expected label={'2' if escape_cnt >= 2 else ('2?' if escape_cnt == 1 else '0/1')}")
        new_rows.extend(cluster_rows)
        cluster_stats.append((cluster_name, len(cluster_rows), escape_cnt))

    print(f"\nTotal new rows: {len(new_rows)} across {len(cluster_stats)} clusters")

    if args.dry_run:
        print("[dry-run] Not writing.")
        return

    if not new_rows:
        print("Nothing to write.")
        return

    updated = existing_rows + new_rows
    with open(rf_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updated)

    print(f"Written {len(updated)} rows to {rf_csv}")
    print("\nNew rows by cluster:")
    for name, n_res, n_esc in cluster_stats:
        print(f"  {name}: {n_res} resources, {n_esc} escape nodes")


if __name__ == "__main__":
    main()
