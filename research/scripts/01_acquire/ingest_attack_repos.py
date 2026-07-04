"""
ingest_attack_repos.py
======================
Ingests newly cloned attack/security demo repos into rf_dataset.csv.

Each repo is treated as one "cluster" for GNN graph construction.
Features are extracted via extract_yaml_features.py (same pipeline as Rahman data).

Repos ingested (from data/raw/attack_repos/):
  kubernetes-goof            snyk-labs: SA token theft + RBAC chains + escape pods
  infra-goof-k8s             snyk-labs/infrastructure-as-code-goof k8s/templates/
  k8s-escape                 KimberleyMsengezi: explicit privileged pod escape chain
  kubernetes-ctf             thedojoseries: OWASP CTF K8s scenarios
  kube-goat                  ksoclabs: ~10 deliberately vulnerable scenarios
  kustomizegoat              bridgecrewio: insecure Kustomize overlays
  k8s-security-lab           anshumaan-10: 10 exploit+fix YAML scenarios
  kube_security_lab          raesene: attacker manifests with privileged pods
  minik8s-ctf                quarkslab: CTF challenges with privileged/hostPath pods
  securekubernetes           securekubernetes demo: hostpath pod + Falco DaemonSet
  kube-pod-escape            danielsagi: hostPath symlink escape + SA token
  gatekeeper_*               OPA gatekeeper: one cluster per PSP policy category
                             (19 clusters — both allowed and disallowed test cases)
  kubeaudit_*                Shopify/kubeaudit: one cluster per auditor type
                             (14 clusters — all fixture files, labels from feature extraction)
  datree-tests               datreeio: pass/ and fail/ manifests across 111 test cases

Split-by-subdir repos use the `split_by_subdir` mechanism: each immediate
subdirectory of `dir` becomes its own cluster. Per-manifest labels are always
derived from feature extraction — the cluster-level label is then assigned by
build_graphs.py from the distribution of node labels.

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
# Cluster definitions
# ---------------------------------------------------------------------------
ATTACK_REPOS_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "attack_repos"

# Each entry is either:
#   Regular:       {"cluster": str, "dir": Path, "recurse": bool, "note": str}
#   Split-by-dir:  {"cluster_prefix": str, "dir": Path, "split_by_subdir": True,
#                   "subdir_fixtures": str|None, "recurse": bool, "note": str}
#
# For split-by-dir entries, expand_clusters() enumerates immediate subdirectories
# of `dir` and creates one cluster per subdir (name = f"{cluster_prefix}_{subdir}").
# If `subdir_fixtures` is set, YAMLs are searched inside that sub-subfolder
# (e.g. "fixtures" → <subdir>/fixtures/).
CLUSTERS: list[dict] = [
    {
        "cluster": "kubernetes-goof",
        "dir":     ATTACK_REPOS_DIR / "kubernetes-goof",
        "recurse": True,
        "note":    "snyk-labs: SA token theft, RBAC chains, privileged pods",
    },
    {
        "cluster": "infra-goof-k8s",
        "dir":     ATTACK_REPOS_DIR / "infrastructure-as-code-goof" / "k8s" / "templates",
        "recurse": False,
        "note":    "snyk-labs iac-goof: 27 security template examples (21 with escape flags)",
    },
    {
        "cluster": "k8s-escape",
        "dir":     ATTACK_REPOS_DIR / "Kubernetes-Container-Escape-Cluster-Breakout",
        "recurse": True,
        "note":    "KimberleyMsengezi: privileged pod + hostPath escape chain",
    },
    {
        "cluster": "kubernetes-ctf",
        "dir":     ATTACK_REPOS_DIR / "kubernetes-ctf",
        "recurse": True,
        "note":    "thedojoseries: OWASP CTF K8s privilege escalation scenarios",
    },
    {
        "cluster": "kube-goat",
        "dir":     ATTACK_REPOS_DIR / "kube-goat",
        "recurse": True,
        "note":    "ksoclabs: deliberately vulnerable K8s cluster scenarios",
    },
    {
        "cluster": "kustomizegoat",
        "dir":     ATTACK_REPOS_DIR / "kustomizegoat",
        "recurse": True,
        "note":    "bridgecrewio: insecure Kustomize overlays",
    },
    {
        "cluster": "k8s-security-lab",
        "dir":     ATTACK_REPOS_DIR / "k8s-security-lab",
        "recurse": True,
        "note":    "anshumaan-10: 10 exploit+fix scenarios",
    },
    {
        "cluster": "kube_security_lab",
        "dir":     ATTACK_REPOS_DIR / "kube_security_lab",
        "recurse": True,
        "note":    "raesene: attacker manifests with privileged pods (hostPID/IPC/NET/privileged/hostPath)",
    },
    {
        "cluster": "minik8s-ctf",
        "dir":     ATTACK_REPOS_DIR / "minik8s-ctf",
        "recurse": True,
        "note":    "quarkslab: CTF challenges with privileged and hostPath pods",
    },
    {
        "cluster": "securekubernetes",
        "dir":     ATTACK_REPOS_DIR / "securekubernetes",
        "recurse": True,
        "note":    "securekubernetes demo: hostPath pod + Falco DaemonSet with host access",
    },
    {
        "cluster": "kube-pod-escape",
        "dir":     ATTACK_REPOS_DIR / "kube-pod-escape",
        "recurse": True,
        "note":    "danielsagi: hostPath /var/log symlink escape + SA token exfiltration",
    },
    # OPA Gatekeeper PSP constraint library — split by policy category.
    # Each category has both allowed (clean) and disallowed (attack) sample pods.
    # Per-manifest labels come from feature extraction, so clean samples receive
    # label=0 and disallowed samples receive label=2 naturally.
    {
        "cluster_prefix": "gatekeeper",
        "dir":            ATTACK_REPOS_DIR / "gatekeeper-library" / "library" / "pod-security-policy",
        "split_by_subdir": True,
        "subdir_fixtures": None,  # recurse directly into each policy subdir
        "recurse":         True,
        "note":            "OPA Gatekeeper PSP library",
    },
    # Shopify/kubeaudit auditor test fixtures — split by auditor type.
    # Fixtures include both clean (flag-absent or annotated-allowed) and
    # attack (flag-set) manifests; feature extraction labels each correctly.
    {
        "cluster_prefix": "kubeaudit",
        "dir":            ATTACK_REPOS_DIR / "kubeaudit-fixtures" / "auditors",
        "split_by_subdir": True,
        "subdir_fixtures": "fixtures",  # YAMLs live in <auditor>/fixtures/
        "recurse":         False,       # fixtures/ is flat
        "note":            "Shopify/kubeaudit auditor fixtures",
    },
    # datree-tests — all 111 test cases in one cluster.
    # Each test has pass/ (compliant) and fail/ (non-compliant) subdirectories;
    # our feature extractor determines per-manifest labels correctly.
    {
        "cluster": "datree-tests",
        "dir":     ATTACK_REPOS_DIR / "datree-tests" / "pkg" / "policy" / "tests",
        "recurse": True,
        "note":    "datreeio policy tests: pass/ and fail/ manifests",
    },
]


# ---------------------------------------------------------------------------
# Escape and misconfig flag sets
# ---------------------------------------------------------------------------
ESCAPE_COLS: frozenset[str] = frozenset({
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET",
    "DOCKERSOCK_PATH", "CAP_SYS_ADMIN", "CAP_SYS_MODULE",
    "SEC_CONT_OVER_PRIVIL", "ALLOW_PRIVI", "HOSTPATH_MOUNT",
})
MISCONFIG_COLS: frozenset[str] = frozenset(FEATURE_COLS) - {"VALID_TAINT_SECRET"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_label(row: dict) -> int:
    """0=clean, 1=misconfig (any flag set)."""
    return 1 if any(int(row.get(c, 0)) for c in MISCONFIG_COLS) else 0


def compute_severity(row: dict) -> int:
    """0=clean, 1=low_medium, 2=high_critical."""
    if any(int(row.get(c, 0)) for c in ESCAPE_COLS):
        return 2
    return compute_label(row)


def find_yamls(directory: Path, recurse: bool) -> list[Path]:
    if recurse:
        yamls: list[Path] = []
        for root, _, files in os.walk(directory):
            for f in files:
                if f.endswith(".yaml") or f.endswith(".yml"):
                    yamls.append(Path(root) / f)
        return sorted(yamls)
    return sorted(list(directory.glob("*.yaml")) + list(directory.glob("*.yml")))


def expand_clusters(cluster_defs: list[dict]) -> list[dict]:
    """
    Expand split_by_subdir entries into one cluster definition per subdir.
    Regular entries are returned unchanged.
    """
    result: list[dict] = []
    for defn in cluster_defs:
        if not defn.get("split_by_subdir"):
            result.append(defn)
            continue

        base_dir: Path = defn["dir"]
        prefix: str = defn["cluster_prefix"]
        fixtures_sub: str | None = defn.get("subdir_fixtures")
        recurse: bool = defn.get("recurse", True)
        note: str = defn.get("note", "")

        if not base_dir.exists():
            result.append({**defn, "cluster": prefix, "dir": base_dir})
            continue

        for subdir in sorted(base_dir.iterdir()):
            if not subdir.is_dir():
                continue
            yaml_dir = subdir / fixtures_sub if fixtures_sub else subdir
            result.append({
                "cluster": f"{prefix}_{subdir.name}",
                "dir":     yaml_dir,
                "recurse": recurse,
                "note":    f"{note}: {subdir.name}",
            })

    return result


def build_row(
    manifest_id: int,
    cluster_name: str,
    ypath: Path,
    feats: dict,
    fieldnames: list[str],
    note: str,
) -> dict:
    row = dict.fromkeys(fieldnames, "")
    row["manifest_id"] = str(manifest_id)
    row["source"]      = "attack_repos"
    row["repo_name"]   = cluster_name
    row["yaml_path"]   = str(ypath)

    for col in FEATURE_COLS:
        row[col] = str(feats.get(col, 0))

    cap_misuse = int(row.get("CAP_SYS_ADMIN", "0") or 0) | int(row.get("CAP_SYS_MODULE", "0") or 0)
    secrets    = int(row.get("WITHIN_MANIFEST_SECRET", "0") or 0) | int(row.get("VALID_TAINT_SECRET", "0") or 0)
    total_mc   = sum(int(row.get(c, 0) or 0) for c in FEATURE_COLS)
    label      = compute_label(row)
    sev        = compute_severity(row)

    row["cap_misuse"]        = str(cap_misuse)
    row["all_secrets"]       = str(secrets)
    row["total_misconfigs"]  = str(total_mc)
    row["risk_score"]        = ""   # recomputed by graph_builder
    row["label"]             = str(label)
    row["severity_class"]    = str(sev)
    row["mitre_technique"]   = ""
    row["attack_description"] = note
    row["has_yaml"]          = "1"
    row["size_bytes"]        = str(ypath.stat().st_size)
    row["age_days"]          = "0"
    row["commits"]           = "0"
    row["devs"]              = "0"
    row["is_deployable"]     = "1"
    row["is_minor"]          = "0"
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    project_root = Path(__file__).parent.parent.parent
    rf_csv = project_root / "data" / "tabular" / "rf_dataset.csv"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with rf_csv.open(newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))
    fieldnames = list(existing_rows[0].keys())
    next_id = max(int(r.get("manifest_id", 0) or 0) for r in existing_rows) + 1

    existing_repos = {r["repo_name"] for r in existing_rows}
    flat_clusters = expand_clusters(CLUSTERS)

    new_rows: list[dict] = []
    cluster_stats: list[tuple[str, int, int]] = []

    for cluster_def in flat_clusters:
        cluster_name: str = cluster_def["cluster"]
        cluster_dir: Path = cluster_def["dir"]

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

        cluster_rows: list[dict] = []
        n_skipped = 0
        for ypath in yamls:
            feats = extract_features_from_file(ypath)
            if feats is None:
                n_skipped += 1
                continue
            cluster_rows.append(
                build_row(next_id, cluster_name, ypath, feats, fieldnames, cluster_def["note"]),
            )
            next_id += 1

        if not cluster_rows:
            print(f"  [skip] {cluster_name}: no workload resources found in {len(yamls)} YAMLs")
            continue

        escape_cnt = sum(
            1 for r in cluster_rows
            if any(int(r.get(c, 0) or 0) for c in ESCAPE_COLS)
        )
        if escape_cnt >= 2:
            expected = "2"
        elif escape_cnt == 1:
            expected = "2?"
        else:
            expected = "0/1"
        print(f"  {cluster_name}: {len(cluster_rows)} resources "
              f"({n_skipped} skipped), {escape_cnt} escape — expected graph label={expected}")
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
    with rf_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updated)

    print(f"Written {len(updated)} rows to {rf_csv}")
    print("\nNew rows by cluster:")
    for name, n_res, n_esc in cluster_stats:
        print(f"  {name}: {n_res} resources, {n_esc} escape nodes")


if __name__ == "__main__":
    main()
