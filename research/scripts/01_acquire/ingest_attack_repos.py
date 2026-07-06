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
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

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
    # OPA Gatekeeper PSP constraint library — one cluster per sample manifest
    # (split_by_file, not split_by_subdir): bundling a policy category's
    # allowed+disallowed samples into one multi-node cluster let unrelated
    # samples that happen to share an unrelated escape flag (e.g. a shared
    # base template) trip the graph-level ">=2 escape nodes" chain rule —
    # see audit/model_fixes.md option 3. Per-file clusters can't do that.
    {
        "cluster_prefix": "gatekeeper",
        "dir":            ATTACK_REPOS_DIR / "gatekeeper-library" / "library" / "pod-security-policy",
        "split_by_file":  True,
        "recurse":        True,
        "note":           "OPA Gatekeeper PSP library",
    },
    # Shopify/kubeaudit auditor test fixtures — one cluster per fixture file,
    # same rationale as gatekeeper above.
    {
        "cluster_prefix": "kubeaudit",
        "dir":            ATTACK_REPOS_DIR / "kubeaudit-fixtures" / "auditors",
        "split_by_file":  True,
        "recurse":        True,
        "note":           "Shopify/kubeaudit auditor fixtures",
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
    # --- Real production workloads (complement the synthetic-lab-heavy corpus
    #     above with legitimate multi-resource deployments that are
    #     structurally attack-chain-shaped: escape-capable DaemonSets/agents
    #     paired with a ServiceAccount bound to a meaningful Role/ClusterRole) ---
    {
        "cluster": "longhorn",
        "dir":     ATTACK_REPOS_DIR / "longhorn" / "deploy",
        "recurse": False,
        "note":    "longhorn/longhorn: production storage manager — privileged "
                   "hostPath-mounting DaemonSet + broad ClusterRole",
    },
    {
        "cluster": "calico",
        "dir":     ATTACK_REPOS_DIR / "calico" / "manifests",
        "recurse": False,
        "note":    "projectcalico/calico: production CNI — hostNetwork+privileged "
                   "node DaemonSet + ClusterRole",
    },
    # Each Dockerfiles/manifests/ subdirectory is a self-contained real
    # deployment variant (own DaemonSet + own rbac.yaml) — split_by_subdir
    # yields one cluster per variant instead of merging unrelated flavours.
    {
        "cluster_prefix": "datadog",
        "dir":            ATTACK_REPOS_DIR / "datadog-agent" / "Dockerfiles" / "manifests",
        "split_by_subdir": True,
        "subdir_fixtures": None,
        "recurse":         False,
        "note":            "DataDog/datadog-agent: production observability agent "
                           "variants — hostPID/capabilities/docker.sock + ClusterRole",
    },
    {
        "cluster": "aws-ebs-csi-driver",
        "dir":     ATTACK_REPOS_DIR / "aws-ebs-csi-driver" / "deploy" / "kubernetes" / "base",
        "recurse": False,
        "note":    "kubernetes-sigs/aws-ebs-csi-driver: production CSI driver — "
                   "privileged node DaemonSet + ClusterRole",
    },
    # --- Purpose-built attack-graph tooling fixtures ---
    # Flat directory, one self-contained SA+Role/ClusterRole+RoleBinding+Pod
    # chain per technique file — split_by_file (no subdirs to split on).
    {
        "cluster_prefix": "kubehound",
        "dir":            ATTACK_REPOS_DIR / "kubehound" / "test" / "setup" / "test-cluster" / "attacks",
        "split_by_file":  True,
        "note":           "DataDog/KubeHound: attack-graph tool's own test fixtures — "
                          "one escalation/lateral-movement technique per file",
    },
    # --- Additional CTF-style scenarios (verified NOT forks of any repo above) ---
    {
        "cluster_prefix": "simulator",
        "dir":            ATTACK_REPOS_DIR / "simulator" / "ansible" / "roles",
        "split_by_subdir": True,
        "subdir_fixtures": "files/manifests",
        "recurse":         False,
        "note":            "controlplaneio/simulator: narrative attack scenarios — "
                           "escape flag + RBAC lateral-movement grant in the same manifest",
    },
    # --- Additional scanner test-fixture corpus — one cluster per
    #     PASSED/FAILED manifest, same split_by_file rationale as above ---
    {
        "cluster_prefix": "checkov",
        "dir":            ATTACK_REPOS_DIR / "checkov-tests" / "tests" / "kubernetes" / "checks",
        "split_by_file":  True,
        "recurse":        True,
        "note":           "bridgecrewio/checkov: per-check PASSED/FAILED Kubernetes fixtures",
    },
    # --- wrongsecrets: real multi-resource K8s app, secrets-exposure focused
    #     (adds clean/isolated diversity, not a chain source) ---
    {
        "cluster": "wrongsecrets-k8s",
        "dir":     ATTACK_REPOS_DIR / "wrongsecrets" / "k8s",
        "recurse": True,
        "note":    "OWASP/wrongsecrets: base k8s manifests + challenge53 sidecar-secret-theft scenario",
    },
    {
        "cluster": "wrongsecrets-aws",
        "dir":     ATTACK_REPOS_DIR / "wrongsecrets" / "aws" / "k8s",
        "recurse": True,
        "note":    "OWASP/wrongsecrets: EKS deployment variant",
    },
    {
        "cluster": "wrongsecrets-azure",
        "dir":     ATTACK_REPOS_DIR / "wrongsecrets" / "azure" / "k8s",
        "recurse": True,
        "note":    "OWASP/wrongsecrets: AKS deployment variant",
    },
    {
        "cluster": "wrongsecrets-gcp",
        "dir":     ATTACK_REPOS_DIR / "wrongsecrets" / "gcp" / "k8s",
        "recurse": True,
        "note":    "OWASP/wrongsecrets: GKE deployment variant",
    },
    {
        "cluster": "wrongsecrets-okteto",
        "dir":     ATTACK_REPOS_DIR / "wrongsecrets" / "okteto" / "k8s",
        "recurse": True,
        "note":    "OWASP/wrongsecrets: Okteto deployment variant",
    },
    # --- KaiMonkey: included for completeness, but its cft/ directory is
    #     CloudFormation (AWS::*), not Kubernetes — expected to contribute
    #     zero rows via the "no workload resources found" skip path.
    {
        "cluster": "kaimonkey",
        "dir":     ATTACK_REPOS_DIR / "KaiMonkey" / "cft",
        "recurse": True,
        "note":    "accurics/KaiMonkey: CloudFormation fixtures, not Kubernetes — expected no-op",
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


def _missing_dir_sentinel(defn: dict, prefix: str, base_dir: Path) -> list[dict]:
    """
    Placeholder cluster for a source directory that doesn't exist (e.g. a repo
    that failed to clone). Reported by main()'s normal "directory not found"
    skip path instead of vanishing silently from the ingestion summary.
    """
    return [{**defn, "cluster": prefix, "dir": base_dir}]


def _expand_split_by_subdir(defn: dict) -> list[dict]:
    """One cluster per immediate subdirectory of `dir`."""
    base_dir: Path = defn["dir"]
    prefix: str = defn["cluster_prefix"]
    fixtures_sub: str | None = defn.get("subdir_fixtures")
    recurse: bool = defn.get("recurse", True)
    note: str = defn.get("note", "")

    if not base_dir.exists():
        return _missing_dir_sentinel(defn, prefix, base_dir)

    result: list[dict] = []
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


def _slugify_relpath(rel: Path) -> str:
    """
    Turn a path relative to a source dir into a collision-safe cluster-name
    slug. Doubles literal underscores in each path segment before joining
    segments with a single underscore, so distinct paths can never collide
    after slugging — a naive '/' -> '_' replacement would map both
    'foo_bar/baz.yaml' and 'foo/bar_baz.yaml' to 'foo_bar_baz', silently
    merging two unrelated fixtures into one cluster (see review discussion,
    audit/model_fixes.md option 3 rationale).
    """
    segments = rel.with_suffix("").as_posix().split("/")
    return "_".join(segment.replace("_", "__") for segment in segments)


def _expand_split_by_file(defn: dict) -> list[dict]:
    """One cluster per YAML file inside `dir` (optionally recursive).

    Used for fixture directories where each file is already a self-contained
    example and grouping by subdirectory would spuriously combine unrelated
    examples into one cluster — e.g. scanner PASSED/FAILED fixture pairs that
    test an unrelated, narrow config setting but happen to share unrelated
    escape flags baked into their shared base template, which would trip the
    graph-level ">=2 escape nodes" chain rule despite representing no real
    escalation chain (see audit/model_fixes.md, option 3). A single-node
    cluster structurally can't satisfy that rule, which is exactly the fix.
    """
    base_dir: Path = defn["dir"]
    prefix: str = defn["cluster_prefix"]
    recurse: bool = defn.get("recurse", False)
    note: str = defn.get("note", "")

    if not base_dir.exists():
        return _missing_dir_sentinel(defn, prefix, base_dir)

    result: list[dict] = []
    for ypath in find_yamls(base_dir, recurse):
        rel_stem = _slugify_relpath(ypath.relative_to(base_dir))
        result.append({
            "cluster": f"{prefix}_{rel_stem}",
            "dir":     ypath.parent,
            "files":   [ypath],
            "recurse": False,
            "note":    f"{note}: {rel_stem}",
        })
    return result


def expand_clusters(cluster_defs: list[dict]) -> list[dict]:
    """
    Expand split_by_subdir/split_by_file entries into one cluster definition
    each. Regular entries are returned unchanged.
    """
    result: list[dict] = []
    for defn in cluster_defs:
        if defn.get("split_by_file"):
            result.extend(_expand_split_by_file(defn))
        elif defn.get("split_by_subdir"):
            result.extend(_expand_split_by_subdir(defn))
        else:
            result.append(defn)
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

    # A cluster's grouping mode (regular/split_by_subdir/split_by_file) or
    # naming can change between runs (e.g. this file's gatekeeper/kubeaudit
    # migration from split_by_subdir to split_by_file). The skip-if-known
    # check below is keyed on repo_name, so rows left behind under an old
    # naming scheme won't be recognised as "already in dataset" and would
    # be silently duplicated under new names on the next run. Surface that
    # instead of letting it happen quietly.
    current_cluster_names = {c["cluster"] for c in flat_clusters}
    orphaned_repos = sorted({
        r["repo_name"] for r in existing_rows
        if r.get("source") == "attack_repos" and r["repo_name"] not in current_cluster_names
    })
    if orphaned_repos:
        logger.warning(
            "%d repo_name(s) already in rf_dataset.csv no longer match any "
            "current CLUSTERS definition — stale from a prior grouping/naming "
            "scheme, or intentionally removed: %s. If a source's split mode "
            "changed, remove its old rows before re-running or duplicates "
            "will be ingested under the new names.",
            len(orphaned_repos),
            ", ".join(orphaned_repos[:20]) + (" ..." if len(orphaned_repos) > 20 else ""),
        )

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

        yamls = cluster_def.get("files") or find_yamls(cluster_dir, cluster_def["recurse"])
        if not yamls:
            print(f"  [skip] {cluster_name}: no YAML files found")
            continue

        cluster_rows: list[dict] = []
        n_skipped = 0
        for ypath in yamls:
            try:
                feats = extract_features_from_file(ypath)
            except Exception:
                logger.warning("%s: unparseable — skipping", ypath, exc_info=True)
                n_skipped += 1
                continue
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
