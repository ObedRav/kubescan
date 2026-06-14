"""
patch_hostpath_column.py
=========================
One-time script: adds HOSTPATH_MOUNT column to rf_dataset.csv.

For rows where a local YAML file is accessible, parses volumes[] to detect
non-docker-sock hostPath mounts. For rows without YAML access, defaults to 0.

This makes HOSTPATH_MOUNT a proper node feature in the GNN (index 24 in the
25-dim binary feature vector, with risk_score at index 25 → NODE_FEATURE_DIM=26).

Usage:
  python scripts/patch_hostpath_column.py
  python scripts/patch_hostpath_column.py --dry-run
"""

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from yaml_feature_extractor import _check_hostpath_mount

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


def _load_docs(path: Path) -> list:
    docs = []
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for doc in yaml.safe_load_all(f):
                if doc and isinstance(doc, dict):
                    docs.append(doc)
    except Exception:
        pass
    return docs


def detect_hostpath(yaml_path: str) -> int:
    """Return 1 if the YAML file has any non-docker-sock hostPath volume."""
    path = Path(yaml_path)
    if not path.exists():
        return 0
    docs = _load_docs(path)
    for doc in docs:
        kind = str(doc.get("kind", ""))
        spec = doc.get("spec", {}) or {}

        # Get pod spec depending on workload kind
        if kind == "Pod":
            pod_spec = spec
        elif kind == "CronJob":
            pod_spec = (spec.get("jobTemplate", {})
                           .get("spec", {})
                           .get("template", {})
                           .get("spec", {})) or {}
        else:
            pod_spec = spec.get("template", {}).get("spec", {}) or {}

        if not pod_spec:
            continue
        if _check_hostpath_mount(pod_spec):
            return 1
    return 0


def _extract_repo_relpath(path: str):
    patterns = [
        r"GITHUB_REPOS(?:_NODEPLOY)?/([^/]+)/(.+)",
        r"GITLAB(?:_K8S_REPOS_RAW_UNFILTERED|_REPOS)/([^/]+)/(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, path)
        if m:
            return m.group(1), m.group(2)
    return None


def _build_gitlab_lookup(project_root: Path) -> dict[str, str]:
    """
    Build a mapping: rf_dataset yaml_path → local file path, for GitLab rows.

    Chain:
      rf_dataset yaml_path (GITLAB_REPOS/<repo>/<rel>)
        → GITLAB-URLS.csv YAML_PATH (GITLAB_K8S_REPOS_RAW_UNFILTERED/<repo>/<rel>)
        → GITLAB-URLS.csv YAML_URL
        → download_manifest yaml_url → local_path
    """
    gitlab_urls_csv = project_root / "original-dataset" / "rahman" / "DATASET" / "GITLAB-URLS.csv"
    dl_manifest_csv = project_root / "data" / "raw" / "rahman" / "download_manifest.csv"

    if not gitlab_urls_csv.exists() or not dl_manifest_csv.exists():
        return {}

    # Step 1: relpath (after GITLAB_K8S_REPOS_RAW_UNFILTERED/) → YAML_URL
    relpath_to_url: dict[str, str] = {}
    with open(gitlab_urls_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = re.search(r"GITLAB_K8S_REPOS_RAW_UNFILTERED/(.+)", row.get("YAML_PATH", ""))
            if m:
                relpath_to_url[m.group(1)] = row["YAML_URL"]

    # Step 2: YAML_URL → local_path (from download manifest)
    url_to_local: dict[str, str] = {}
    with open(dl_manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("source") == "gitlab":
                url_to_local[row["yaml_url"]] = row["local_path"]

    # Step 3: rf yaml_path → local_path
    result: dict[str, str] = {}
    for relpath, url in relpath_to_url.items():
        local = url_to_local.get(url)
        if local and Path(local).exists():
            # rf_dataset yaml_path uses GITLAB_REPOS prefix
            rf_path_key = f"GITLAB_REPOS/{relpath}"
            result[rf_path_key] = local

    return result


def main():
    project_root = Path(__file__).parent.parent.parent
    default_csv  = project_root / "data" / "tabular" / "rf_dataset.csv"
    downloads_dir = project_root / "dataset" / "downloads"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rf-dataset", type=Path, default=default_csv)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Build url→local_path lookup from downloads manifest if available
    manifest_ok: dict[str, str] = {}
    key_to_url:  dict[tuple, str] = {}
    dl_manifest = downloads_dir / "manifest.csv" if downloads_dir.exists() else None
    if dl_manifest and dl_manifest.exists():
        with open(dl_manifest, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    manifest_ok[row["url"]] = row["local_path"]
                    rp = _extract_repo_relpath(row.get("yaml_path", ""))
                    if rp:
                        key_to_url[rp] = row["url"]

    # Build GitLab lookup via GITLAB-URLS.csv chain
    gitlab_lookup = _build_gitlab_lookup(project_root)
    print(f"GitLab lookup: {len(gitlab_lookup)} paths resolved to local files")

    print(f"Loading {args.rf_dataset}...")
    with open(args.rf_dataset, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    if "HOSTPATH_MOUNT" in fieldnames:
        print("  HOSTPATH_MOUNT column already exists — updating values only")
    else:
        # Insert after UNTRUSTED_REGISTRY
        if "UNTRUSTED_REGISTRY" in fieldnames:
            idx = fieldnames.index("UNTRUSTED_REGISTRY") + 1
            fieldnames.insert(idx, "HOSTPATH_MOUNT")
        else:
            fieldnames.append("HOSTPATH_MOUNT")
        print(f"  Added HOSTPATH_MOUNT column at position {fieldnames.index('HOSTPATH_MOUNT')}")

    n_set = n_skip = n_gitlab = 0
    for row in rows:
        yaml_path = row.get("yaml_path", "")
        local = None

        # Direct local path (badpods, kubernetes_goat)
        if Path(yaml_path).exists():
            local = yaml_path
        else:
            # Try GitHub download manifest chain
            rp = _extract_repo_relpath(yaml_path)
            if rp:
                url = key_to_url.get(rp)
                if url:
                    local = manifest_ok.get(url)

        # Try GitLab lookup chain (GITLAB-URLS.csv → download_manifest)
        if not local and row.get("source") == "gitlab":
            m = re.search(r"GITLAB_REPOS/(.+)", yaml_path)
            if m:
                local = gitlab_lookup.get(f"GITLAB_REPOS/{m.group(1)}")
                if local:
                    n_gitlab += 1

        if local:
            val = detect_hostpath(local)
            row["HOSTPATH_MOUNT"] = str(val)
            if val:
                n_set += 1
        else:
            row.setdefault("HOSTPATH_MOUNT", "0")
            n_skip += 1

    print(f"  HOSTPATH_MOUNT=1: {n_set} rows")
    print(f"  Resolved via GitLab lookup: {n_gitlab} rows")
    print(f"  HOSTPATH_MOUNT=0 (no YAML):  {n_skip} rows")

    if args.dry_run:
        print("[dry-run] Not writing.")
        return

    with open(args.rf_dataset, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Written: {args.rf_dataset}")


if __name__ == "__main__":
    main()
