"""
scan_security_tools.py
=======================
Run Checkov + kube-linter on all downloaded YAML files and merge results into
rf_dataset.csv as new feature columns.

This replaces the unavailable GenKubeSec dataset with equivalent multi-tool
validation, adding:
  - checkov_failed_count   : number of Checkov checks failed (out of ~90 K8s checks)
  - checkov_passed_count   : number of Checkov checks passed
  - checkov_umi_score      : failed / (failed + passed) — normalized severity index
  - kl_failed_count        : number of kube-linter checks failed
  - [per-check flags]      : binary columns for specific Checkov check IDs that
                             fill feature gaps (see CHECKOV_FEATURE_MAP below)

Gap-filling Checkov checks (map check ID → new feature column):
  CKV_K8S_30 / CKV_K8S_40  →  checkov_no_run_as_nonroot
  CKV_K8S_22                →  checkov_no_readonly_rootfs
  CKV_K8S_39                →  checkov_image_latest
  CKV_K8S_35                →  checkov_sa_automount
  CKV_K8S_36                →  checkov_default_sa
  CKV_K8S_15                →  checkov_untrusted_registry
  CKV_K8S_31                →  checkov_no_seccomp       (fills SECCOMP_UNCONFINED gap)

Usage:
  python scripts/scan_with_tools.py

  Optional:
    --yamls-dir      Base directory of downloaded YAMLs
    --rf-dataset     Path to rf_dataset.csv
    --urls-csv       Path to GITHUB-URLS.csv (for path→URL mapping)
    --manifest-csv   Path to download_manifest.csv (URL→local_path mapping)
    --workers N      Parallel Checkov processes (default: 4)
    --dry-run        Scan without writing to rf_dataset.csv
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_print_lock = threading.Lock()

def log(msg):
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Checkov check IDs → new dataset feature columns (gap-filling)
# ---------------------------------------------------------------------------
CHECKOV_FEATURE_MAP: dict[str, str] = {
    "CKV_K8S_30": "checkov_no_run_as_nonroot",   # runAsUser != 0
    "CKV_K8S_40": "checkov_no_run_as_nonroot",   # runAsNonRoot not set (same feature, either check)
    "CKV_K8S_22": "checkov_no_readonly_rootfs",  # readOnlyRootFilesystem
    "CKV_K8S_39": "checkov_image_latest",         # image uses latest/no tag
    "CKV_K8S_35": "checkov_sa_automount",         # SA token automount
    "CKV_K8S_36": "checkov_default_sa",           # uses default SA
    "CKV_K8S_15": "checkov_untrusted_registry",   # untrusted image registry
    "CKV_K8S_31": "checkov_no_seccomp",           # seccomp not set (fills SECCOMP_UNCONFINED)
    "CKV_K8S_32": "checkov_no_apparmor",          # AppArmor not set
    "CKV_K8S_20": "checkov_allow_privi_esc",      # allowPrivilegeEscalation (validates ALLOW_PRIVI)
    "CKV_K8S_6":  "checkov_allow_privi_esc",      # allowPrivilegeEscalation (older check ID)
    "CKV_K8S_37": "checkov_caps_not_dropped",     # capabilities not dropped
}

# All new columns added to rf_dataset.csv by this script
NEW_COLUMNS = [
    "checkov_failed_count",
    "checkov_passed_count",
    "checkov_umi_score",
    "kl_failed_count",
    "checkov_no_run_as_nonroot",
    "checkov_no_readonly_rootfs",
    "checkov_image_latest",
    "checkov_sa_automount",
    "checkov_default_sa",
    "checkov_untrusted_registry",
    "checkov_no_seccomp",
    "checkov_no_apparmor",
    "checkov_allow_privi_esc",
    "checkov_caps_not_dropped",
    "has_yaml",   # 1 if the YAML was downloaded, 0 if only Rahman CSV data
]


# ---------------------------------------------------------------------------
# Checkov runner
# ---------------------------------------------------------------------------

def run_checkov(yaml_path: Path) -> dict:
    """
    Run Checkov on a single YAML file and return parsed results.
    Returns a dict with counts and per-check flags.
    """
    result = dict.fromkeys(NEW_COLUMNS, 0)
    result["has_yaml"] = 1

    try:
        proc = subprocess.run(
            ["checkov", "-f", str(yaml_path), "--framework", "kubernetes",
             "-o", "json", "--quiet", "--compact"],
            capture_output=True, text=True, timeout=30
        )
        if not proc.stdout.strip():
            return result

        data = json.loads(proc.stdout)
        scan_results = data.get("results", {})
        passed = scan_results.get("passed_checks", [])
        failed = scan_results.get("failed_checks", [])

        result["checkov_passed_count"] = len(passed)
        result["checkov_failed_count"] = len(failed)
        total = len(passed) + len(failed)
        result["checkov_umi_score"] = round(len(failed) / total, 4) if total > 0 else 0.0

        # Map specific failed checks to feature columns
        for check in failed:
            check_id = check.get("check_id", "")
            col = CHECKOV_FEATURE_MAP.get(check_id)
            if col:
                result[col] = 1

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        pass

    return result


def run_kube_linter(yaml_path: Path) -> int:
    """Run kube-linter and return the number of failed checks."""
    try:
        proc = subprocess.run(
            ["kube-linter", "lint", str(yaml_path)],
            capture_output=True, text=True, timeout=30
        )
        # kube-linter outputs one line per error
        error_lines = [line for line in proc.stdout.splitlines() if "check:" in line.lower()]
        return len(error_lines)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Path mapping: rf_dataset yaml_path → downloaded local path
# ---------------------------------------------------------------------------

def _extract_repo_relpath(path: str, source: str) -> tuple[str, str] | None:
    """
    Extract (repo_name, relative_path) from a Rahman dataset path.
    Handles two path conventions used by different researchers in the dataset.

    FINAL-COUNT (arahman's machine):  .../GITHUB_REPOS[_NODEPLOY]/<repo>/<relpath>
    URLS.csv    (akondrahman's machine): .../GITHUB_K8S_REPOS_RAW_UNFILTERED/<repo>/<relpath>
    GitLab (both):                     .../GITLAB_K8S_REPOS_RAW_UNFILTERED/<repo>/<relpath>
    """
    if source in ("count", "rf_dataset"):
        patterns = [
            r"GITHUB_REPOS(?:_NODEPLOY)?/([^/]+)/(.+)",
            r"GITLAB_K8S_REPOS_RAW_UNFILTERED/([^/]+)/(.+)",
        ]
    else:  # urls_csv
        patterns = [
            r"GITHUB_K8S_REPOS_RAW_UNFILTERED/([^/]+)/(.+)",
            r"GITLAB_K8S_REPOS_RAW_UNFILTERED/([^/]+)/(.+)",
        ]
    for pat in patterns:
        m = re.search(pat, path)
        if m:
            return m.group(1), m.group(2)
    return None


def build_path_lookup(
    urls_csv: Path,
    manifest_csv: Path,
) -> dict[str, str]:
    """
    Build a map: rf_dataset.yaml_path → downloaded local_path.

    The FINAL-COUNT and URLS.csv use different path prefixes (different researcher
    machines), but the (repo_name, relative_path) tuple is consistent across both.

    Join chain:
      rf_dataset.yaml_path
        → (repo, relpath)  via GITHUB_REPOS/ prefix extraction
        → yaml_url         via URLS.csv keyed on (repo, relpath) from GITHUB_K8S_REPOS_RAW_UNFILTERED/
        → local_path       via download_manifest.csv keyed on yaml_url
    """
    # Step 1: (repo, relpath) → yaml_url  (from GITHUB/GITLAB-URLS.csv)
    key_to_url: dict[tuple, str] = {}
    for csv_path in [urls_csv, urls_csv.parent / "GITLAB-URLS.csv"]:
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                orig_path = row.get("YAML_PATH", "").strip()
                yaml_url  = row.get("YAML_URL", "").strip()
                result = _extract_repo_relpath(orig_path, "urls_csv")
                if result and yaml_url:
                    key_to_url[result] = yaml_url

    # Step 2: yaml_url → local_path  (from download_manifest.csv)
    url_to_local: dict[str, str] = {}
    if manifest_csv.exists():
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url   = row.get("yaml_url", "").strip()
                local = row.get("local_path", "").strip()
                if url and local and row.get("status") == "ok":
                    url_to_local[url] = local

    # Step 3: rf_dataset.yaml_path → (repo, relpath) → yaml_url → local_path
    orig_to_local: dict[str, str] = {}
    for key, url in key_to_url.items():
        local = url_to_local.get(url)
        if local:
            # Reconstruct an artificial "original path" key using the GITHUB_REPOS convention
            # that rf_dataset uses. We build it as: GITHUB_REPOS/<repo>/<relpath>
            # This is matched against rf_dataset rows using the same extraction.
            orig_to_local[("GITHUB_REPOS/" + "/".join(key))] = local
            orig_to_local[("GITLAB_K8S_REPOS_RAW_UNFILTERED/" + "/".join(key))] = local

    return key_to_url, url_to_local


def resolve_local_path(yaml_path: str, key_to_url: dict, url_to_local: dict) -> str | None:
    """Resolve a rf_dataset yaml_path to its downloaded local path."""
    result = _extract_repo_relpath(yaml_path, "rf_dataset")
    if not result:
        return None
    url = key_to_url.get(result)
    if not url:
        return None
    return url_to_local.get(url)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent

    default_rf       = project_root / "data" / "tabular" / "rf_dataset.csv"
    default_data_dir = project_root / "original-dataset" / "rahman" / "DATASET"
    default_manifest = project_root / "data" / "raw" / "rahman" / "download_manifest.csv"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rf-dataset",   type=Path, default=default_rf)
    parser.add_argument("--urls-csv",     type=Path, default=default_data_dir / "GITHUB-URLS.csv")
    parser.add_argument("--manifest-csv", type=Path, default=default_manifest)
    parser.add_argument("--workers",      type=int,  default=4)
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    # Build path lookup
    print("Building path lookup (original path → downloaded local path)...")
    key_to_url, url_to_local = build_path_lookup(args.urls_csv, args.manifest_csv)
    print(f"  URL keys indexed: {len(key_to_url)}")
    print(f"  Downloaded files: {len(url_to_local)}")

    # Load dataset
    print(f"Loading {args.rf_dataset}...")
    with open(args.rf_dataset, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"  {len(rows)} rows")

    # Initialize new columns on all rows
    for row in rows:
        for col in NEW_COLUMNS:
            if col not in row:
                row[col] = ""
        row["has_yaml"] = "0"  # default until we scan it

    # Build tasks: only rows that have a downloaded YAML and haven't been scanned
    tasks = []
    for i, row in enumerate(rows):
        orig_path = row.get("yaml_path", "")
        local = resolve_local_path(orig_path, key_to_url, url_to_local)
        if local and Path(local).exists():
            tasks.append((i, Path(local)))

    print(f"  {len(tasks)} rows have downloaded YAMLs available for scanning")
    print(f"  {len(rows) - len(tasks)} rows will get has_yaml=0 (download pending)")

    if not tasks:
        print("No YAMLs to scan. Run download_yamls.py first.")
        sys.exit(0)

    # Run Checkov + kube-linter in parallel
    print(f"\nScanning with Checkov + kube-linter ({args.workers} workers)...")
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(lambda t=t: (t[0], run_checkov(t[1]), run_kube_linter(t[1]))): t for t in tasks}
        for future in as_completed(future_map):
            row_idx, checkov_res, kl_failed = future.result()
            row = rows[row_idx]
            for col, val in checkov_res.items():
                row[col] = val
            row["kl_failed_count"] = kl_failed
            row["has_yaml"] = "1"
            done += 1
            if done % 50 == 0 or done == len(tasks):
                log(f"  Scanned {done}/{len(tasks)}")

    # Summary
    has_yaml = sum(1 for r in rows if str(r["has_yaml"]) == "1")
    avg_umi  = sum(float(r["checkov_umi_score"] or 0) for r in rows if str(r["has_yaml"])=="1") / max(has_yaml, 1)
    avg_kl   = sum(int(r["kl_failed_count"] or 0) for r in rows if str(r["has_yaml"])=="1") / max(has_yaml, 1)

    print(f"\n{'='*60}")
    print("SCAN SUMMARY")
    print(f"{'='*60}")
    print(f"  Rows scanned       : {has_yaml}")
    print(f"  Rows without YAML  : {len(rows) - has_yaml}")
    print(f"  Avg Checkov UMI    : {avg_umi:.3f}")
    print(f"  Avg kube-linter    : {avg_kl:.1f} failed checks")

    # Show gap-fill effectiveness
    print("\n  Gap-fill check coverage (among scanned rows):")
    scanned = [r for r in rows if str(r["has_yaml"])=="1"]
    for col in ["checkov_no_run_as_nonroot","checkov_no_readonly_rootfs",
                "checkov_image_latest","checkov_sa_automount",
                "checkov_default_sa","checkov_no_seccomp","checkov_no_apparmor"]:
        cnt = sum(1 for r in scanned if str(r.get(col,"0"))=="1")
        pct = 100*cnt/max(len(scanned),1)
        print(f"    {col:35s}: {cnt} ({pct:.1f}%)")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # Write updated CSV
    fieldnames = list(rows[0].keys())
    with open(args.rf_dataset, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved: {args.rf_dataset}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
