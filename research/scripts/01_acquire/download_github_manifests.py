"""
download_github_manifests.py
=============================
Download the actual Kubernetes YAML manifest files referenced in Rahman et al.'s dataset.

The Rahman dataset provides misconfiguration labels but NOT the raw YAML files.
This script fetches them from GitHub (and optionally GitLab) using the commit-pinned
URLs in GITHUB-URLS.csv and GITLAB-URLS.csv.

Why we need the YAMLs
----------------------
  1. GNN graph building requires the full manifest content to extract node/edge structure
     (pod → service account → role → rolebinding relationships)
  2. Running Checkov + kube-linter on the raw YAMLs replaces the unavailable GenKubeSec
     dataset with equivalent multi-tool labels (UMI proxy score)

Output layout
--------------
  dataset/raw/rahman/yamls/
    github/
      <repo_name>/           # one dir per unique repo (= one cluster snapshot)
        <file>.yaml
        ...
    gitlab/
      <repo_name>/
        ...
  dataset/raw/rahman/download_manifest.csv   # log: path, url, status, size_bytes

Rate limiting
--------------
  GitHub API: 60 req/hour unauthenticated, 5000 req/hour with token.
  Set GITHUB_TOKEN env var for authenticated access (strongly recommended).
  Default: 1 request/second, respects Retry-After headers.

Usage
------
  python scripts/download_yamls.py

  Optional flags:
    --data-dir      Path to rahman/DATASET/           (default: auto-detected)
    --out-dir       Where to save YAMLs               (default: dataset/raw/rahman/yamls/)
    --manifest-csv  Path for the download log CSV     (default: dataset/raw/rahman/download_manifest.csv)
    --max-workers   Parallel download threads         (default: 4 with token, 1 without)
    --github-only   Skip GitLab manifests
    --limit N       Download only first N files (useful for testing)
    --retry         Re-attempt previously failed downloads
"""

import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
DEFAULT_DELAY = 1.0   # seconds between requests (unauthenticated)
AUTH_DELAY    = 0.05  # seconds between requests (authenticated — ~20 req/s, well within 5000/hr)
MAX_RETRIES   = 3
REQUEST_TIMEOUT = 30  # seconds

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# URL conversion helpers
# ---------------------------------------------------------------------------

def github_blob_to_raw(url: str) -> str | None:
    """
    Convert a GitHub blob URL to a raw.githubusercontent.com URL.

    Input:  https://github.com/owner/repo/blob/<sha>/path/to/file.yaml
    Output: https://raw.githubusercontent.com/owner/repo/<sha>/path/to/file.yaml
    """
    # Handle already-raw URLs
    if "raw.githubusercontent.com" in url:
        return url
    m = re.match(
        r"https://github\.com/([^/]+/[^/]+)/blob/([0-9a-f]+)/(.+)",
        url,
        re.IGNORECASE,
    )
    if not m:
        return None
    repo, sha, path = m.group(1), m.group(2), m.group(3)
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"


def gitlab_blob_to_raw(url: str, commit: str) -> str | None:
    """
    Convert a GitLab blob URL to a raw file URL.

    Input:  https://gitlab.com/owner/repo/blob/<sha>/path/to/file.yaml
    Output: https://gitlab.com/owner/repo/raw/<sha>/path/to/file.yaml
    """
    m = re.match(
        r"https://gitlab\.com/([^/]+(?:/[^/]+)*)/blob/([0-9a-f]+)/(.+)",
        url,
        re.IGNORECASE,
    )
    if not m:
        return None
    repo_path, sha, file_path = m.group(1), m.group(2), m.group(3)
    return f"https://gitlab.com/{repo_path}/raw/{sha}/{file_path}"


# ---------------------------------------------------------------------------
# Repo name extraction
# ---------------------------------------------------------------------------

def repo_name_from_url(url: str) -> str:
    """Extract 'owner__repo' string from a GitHub or GitLab URL."""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2:
        # Sanitise for filesystem use
        return f"{parts[0]}__{parts[1]}".replace(".", "_")
    return "unknown_repo"


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

class YamlDownloader:

    def __init__(
        self,
        out_dir: Path,
        manifest_csv: Path,
        delay: float,
        max_workers: int,
        retry_failed: bool = False,
    ):
        self.out_dir = out_dir
        self.manifest_csv = manifest_csv
        self.delay = delay
        self.max_workers = max_workers
        self.retry_failed = retry_failed

        self.session = requests.Session()
        if GITHUB_TOKEN:
            self.session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
            log("[auth] Using GitHub token — higher rate limit enabled")
        else:
            log("[auth] No GITHUB_TOKEN found — using unauthenticated (60 req/hr limit)")

        # Load existing manifest to support resume
        self.completed: dict[str, str] = {}  # url → status
        if manifest_csv.exists():
            with open(manifest_csv, newline="") as f:
                for row in csv.DictReader(f):
                    self.completed[row["yaml_url"]] = row["status"]

        self._manifest_lock = threading.Lock()
        self._manifest_file = open(manifest_csv, "a", newline="", encoding="utf-8")
        self._manifest_writer = None  # initialised lazily after header check

        # Write header if file is new
        if manifest_csv.stat().st_size == 0:
            self._manifest_writer = csv.DictWriter(
                self._manifest_file,
                fieldnames=["yaml_url", "raw_url", "local_path", "repo_name", "source", "status", "size_bytes", "error"],
            )
            self._manifest_writer.writeheader()
        else:
            self._manifest_writer = csv.DictWriter(
                self._manifest_file,
                fieldnames=["yaml_url", "raw_url", "local_path", "repo_name", "source", "status", "size_bytes", "error"],
            )

    def _log_result(self, row: dict) -> None:
        with self._manifest_lock:
            self._manifest_writer.writerow(row)
            self._manifest_file.flush()

    def _should_skip(self, yaml_url: str, local_path: Path) -> bool:
        prev_status = self.completed.get(yaml_url)
        if prev_status == "ok" and local_path.exists():
            return True
        if prev_status == "failed" and not self.retry_failed:
            return True
        return False

    def download_one(self, task: dict) -> dict:
        """Download a single YAML file. Returns a result dict."""
        yaml_url = task["yaml_url"]
        raw_url  = task["raw_url"]
        local_path = Path(task["local_path"])
        source   = task["source"]
        repo_name = task["repo_name"]

        result = {
            "yaml_url":   yaml_url,
            "raw_url":    raw_url,
            "local_path": str(local_path),
            "repo_name":  repo_name,
            "source":     source,
            "status":     "failed",
            "size_bytes": 0,
            "error":      "",
        }

        if self._should_skip(yaml_url, local_path):
            result["status"] = "skipped"
            return result

        local_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.get(raw_url, timeout=REQUEST_TIMEOUT)

                if resp.status_code == 200:
                    local_path.write_bytes(resp.content)
                    result["status"] = "ok"
                    result["size_bytes"] = len(resp.content)
                    log(f"  [ok]  {repo_name}/{local_path.name} ({len(resp.content)} bytes)")
                    break

                elif resp.status_code == 404:
                    result["error"] = "404 Not Found"
                    result["status"] = "not_found"
                    log(f"  [404] {raw_url}")
                    break  # Don't retry 404s

                elif resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    log(f"  [rate-limit] Sleeping {retry_after}s...")
                    time.sleep(retry_after)

                elif resp.status_code in (500, 502, 503):
                    wait = 5 * attempt
                    log(f"  [{resp.status_code}] Attempt {attempt}/{MAX_RETRIES}, retry in {wait}s")
                    time.sleep(wait)

                else:
                    result["error"] = f"HTTP {resp.status_code}"
                    log(f"  [err] {raw_url} → HTTP {resp.status_code}")
                    break

            except requests.RequestException as e:
                result["error"] = str(e)
                log(f"  [exc] {raw_url} → {e} (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    time.sleep(5)

        time.sleep(self.delay)
        return result

    def run(self, tasks: list[dict]) -> dict:
        """Execute all download tasks. Returns summary counts."""
        counts = {"ok": 0, "skipped": 0, "not_found": 0, "failed": 0}
        total = len(tasks)

        log(f"\n[download] {total} files to process | workers={self.max_workers} | delay={self.delay}s\n")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.download_one, t): t for t in tasks}
            done = 0
            for future in as_completed(futures):
                done += 1
                result = future.result()
                status = result["status"]
                counts[status] = counts.get(status, 0) + 1
                self._log_result(result)
                if done % 50 == 0 or done == total:
                    log(f"  Progress: {done}/{total} | ok={counts['ok']} skip={counts['skipped']} 404={counts['not_found']} fail={counts['failed']}")

        self._manifest_file.close()
        return counts

    def close(self):
        if not self._manifest_file.closed:
            self._manifest_file.close()


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def build_github_tasks(urls_csv: Path, out_dir: Path, limit: int | None) -> list[dict]:
    tasks = []
    with open(urls_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    for row in rows:
        yaml_url  = row.get("YAML_URL", "").strip()
        if not yaml_url:
            continue

        raw_url = github_blob_to_raw(yaml_url)
        if not raw_url:
            log(f"  [warn] Cannot convert URL: {yaml_url}")
            continue

        repo_name = repo_name_from_url(yaml_url)
        # Derive a clean filename from the blob URL path
        parsed_path = urlparse(yaml_url).path
        # Strip /owner/repo/blob/sha/ prefix
        path_parts = parsed_path.strip("/").split("/")
        if len(path_parts) > 4:
            rel_path = "/".join(path_parts[4:])  # everything after /blob/sha/
        else:
            rel_path = path_parts[-1]

        local_path = out_dir / "github" / repo_name / rel_path

        tasks.append({
            "yaml_url":   yaml_url,
            "raw_url":    raw_url,
            "local_path": str(local_path),
            "repo_name":  repo_name,
            "source":     "github",
        })

    return tasks


def build_gitlab_tasks(urls_csv: Path, out_dir: Path, limit: int | None) -> list[dict]:
    tasks = []
    with open(urls_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if limit:
        rows = rows[:limit]

    for row in rows:
        yaml_url = row.get("YAML_URL", "").strip()
        commit   = row.get("COMMIT", "").strip()
        if not yaml_url:
            continue

        raw_url = gitlab_blob_to_raw(yaml_url, commit)
        if not raw_url:
            log(f"  [warn] Cannot convert GitLab URL: {yaml_url}")
            continue

        repo_name = repo_name_from_url(yaml_url)
        parsed_path = urlparse(yaml_url).path
        path_parts = parsed_path.strip("/").split("/")
        if len(path_parts) > 4:
            rel_path = "/".join(path_parts[4:])
        else:
            rel_path = path_parts[-1]

        local_path = out_dir / "gitlab" / repo_name / rel_path

        tasks.append({
            "yaml_url":   yaml_url,
            "raw_url":    raw_url,
            "local_path": str(local_path),
            "repo_name":  repo_name,
            "source":     "gitlab",
        })

    return tasks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent

    default_data_dir    = project_root / "original-dataset" / "rahman" / "DATASET"
    default_out_dir     = project_root / "data" / "raw" / "rahman" / "yamls"
    default_manifest    = project_root / "data" / "raw" / "rahman" / "download_manifest.csv"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir",     type=Path, default=default_data_dir)
    parser.add_argument("--out-dir",      type=Path, default=default_out_dir)
    parser.add_argument("--manifest-csv", type=Path, default=default_manifest)
    parser.add_argument("--max-workers",  type=int,  default=4 if GITHUB_TOKEN else 1)
    parser.add_argument("--github-only",  action="store_true")
    parser.add_argument("--limit",        type=int,  default=None, help="Download only first N files")
    parser.add_argument("--retry",        action="store_true", help="Retry previously failed downloads")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data directory not found: {args.data_dir}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    delay = AUTH_DELAY if GITHUB_TOKEN else DEFAULT_DELAY

    # Build task lists
    tasks = []
    github_urls = args.data_dir / "GITHUB-URLS.csv"
    if github_urls.exists():
        gh_tasks = build_github_tasks(github_urls, args.out_dir, args.limit)
        log(f"[github] {len(gh_tasks)} tasks queued")
        tasks.extend(gh_tasks)
    else:
        log(f"[warn] GITHUB-URLS.csv not found at {github_urls}")

    if not args.github_only:
        gitlab_urls = args.data_dir / "GITLAB-URLS.csv"
        if gitlab_urls.exists():
            limit = args.limit - len(tasks) if args.limit else None
            gl_tasks = build_gitlab_tasks(gitlab_urls, args.out_dir, limit)
            log(f"[gitlab] {len(gl_tasks)} tasks queued")
            tasks.extend(gl_tasks)

    if not tasks:
        log("No tasks to process. Exiting.")
        sys.exit(0)

    downloader = YamlDownloader(
        out_dir=args.out_dir,
        manifest_csv=args.manifest_csv,
        delay=delay,
        max_workers=args.max_workers,
        retry_failed=args.retry,
    )

    try:
        counts = downloader.run(tasks)
    finally:
        downloader.close()

    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    for status, count in sorted(counts.items()):
        print(f"  {status:12s}: {count}")
    print(f"\n  YAMLs saved to : {args.out_dir}")
    print(f"  Manifest log   : {args.manifest_csv}")
    print("=" * 50)

    if GITHUB_TOKEN:
        print("\nTip: set GITHUB_TOKEN env var for 5000 req/hr (vs 60/hr unauthenticated)")


if __name__ == "__main__":
    main()
