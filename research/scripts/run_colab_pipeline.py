"""
run_colab_pipeline.py
======================
Orchestrates the research training pipeline on a remote Colab CPU session,
launching each stage as a detached process on the VM and downloading its
artifacts locally the moment they are written.

Rationale: a single long-lived `colab exec` call that blocks for the whole
duration of a stage (e.g. ~70 minutes of 5-fold GNN training) has, in
practice, coincided with the remote session being reclaimed shortly after
the call returns -- losing every artifact that wasn't already downloaded.
Detaching each stage and polling it with short, frequent `colab exec` /
`colab download` calls means a session loss costs at most the in-flight
stage, not the whole run.

Preconditions (see research/notebooks/colab_reproduce.ipynb):
  - The named Colab session already exists (`colab new -s <name>`).
  - The kubescan monorepo is cloned to /content/kubescan on the VM.
  - `kubescan/` is pip-installed editable and the kernel has been
    restarted since (so the .pth file is picked up).
  - torch (CPU wheel), torch-geometric, scikit-learn, skops, networkx,
    pyyaml are installed on the VM.

Usage:
  python research/scripts/run_colab_pipeline.py --session kubescan-repro4
  python research/scripts/run_colab_pipeline.py --session my-session \\
      --stages gnn_cv,ga_ensemble,test_evaluation --poll-interval 15
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

DEFAULT_POLL_INTERVAL_SECONDS: Final[int] = 15
DEFAULT_EXEC_TIMEOUT_SECONDS: Final[int] = 60
LAUNCH_SETTLE_SECONDS: Final[float] = 2.0
EXEC_TIMEOUT_BUFFER_SECONDS: Final[int] = 30
MAX_CONSECUTIVE_EXEC_FAILURES: Final[int] = 3
EXEC_RETRY_BACKOFF_SECONDS: Final[float] = 5.0
LAUNCH_MARKER: Final[str] = "LAUNCHED_PID"
MAX_STAGE_SECONDS: Final[int] = 6 * 60 * 60
FAILURE_LOG_TAIL_CHARS: Final[int] = 2000
FAILURE_SCAN_CARRY_CHARS: Final[int] = 64

REMOTE_REPO_ROOT: Final[str] = "/content/kubescan"
REMOTE_CHECKPOINTS_DIR: Final[str] = f"{REMOTE_REPO_ROOT}/research/models/checkpoints"
REMOTE_LOG_DIR: Final[str] = "/content"

DEFAULT_SEED: Final[int] = 42
_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _SCRIPT_DIR.parent.parent
DEFAULT_DEST_DIR: Final[Path] = _REPO_ROOT / "research" / "models" / "checkpoints"


class ColabPipelineError(Exception):
    """Raised when a remote stage fails or its state cannot be verified."""


class ColabExecTimeout(ColabPipelineError):
    """Raised when a single `colab exec` call hangs past its hard deadline."""


class ColabExecFailed(ColabPipelineError):
    """Raised when `colab exec` returns without hanging but exits non-zero."""


class RemoteState(TypedDict):
    log: str
    exit_code: int | None
    process_alive: bool
    existing_sizes: dict[str, int]


@dataclass(frozen=True)
class PipelineStage:
    name: str
    remote_command: str
    log_filename: str
    failure_pattern: str
    progress_pattern: str
    expected_artifacts: tuple[str, ...] = ()


def build_stages(seed: int) -> tuple[PipelineStage, ...]:
    gnn_fold_checkpoints = tuple(f"gnn_fold_{i}.pt" for i in range(5))
    return (
        PipelineStage(
            name="augment_graphs",
            remote_command=f"python3 -u research/scripts/03_augment/augment_graphs.py --seed {seed}",
            log_filename="augment.log",
            failure_pattern=r"Traceback|Error",
            progress_pattern=r"^===|Done",
        ),
        PipelineStage(
            name="build_graph_cache",
            remote_command="python3 -u research/scripts/04_build_datasets/build_graph_cache.py",
            log_filename="build_cache.log",
            failure_pattern=r"Traceback|Error",
            progress_pattern=r"^===|Done",
        ),
        PipelineStage(
            name="create_splits",
            remote_command=f"python3 -u research/scripts/05_split/create_splits.py --seed {seed}",
            log_filename="splits.log",
            failure_pattern=r"Traceback|Error",
            progress_pattern=r"^===|Done",
        ),
        PipelineStage(
            name="train_rf",
            remote_command=f"python3 -u research/models/train_rf.py --seed {seed}",
            log_filename="train_rf.log",
            failure_pattern=r"Traceback|Error",
            progress_pattern=r"F1|Accuracy",
            expected_artifacts=("rf_model.skops", "rf_severity.skops", "rf_results.json"),
        ),
        PipelineStage(
            name="gnn_cv",
            remote_command=(
                f"python3 -u research/models/train_gnn.py --cv-folds 5 --epochs 300 "
                f"--hidden 64 --heads 4 --layers 3 --seed {seed}"
            ),
            log_filename="gnn_train.log",
            failure_pattern=r"Traceback|Killed|OOM",
            progress_pattern=r"^  Fold [0-9]+:|Fold [0-9]+ result|CROSS-VALIDATION SUMMARY",
            # gnn_best.pt is only written by train_gnn.py's fixed-split branch
            # (--cv-folds 0); it is never produced with --cv-folds 5 above, so
            # it is deliberately excluded here.
            expected_artifacts=gnn_fold_checkpoints + ("gnn_config.json", "cv_results.json"),
        ),
        PipelineStage(
            name="ga_ensemble",
            remote_command=f"python3 -u research/models/run_ga_ensemble.py --oof --seed {seed}",
            log_filename="ga_ensemble.log",
            failure_pattern=r"Traceback|Killed|OOM",
            progress_pattern=r"Generation|Best fitness",
            expected_artifacts=("ga_weights.json", "ga_results.json"),
        ),
        PipelineStage(
            name="test_evaluation",
            remote_command=f"python3 -u research/models/evaluate_test_set.py --show-rankings --seed {seed}",
            log_filename="evaluate_test.log",
            failure_pattern=r"Traceback|Killed|OOM",
            progress_pattern=r"F1|Precision",
            expected_artifacts=("test_results.json",),
        ),
        PipelineStage(
            name="snapshot_manifest",
            remote_command="python3 -u research/scripts/snapshot_run_manifest.py",
            log_filename="manifest.log",
            failure_pattern=r"Traceback|Error",
            progress_pattern=r"^===|written",
            expected_artifacts=("run_manifest.json",),
        ),
    )


class ColabPipelineRunner:
    def __init__(
        self,
        session: str,
        dest_dir: Path,
        poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
        exec_timeout: int = DEFAULT_EXEC_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._dest_dir = dest_dir
        self._poll_interval = poll_interval
        self._exec_timeout = exec_timeout
        self._dest_dir.mkdir(parents=True, exist_ok=True)

    def run(self, stages: tuple[PipelineStage, ...]) -> None:
        for stage in stages:
            print(f"=== STAGE START: {stage.name} ===", flush=True)
            self._launch_stage(stage)
            self._poll_stage(stage)
            print(f"=== STAGE DONE: {stage.name} ===", flush=True)
        print("=== PIPELINE COMPLETE ===", flush=True)

    def _run_colab_cli(self, args: list[str], code: str | None = None) -> str:
        """Run one `colab` CLI invocation with a hard Python-side deadline.

        `colab exec --timeout` only bounds remote code execution; the local
        CLI's own connection can still hang indefinitely on a dead session
        (observed directly: a poll loop blocked ~54 minutes on one hung call
        before the session was noticed to be gone). `subprocess.run(timeout=)`
        guarantees this call itself cannot block the caller forever. A fast
        (non-hanging) nonzero exit is also treated as a failure rather than
        silently returning whatever partial/garbage stdout came back.
        """
        hard_timeout = self._exec_timeout + EXEC_TIMEOUT_BUFFER_SECONDS
        try:
            result = subprocess.run(
                args, input=code, capture_output=True, text=True, timeout=hard_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ColabExecTimeout(f"{args[1]} hung past {hard_timeout}s") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise ColabExecFailed(f"{args[1]} exited {result.returncode}: {detail}")
        return result.stdout

    def _exec(self, code: str) -> str:
        return self._run_colab_cli(
            ["colab", "exec", "-s", self._session, "--timeout", str(self._exec_timeout)], code,
        )

    def _exec_resilient(self, code: str) -> str:
        """Retry transient exec failures (hangs or fast nonzero exits); give
        up after too many in a row. Safe to retry unconditionally because
        every snippet run through this method is idempotent -- in particular
        `_launch_stage`'s snippet checks a remote lock file before spawning
        anything, so a retry after a lost response never double-launches."""
        failures = 0
        while True:
            try:
                return self._exec(code)
            except (ColabExecTimeout, ColabExecFailed) as exc:
                failures += 1
                print(
                    f"  WARNING: exec failed ({failures}/{MAX_CONSECUTIVE_EXEC_FAILURES}): {exc}",
                    flush=True,
                )
                if failures >= MAX_CONSECUTIVE_EXEC_FAILURES:
                    raise ColabPipelineError(
                        f"exec failed {failures} times in a row, giving up",
                    ) from exc
                time.sleep(EXEC_RETRY_BACKOFF_SECONDS)

    def _download(self, filename: str) -> bool:
        """Download one artifact. Only called for filenames already confirmed
        to exist remotely, so a nonzero return code here is a genuine error
        (auth, path, quota) rather than "not written yet" and is surfaced.

        A local file that already exists is re-validated, not trusted
        outright -- it may be a stale/partial leftover from an interrupted
        earlier run targeting the same --dest."""
        local_path = self._dest_dir / filename
        if local_path.exists():
            if self._is_complete(local_path):
                return True
            local_path.unlink()
        remote_path = f"{REMOTE_CHECKPOINTS_DIR}/{filename}"
        try:
            self._run_colab_cli(["colab", "download", "-s", self._session, remote_path, str(local_path)])
        except ColabExecTimeout:
            print(f"  WARNING: download of {filename} timed out, will retry", flush=True)
            return False
        except ColabExecFailed as exc:
            print(f"  WARNING: download of {filename} failed: {exc}", flush=True)
            return False
        if not (local_path.exists() and local_path.stat().st_size > 0):
            return False
        if not self._is_complete(local_path):
            local_path.unlink()
            return False
        return True

    def _is_complete(self, local_path: Path) -> bool:
        """Guard against a download racing an in-progress write on the VM.

        These scripts write output via plain `open(path, "w")` + `json.dump`
        or `torch.save`/`skops.dump`, not write-then-atomic-rename, so a file
        can exist and be non-empty while only partially written. JSON files
        are validated by parsing; every file type additionally has to pass
        the two-consecutive-polls size-stability check in `_stable_artifacts`
        before a download is even attempted, which catches binary files too.
        """
        if local_path.suffix == ".json":
            try:
                json.loads(local_path.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError):
                return False
        return True

    def _remote_paths(self, stage: PipelineStage) -> tuple[str, str, str]:
        remote_log = f"{REMOTE_LOG_DIR}/{stage.log_filename}"
        return remote_log, f"{remote_log}.exit", f"{remote_log}.pid"

    def _launch_stage(self, stage: PipelineStage) -> None:
        """Launch a stage via a remote snippet that is safe to retry: it
        first checks a PID lock file and, if a live process already holds
        it, reports that PID instead of spawning a second instance. Only
        when no live process is found does it clear stale state (log, exit
        flag, lock) from a previous run and start a fresh one."""
        remote_log, remote_exit_flag, remote_lock = self._remote_paths(stage)
        snippet = (
            "import subprocess, os\n"
            f"lock_path = {remote_lock!r}\n"
            "already_running = False\n"
            "old_pid = None\n"
            "if os.path.exists(lock_path):\n"
            "    try:\n"
            "        old_pid = int(open(lock_path).read().strip())\n"
            "        os.kill(old_pid, 0)\n"
            "        already_running = True\n"
            "    except (ValueError, ProcessLookupError, PermissionError):\n"
            "        already_running = False\n"
            "if already_running:\n"
            f"    print({LAUNCH_MARKER!r}, old_pid)\n"
            "else:\n"
            f"    for stale in ({remote_log!r}, {remote_exit_flag!r}, lock_path):\n"
            "        if os.path.exists(stale):\n"
            "            os.remove(stale)\n"
            f"    log = open({remote_log!r}, 'wb')\n"
            "    p = subprocess.Popen(\n"
            f"        ['bash', '-c', {stage.remote_command!r} + '; echo $? > ' + {remote_exit_flag!r}],\n"
            f"        cwd={REMOTE_REPO_ROOT!r}, stdin=subprocess.DEVNULL,\n"
            "        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,\n"
            "    )\n"
            "    open(lock_path, 'w').write(str(p.pid))\n"
            f"    print({LAUNCH_MARKER!r}, p.pid)\n"
        )
        output = self._exec_resilient(snippet)
        if LAUNCH_MARKER not in output:
            raise ColabPipelineError(f"{stage.name}: launch failed:\n{output}")
        time.sleep(LAUNCH_SETTLE_SECONDS)

    def _poll_stage(self, stage: PipelineStage) -> None:
        remote_log, remote_exit_flag, remote_lock = self._remote_paths(stage)
        offset_bytes = 0
        grabbed: set[str] = set()
        stable_sizes: dict[str, int] = {}
        failure_carry = ""
        started_at = time.monotonic()
        while True:
            if time.monotonic() - started_at > MAX_STAGE_SECONDS:
                raise ColabPipelineError(f"{stage.name}: exceeded {MAX_STAGE_SECONDS}s wall-clock limit")
            pending = [f for f in stage.expected_artifacts if f not in grabbed]
            state = self._poll_remote_state(remote_log, remote_exit_flag, remote_lock, pending, offset_bytes)
            new_text = state["log"]
            offset_bytes += len(new_text.encode("utf-8"))
            self._echo_progress(stage, new_text)
            ready = self._stable_artifacts(state["existing_sizes"], stable_sizes)
            self._grab_ready_artifacts(stage, ready, grabbed)
            scan_text = failure_carry + new_text
            if re.search(stage.failure_pattern, scan_text, re.IGNORECASE):
                raise ColabPipelineError(f"{stage.name}: failure detected:\n{scan_text[-FAILURE_LOG_TAIL_CHARS:]}")
            failure_carry = scan_text[-FAILURE_SCAN_CARRY_CHARS:]
            if state["exit_code"] is not None:
                if state["exit_code"] != 0:
                    raise ColabPipelineError(
                        f"{stage.name}: exited with code {state['exit_code']}:"
                        f"\n{scan_text[-FAILURE_LOG_TAIL_CHARS:]}",
                    )
                self._finalize_stage(stage, remote_log, remote_exit_flag, remote_lock, offset_bytes, grabbed)
                return
            if not state["process_alive"]:
                raise ColabPipelineError(
                    f"{stage.name}: remote process is no longer running and never wrote an exit code",
                )
            time.sleep(self._poll_interval)

    def _stable_artifacts(self, current_sizes: dict[str, int], stable_sizes: dict[str, int]) -> list[str]:
        """A filename is only "ready" once its remote size is nonzero and
        unchanged across two consecutive polls -- these writer scripts don't
        write via temp-file + atomic rename, so a file can exist mid-write
        for any format, not just JSON. `stable_sizes` is mutated in place to
        remember this poll's sizes for the next comparison."""
        ready = [f for f, size in current_sizes.items() if size > 0 and stable_sizes.get(f) == size]
        stable_sizes.clear()
        stable_sizes.update(current_sizes)
        return ready

    def _poll_remote_state(
        self,
        remote_log: str,
        remote_exit_flag: str,
        remote_lock: str,
        pending: list[str],
        offset_bytes: int,
    ) -> RemoteState:
        """One round trip: fetch only log bytes written since `offset_bytes`
        (not the whole file -- re-reading from byte 0 every 15s made poll
        payloads grow quadratically over a ~70-minute stage), the exit code
        if the process has finished, whether the launched process is still
        alive (via the PID lock file `_launch_stage` writes), and the size
        of each `pending` artifact that already exists on the VM."""
        snippet = (
            "import subprocess, os, json\n"
            f"log_bytes = subprocess.run(['tail', '-c', '+{offset_bytes + 1}', {remote_log!r}], "
            "capture_output=True).stdout\n"
            "log_text = log_bytes.decode('utf-8', errors='replace')\n"
            "exit_code = None\n"
            f"if os.path.exists({remote_exit_flag!r}):\n"
            "    try:\n"
            f"        exit_code = int(open({remote_exit_flag!r}).read().strip())\n"
            "    except ValueError:\n"
            "        pass  # flag file mid-write, treat as not-yet-finished\n"
            "process_alive = False\n"
            f"if os.path.exists({remote_lock!r}):\n"
            "    try:\n"
            f"        pid = int(open({remote_lock!r}).read().strip())\n"
            "        os.kill(pid, 0)\n"
            "        process_alive = True\n"
            "    except (ValueError, ProcessLookupError, PermissionError):\n"
            "        process_alive = False\n"
            "existing_sizes = {}\n"
            f"for f in {pending!r}:\n"
            f"    p = os.path.join({REMOTE_CHECKPOINTS_DIR!r}, f)\n"
            "    if os.path.exists(p):\n"
            "        existing_sizes[f] = os.path.getsize(p)\n"
            "print(json.dumps({'log': log_text, 'exit_code': exit_code, "
            "'process_alive': process_alive, 'existing_sizes': existing_sizes}))\n"
        )
        output = self._exec_resilient(snippet)
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise ColabPipelineError(f"malformed poll response:\n{output[-FAILURE_LOG_TAIL_CHARS:]}") from exc

    def _finalize_stage(
        self,
        stage: PipelineStage,
        remote_log: str,
        remote_exit_flag: str,
        remote_lock: str,
        offset_bytes: int,
        grabbed: set[str],
    ) -> None:
        """Process already exited cleanly (exit_code == 0): one last check
        catches anything written between the previous poll and exit. With no
        writer left alive there is no more write-race to guard against, so
        any remaining pending file that exists at all is complete. Any
        artifact still missing after that is a hard failure, not a warning
        -- a downstream stage may depend on it."""
        pending = [f for f in stage.expected_artifacts if f not in grabbed]
        if pending:
            state = self._poll_remote_state(remote_log, remote_exit_flag, remote_lock, pending, offset_bytes)
            self._echo_progress(stage, state["log"])
            ready = [f for f, size in state["existing_sizes"].items() if size > 0]
            self._grab_ready_artifacts(stage, ready, grabbed)
        missing = [f for f in stage.expected_artifacts if f not in grabbed]
        if missing:
            raise ColabPipelineError(f"{stage.name}: exited cleanly but never produced: {missing}")

    def _echo_progress(self, stage: PipelineStage, new_text: str) -> None:
        for line in new_text.splitlines():
            if re.search(stage.progress_pattern, line):
                print(f"  [{stage.name}] {line}", flush=True)

    def _grab_ready_artifacts(
        self, stage: PipelineStage, ready: list[str], grabbed: set[str],
    ) -> None:
        for filename in ready:
            if filename in grabbed:
                continue
            if self._download(filename):
                grabbed.add(filename)
                print(f"  [{stage.name}] grabbed {filename}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Colab session name (colab new -s NAME)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST_DIR,
                         help="Local directory to download artifacts into")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--exec-timeout", type=int, default=DEFAULT_EXEC_TIMEOUT_SECONDS)
    parser.add_argument("--stages", type=str, default=None,
                         help="Comma-separated subset of stage names to run, in order")
    return parser.parse_args()


def select_stages(all_stages: tuple[PipelineStage, ...], names: str | None) -> tuple[PipelineStage, ...]:
    if names is None:
        return all_stages
    wanted = names.split(",")
    by_name = {stage.name: stage for stage in all_stages}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ColabPipelineError(f"Unknown stage(s): {unknown}. Known: {list(by_name)}")
    return tuple(by_name[name] for name in wanted)


def main() -> None:
    args = parse_args()
    stages = select_stages(build_stages(args.seed), args.stages)
    runner = ColabPipelineRunner(
        session=args.session,
        dest_dir=args.dest,
        poll_interval=args.poll_interval,
        exec_timeout=args.exec_timeout,
    )
    try:
        runner.run(stages)
    except ColabPipelineError as exc:
        print(f"PIPELINE FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
