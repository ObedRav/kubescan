"""
test_cli_scan.py
================
Integration tests for the kubescan CLI scan command.

Runs the full pipeline (extract → RF → graph → GNN ensemble → GA scorer)
once against the trained checkpoints in checkpoints/trained, then asserts on
the cached JSON report. Skipped only when the checkpoints are absent
(e.g. fresh clone before training).
"""
from __future__ import annotations

__all__: list[str] = []

import json
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from kubescan.cli import main
from kubescan.model.ga_ensemble import LABEL_NAMES

# Prefer the committed CI fixtures (tiny, deterministic, always present) over the
# large trained checkpoints, so this suite runs in CI. Set KUBESCAN_CHECKPOINTS
# to point at real checkpoints for a higher-fidelity local run.
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "checkpoints"
_TRAINED  = Path(__file__).parents[2] / "checkpoints" / "trained"
_CHECKPOINTS = _FIXTURES if (_FIXTURES / "ga_weights.json").exists() else _TRAINED

pytestmark = pytest.mark.skipif(
    not (_CHECKPOINTS / "ga_weights.json").exists(),
    reason="no checkpoints found — run kubescan/tests/fixtures/make_fixtures.py",
)

_CLEAN_MANIFEST = textwrap.dedent("""\
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: safe-app
      namespace: production
    spec:
      template:
        spec:
          serviceAccountName: safe-sa
          containers:
            - name: app
              image: gcr.io/myproject/app:1.2.3
              resources:
                requests: {cpu: "100m", memory: "128Mi"}
              securityContext:
                runAsNonRoot: true
                readOnlyRootFilesystem: true
                allowPrivilegeEscalation: false
""")

_ATTACK_MANIFEST = textwrap.dedent("""\
    apiVersion: v1
    kind: Pod
    metadata:
      name: bad-pod
    spec:
      hostPID: true
      hostNetwork: true
      containers:
        - name: pwn
          image: ubuntu:latest
          securityContext:
            privileged: true
            capabilities:
              add: ["SYS_ADMIN", "ALL"]
          volumeMounts:
            - name: docker
              mountPath: /var/run/docker.sock
      volumes:
        - name: docker
          hostPath:
            path: /var/run/docker.sock
""")


@pytest.fixture(scope="module")
def scan_report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run `kubescan scan --format json` once; all tests assert on the result."""
    cluster_dir = tmp_path_factory.mktemp("cluster")
    (cluster_dir / "clean.yaml").write_text(_CLEAN_MANIFEST)
    (cluster_dir / "attack.yaml").write_text(_ATTACK_MANIFEST)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["scan", str(cluster_dir), "--format", "json",
         "--checkpoints-dir", str(_CHECKPOINTS)],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.output[result.output.index("{"):])


def test_cli_scan_emits_valid_json_with_verdict(scan_report: dict) -> None:
    assert scan_report["verdict"] in LABEL_NAMES.values()


def test_cli_scan_finds_both_manifests(scan_report: dict) -> None:
    assert scan_report["n_manifests"] == 2


def test_cli_scan_ensemble_score_in_unit_interval(scan_report: dict) -> None:
    assert 0.0 <= scan_report["ensemble_score"] <= 1.0


def test_cli_scan_attack_manifest_scores_above_clean(scan_report: dict) -> None:
    risk = {m["file"]: m["risk_score"] for m in scan_report["manifests"]}
    assert risk["attack.yaml"] > risk["clean.yaml"]


def test_cli_scan_attack_manifest_is_escape_capable(scan_report: dict) -> None:
    flags = {m["file"]: m["escape_capable"] for m in scan_report["manifests"]}
    assert flags["attack.yaml"] is True and flags["clean.yaml"] is False


def test_cli_scan_attack_cluster_not_verdicted_clean(scan_report: dict) -> None:
    # A cluster containing an escape-capable pod must never be reported CLEAN:
    # the binary escape signal alone contributes w_escape to the score.
    assert scan_report["verdict"] != "CLEAN"
