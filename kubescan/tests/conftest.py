"""
conftest.py
===========
Shared pytest fixtures for kubescan tests.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

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


@pytest.fixture()
def clean_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "clean.yaml"
    p.write_text(_CLEAN_MANIFEST)
    return p


@pytest.fixture()
def attack_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "attack.yaml"
    p.write_text(_ATTACK_MANIFEST)
    return p


@pytest.fixture()
def cluster_dir(tmp_path: Path, clean_yaml: Path, attack_yaml: Path) -> Path:
    """Directory containing both a clean and an attack manifest."""
    return tmp_path
