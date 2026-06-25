"""
test_rf_classifier.py
=====================
Unit tests for kubescan/model/rf_classifier.py.
"""
from __future__ import annotations

__all__: list[str] = []

from pathlib import Path

import pytest

from kubescan.exceptions import ModelLoadError
from kubescan.model.rf_classifier import (
    RFClassifier,
    _validate_skops_types,
)

# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------

def test_validate_skops_types_accepts_sklearn_prefix() -> None:
    _validate_skops_types(Path("dummy.skops"), frozenset({"sklearn.ensemble._forest.RandomForestClassifier"}))


def test_validate_skops_types_accepts_numpy_prefix() -> None:
    _validate_skops_types(Path("dummy.skops"), frozenset({"numpy.core.multiarray"}))


def test_validate_skops_types_rejects_unsafe_prefix() -> None:
    with pytest.raises(ModelLoadError, match="unsafe types"):
        _validate_skops_types(Path("dummy.skops"), frozenset({"os.system"}))


def test_validate_skops_types_rejects_mixed_unsafe() -> None:
    with pytest.raises(ModelLoadError, match="unsafe types"):
        _validate_skops_types(
            Path("dummy.skops"),
            frozenset({"sklearn.ensemble._forest.RandomForestClassifier", "subprocess.Popen"}),
        )


# ---------------------------------------------------------------------------
# predict_risk_scores
# ---------------------------------------------------------------------------

def test_predict_risk_scores_returns_list_of_floats(fixture_checkpoints: Path) -> None:
    rf = RFClassifier.from_checkpoints(fixture_checkpoints)
    feats = [{"TRUE_HOST_PID": 0, "DOCKERSOCK_PATH": 1}]
    scores = rf.predict_risk_scores(feats)
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert isinstance(scores[0], float)


def test_predict_risk_scores_in_unit_interval(fixture_checkpoints: Path) -> None:
    rf = RFClassifier.from_checkpoints(fixture_checkpoints)
    feats = [{"SEC_CONT_OVER_PRIVIL": 1, "ALLOW_PRIVI": 1}]
    scores = rf.predict_risk_scores(feats)
    assert 0.0 <= scores[0] <= 1.0


def test_predict_risk_scores_clean_manifest_low_risk(fixture_checkpoints: Path) -> None:
    rf = RFClassifier.from_checkpoints(fixture_checkpoints)
    clean = [dict.fromkeys(["TRUE_HOST_PID", "CAP_SYS_ADMIN", "WITHIN_MANIFEST_SECRET"], 0)]
    scores = rf.predict_risk_scores(clean)
    assert scores[0] < 0.9


def test_predict_risk_scores_multiple_manifests(fixture_checkpoints: Path) -> None:
    rf = RFClassifier.from_checkpoints(fixture_checkpoints)
    feats = [{"TRUE_HOST_PID": 0}, {"DOCKERSOCK_PATH": 1}]
    scores = rf.predict_risk_scores(feats)
    assert len(scores) == 2


# ---------------------------------------------------------------------------
# from_checkpoints error handling
# ---------------------------------------------------------------------------

def test_from_checkpoints_raises_when_no_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match=r"rf_model\.skops"):
        RFClassifier.from_checkpoints(tmp_path)


def test_pickle_fallback_raises_without_allow_pickle(tmp_path: Path) -> None:
    (tmp_path / "rf_model.pkl").write_bytes(b"fake")
    with pytest.raises(ModelLoadError, match="Refusing"):
        RFClassifier.from_checkpoints(tmp_path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_checkpoints() -> Path:
    return Path(__file__).parent.parent / "fixtures" / "checkpoints"
