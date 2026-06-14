"""
rf_classifier.py
================
Inference-only Random Forest wrapper.

Loads rf_model.skops (binary classifier) and exposes predict_risk_scores().
Keeps the same derived-feature logic as research/models/train_rf.py so that
features extracted by yaml_parser map correctly to the RF input.

Serialization: skops is the primary format because it validates types at load
time instead of executing arbitrary pickle bytecode. A legacy rf_model.pkl is
still accepted with a warning — pickle checkpoints must only ever be loaded
from a trusted source, since unpickling runs arbitrary code.
"""
from __future__ import annotations

__all__ = ["RFClassifier"]

import logging
import pickle
from pathlib import Path

import numpy as np
from skops.io import get_untrusted_types
from skops.io import load as skops_load

from ..exceptions import ModelLoadError
from ..utils.yaml_parser import FEATURE_COLS

logger = logging.getLogger(__name__)

# Feature layout — must exactly match research/models/train_rf.py
_RAHMAN_FEATURES: list[str] = [
    "TRUE_HOST_PID", "TRUE_HOST_IPC", "TRUE_HOST_NET", "DOCKERSOCK_PATH",
    "CAP_SYS_ADMIN", "CAP_SYS_MODULE", "WITHIN_MANIFEST_SECRET",
    "SEC_CONT_OVER_PRIVIL", "ALLOW_PRIVI", "INSECURE_HTTP",
    "NO_SECU_CONTEXT", "HOST_ALIAS", "NO_DEFAULT_NSPACE",
    "NO_RESO", "NO_ROLLING_UPDATE",
]
_EXTENDED_FEATURES: list[str] = [
    "NO_RUN_AS_NON_ROOT", "NO_READ_ONLY_ROOT_FS", "IMAGE_USES_LATEST",
    "SA_AUTOMOUNT_TOKEN", "USES_DEFAULT_SA", "UNTRUSTED_REGISTRY", "HOSTPATH_MOUNT",
]
_ALL_RF_FEATURES: list[str] = [
    *_RAHMAN_FEATURES,
    "cap_misuse", "all_secrets", "total_misconfigs",
    *_EXTENDED_FEATURES,
]

# All extractor columns — single source of truth: yaml_parser.FEATURE_COLS
_EXTRACTOR_COLS: list[str] = list(FEATURE_COLS)


def _compute_derived_features(feats: dict[str, int | float]) -> dict[str, int]:
    """Compute cap_misuse, all_secrets, total_misconfigs from raw feature dict."""
    cap_misuse       = int(feats.get("CAP_SYS_ADMIN", 0)) | int(feats.get("CAP_SYS_MODULE", 0))
    all_secrets      = int(feats.get("WITHIN_MANIFEST_SECRET", 0)) | int(feats.get("VALID_TAINT_SECRET", 0))
    total_misconfigs = sum(int(feats.get(c, 0)) for c in _EXTRACTOR_COLS)
    return {
        "cap_misuse":       cap_misuse,
        "all_secrets":      all_secrets,
        "total_misconfigs": total_misconfigs,
    }


def _feats_to_rf_vec(feats: dict[str, int | float]) -> np.ndarray:
    """Map yaml_parser output dict → 25-dim RF input vector."""
    merged = {**feats, **_compute_derived_features(feats)}
    return np.array(
        [float(merged.get(col, 0) or 0) for col in _ALL_RF_FEATURES],
        dtype=np.float32,
    )


class RFClassifier:
    """
    Wraps the trained Random Forest for inference.

    predict_risk_scores(feats_list) → list of float
        Each float is the RF's probability that the manifest is misconfigured (class 1).
        This becomes the risk_score node feature (index 25) in the graph.
    """

    def __init__(self, model_path: Path) -> None:
        if model_path.suffix == ".skops":
            trusted = get_untrusted_types(file=model_path)
            self._model = skops_load(model_path, trusted=trusted)
        else:
            logger.warning(
                "Loading RF from pickle (%s) — pickle executes arbitrary code on "
                "load; only use checkpoints from a trusted source. Prefer .skops.",
                model_path,
            )
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
        logger.debug("RF model loaded from %s", model_path)

    def predict_risk_scores(self, feats_list: list[dict[str, int | float]]) -> list[float]:
        """Return per-manifest risk scores in [0, 1]."""
        X     = np.stack([_feats_to_rf_vec(f) for f in feats_list])
        proba = self._model.predict_proba(X)
        return proba[:, 1].tolist()

    @classmethod
    def from_checkpoints(cls, checkpoints_dir: Path) -> RFClassifier:
        skops_path  = Path(checkpoints_dir) / "rf_model.skops"
        pickle_path = Path(checkpoints_dir) / "rf_model.pkl"
        if skops_path.exists():
            return cls(skops_path)
        if pickle_path.exists():
            return cls(pickle_path)
        raise ModelLoadError(
            skops_path,
            "rf_model.skops not found — run research/models/train_rf.py first.",
        )
