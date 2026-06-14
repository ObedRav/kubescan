"""
test_properties.py
==================
Property-based tests (Hypothesis) for the two components that face untrusted
input: the YAML feature extractor and the cluster graph builder. Unit tests
pin specific cases; these assert invariants over a wide space of generated
inputs — the fuzzing-class bugs (crashes, out-of-range features, malformed
graphs) that example-based tests miss.
"""
from __future__ import annotations

__all__: list[str] = []

from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from kubescan.utils.graph_builder import (
    NODE_FEATURE_DIM,
    build_cluster_graph,
    graph_to_pyg,
)
from kubescan.utils.yaml_parser import FEATURE_COLS, extract_features_from_file

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Arbitrary nested JSON-ish YAML values (dicts/lists/scalars), plus garbage text.
_yaml_scalars = st.one_of(
    st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=20),
)
_yaml_values = st.recursive(
    _yaml_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(min_size=1, max_size=12), children, max_size=4),
    ),
    max_leaves=15,
)


@st.composite
def _pod_manifest(draw) -> str:
    """A Pod manifest with a randomly-populated, possibly-malformed spec."""
    import yaml as _yaml
    spec = draw(st.dictionaries(st.text(min_size=1, max_size=12), _yaml_values, max_size=6))
    doc = {"apiVersion": "v1", "kind": "Pod",
           "metadata": {"name": draw(st.text(min_size=1, max_size=10))},
           "spec": spec}
    return _yaml.safe_dump(doc)


# ---------------------------------------------------------------------------
# yaml_parser invariants
# ---------------------------------------------------------------------------

@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.text(max_size=400))
def test_extractor_never_crashes_on_arbitrary_text(tmp_path_factory, raw: str) -> None:
    p: Path = tmp_path_factory.mktemp("p") / "x.yaml"
    p.write_text(raw)
    # Must return a dict or None — never raise — on any byte soup.
    result = extract_features_from_file(p)
    assert result is None or isinstance(result, dict)


@settings(max_examples=120, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(_pod_manifest())
def test_extractor_features_are_binary_and_complete(tmp_path_factory, manifest: str) -> None:
    p: Path = tmp_path_factory.mktemp("p") / "pod.yaml"
    p.write_text(manifest)
    result = extract_features_from_file(p)
    if result is None:
        return
    for col in FEATURE_COLS:
        assert col in result, f"missing feature {col}"
        assert result[col] in (0, 1), f"{col} not binary: {result[col]!r}"


# ---------------------------------------------------------------------------
# graph_builder invariants
# ---------------------------------------------------------------------------

def _feat_dict(flags: dict[str, int]) -> dict[str, object]:
    return {**dict.fromkeys(FEATURE_COLS, 0), **flags}


@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    n=st.integers(min_value=1, max_value=12),
    flag_data=st.data(),
)
def test_graph_to_pyg_well_formed(tmp_path_factory, n: int, flag_data) -> None:
    base = tmp_path_factory.mktemp("cluster")
    feats_list, risk_scores, yaml_paths = [], [], []
    for i in range(n):
        active = flag_data.draw(st.lists(st.sampled_from(FEATURE_COLS), max_size=5))
        feats_list.append(_feat_dict(dict.fromkeys(active, 1)))
        risk_scores.append(flag_data.draw(st.floats(min_value=0.0, max_value=1.0)))
        yaml_paths.append(base / f"m{i}.yaml")

    pyg = graph_to_pyg(build_cluster_graph(feats_list, risk_scores, yaml_paths))

    # Node feature matrix has the expected shape and finite values.
    assert pyg.x.shape == (n, NODE_FEATURE_DIM)
    assert bool(np.isfinite(pyg.x.numpy()).all())
    # edge_index is a valid [2, E] COO list referencing existing nodes.
    assert pyg.edge_index.shape[0] == 2
    assert pyg.edge_index.shape[1] == pyg.edge_attr.shape[0]
    if pyg.edge_index.numel():
        assert int(pyg.edge_index.max()) < n
        assert int(pyg.edge_index.min()) >= 0
    # batch vector marks every node as belonging to the single graph.
    assert pyg.batch.shape == (n,)
    assert int(pyg.batch.max()) == 0
