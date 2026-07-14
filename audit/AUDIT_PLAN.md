# kubescan — Full Codebase Audit Plan

**Purpose:** Exhaustive correctness, security, and train/inference parity audit of the
entire kubescan repository (~43 Python files). This document is the single source of
truth for the audit: status tracker, execution instructions, known context, and
agent briefing notes.

**Status legend:** `[ ]` Not started · `[~]` In progress · `[x]` Complete · `[!]` Has findings

---

## 0. Read This First (Agent Orientation)

If you are reading this after a context reset, here is everything you need to know
before running any phase.

### 0.1 What is kubescan?

A 3-layer Kubernetes attack-chain risk scanner:

```
Layer 1 — Random Forest (RF)
  Input:  25 binary security flags per manifest  (+ 3 derived = 28 RF features)
  Output: risk_score ∈ [0, 1]  (prob of being misconfigured)

Layer 2 — Graph Attention Network (GAT / KubeGAT)
  Input:  cluster graph, node features = [25 flags | risk_score] → dim 26
  Output: chain_probability ∈ [0, 1]

Layer 3 — GA-optimised ensemble
  score = w_rf · mean_rf_risk + w_gnn · chain_prob + w_escape · escape_signal
  escape_signal is BINARY (1.0 / 0.0) — never the fraction
  Verdict: score ≥ 0.60 → ATTACK_CHAIN · ≥ 0.30 → ISOLATED_MISCONFIG · else CLEAN
```

### 0.2 The three-pillar structure

```
TFE/
├── kubescan/          ← distributable package (pip install -e kubescan/)
│   ├── src/kubescan/
│   │   ├── cli.py                  entry point: kubescan scan <dir>
│   │   ├── exceptions.py           typed error hierarchy
│   │   ├── model/
│   │   │   ├── gat_encoder.py      KubeGAT architecture (SINGLE SOURCE OF TRUTH)
│   │   │   ├── rf_classifier.py    RF wrapper (inference only)
│   │   │   └── ga_ensemble.py      ensemble scorer + escape helpers
│   │   └── utils/
│   │       ├── yaml_parser.py      25-flag extractor (SINGLE SOURCE OF TRUTH)
│   │       └── graph_builder.py    cluster graph builder (SINGLE SOURCE OF TRUTH)
│   └── tests/                      61 tests, all must pass
│
├── research/          ← training pipeline (NOT importable as package)
│   ├── models/
│   │   ├── train_rf.py             Layer 1 training
│   │   ├── train_gnn.py            Layer 2 training (imports KubeGAT from kubescan)
│   │   ├── run_ga_ensemble.py      Layer 3 GA weight optimisation
│   │   ├── evaluate_test_set.py    held-out test evaluation
│   │   ├── predict.py              research-side end-to-end inference
│   │   └── provenance.py           checkpoint metadata logging
│   └── scripts/
│       ├── 01_acquire/             download + ingest raw manifests
│       ├── 02_extract/             YAML features + graph construction
│       ├── 03_augment/             attack-chain graph augmentation
│       ├── 04_build_datasets/      assemble RF and GNN datasets
│       ├── 05_split/               stratified splits + 5-fold CV
│       └── fixes/                  one-off data patches (never in main pipeline)
│
└── audit/             ← THIS directory
    └── AUDIT_PLAN.md  ← YOU ARE HERE
```

### 0.3 The cardinal rule of this codebase

**The same logical feature must be extracted identically in two places:**

| Logic | Training (research) | Inference (kubescan) |
|-------|--------------------|-----------------------|
| 25 YAML flags | `02_extract/extract_yaml_features.py` | `kubescan/utils/yaml_parser.py` |
| Cluster graph | `02_extract/build_graphs.py` | `kubescan/utils/graph_builder.py` |
| KubeGAT model | `train_gnn.py` imports from kubescan | `kubescan/model/gat_encoder.py` |
| RF feature vec | `train_rf.py` | `kubescan/model/rf_classifier.py` |
| Ensemble score | `run_ga_ensemble.py` | `kubescan/model/ga_ensemble.py` |

Any divergence between a training implementation and its inference counterpart
is a **silent correctness bug** — the model was optimised for signal A but the
CLI delivers signal B.

### 0.4 Canonical constants (never re-derive; grep for deviations)

```python
# kubescan/utils/yaml_parser.py
FEATURE_COLS: list[str]  # 25 strings — THE feature order for RF + GNN
TRUSTED_REGISTRIES: frozenset[str]  # 8 entries
WORKLOAD_KINDS: frozenset[str]      # 8 entries

# kubescan/utils/graph_builder.py
NODE_FEATURE_DIM = 26     # 25 flags + risk_score at index 25
RISK_SCORE_INDEX = 25
EdgeType(IntEnum): DIR_PROXIMITY=0, PRIV_REACH=1, SA_LATERAL=2, SEMANTIC_NS=3, RBAC_PRIV=4
ESCAPE_FLAGS: frozenset  # 8 flag names that make a node escape-capable
LATERAL_FLAGS: frozenset # 4 flag names that make a node lateral-capable

# kubescan/model/ga_ensemble.py
SCORE_HIGH_THRESHOLD     = 0.60
SCORE_MODERATE_THRESHOLD = 0.30
ESCAPE_FLAG_INDICES: list[int]  # derived from FEATURE_COLS ∩ ESCAPE_FLAGS
LATERAL_FLAG_INDICES: list[int] # derived from FEATURE_COLS ∩ LATERAL_FLAGS
```

### 0.5 Bugs already found and fixed (do not re-report these)

These were confirmed and fixed across PR review rounds 1–4. All are committed on
`feat/first-commit`.

| # | File | Bug | Fix commit |
|---|------|-----|------------|
| 1 | `cli.py` | Checkpoint path off-by-one parent | `cf34720` |
| 2 | `cli.py` | YAMLError swallowed silently | `cf34720` |
| 3 | `yaml_parser.py` | `is False` → `is not True` (4 occurrences) | `cf34720` |
| 4 | `yaml_parser.py` | `NO_SECU_CONTEXT`: invert logic (any_missing not all_missing) | `cf34720` |
| 5 | `yaml_parser.py` | `NO_RESO`: only checked resources key, not limits specifically | `cf34720` |
| 6 | `run_ga_ensemble.py` | `mask.cpu()` before indexing GPU tensor (device mismatch) | `cf34720` |
| 7 | `extract_yaml_features.py` | Local TRUSTED_REGISTRIES had extra entry vs canonical | `cf34720` |
| 8 | `ga_ensemble.py` | Module docstring said escape_fraction not escape_signal | `cf34720` |
| 9 | `ga_ensemble.py` | Thresholds were private; predict.py duplicated literals | `cf34720` |
| 10 | `predict.py` | escape_fraction passed to ensemble instead of escape_signal | `cf34720` |
| 11 | `predict.py` | Verdict used chain_prob thresholds not ensemble_score | `cf34720` |
| 12 | `predict.py` | Node table sort preserved wrong escape index | `cf34720` |
| 13 | `predict.py` | Default weights hardcoded as literals (0.36, 0.64) | `9b2fcfd` |
| 14 | `cli.py` | `timeout=30` magic number | `9b2fcfd` |
| 15 | `yaml_parser.py` | Pod-level `runAsNonRoot=True` not inherited by container check | `19da4ea` |
| 16 | `yaml_parser.py` | INSECURE_HTTP: probe-only vs recursive spec scan | `19da4ea` |
| 17 | `yaml_parser.py` | NO_ROLLING_UPDATE fires on CronJob/Job/Pod (no kind guard) | `19da4ea` |
| 18 | `yaml_parser.py` | `strategy: {}` (falsy empty dict) wrongly flags NO_ROLLING_UPDATE | `19da4ea` |
| 19 | `predict.py` | ESCAPE_FLAG_INDICES hardcoded literal, not imported | `19da4ea` |
| 20 | `predict.py` | PRIV_REACH edges guarded by has_edge(); graph_builder overwrites | `19da4ea` |
| 21 | `predict.py` | Local `graph_to_pyg()` duplicated; now imported from graph_builder | `19da4ea` |
| 22 | `rf_classifier.py` | `get_untrusted_types()` passed directly as trusted= (security) | `19da4ea` |

### 0.6 Running the test suite and linter

```bash
# From TFE/
cd kubescan && python -m pytest tests/ -q          # must show 61 passed
cd .. && ruff check kubescan/src kubescan/tests research/models research/scripts
```

Both must be clean before committing any finding's fix.

---

## 1. Audit Tracker

### Phase 0 — Automated baseline

| Task | Status | Output file | Notes |
|------|--------|-------------|-------|
| `mypy --strict` on kubescan/ | `[x]` | `audit/phase0/mypy.txt` | 37 errors in 6 files |
| `ruff --select ALL` (broad lint) | `[x]` | `audit/phase0/ruff_strict.txt` | 334 violations (51 auto-fixable) |
| `pytest --cov --cov-report=term-missing` | `[x]` | `audit/phase0/coverage.txt` | 61 passed, 79% total coverage |
| `radon cc` complexity report | `[x]` | `audit/phase0/complexity.txt` | avg C (12.37); 2 functions rated F |

---

### Phase 1 — Package core (8 agents, run in parallel)

| Unit | File(s) | Status | Finding count | Notes |
|------|---------|--------|--------------|-------|
| A1 | `kubescan/utils/yaml_parser.py` | `[!]` | 12 (6C 4H 2M) | WITHIN_MANIFEST_SECRET completely wrong semantics |
| A2 | `kubescan/utils/graph_builder.py` | `[!]` | 4 (2C 1H 1M) | RBAC edge logic diverges from training |
| A3 | `kubescan/model/ga_ensemble.py` | `[!]` | 4 (0C 1H 3L) | crash on empty fold_models |
| A4 | `kubescan/model/gat_encoder.py` | `[!]` | 5 (0C 1H 2M 2L) | load_fold_ensemble incomplete param surface |
| A5 | `kubescan/model/rf_classifier.py` | `[!]` | 4 (1C 0H 2M 1L) | total_misconfigs sums 25 flags vs training 18 |
| A6 | `kubescan/cli.py` | `[!]` | 3 (0C 2H 1L) | inference pipeline not wrapped in try/except |
| A7 | `kubescan/exceptions.py` + `utils/device_utils.py` | `[!]` | 5 (0C 2H 2M 1L) | KubescanError not exported; torch import unsafe |
| A8 | All 8 test files | `[!]` | 15 (0C 7H 5M 3L) | no test_rf_classifier.py, no test_gat_encoder.py |

---

### Phase 2 — Research models (5 agents, run in parallel)

| Unit | File(s) | Status | Finding count | Notes |
|------|---------|--------|--------------|-------|
| B1 | `research/models/train_rf.py` | `[!]` | 8 (2C 0H 3M 3L) | 3 Rahman flags silently dropped from X matrix |
| B2 | `research/models/train_gnn.py` | `[!]` | 6 (0C 1H 4M 1L) | checkpoint has no arch config — shape mismatch on load |
| B3 | `research/models/run_ga_ensemble.py` | `[!]` | 6 (0C 0H 3M 3L) | escape signal computed inline (DRY); bare open() ×2 |
| B4 | `research/models/evaluate_test_set.py` | `[!]` | 6 (2C 1H 2M 1L) | **P@5 formula uses /k not /min(k,n_chains)** |
| B5 | `research/models/predict.py` + `provenance.py` | `[!]` | 9 (0C 0H 4M 5L) | run_gnn_ensemble still local; 3 other DRY re-impls |

---

### Phase 3 — Research pipeline (6 agents, run in parallel)

| Unit | File(s) | Status | Finding count | Notes |
|------|---------|--------|--------------|-------|
| C1 | `01_acquire/download_github_manifests.py` + `ingest_attack_repos.py` | `[!]` | 11 (2C 3H 4M 2L) | Fixture repos contain contradictory labels; path traversal |
| C2 | `02_extract/extract_yaml_features.py` | `[!]` | 15 (6C 2H 5M 2L) | 8 NO-match flags in full parity table; 7 PARTIAL |
| C3 | `02_extract/build_graphs.py` + `scan_security_tools.py` | `[!]` | 10 (1C 2H 6M 1L) | HOSTPATH_MOUNT node feature not updated; stale docstrings |
| C4 | `04_build_datasets/*.py` (4 files) | `[!]` | 14 (4C 4H 5M 1L) | total_misconfigs confirmed; Goat labeling wrong; non-atomic cache |
| C5 | `05_split/create_splits.py` | `[!]` | 8 (0C 1H 3M 4L) | val.txt double-dipping inflates val metrics |
| C6 | `03_augment/augment_graphs.py` + `fixes/patch_hostpath_column.py` | `[!]` | 5 (0C 1H 2M 2L) | patch crashes on import (wrong sys.path) |

---

### Phase 4 — Cross-cutting parity (7 agents, run in parallel)

| Unit | Scope | Status | Finding count | Notes |
|------|-------|--------|--------------|-------|
| X1 | Feature flag parity (yaml_parser vs extract_yaml_features, all 25) | `[!]` | 10 (2C 4H 4M) | Parity table: 8 NO / 7 PARTIAL / 10 YES confirmed |
| X2 | Graph construction parity (graph_builder vs build_graphs, all 5 edge types) | `[!]` | 7 (1C 2H 2M 2L) | Confirmed F-A2-001, F-A2-002, F-C3-001; edge table complete |
| X3 | KubeGAT architecture parity (gat_encoder vs train_gnn call sites) | `[!]` | 7 (0C 3H 2M 2L) | Checkpoint key format consistent; config JSON absent |
| X4 | Security audit (subprocess, pickle, eval, yaml.load, shell=True, path concat) | `[!]` | 5 (0C 2H 3M) | No CRITICAL; 2 pickle.load HIGH; bare open() ×3; all torch.load use weights_only=True |
| X5 | Data leakage audit (can augmented graphs reach clean test splits?) | `[!]` | 5 (0C 1H 2M 2L) | RF split manifest-level → risk_score leakage; aug exclusion confirmed correct |
| X6 | Seed / reproducibility (all three RNG libraries seeded before use?) | `[!]` | 11 (1C 5H 4M 1L) | evaluate_test_set.py hardcoded seed; set_global_seed() missing; workers unseeded |
| X7 | Checkpoint format contract (save key names = load key names?) | `[!]` | 3 (0C 0H 0M 3L) | Format consistent across all readers/writers; all torch.load use weights_only=True |

---

### Phase 5 — Synthesis

| Task | Status | Output |
|------|--------|--------|
| Collect all phase 1–4 findings | `[x]` | Sourced from §5 of this file (198 raw findings) |
| Deduplicate | `[x]` | Merged per cluster rules; 198 → 50 unique findings |
| Classify by severity and group by file | `[x]` | `audit/findings_final.md` |
| Identify which fixes require model retraining | `[x]` | Section "Findings requiring model retraining" in `audit/findings_final.md` |

---

## 2. Phase Execution Instructions

### How to run Phase 0 (automated baseline)

```bash
# Mypy
mypy kubescan/src --strict --ignore-missing-imports 2>&1 | tee audit/phase0/mypy.txt

# Ruff strict (broadened rule set)
ruff check kubescan/src kubescan/tests research/models research/scripts \
  --select ALL \
  --ignore ANN,D,ERA,FBT,TRY,RUF,N,EM,G,T20,S101,PLR2004,SLF001,PD,NPY \
  2>&1 | tee audit/phase0/ruff_strict.txt

# Coverage
cd kubescan && python -m pytest tests/ -q \
  --cov=kubescan --cov-report=term-missing 2>&1 | tee ../audit/phase0/coverage.txt
cd ..

# Radon complexity (flag CC > 10)
radon cc kubescan/src research/models research/scripts -s -a \
  --min B 2>&1 | tee audit/phase0/complexity.txt
```

---

### How to brief each Phase 1 / 2 / 3 agent

Use this template. Fill in `{UNIT}`, `{FILE}`, and the unit-specific questions.

```
You are performing a deep correctness audit of a Kubernetes attack-chain scanner
called kubescan. Read the context below, then read the target file(s) completely
(every line), and return all findings.

=== CODEBASE CONTEXT ===

Architecture: 3-layer ensemble (RF → GAT → GA weights).
Canonical feature order: FEATURE_COLS (25 flags) defined in yaml_parser.py.
Node feature vector: 26-dim (FEATURE_COLS + risk_score at index 25).
Edge types: DIR_PROXIMITY=0, PRIV_REACH=1, SA_LATERAL=2, SEMANTIC_NS=3, RBAC_PRIV=4.
escape_signal: BINARY (1.0/0.0), never a fraction.
Thresholds: SCORE_HIGH=0.60, SCORE_MODERATE=0.30.

=== BUGS ALREADY FIXED (do not re-report) ===
[paste §0.5 table here]

=== AUDIT UNIT: {UNIT} ===
File(s): {FILE}

Working directory: /Users/obedrayo/Documents/UNIR/TFE

=== TASK ===
1. Read {FILE} completely.
2. For each function/class, ask:
   a. Does it do exactly what its name says?
   b. Is the edge-case behaviour correct (empty input, None, zero, falsy-but-non-null)?
   c. Does it violate any rule in §0.4 (wrong constant, missing source-of-truth import)?
3. Unit-specific questions: {UNIT_SPECIFIC_QUESTIONS}
4. Return findings in this format:
   file: <path>
   line: <line number>
   severity: CRITICAL | HIGH | MEDIUM | LOW
   summary: <one sentence>
   failure_scenario: <concrete input → wrong output or crash>

CRITICAL = silent wrong ML output (wrong feature value, wrong model output)
HIGH     = crash or data corruption
MEDIUM   = DRY violation / wrong constant / missing type hint
LOW      = style / CLAUDE.md formatting rule
```

---

### Unit-specific questions (fill into template above)

**A1 — yaml_parser.py**
- Do all 25 extractors match the semantics in `§0.5 canon` and `extract_yaml_features.py`?
- Does `_extract_workload_metadata` guard `NO_ROLLING_UPDATE` with `_ROLLING_UPDATE_KINDS`?
- Does `_spec_has_insecure_http` exclude localhost correctly?
- Does pod-level `runAsNonRoot` inheritance propagate to all containers (not just first)?
- Are there any remaining `is False` patterns that should be `is not True`?
- Does `_extract_file` handle multi-doc YAML files (`---` separator) correctly?

**A2 — graph_builder.py**
- Does `_add_privilege_edges` overwrite (not guard with has_edge) for PRIV_REACH?
- Does `_add_lateral_edges` correctly guard with has_edge (lower priority than PRIV_REACH)?
- Does `graph_to_pyg` produce `edge_attr` of shape `[E, 1]` with dtype `LongTensor`?
- Does `_enrich_with_yaml_semantics` read each file exactly once (no double I/O)?
- Does `ESCAPE_FLAGS` include all 8 correct flag names?
- Do `ESCAPE_FLAG_INDICES` and `LATERAL_FLAG_INDICES` in ga_ensemble.py derive to the correct values given current `FEATURE_COLS`?

**A3 — ga_ensemble.py**
- Does `EnsembleScorer.score()` formula match `run_ga_ensemble.py:compute_objective()`?
- Does `predict_label()` use `SCORE_HIGH_THRESHOLD` / `SCORE_MODERATE_THRESHOLD` (not literals)?
- Is `weights_path.open()` used (not bare `open()`)?
- Does `run_gnn_ensemble()` return `(chain_prob, clean_prob, isolated_prob)` in that exact order?
- Are `compute_escape_fraction` and `compute_escape_signal` both needed, or can one delegate to the other?

**A4 — gat_encoder.py**
- What are the exact `__init__` parameters? (in_channels, hidden, heads, num_layers, num_classes, dropout)
- Does the forward pass handle `edge_attr` as `[E, 1]` LongTensor (embedding lookup)?
- Does the model use `global_mean_pool` or `global_max_pool` or both?
- Is there any training-specific code (`torch.optim`, `nn.Dropout` in training mode) that should not be here per the inference-only boundary?
- Are all dimensions consistent: `in_channels` → hidden → out = 3 classes?

**A5 — rf_classifier.py**
- Does `_feats_to_rf_vec` produce the same 25+3 = 28-dim vector as `train_rf.py`?
- Does `_compute_derived_features` use `FEATURE_COLS` (not a local list) for `total_misconfigs`?
- Does `_validate_skops_types` check prefixes `{"sklearn.", "numpy."}` only?
- Does `from_checkpoints()` prefer `.skops` over `.pkl` correctly?

**A6 — cli.py**
- Does `_resolve_checkpoints()` follow the exact fallback chain in CLAUDE.md?
  (`--checkpoints-dir` → `KUBESCAN_CHECKPOINTS` env → symlink → `CheckpointNotFoundError`)
- Does the subprocess call use `_KUBECTL_TIMEOUT_SECS` everywhere (not a literal)?
- Is `ManifestParseError` raised (not swallowed) for YAMLError from kubectl output?
- Does `EnsembleScorer` receive `escape_signal` (binary) not `escape_fraction`?
- Are all `print()` calls gone from library functions (only in entry point)?

**A7 — exceptions.py + device_utils.py**
- Does `KubescanError` form the base for ALL package errors?
- Does `ModelLoadError` inherit `KubescanError` and accept `(path, reason)`?
- Does `CheckpointNotFoundError` exist and inherit correctly?
- Does `resolve_device()` handle MPS (Apple Silicon) and CUDA and CPU without crashing?

**A8 — Test files**
- Does every public function in yaml_parser, graph_builder, ga_ensemble, gat_encoder, rf_classifier have at least one test?
- Do tests for yaml_parser cover: `strategy: {}`, `runAsNonRoot` pod-level inheritance, `NO_ROLLING_UPDATE` on CronJob, INSECURE_HTTP in env vars?
- Do tests check `compute_escape_signal` vs `compute_escape_fraction` distinction?
- Are any fixtures using wrong/outdated feature counts (e.g., 25 vs 26 dims)?
- Is the integration test (`test_cli_scan.py`) checking that `risk["attack.yaml"] > risk["clean.yaml"]` still valid after the NO_RESO semantics change?

**B1 — train_rf.py**
- Does it use `FEATURE_COLS` from `yaml_parser.py` (not a local list)?
- Does `total_misconfigs` sum all 25 flags (not just 18 Rahman flags)?
- Does it call `set_global_seed()` before any sklearn/numpy randomness?
- Are `cap_misuse` and `all_secrets` derived identically to `rf_classifier._compute_derived_features`?
- Does it save as `.skops` (not `.pkl`) as the primary format?
- Does the CV protocol match the split files in `05_split/` (same cluster IDs, same fold assignments)?

**B2 — train_gnn.py**
- Does it `from kubescan.model.gat_encoder import KubeGAT`? (never redefine it)
- Are `in_channels`, `hidden`, `heads`, `num_layers` consistent with the CLI defaults?
- Does `best_state = {k: v.cpu().clone() ...}` delete the old dict before replacing?
- Does `torch.save` use keys that `gat_encoder.py` / `cli.py` / `predict.py` expect at load time?
- Does the data loader use the group-aware splits from `05_split/`?
- Is `edge_attr` correctly passed to `model(x, edge_index, edge_attr, batch)`?

**B3 — run_ga_ensemble.py**
- Does `compute_objective(w_rf, w_gnn, w_escape)` use the BINARY escape signal, not fraction?
- Does `_infer_dataset` call `compute_escape_signal()` from `ga_ensemble` (not inline)?
- Does the GA weight output JSON contain keys `w_rf`, `w_gnn`, `w_escape`?  (keys consumed by `EnsembleScorer.__init__`)
- Does the comparison table correctly assign weights to the right column labels?
- Does the mask stay on device when indexing `batch.x[mask]`?

**B4 — evaluate_test_set.py**
- Is P@5 computed as: sort clusters by score descending, count attack-chains in top 5, divide by min(5, n_attack_chains)?
- Does it use the held-out test split (never touching val or train)?
- Does it load the same checkpoint format that `run_ga_ensemble.py` writes?
- Are metrics logged with the seed used?

**B5 — predict.py + provenance.py**
- After round-4 fixes: does any `graph_to_pyg` local definition remain?
- Is `ESCAPE_FLAG_INDICES` imported (not hardcoded)?
- Does it import `run_gnn_ensemble` from `ga_ensemble` or still define locally?
- Does `provenance.py` log all relevant parameters (seed, feature count, checkpoint hash)?

**C1 — 01_acquire/**
- How is the attack/clean label assigned per cluster? Is the boundary unambiguous?
- Are there any clusters that could plausibly be both attack and clean?
- Is there deduplication across repos to prevent a cluster from appearing twice?

**C2 — extract_yaml_features.py**
- Produce a parity table: for each of the 25 flags, state the trigger condition in THIS
  file and in `yaml_parser.py`. Flag any divergence.
- Does `_check_no_rolling_update` guard by kind (`deployment|statefulset|daemonset` only)?
- Does `_check_no_run_as_non_root` short-circuit on pod-level `runAsNonRoot=True`?
- Does `_check_insecure_http` scan recursively (not just probes)?
- Does it import `TRUSTED_REGISTRIES` from `yaml_parser` (not redefine it)?

**C3 — build_graphs.py + scan_security_tools.py**
- Does `build_graphs.py` use the same 5 edge type integers as `graph_builder.py`?
- Does PRIV_REACH overwrite DIR_PROXIMITY (no `has_edge` guard)?
- Does `graph_to_pyg` produce `edge_attr` of shape `[E, 1]`?
- Does `dir_key()` use the same depth (DEFAULT_DIR_KEY_DEPTH=2) as `graph_builder.py`?
- Does `scan_security_tools.py` output feed into any training feature? (or is it research-only)

**C4 — 04_build_datasets/**
- Does `enrich_rf_dataset.py` compute `total_misconfigs = sum(flags[c] for c in FEATURE_COLS)`
  (all 25, not 18)?
- Does `build_graph_cache.py` produce `.npz` files with exactly: `x [N,26]`, `edge_index [2,E]`, `y`, `cluster_id`?
- Does `gnn_dataset.py` load `.npz` and produce `Data` objects whose `edge_attr` shape is `[E,1]`?
- Is `risk_score` at node feature index 25 (not 0, not appended differently)?

**C5 — create_splits.py**
- Is the split key the cluster ID (not the individual graph/file ID)?
- Can a cluster appear in both train and test?
- Is stratification by label, not by repo (to avoid domain shift)?
- Are the 5 CV folds written as separate `.txt` files that `train_gnn.py` reads by fold index?

**C6 — augment_graphs.py + patch_hostpath_column.py**
- Does augmentation only touch graphs that are in the training split?
- Are augmented graph IDs distinguishable from original graph IDs (e.g., `_aug_` infix)?
- Does `patch_hostpath_column.py` idempotently update the CSV (safe to run twice)?

**X1 — Feature flag parity (25 flags)**
Read BOTH `yaml_parser.py` AND `extract_yaml_features.py`. For each of the 25 flags
in `FEATURE_COLS`, fill this table:

| Flag | yaml_parser.py trigger | extract_yaml_features.py trigger | Match? |
|------|------------------------|----------------------------------|--------|
| TRUE_HOST_PID | | | |
| TRUE_HOST_IPC | | | |
| TRUE_HOST_NET | | | |
| DOCKERSOCK_PATH | | | |
| CAP_SYS_ADMIN | | | |
| CAP_SYS_MODULE | | | |
| WITHIN_MANIFEST_SECRET | | | |
| SEC_CONT_OVER_PRIVIL | | | |
| ALLOW_PRIVI | | | |
| SECCOMP_UNCONFINED | | | |
| VALID_TAINT_SECRET | | | |
| INSECURE_HTTP | | | |
| NO_SECU_CONTEXT | | | |
| NO_NETWORK_POLICY | | | |
| HOST_ALIAS | | | |
| NO_DEFAULT_NSPACE | | | |
| NO_RESO | | | |
| NO_ROLLING_UPDATE | | | |
| NO_RUN_AS_NON_ROOT | | | |
| NO_READ_ONLY_ROOT_FS | | | |
| IMAGE_USES_LATEST | | | |
| SA_AUTOMOUNT_TOKEN | | | |
| USES_DEFAULT_SA | | | |
| UNTRUSTED_REGISTRY | | | |
| HOSTPATH_MOUNT | | | |

**X2 — Graph construction parity (5 edge types)**
Read BOTH `graph_builder.py` AND `build_graphs.py`. For each edge type, compare:
- Trigger condition (what makes two nodes get this edge)
- Priority (does it overwrite lower-priority edges or guard with has_edge?)
- Direction (directed or bidirectional?)
- `dir_key` depth consistent?
- RBAC privileged-role detection logic identical?

**X3 — KubeGAT architecture parity**
Read `gat_encoder.py` and the `KubeGAT(...)` instantiation in `train_gnn.py`, `cli.py`,
and `predict.py`. All must use the same `in_channels=26`, `hidden`, `heads`,
`num_layers`, `num_classes=3`, `dropout`. Check the `forward()` signature matches
how it is called at every call site.

**X4 — Security audit**
Grep for and read context around:
```bash
grep -rn "subprocess\|pickle\.load\|eval(\|exec(\|yaml\.load[^_]\|shell=True" \
  kubescan/src research/models research/scripts
grep -rn "open(" kubescan/src  # bare open() should be path.open()
grep -rn "\.format(\|f\".*{.*}\|%" kubescan/src  # potential injection in log messages
```
For each hit: Is the input from a trusted source? Could a crafted YAML file reach it?

**X5 — Data leakage**
Read `create_splits.py` and `augment_graphs.py`. Trace:
1. How are cluster IDs assigned?
2. Are augmented graphs tagged with the same cluster ID as originals?
3. Can the same cluster ID appear in train and test?
4. Does augmentation happen before or after splitting?

**X6 — Seed / reproducibility**
Grep for: `random.seed`, `np.random.seed`, `torch.manual_seed`, `set_global_seed`.
- Is `set_global_seed()` called at the start of every training script?
- Is the seed logged at INFO level?
- Are there any numpy/torch random calls that happen before seeding?
- Does `DataLoader(shuffle=True)` use a worker seed?

**X7 — Checkpoint format contract**
Trace every `torch.save(...)` in `train_gnn.py` and every `torch.load(...)` in
`cli.py`, `predict.py`, and `evaluate_test_set.py`. Key names must match exactly.
Same for `ga_weights.json`: keys written by `run_ga_ensemble.py` must be the keys
read by `EnsembleScorer.__init__()` and `compute_ensemble_score()` in `predict.py`.

---

## 3. Severity Classification

Use these definitions consistently across all phases.

| Severity | Definition | Example |
|----------|------------|---------|
| **CRITICAL** | Silent wrong ML output — the model produces a number but it is computed from the wrong signal. User cannot detect this without ground truth. | Feature parity bug: training sets flag=1, inference sets flag=0 for the same manifest |
| **HIGH** | Crash or data corruption — the program raises an exception or writes incorrect data. | `FileNotFoundError` on a missing checkpoint, `RuntimeError` from device mismatch |
| **MEDIUM** | Structural violation — DRY, SOLID, missing import from canonical source, magic literal, missing type hint. Observable effect possible but not guaranteed. | `ESCAPE_FLAG_INDICES` hardcoded instead of imported |
| **LOW** | Style only — CLAUDE.md formatting rule, docstring quality, bare `open()`. Zero observable effect. | `open(path)` instead of `path.open()` |

Fixes that affect feature extraction or graph construction require **retraining** all
downstream models. Flag these explicitly.

---

## 4. Finding Template

When recording a finding from any phase, use this format exactly:

```
### F-<phase><unit>-<seq>
- **File:** path/to/file.py
- **Line:** 123
- **Severity:** CRITICAL | HIGH | MEDIUM | LOW
- **Requires retraining:** Yes | No | Maybe
- **Summary:** One sentence.
- **Failure scenario:** Concrete input → wrong output or crash.
- **Fix sketch:** What the correct code should do.
```

Example:
```
### F-X1-003
- **File:** kubescan/src/kubescan/utils/yaml_parser.py
- **Line:** 269
- **Severity:** CRITICAL
- **Requires retraining:** No (inference-only fix)
- **Summary:** INSECURE_HTTP only checked httpGet.scheme in probes; training extractor
  scans all spec values recursively.
- **Failure scenario:** A manifest with http:// in an env var → INSECURE_HTTP=0 at
  inference while training data had it =1. RF risk score is lower than the model expects.
- **Fix sketch:** Replace probe loop with `_spec_has_insecure_http(doc["spec"])` recursive scan.
```

---

## 5. Findings Log

*Populated during execution. One section per completed phase.*

### Phase 0 findings

**mypy --strict — 37 errors in 6 files** (`audit/phase0/mypy.txt`)

Key clusters:
- `cli.py` (11 errors, lines 303–353): dict access returns `object`; `node_data`, `escape_nodes`, `sa_nodes` typed as `object` instead of `list[...]` after `.get()`. Callers of `_print_text_report` pass `object` where typed lists are expected.
- `graph_builder.py` (10 errors): `object` not iterable at lines 145, 172, 185, 191, 285; `float(object)` at 232; `len(object)` at 409; `.nodes`/`.edges` on `object` at 411–413.
- `yaml_parser.py` (5 errors): `object` not iterable at lines 96, 194, 266, 295; incompatible assignment at 149.
- `ga_ensemble.py` (3 errors): `ndarray` missing type args at 166, 177, 194; `Returning Any` at 127.
- `rf_classifier.py` (2 errors): `ndarray` missing type args at 83; `Returning Any` at 120.
- `gat_encoder.py` (1 error): `Returning Any` at 132.

Root cause: YAML dict access (`doc.get("key")`) returns `object` under mypy strict; callers do not narrow with `isinstance`.

**ruff --select ALL — 334 violations** (`audit/phase0/ruff_strict.txt`)

Dominant categories (all files, kubescan/ + research/):
- `PTH123` — bare `open()` instead of `path.open()` (11 occurrences across 6 files)
- `COM812` — trailing comma missing (~30 occurrences)
- `PLR0913` — too many arguments in function definition (>5 params): affects `_print_text_report` (14!), `build_graph` (many), `run_ga`, `load_oof_predictions`, etc.
- `TID252` — relative imports from parent modules (`from ..exceptions import`) in all model files
- `TC002`/`TC003` — third-party/stdlib imports not in `TYPE_CHECKING` block (torch, pathlib)
- `C901` — too complex: `_extract_container_features` (16), `_extract_file` (11), `_parse_yaml_semantics` (14), `build_graph` in predict.py (26), `main` in evaluate/train/predict (11–14)
- `BLE001` — bare `except Exception` in yaml_parser, provenance, train_rf
- `S301` — `pickle.load` security warning in rf_classifier and predict.py (expected; pickle fallback path)
- `INP001` — research/models/*.py lack `__init__.py` (implicit namespace package)
- `PLW2901` — `batch = batch.to(device)` overwrites loop variable in 4 training loops
- 51 errors are auto-fixable (`ruff check --fix`)

**pytest --cov — 61 passed, 79% total coverage** (`audit/phase0/coverage.txt`)

Files with lowest coverage:
| File | Coverage | Key missed lines |
|------|----------|-----------------|
| `cli.py` | 50% | 95 missed: lines 71–76, 88, 99–124, 143–165, 200–254, 277–282, 287, 340, 406–415, 462–490 |
| `exceptions.py` | 63% | 28–35, 40–41, 46, 51–52 |
| `graph_builder.py` | 78% | 114–120, 137–146, 174, 179, 181–191, 199, 277–279, 286–287, 325, 337, 339, 350–354 |
| `rf_classifier.py` | 87% | pickle fallback path (107–113), skops validation (42) |
| `ga_ensemble.py` | 92% | 84–85, 90, 106, 188, 207 |
| `yaml_parser.py` | 92% | edge-case paths in extractors |

`cli.py` at 50% is the largest gap — the live-mode path and most error branches are untested.

**radon cc — avg C (12.37), 81 blocks** (`audit/phase0/complexity.txt`)

Worst offenders:
| Function | File | CC | Grade |
|----------|------|----|-------|
| `build_cluster_graph` | `research/scripts/02_extract/build_graphs.py` | 54 | **F** |
| `build_graph` | `research/models/predict.py` | 41 | **F** |
| `main` | `research/models/evaluate_test_set.py` | 32 | **E** |
| `main` | `research/scripts/01_acquire/ingest_attack_repos.py` | 30 | D |
| `_extract_container_features` | `kubescan/utils/yaml_parser.py` | 26 | D |
| `main` | `research/models/train_gnn.py` | 24 | D |
| `load_dataset` | `research/models/train_rf.py` | 23 | D |
| `_parse_yaml_semantics` | `kubescan/utils/graph_builder.py` | 21 | D |

Both F-rated functions are in research/, not in the inference package. `build_cluster_graph` CC=54 is a direct refactoring target for Phase 3 / C3.

### Phase 1 findings

**Total: 52 findings — 9 CRITICAL · 18 HIGH · 13 MEDIUM · 12 LOW**

---

#### A1 — yaml_parser.py (12 findings)

### F-A1-001
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~222
- **Severity:** CRITICAL
- **Requires retraining:** No (inference-only fix)
- **Summary:** `NO_READ_ONLY_ROOT_FS` always fires because `readOnlyRootFilesystem` is not a valid `PodSecurityContext` field — `pod_sc.get("readOnlyRootFilesystem")` is always `None`, so `pod_writable_fs` is always `True`, forcing the flag to 1 for every manifest regardless of container settings.
- **Failure scenario:** A Pod with every container setting `readOnlyRootFilesystem: true` → training correctly returns 0; inference returns 1. Every manifests RF risk score is inflated.
- **Fix sketch:** Remove the pod-level `readOnlyRootFilesystem` check entirely; only check per-container `sc.get("readOnlyRootFilesystem")`.

### F-A1-002
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~221
- **Severity:** CRITICAL
- **Requires retraining:** No (inference-only fix)
- **Summary:** `pod_run_as_root` is computed incorrectly: `pod_sc.get("runAsNonRoot") is not True` is always True when only `runAsUser` is set, so any pod that enforces non-root via `runAsUser: <non-zero>` (not via `runAsNonRoot: true`) still sets `NO_RUN_AS_NON_ROOT=1` at inference while training skips those containers.
- **Failure scenario:** Pod has `securityContext: {runAsUser: 1000}`, containers have no sc → training: uid=1000 ≠ 0 → skip → flag=0; inference: `pod_run_as_root = True OR False = True` → flag=1.
- **Fix sketch:** Pod-level user enforcement should mirror training: if `pod_run_as_user is not None and pod_run_as_user != 0`, treat it as equivalent to `runAsNonRoot: true` for the container-loop short-circuit.

### F-A1-003
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~296
- **Severity:** CRITICAL
- **Requires retraining:** No (inference-only fix — training behavior is the ground truth)
- **Summary:** `WITHIN_MANIFEST_SECRET` checks for `secretKeyRef` presence (a safe Kubernetes-native secret reference pattern), while training checks for hardcoded credential values using a regex on key names. The two conditions are semantically opposite.
- **Failure scenario:** `env: [{name: DB_PASS, value: "MyS3cr3tP@ssword"}]` → training flags it (hardcoded secret); inference does not (no secretKeyRef). `env: [{name: TOKEN, valueFrom: {secretKeyRef: ...}}]` → inference flags it; training does not. Feature signal is inverted for the majority of real manifests.
- **Fix sketch:** Replace the `secretKeyRef` check with a regex scan of env `value` fields matching `_SECRET_KEY_PATTERNS` (import the pattern set from yaml_parser or define identically to training's `extract_yaml_features.py`).

### F-A1-004
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~324
- **Severity:** CRITICAL
- **Requires retraining:** No
- **Summary:** `NO_ROLLING_UPDATE` is not set when `strategy` is an explicit empty dict `{}`; the fix (bug #18) only prevented the false-positive from an empty dict at inference, but training still flags empty dict via `not strategy` → True. The parity gap persists in the opposite direction.
- **Failure scenario:** `Deployment.spec.strategy: {}` → training: `not {}` = True → flag=1; inference: `strategy_raw is not None` → falls through to type check → `"" != "recreate"` → flag=0.
- **Fix sketch:** Add `or not strategy_raw` to the condition: `if strategy_raw is None or not strategy_raw or str(...).lower() == "recreate":` — this matches training's `not strategy` semantics.

### F-A1-005
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~219
- **Severity:** CRITICAL
- **Requires retraining:** No (inference-only fix)
- **Summary:** `SECCOMP_UNCONFINED` only checks the Kubernetes 1.19+ `seccompProfile.type` field; training's extractor also scans the old-style pod annotation (`seccomp.security.alpha.kubernetes.io/*: unconfined`). Pre-1.19 manifests are always 0 at inference even if training set them to 1.
- **Failure scenario:** Manifest has `metadata.annotations: {"seccomp.security.alpha.kubernetes.io/pod": "unconfined"}` with no `seccompProfile` field → training flags it; inference returns 0.
- **Fix sketch:** Add an annotation scan loop in `_extract_workload_metadata`: iterate `doc.get("metadata", {}).get("annotations", {})` and flag if any key contains `"seccomp"` and corresponding value contains `"unconfined"`.

### F-A1-006
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~284
- **Severity:** CRITICAL
- **Requires retraining:** No
- **Summary:** `SA_AUTOMOUNT_TOKEN` checks both pod-level and per-container `automountServiceAccountToken`; training only checks the pod-level field. A container-level `true` override on a pod that disables it causes inference to flag but training not to.
- **Failure scenario:** `pod.spec.automountServiceAccountToken: false`, container sets `automountServiceAccountToken: true` → training: pod=False → flag=0; inference: `ctr_mount is True` fires → flag=1.
- **Fix sketch:** Check only `pod_spec.get("automountServiceAccountToken") is not False` to match training behavior exactly.

### F-A1-007
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~259
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `SEC_CONT_OVER_PRIVIL` and `ALLOW_PRIVI` use truthy checks (`if sc.get("privileged")`), while training uses strict `is True`; integer `1` from PyYAML would cause inference to flag while training would not.
- **Failure scenario:** `securityContext.privileged: 1` (integer) → training: `1 is True` = False → not flagged; inference: `if 1` = True → flagged.
- **Fix sketch:** Change both checks to `sc.get("privileged") is True` to match training's strict `is True` pattern.

### F-A1-008
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~179
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `TRUE_HOST_PID`, `TRUE_HOST_IPC`, `TRUE_HOST_NET` use truthy checks; training uses strict `is True`. Integer `1` diverges.
- **Failure scenario:** `pod.spec.hostPID: 1` → training: `1 is True` = False; inference: `if 1` = True → flagged.
- **Fix sketch:** Use `pod_spec.get("hostPID") is True` for all three host-namespace flags.

### F-A1-009
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~193 (`_extract_volume_features`)
- **Severity:** HIGH
- **Requires retraining:** No (inference-only fix)
- **Summary:** `DOCKERSOCK_PATH` only scans `pod_spec["volumes"]`; training also scans container `volumeMounts` for the docker socket path. A mount declared only at container level is missed at inference.
- **Failure scenario:** Container has `volumeMounts: [{mountPath: "/var/run/docker.sock"}]` but no corresponding `volumes` entry → training flags it; inference returns 0.
- **Fix sketch:** Add a container `volumeMounts` scan in `_extract_volume_features`, checking `mountPath` for `/docker.sock` or `/var/run/docker.sock`.

### F-A1-010
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~133
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `UNTRUSTED_REGISTRY` diverges for `host:port/image` format — inference correctly extracts the registry domain; training does `image.split(":")[0]` before splitting on `/`, losing the port and treating the hostname alone as the registry check target.
- **Failure scenario:** Image `myprivateregistry:5000/myapp:v1` → training: splits on `:` first → `"myprivateregistry"` → not in TRUSTED_REGISTRIES but checks by wrong string → UNTRUSTED=0 (depending on logic); inference: correctly extracts `"myprivateregistry"` → not trusted → UNTRUSTED=1.
- **Fix sketch:** Align the registry extraction logic in both files. Inference appears correct; training's `extract_yaml_features.py` needs the same fix.

### F-A1-011
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~312
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `USES_DEFAULT_SA` and `NO_DEFAULT_NSPACE` comparisons are case-sensitive `== "default"`, while training uses `.lower() == "default"`.
- **Failure scenario:** `metadata.namespace: "Default"` → training flags it; inference does not.
- **Fix sketch:** `.lower() == "default"` on both SA name and namespace strings.

### F-A1-012
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~219
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `SECCOMP_UNCONFINED` type string comparison is case-sensitive `== "Unconfined"`; training uses `.lower() == "unconfined"`.
- **Failure scenario:** `seccompProfile.type: "UNCONFINED"` → training flags; inference does not.
- **Fix sketch:** `str(profile.get("type", "")).lower() == "unconfined"`.

---

#### A2 — graph_builder.py (4 findings)

### F-A2-001
- **File:** `kubescan/src/kubescan/utils/graph_builder.py` vs `research/scripts/02_extract/build_graphs.py`
- **Line:** ~311 (graph_builder.py)
- **Severity:** CRITICAL
- **Requires retraining:** No (but see note)
- **Summary:** `graph_builder.py:_add_privilege_edges` does NOT use `has_edge` guard (PRIV_REACH overwrites), but the training script `build_graphs.py` DOES guard with `not G.has_edge(src, dst)` (DIR_PROXIMITY wins). The training data was generated with DIR_PROXIMITY preserved; inference produces PRIV_REACH for the same edges.
- **Failure scenario:** Node A (escape-capable) and Node B share a directory. Training edge A→B: `edge_type=0` (DIR_PROXIMITY, preserved by guard). Inference: edge A→B: `edge_type=1` (PRIV_REACH, overwritten). The GNN edge embedding differs → wrong chain probability.
- **Fix sketch:** Add `if not G.has_edge(src, dst):` guard in `_add_privilege_edges` (to match how training data was actually generated), OR retrain after removing the guard from `build_graphs.py` (to match inference). Must choose one canonical behaviour.

### F-A2-002
- **File:** `kubescan/src/kubescan/utils/graph_builder.py`
- **Line:** ~180 (`_parse_yaml_semantics`) / ~264
- **Severity:** CRITICAL
- **Requires retraining:** No (inference-only fix to match training)
- **Summary:** Inference only adds subjects to `elevated_sas` when their `roleRef` is in a filtered privileged-role set. Training (`build_graphs.py`) adds subjects from EVERY `RoleBinding`/`ClusterRoleBinding` unconditionally. Clusters using custom admin roles get RBAC_PRIV edges in training data but not at inference.
- **Failure scenario:** Cluster has a `RoleBinding` to a custom role `"custom-admin"` (not `cluster-admin`). Training: the bound SA lands in `elevated_sas` → gets RBAC_PRIV edges to all other nodes. Inference: custom role not in `_PRIVILEGED_ROLE_NAMES`, not in escalation verbs → no RBAC_PRIV edges. Attack chain via RBAC silently missed.
- **Fix sketch:** Remove the privileged-role filtering from `_parse_yaml_semantics` (or make it identical to training's unconditional collection). Alternatively, retrain with the filtered version — but training must be updated first.

### F-A2-003
- **File:** `kubescan/src/kubescan/utils/graph_builder.py`
- **Line:** ~405 (`graph_to_pyg`)
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `graph_to_pyg` calls `np.stack([...])` with no zero-node guard; if called with an empty cluster (0 nodes), `np.stack([])` raises `ValueError`.
- **Failure scenario:** `feats_list=[]`, `risk_scores=[]` → `build_cluster_graph` returns result dict → `graph_to_pyg` called → `ValueError: need at least one array to stack`.
- **Fix sketch:** Add early return in `build_cluster_graph` when `feats_list` is empty (raise `KubescanError` or return a sentinel), matching `build_graphs.py:if not rows: return None`.

### F-A2-004
- **File:** `kubescan/src/kubescan/utils/graph_builder.py`
- **Line:** ~419
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** Magic literal `0` as default `edge_type` in `e[2].get("edge_type", 0)` instead of `EdgeType.DIR_PROXIMITY`.
- **Failure scenario:** An edge added without `edge_type` attribute silently becomes `DIR_PROXIMITY=0` rather than raising an error.
- **Fix sketch:** `e[2].get("edge_type", EdgeType.DIR_PROXIMITY)`.

---

#### A3 — ga_ensemble.py (4 findings)

### F-A3-001
- **File:** `kubescan/src/kubescan/model/ga_ensemble.py`
- **Line:** ~173
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `run_gnn_ensemble` crashes with `ValueError` when `fold_models=[]` because `np.mean([], axis=0)` raises on an empty sequence.
- **Failure scenario:** All checkpoint files missing or unreadable → `load_fold_ensemble` raises → caller catches → passes `[]` → `run_gnn_ensemble` crashes with unhandled `ValueError` instead of a clean `ModelLoadError`.
- **Fix sketch:** Add `if not fold_models: raise ModelLoadError(checkpoints_dir, "no fold models loaded")` at the top of `run_gnn_ensemble`.

### F-A3-002
- **File:** `kubescan/src/kubescan/model/ga_ensemble.py`
- **Line:** 80
- **Severity:** LOW
- **Summary:** `open(weights_path)` — bare built-in instead of `weights_path.open()`.
- **Fix sketch:** `with weights_path.open() as f:`.

### F-A3-003
- **File:** `kubescan/src/kubescan/model/ga_ensemble.py`
- **Line:** 79
- **Severity:** LOW
- **Summary:** `GAWeights` referenced in CLAUDE.md architecture examples is never declared as a typed dataclass; weights stored as raw floats on `EnsembleScorer`.
- **Fix sketch:** `@dataclass(frozen=True) class GAWeights: w_rf: float; w_gnn: float; w_escape: float`.

### F-A3-004
- **File:** `kubescan/src/kubescan/model/ga_ensemble.py`
- **Line:** 79
- **Severity:** LOW
- **Summary:** Extra keys in `ga_weights.json` (metadata keys from `run_ga_ensemble.py`) are silently ignored without a log warning.
- **Fix sketch:** Log unexpected keys at `DEBUG` level: `unexpected = set(w) - {"w_rf", "w_gnn", "w_escape"}; if unexpected: logger.debug(...)`.

---

#### A4 — gat_encoder.py (5 findings)

### F-A4-001
- **File:** `kubescan/src/kubescan/model/gat_encoder.py`
- **Line:** ~157
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `load_fold_ensemble` hardwires `num_classes`, `dropout`, `num_edge_types`, `edge_emb_dim` to `GATConfig` defaults; if these differ between a checkpoint and the current config, `load_state_dict` raises a cryptic `RuntimeError: size mismatch` with no guidance on which hyperparameter to fix.
- **Failure scenario:** User retrains with `edge_emb_dim=16`; existing `gnn_fold_0.pt` was saved with `edge_emb_dim=8`. `load_fold_ensemble` builds a model with `[5, 16]` embedding → `load_state_dict` gets `[5, 8]` → `RuntimeError`.
- **Fix sketch:** Accept `**kwargs` forwarded to `KubeGAT.__init__`, or accept a `GATConfig` object directly, so all hyperparameters can be overridden by the caller.

### F-A4-002
- **File:** `kubescan/src/kubescan/model/gat_encoder.py`
- **Line:** ~135
- **Severity:** MEDIUM
- **Summary:** `load_fold_ensemble` signature exposes only 4 of 8 `KubeGAT` hyperparameters — the exposed set (`in_channels`, `hidden`, `heads`, `num_layers`) is asymmetric and incomplete.
- **Fix sketch:** Replace the four named parameters with a single `config: GATConfig = GATConfig()` parameter.

### F-A4-003
- **File:** `kubescan/src/kubescan/model/gat_encoder.py`
- **Line:** ~82
- **Severity:** MEDIUM
- **Summary:** `in_dim` is unconditionally set to `hidden * heads` for every layer regardless of the preceding layer's `concat` state; the logic is fragile and undocumented, making future changes to pooling or layer structure error-prone.
- **Fix sketch:** Add an inline comment explaining that `input_proj` always produces `hidden * heads` so `in_dim` is constant, or compute `in_dim` from the previous layer's actual output size.

### F-A4-004
- **File:** `research/models/train_gnn.py`
- **Line:** 12
- **Severity:** LOW
- **Summary:** Module docstring says "18 Rahman flags + 6 extended" giving 25 dim total, but there are 7 extended flags (25 flags + risk_score = 26 dim).
- **Fix sketch:** Update docstring to "25 binary flags + risk_score = 26-dim node features; 5 edge types (0–4)".

### F-A4-005
- **File:** `research/models/train_gnn.py`
- **Line:** 12
- **Severity:** LOW
- **Summary:** Docstring states "Edge types: 0–3" but `EdgeType` defines 5 types (0–4, including `RBAC_PRIV=4`).
- **Fix sketch:** Same docstring update as F-A4-004.

---

#### A5 — rf_classifier.py (4 findings)

### F-A5-001
- **File:** `kubescan/src/kubescan/model/rf_classifier.py`
- **Line:** 75
- **Severity:** CRITICAL
- **Requires retraining:** No (inference must match what the RF was actually trained on)
- **Summary:** `_compute_derived_features` sums all 25 `FEATURE_COLS` for `total_misconfigs`, but the RF was trained on a CSV where `total_misconfigs` = sum of only the 18 Rahman-category flags (excluding the 7 extended flags). Every manifest with extended flags set gets a wrong `total_misconfigs` value, corrupting the RF risk score.
- **Failure scenario:** Manifest with `HOSTPATH_MOUNT=1, IMAGE_USES_LATEST=1` and no Rahman flags → inference: `total_misconfigs=2`; training value: `total_misconfigs=0`. RF split point fires differently → wrong risk score.
- **Fix sketch:** Define `_TOTAL_MISCONFIGS_COLS = _RAHMAN_FEATURES` (18 flags) and sum only those in `_compute_derived_features`. Alternatively fix the training CSV to use 25-flag totals and retrain.

### F-A5-002
- **File:** `research/models/train_rf.py`
- **Line:** ~52
- **Severity:** MEDIUM
- **Summary:** `train_rf.py` defines local feature lists (`RAHMAN_FEATURES`, `EXTENDED_FEATURES`) instead of importing from `kubescan.utils.yaml_parser.FEATURE_COLS` — violates the canonical-source-of-truth rule.
- **Fix sketch:** `from kubescan.utils.yaml_parser import FEATURE_COLS` and derive `_RAHMAN_FEATURES` and `_EXTENDED_FEATURES` from it.

### F-A5-003
- **File:** `kubescan/src/kubescan/model/rf_classifier.py`
- **Line:** 112
- **Severity:** MEDIUM
- **Summary:** Pickle fallback path uses bare `open(model_path, "rb")` instead of `model_path.open("rb")`.
- **Fix sketch:** `with model_path.open("rb") as f:`.

### F-A5-004
- **File:** `research/models/train_rf.py`
- **Line:** 6
- **Severity:** LOW
- **Summary:** Module docstring says "24 columns" and "6 extended" but there are 25 columns (15 Rahman + 3 derived + 7 extended = 25).
- **Fix sketch:** Update docstring.

---

#### A6 — cli.py (3 findings)

### F-A6-001
- **File:** `kubescan/src/kubescan/cli.py`
- **Line:** ~418
- **Severity:** HIGH
- **Summary:** `_run_inference_pipeline` is called inside the `scan` command without a `try/except KubescanError` wrapper; any `KubescanError` or unexpected exception from feature extraction, GNN inference, or RF prediction propagates as a raw Python traceback.
- **Failure scenario:** Corrupt `.pt` checkpoint causes `RuntimeError` inside `run_gnn_ensemble`; user sees full internal traceback with local paths instead of a clean error.
- **Fix sketch:** Wrap both `_run_inference_pipeline` call sites (lines ~418 and ~490) with `try: ... except KubescanError as exc: raise click.ClickException(str(exc)) from exc`.

### F-A6-002
- **File:** `kubescan/src/kubescan/cli.py`
- **Line:** ~490
- **Severity:** HIGH
- **Summary:** Same as F-A6-001 — `_run_inference_pipeline` in the `live` command is also unwrapped.
- **Fix sketch:** Same wrapper as above.

### F-A6-003
- **File:** `kubescan/src/kubescan/cli.py`
- **Line:** ~291
- **Severity:** LOW
- **Summary:** `graph_result` dict access returns `object` under mypy strict; `node_data`, `escape_nodes`, `sa_nodes` are not narrowed, producing 11 mypy errors (lines 303–353). No runtime risk since `build_cluster_graph` always returns lists.
- **Fix sketch:** Add `assert isinstance(node_data, list)` guards or change `build_cluster_graph` return type to a typed `TypedDict`.

---

#### A7 — exceptions.py + device_utils.py (5 findings)

### F-A7-001
- **File:** `kubescan/src/kubescan/__init__.py`
- **Line:** ~14
- **Severity:** HIGH
- **Summary:** `KubescanError` is not exported from the package's public `__all__`; `from kubescan import KubescanError` raises `ImportError`, making the CLI-boundary catch contract unusable for external callers.
- **Failure scenario:** User: `from kubescan import KubescanError` → `ImportError: cannot import name 'KubescanError'`.
- **Fix sketch:** Add `from .exceptions import KubescanError` and include in `__all__`.

### F-A7-002
- **File:** `kubescan/src/kubescan/utils/device_utils.py`
- **Line:** 8
- **Severity:** HIGH
- **Summary:** `import torch` is unconditional at module level; if `torch` is not installed, importing `device_utils` raises `ModuleNotFoundError` which does NOT inherit `KubescanError`, escaping the CLI boundary.
- **Failure scenario:** CI without torch installed → `kubescan scan ./dir` → unformatted `ModuleNotFoundError` traceback.
- **Fix sketch:** Lazy-import torch inside `resolve_device()` and wrap with `try/except ImportError: raise KubescanDependencyError("torch is required")`.

### F-A7-003
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~30
- **Severity:** MEDIUM
- **Summary:** `ImportError` for missing PyYAML at module top does not inherit `KubescanError`, escaping the CLI-boundary catch.
- **Fix sketch:** Wrap import in `try/except ImportError: raise KubescanDependencyError("pyyaml is required") from exc`.

### F-A7-004
- **File:** `kubescan/src/kubescan/utils/device_utils.py`
- **Line:** ~19
- **Severity:** MEDIUM
- **Summary:** `resolve_device()` does not log the selected device at `DEBUG` level, violating the project logging rule and making device-selection debugging impossible.
- **Fix sketch:** `logger.debug("Using device: %s", device)` before `return device`.

### F-A7-005
- **File:** `kubescan/src/kubescan/exceptions.py`
- **Line:** 19
- **Severity:** LOW
- **Summary:** `from pathlib import Path` is a runtime import; with `from __future__ import annotations` active it is annotation-only and should be under `if TYPE_CHECKING:`.
- **Fix sketch:** Move to `if TYPE_CHECKING: from pathlib import Path`.

---

#### A8 — Test files (15 findings)

### F-A8-001
- **File:** `kubescan/tests/unit/test_yaml_parser.py`
- **Line:** — (missing test)
- **Severity:** HIGH
- **Summary:** No test covers `strategy: {}` (explicit empty dict) triggering `NO_ROLLING_UPDATE`.
- **Fix sketch:** `def test_extract_features_empty_strategy_dict_sets_no_rolling_update()`.

### F-A8-002
- **File:** `kubescan/tests/unit/test_yaml_parser.py`
- **Line:** — (missing test)
- **Severity:** HIGH
- **Summary:** No test verifies `NO_ROLLING_UPDATE` does NOT fire on `CronJob`, `Job`, or `Pod`.
- **Fix sketch:** `def test_extract_features_cronjob_kind_does_not_set_no_rolling_update()`.

### F-A8-003
- **File:** `kubescan/tests/unit/test_yaml_parser.py`
- **Line:** — (missing test)
- **Severity:** HIGH
- **Summary:** No test covers pod-level `runAsNonRoot: true` suppressing `NO_RUN_AS_NON_ROOT` on containers that omit their own setting.
- **Fix sketch:** `def test_extract_features_pod_level_run_as_non_root_inherited_by_container_clears_flag()`.

### F-A8-004
- **File:** `kubescan/tests/unit/test_yaml_parser.py`
- **Line:** — (missing test)
- **Severity:** HIGH
- **Summary:** No test verifies `INSECURE_HTTP` is detected in container `env` var values, not just `httpGet` probes.
- **Fix sketch:** `def test_extract_features_http_url_in_env_var_sets_insecure_http()`.

### F-A8-005
- **File:** `kubescan/tests/unit/test_graph_builder.py`
- **Line:** — (missing test)
- **Severity:** HIGH
- **Summary:** No test verifies `PRIV_REACH` edges overwrite `DIR_PROXIMITY` edges (the no-`has_edge`-guard path).
- **Fix sketch:** `def test_build_cluster_graph_priv_reach_overwrites_dir_proximity_edge_type()`.

### F-A8-006
- **File:** `kubescan/tests/unit/`
- **Line:** — (missing file)
- **Severity:** HIGH
- **Summary:** `test_rf_classifier.py` does not exist; `RFClassifier`, `_validate_skops_types`, `predict_risk_scores`, and `from_checkpoints` have zero unit tests. The security-critical `_validate_skops_types` is completely untested.
- **Fix sketch:** Create `kubescan/tests/unit/test_rf_classifier.py` with tests for skops loading, type validation rejection of unsafe prefixes, and correct risk score shape.

### F-A8-007
- **File:** `kubescan/tests/unit/`
- **Line:** — (missing file)
- **Severity:** HIGH
- **Summary:** `test_gat_encoder.py` does not exist; `KubeGAT.forward`, `load_fold_ensemble` (including the zero-folds error path), and `GATConfig` have no unit tests.
- **Fix sketch:** Create `kubescan/tests/unit/test_gat_encoder.py` with at least: forward pass produces `[batch, 3]` output; `load_fold_ensemble` raises `ModelLoadError` on missing directory.

### F-A8-008
- **File:** `kubescan/tests/unit/test_ga_ensemble.py`
- **Line:** ~49
- **Severity:** MEDIUM
- **Summary:** `compute_escape_signal` tests assert `== 1.0` but not `isinstance(result, float)`; a numpy `bool_` or Python `bool` return would pass silently.
- **Fix sketch:** `assert result == 1.0 and isinstance(result, float)` or split into two assertions per AAA rule.

### F-A8-009
- **File:** `kubescan/tests/unit/test_properties.py`
- **Line:** ~77
- **Severity:** MEDIUM
- **Summary:** The property test `test_extractor_features_are_binary_and_complete` silently returns (`if result is None: return`) when `_extract_file` returns `None` for a well-formed Pod manifest, masking extractor regressions.
- **Fix sketch:** Replace with `assert result is not None, "extractor returned None for a valid Pod manifest"`.

### F-A8-010
- **File:** `kubescan/tests/fixtures/make_fixtures.py`
- **Line:** ~58
- **Severity:** MEDIUM
- **Summary:** Creates only 2 GNN fold checkpoints while `NUM_FOLDS=5`; the "degraded ensemble" warning path fires on every CI run and the full 5-fold path is never exercised.
- **Fix sketch:** Change loop to `for fold in range(NUM_FOLDS):` to create all 5 folds (tiny models are fast to create).

### F-A8-011
- **File:** `kubescan/tests/integration/test_cli_scan.py`
- **Line:** — (missing test variants)
- **Severity:** MEDIUM
- **Summary:** Integration tests only exercise `--format json`; text output, `--show-nodes`, `KUBESCAN_CHECKPOINTS` env-var resolution, and the `live` command are completely untested (cli.py is at 50% coverage).
- **Fix sketch:** Add parametrized tests for `--format text` and `--show-nodes`, and a test that sets `KUBESCAN_CHECKPOINTS` env var.

### F-A8-012
- **File:** `kubescan/tests/unit/test_yaml_parser.py`
- **Line:** — (missing test)
- **Severity:** MEDIUM
- **Summary:** No test for YAML edge cases: empty file, file with only comments, or multi-document YAML (`---`) mixing workload and non-workload kinds.
- **Fix sketch:** `def test_extract_features_from_file_empty_file_returns_none()`, `def test_extract_features_from_file_multidoc_yaml_returns_first_workload()`.

### F-A8-013
- **File:** `kubescan/tests/conftest.py` + `kubescan/tests/integration/test_cli_scan.py`
- **Line:** 13 / 37
- **Severity:** LOW
- **Summary:** `_CLEAN_MANIFEST` and `_ATTACK_MANIFEST` string constants are defined in both files; DRY violation at the test layer.
- **Fix sketch:** Move to `conftest.py` only, use as fixtures or imported constants in the integration test.

### F-A8-014
- **File:** `kubescan/tests/unit/test_graph_builder.py`
- **Line:** ~41
- **Severity:** LOW
- **Summary:** `test_graph_to_pyg_node_feature_shape` and `test_graph_to_pyg_edge_index_has_two_rows` test two unrelated things using the same setup without asserting edges actually exist.
- **Fix sketch:** Add `assert pyg_data.edge_index.shape[1] > 0` and split into properly isolated tests.

### F-A8-015
- **File:** `kubescan/tests/unit/test_ga_ensemble.py`
- **Line:** ~66
- **Severity:** LOW
- **Summary:** `_scorer` is a plain helper function rather than a `@pytest.fixture`, limiting pytest's scope management and dependency injection.
- **Fix sketch:** Decorate with `@pytest.fixture` and accept `tmp_path` as a pytest-injected argument.

### Phase 2 findings

**Total: 35 findings — 4 CRITICAL · 2 HIGH · 16 MEDIUM · 13 LOW**

---

#### B1 — train_rf.py (8 findings)

### F-B1-001
- **File:** `research/models/train_rf.py`
- **Line:** ~52
- **Severity:** CRITICAL
- **Requires retraining:** N/A (this IS training — but exposes what the model was actually trained on)
- **Summary:** `RAHMAN_FEATURES` (15 items) is missing 3 flags that are present in every CSV row: `SECCOMP_UNCONFINED`, `VALID_TAINT_SECRET`, `NO_NETWORK_POLICY`. These columns exist in `rf_dataset.csv` and are produced by `yaml_parser.py`, but are silently dropped when building the X matrix — the trained RF never sees them as standalone signals.
- **Failure scenario:** A manifest with only `SECCOMP_UNCONFINED=1` gets `risk_score≈0` from the RF because that flag is not in the input vector. The model was structurally unable to learn from 3 security features the extractor produces.
- **Fix sketch:** Audit `RAHMAN_FEATURES` against the CSV column headers and against `FEATURE_COLS`. Add missing flags or formally document that they are excluded and explain why.

### F-B1-002
- **File:** `research/models/train_rf.py` / `research/scripts/04_build_datasets/`
- **Line:** ~79 (train_rf.py `ALL_FEATURES`) / `build_rf_dataset.py` line ~197 / `enrich_rf_dataset.py` line ~205
- **Severity:** CRITICAL
- **Requires retraining:** Depends on resolution
- **Summary:** `total_misconfigs` is computed inconsistently across CSV rows: Rahman-sourced rows use 18-flag sum (SEVERITY_WEIGHTS keys in `build_rf_dataset.py`); BadPods/Goat/attack rows use 25-flag sum (`FEATURE_COLS` in `enrich_rf_dataset.py`); inference (`rf_classifier.py`) always uses 25-flag sum. One column in the training matrix means three different things for different rows.
- **Failure scenario:** The RF learns a `total_misconfigs` threshold calibrated on mixed-basis values. At inference every manifest is scored with the 25-flag basis, producing systematically different values for Rahman-type manifests (which dominate the training set). This corrupts the weight the RF assigns to `total_misconfigs` splits.
- **Fix sketch:** Rebuild `rf_dataset.csv` with a single consistent `total_misconfigs` definition across all sources. The correct definition (18 vs 25) must match inference (`rf_classifier.py`), then retrain.

### F-B1-003
- **File:** `research/models/train_rf.py`
- **Line:** ~316
- **Severity:** MEDIUM
- **Summary:** No `set_global_seed()` call; Python's `random` module is never seeded; seed is not logged at `INFO` level — only printed as part of a combined status line.
- **Fix sketch:** Call `random.seed(args.seed)`, `np.random.seed(args.seed)`, and `logger.info("seed=%d", args.seed)` before any randomness.

### F-B1-004
- **File:** `research/models/train_rf.py`
- **Line:** ~195
- **Severity:** MEDIUM
- **Summary:** `run_cv()` uses `StratifiedKFold` on the full dataset without cluster-level grouping; manifests from the same repo can appear in both train and val folds, inflating CV metrics via within-repo pattern memorisation.
- **Failure scenario:** `foo-k8s` has 20 manifests split across fold 0 train and val. RF learns repo-specific image/naming conventions. CV F1=0.9935 is optimistic vs unseen-repo performance.
- **Fix sketch:** Use `GroupKFold` or `StratifiedGroupKFold` with `groups=repo_names` to match the cluster-aware GNN split protocol.

### F-B1-005
- **File:** `research/models/train_rf.py`
- **Line:** ~354
- **Severity:** MEDIUM
- **Summary:** All RF hyperparameters (`n_estimators=500`, `min_samples_leaf=2`, `max_features="sqrt"`, `test_size=0.20`, `n_splits=5`) are magic literals, not a Config dataclass — violates CLAUDE.md rules 1 and 5.
- **Fix sketch:** `@dataclass(frozen=True) class RFConfig` with all these fields; load from `--config research/configs/rf_config.yaml`.

### F-B1-006
- **File:** `research/models/train_rf.py`
- **Line:** ~507
- **Severity:** LOW
- **Summary:** Magic literal `0.85` used as `binary_f1_target` threshold and in comparisons at lines ~509 and ~523.
- **Fix sketch:** `_F1_TARGET: Final[float] = 0.85`.

### F-B1-007
- **File:** `research/models/train_rf.py`
- **Line:** ~125
- **Severity:** LOW
- **Summary:** Median override values `1.0` and `0.0` for `SA_AUTOMOUNT_TOKEN` and `UNTRUSTED_REGISTRY` are bare float literals not loaded from the referenced `dataset_config.json`.
- **Fix sketch:** Load from config or define as named constants referencing the config file.

### F-B1-008
- **File:** `research/models/train_rf.py`
- **Line:** ~187
- **Severity:** LOW
- **Summary:** Bare `except Exception: pass` silently swallows AUC-ROC computation failures (e.g., single-class fold) with no log warning.
- **Fix sketch:** `except Exception as exc: logger.debug("AUC-ROC skipped: %s", exc)`.

---

#### B2 — train_gnn.py (6 findings)

### F-B2-001
- **File:** `research/models/train_gnn.py`
- **Line:** ~418
- **Severity:** HIGH
- **Summary:** Each fold checkpoint is saved as a raw `state_dict` with no accompanying config JSON recording the architecture hyperparameters (`hidden`, `heads`, `layers`, `dropout`, `edge_emb_dim`) used during that run. `load_fold_ensemble` defaults to `GATConfig` values; training with non-default args produces checkpoints that crash `load_state_dict` with an uninformative size-mismatch error.
- **Failure scenario:** `python train_gnn.py --hidden 128` → saves 128-dim weight tensors → `kubescan scan` calls `load_fold_ensemble` with `hidden=64` → `RuntimeError: size mismatch for input_proj.weight`.
- **Fix sketch:** Save a `gnn_config.json` alongside each checkpoint: `{"hidden": args.hidden, "heads": args.heads, "num_layers": args.layers, "dropout": args.dropout, "edge_emb_dim": GATConfig.edge_emb_dim, "num_edge_types": GATConfig.num_edge_types}`. `load_fold_ensemble` should read this file and build the model from it instead of defaulting to `GATConfig`.

### F-B2-002
- **File:** `research/models/train_gnn.py`
- **Line:** ~138
- **Severity:** MEDIUM
- **Summary:** `make_model` never passes `num_edge_types` or `edge_emb_dim` to `KubeGAT`; they silently default to `GATConfig` values. If `GATConfig` defaults are ever changed, training and inference will silently diverge.
- **Fix sketch:** Pass all hyperparameters explicitly: `KubeGAT(in_channels=in_ch, hidden=hidden, heads=heads, num_layers=layers, dropout=dropout, num_edge_types=GATConfig.num_edge_types, edge_emb_dim=GATConfig.edge_emb_dim, num_classes=NUM_CLASSES)`.

### F-B2-003
- **File:** `research/models/train_gnn.py`
- **Line:** ~379
- **Severity:** MEDIUM
- **Summary:** Python `random` module never seeded; no `set_global_seed()` call; seed is printed in a combined status line, never logged at `logging.INFO` — violates CLAUDE.md reproducibility and logging rules.
- **Fix sketch:** `random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed); logger.info("seed=%d", args.seed)`.

### F-B2-004
- **File:** `research/models/train_gnn.py`
- **Line:** ~264
- **Severity:** MEDIUM
- **Summary:** `DataLoader(train_set, shuffle=True)` does not pass `generator=torch.Generator().manual_seed(args.seed)`, so per-fold batch ordering is driven by accumulated global RNG state, making isolated fold reproduction impossible.
- **Fix sketch:** `g = torch.Generator(); g.manual_seed(args.seed + fold_idx); DataLoader(..., shuffle=True, generator=g)`.

### F-B2-005
- **File:** `research/models/train_gnn.py`
- **Line:** ~223
- **Severity:** MEDIUM
- **Summary:** `precision_at_k` divides by `k` (fixed), not `min(k, n_positives)`. Under the thesis spec ("divide by min(5, n_attack_chains)"), the formula underreports precision for eval sets with fewer than 5 attack chains.
- **Failure scenario:** Eval set has 3 attack chains, all ranked in top 3. Thesis spec: P@5 = 3/3 = 1.0. Code: P@5 = 3/5 = 0.60. The model's ranking quality is understated during training-time evaluation, potentially causing early stopping before the optimal model is reached.
- **Fix sketch:** `p_at_k = sum(1 for i in top_k_idx if ...) / min(k, sum(1 for l in labels if l == chain_class))`.

### F-B2-006
- **File:** `research/models/train_gnn.py`
- **Line:** ~63
- **Severity:** LOW
- **Summary:** `__all__` re-exports `KubeGAT` (defined in `gat_encoder.py`, only imported here), implying this training script owns the class definition.
- **Fix sketch:** Remove `KubeGAT` from `__all__`; it should not be re-exported from a training script.

---

#### B3 — run_ga_ensemble.py (6 findings)

### F-B3-001
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~111
- **Severity:** MEDIUM
- **Summary:** Escape signal computed inline in `_infer_dataset` instead of calling `ga_ensemble.compute_escape_signal()` — DRY violation that risks silent divergence if the canonical function is updated.
- **Failure scenario:** If `compute_escape_signal` adds a threshold or flag-set change, the GA will optimise with the old definition while inference uses the new one — GA weights will be wrong for the updated signal.
- **Fix sketch:** `from kubescan.model.ga_ensemble import compute_escape_signal` and call it on each cluster's feature matrix slice.

### F-B3-002
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~562
- **Severity:** MEDIUM
- **Summary:** `ga_weights.json` and `ga_results.json` written with bare `open()` instead of `weights_out.open()` / `results_out.open()` — same violation as F-A3-002 in ga_ensemble.py (two separate occurrences here).
- **Fix sketch:** `with weights_out.open("w") as f:` and `with results_out.open("w") as f:`.

### F-B3-003
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~572
- **Severity:** MEDIUM
- **Summary:** Second bare `open()` for `ga_results.json` (same fix as F-B3-002 — listed separately for tracking).
- **Fix sketch:** `with results_out.open("w") as f:`.

### F-B3-004
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~19
- **Severity:** LOW
- **Summary:** Module docstring hardcodes escape flag indices `[0,1,2,3,4,5,7,24]` instead of referencing the imported `ESCAPE_FLAG_INDICES` constant — will silently show wrong indices if `ESCAPE_FLAGS` changes.
- **Fix sketch:** Remove the hardcoded list; add `# see ga_ensemble.ESCAPE_FLAG_INDICES`.

### F-B3-005
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~199 and ~452
- **Severity:** LOW
- **Summary:** `from collections import Counter` appears twice inside function bodies (inside a loop at line ~199, inside `main()` at ~452) instead of at the module top level — PLC0415 violation.
- **Fix sketch:** Move both to top-level imports.

### F-B3-006
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~507
- **Severity:** LOW
- **Summary:** For-loop unpacks `(label, w_gnn, w_rf, w_esc)` — reversed order vs the universal `(w_rf, w_gnn, w_esc)` convention used everywhere else. A future refactor to positional args would silently swap RF and GNN weights in the comparison table.
- **Fix sketch:** Reorder tuple elements to `(label, w_rf, w_gnn, w_esc)` matching the function signature of `compute_objective`.

---

#### B4 — evaluate_test_set.py (6 findings)

### F-B4-001
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~138 (`precision_at_k`)
- **Severity:** CRITICAL
- **Requires retraining:** No (evaluation-only script)
- **Summary:** `precision_at_k` divides by `k` (always 5), not `min(k, n_attack_chains)`. When the test set has fewer than 5 attack chains, the formula computes hits/5, which is Recall@K, not standard Precision@K. The same bug exists in `train_gnn.py:precision_at_k` and `run_ga_ensemble.py:compute_objective`, so the GA was optimised with the same non-standard metric — but the reported thesis figure P@5=0.880 must be verified against the actual test set chain count.
- **Failure scenario:** Test set has n_chains attack chains < 5, all ranked in top-n_chains. Standard P@5 = 1.00 (perfect recall in top 5). Code reports n_chains/5 < 1.00. If the thesis claims P@5=0.880 using the standard definition, the actual achievable ceiling under the current formula may differ.
- **Fix sketch:** `n_pos = sum(1 for l in labels if l == chain_label); return hits / min(k, n_pos) if n_pos > 0 else 0.0`. Apply the same fix to `train_gnn.py` and `run_ga_ensemble.py` for consistency, then re-evaluate to confirm the reported figure.

### F-B4-002
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~313
- **Severity:** CRITICAL
- **Summary:** `p_at_1`, `p_at_3`, `p_at_5` computed by calling `precision_at_k` directly — bypasses the `k_eff = min(k, len(labels))` guard present in `rank_metrics`. When the test set has fewer clusters than `k`, the result is wrong.
- **Failure scenario:** Test set has 3 clusters total; `precision_at_k(ranked_true, 5)` slices `ranked_true[:5]` → gets 3 items → divides by 5. A result of 2 hits returns 0.40, not 0.67.
- **Fix sketch:** Route all P@K calls through `rank_metrics` or apply the `k_eff` guard inline.

### F-B4-003
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~254
- **Severity:** HIGH
- **Summary:** `p5_ceiling` is `min(n_chains, 5) / 5` and `hits_ceiling = p_at_5 >= p5_ceiling`, but `p_at_5` uses denominator 5 while `p5_ceiling` also uses denominator 5 — so `hits_ceiling` is always True when the ranker puts all chains in the top-5, regardless of how many chains there are. The field is not vacuous, but the derivation is only correct when both numerator and denominator use the same `k`.
- **Fix sketch:** Once F-B4-001 is fixed to use `min(k, n_chains)`, `p5_ceiling` should simply be `1.0` (a perfect ranker achieves 1.0 under the corrected formula when all chains are in the top 5). Update the ceiling computation accordingly.

### F-B4-004
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~120
- **Severity:** MEDIUM
- **Summary:** RF risk score read from last column of node feature matrix as `feats[:, -1].mean()` using implicit index `-1` instead of the named constant `RISK_SCORE_INDEX` (or `NODE_FEATURE_DIM - 1`) from `graph_builder.py`.
- **Fix sketch:** `from kubescan.utils.graph_builder import RISK_SCORE_INDEX; feats[:, RISK_SCORE_INDEX].mean()`.

### F-B4-005
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~231
- **Severity:** MEDIUM
- **Summary:** Score thresholds (`SCORE_HIGH_THRESHOLD`, `SCORE_MODERATE_THRESHOLD`) never imported from `ga_ensemble`; script uses `argmax()` for class prediction instead of threshold-based `predict_label()`, producing classification verdicts that differ from the CLI for borderline scores near the threshold.
- **Failure scenario:** A cluster with `ensemble_score=0.55` (between moderate 0.30 and high 0.60): `argmax` on GNN softmax may return ISOLATED_MISCONFIG while threshold-based `predict_label` returns ATTACK_CHAIN. The reported confusion matrix does not match what the CLI would produce.
- **Fix sketch:** Import `predict_label` from `ga_ensemble` and use it for classification verdicts.

### F-B4-006
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~231
- **Severity:** LOW
- **Summary:** `w_escape = weights.get("w_escape", 0.0)` silently defaults to 0 if key missing; no log warning emitted when the escape component is zeroed out.
- **Fix sketch:** `if "w_escape" not in weights: logger.warning("w_escape missing from weights file — defaulting to 0.0")`.

---

#### B5 — predict.py + provenance.py (9 findings)

### F-B5-001
- **File:** `research/models/predict.py`
- **Line:** ~297
- **Severity:** MEDIUM
- **Summary:** `run_gnn_ensemble()` is re-implemented locally — formula is currently identical to `ga_ensemble.run_gnn_ensemble`, but is a DRY violation with silent divergence risk.
- **Failure scenario:** If `ga_ensemble.run_gnn_ensemble` is updated (e.g. temperature scaling), research-side predict.py silently uses the old formula.
- **Fix sketch:** `from kubescan.model.ga_ensemble import run_gnn_ensemble` (alongside the already-imported constants). The local definition is then deleted.

### F-B5-002
- **File:** `research/models/predict.py`
- **Line:** ~107
- **Severity:** MEDIUM
- **Summary:** `LABEL_NAMES` dict re-defined locally instead of imported from `ga_ensemble`.
- **Fix sketch:** `from kubescan.model.ga_ensemble import LABEL_NAMES`.

### F-B5-003
- **File:** `research/models/predict.py`
- **Line:** ~327
- **Severity:** MEDIUM
- **Summary:** `compute_ensemble_score()` re-implements `EnsembleScorer.score()` locally (normalises weights then applies the linear formula) instead of constructing an `EnsembleScorer` from the loaded weights and calling `.score()`.
- **Fix sketch:** Replace the local function and its call sites with `EnsembleScorer(weights_path).score(rf_risk, chain_prob, escape_signal)`.

### F-B5-004
- **File:** `research/models/predict.py`
- **Line:** ~532
- **Severity:** MEDIUM
- **Summary:** `escape_signal` and `escape_fraction` computed inline via dict lookups instead of calling `compute_escape_signal` / `compute_escape_fraction` from `ga_ensemble`.
- **Fix sketch:** Import both functions from `ga_ensemble` and call them on the node feature list.

### F-B5-005
- **File:** `research/models/predict.py`
- **Line:** ~387
- **Severity:** LOW
- **Summary:** `len(weights.get('mode', 'oof'))` prints 3 (length of string `"oof"`) instead of the actual fold model count.
- **Fix sketch:** `len(fold_models)` — the variable is in scope at the call site.

### F-B5-006
- **File:** `research/models/predict.py`
- **Line:** ~175 and ~469
- **Severity:** LOW
- **Summary:** Magic literal `26` used twice for `NODE_FEATURE_DIM` instead of importing the constant from `graph_builder`.
- **Fix sketch:** `from kubescan.utils.graph_builder import NODE_FEATURE_DIM`.

### F-B5-007
- **File:** `research/models/predict.py`
- **Line:** ~480
- **Severity:** LOW
- **Summary:** `dropout=0.3` magic literal; should be `GATConfig.dropout`.
- **Fix sketch:** `from kubescan.model.gat_encoder import GATConfig; dropout=GATConfig.dropout`.

### F-B5-008
- **File:** `research/models/predict.py`
- **Line:** ~461
- **Severity:** LOW
- **Summary:** `pickle.load()` used without a security warning; `rf_classifier.py` emits a `logger.warning` for the same code path. Research script should at minimum `print()` a warning.
- **Fix sketch:** Add `print("WARNING: loading RF from pickle — only use checkpoints from a trusted source.")`.

### F-B5-009
- **File:** `research/models/provenance.py`
- **Line:** ~44
- **Severity:** LOW
- **Summary:** Provenance block does not record node feature dimension (26), FEATURE_COLS length, or checkpoint file paths/hashes — schema drift is invisible from the provenance block alone.
- **Fix sketch:** Add `"feature_dim": NODE_FEATURE_DIM, "feature_cols": len(FEATURE_COLS), "checkpoint_paths": [str(p) for p in checkpoint_files]` to the provenance dict.

### Phase 3 findings

**Total: 63 findings — 13 CRITICAL · 13 HIGH · 25 MEDIUM · 12 LOW**

---

#### C1 — 01_acquire/ (11 findings)

### F-C1-001
- **File:** `research/scripts/01_acquire/ingest_attack_repos.py`
- **Line:** ~1 (repo-level)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** Three fixture repos (`gatekeeper-library`, `kubeaudit-fixtures`, `datree-tests`) contain a mix of intentionally clean and intentionally vulnerable manifests within the same directory tree, but are ingested with a single repo-level label (attack or clean). This gives the RF and GNN contradictory training signal — some `y=1` clusters contain only benign manifests and vice versa.
- **Failure scenario:** `gatekeeper-library/examples/PSPAllowPrivilegeEscalationContainer/` is a constraint violation example (clean) co-ingested with `gatekeeper-library/library/pod-security-policy/privileged-containers/` (attack). Both reach the training set under the same cluster label. The RF learns a mixed signal for `ALLOW_PRIVI=1` manifests.
- **Fix sketch:** Filter by subdirectory or split the repos into per-subfolder clusters with independent labels; or exclude these ambiguous repos from training entirely.

### F-C1-002
- **File:** `research/scripts/01_acquire/ingest_attack_repos.py`
- **Line:** ~85
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** The cluster-level label is assigned at repo granularity before individual manifest content is examined; a repo that contains both attack-chain manifests and harmless RBAC yamls gets labeled `y=2` (attack chain), causing benign manifests to be wrongly labeled as attack chain nodes in the GNN.
- **Failure scenario:** `kubernetes-goat` has both `sensitive-keys-challenge/deployment.yaml` (attack) and `health-check/configmap.yaml` (clean). Both get `y=2` in the graph dataset. The GNN is trained to classify the configmap as an attack-chain participant.
- **Fix sketch:** Label at the subdirectory / challenge level (each challenge = one cluster) or use manifest-level heuristics to assign per-node labels before merging.

### F-C1-003
- **File:** `research/scripts/01_acquire/download_github_manifests.py`
- **Line:** ~62
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `rel_path = Path(root).relative_to(base_dir)` is used to construct cluster IDs, but no sanitization prevents a crafted repo name or directory name containing `..` from producing a cluster ID that escapes the intended base directory, potentially overwriting an existing cluster's data file.
- **Failure scenario:** A GitHub repo with a directory named `../../other_cluster/` causes `rel_path` to resolve outside `data/raw/`, silently overwriting a previously ingested cluster's YAML files.
- **Fix sketch:** Validate that `rel_path` resolves within `base_dir` after construction: `assert (base_dir / rel_path).resolve().is_relative_to(base_dir.resolve())`.

### F-C1-004
- **File:** `research/scripts/01_acquire/download_github_manifests.py`
- **Line:** ~112
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** No cross-source deduplication: the same manifest (e.g., a widely-copied `nginx.yaml` from the Kubernetes docs) can appear in multiple ingested repos, causing the same YAML content to appear in both training and test clusters under different IDs.
- **Failure scenario:** `nginx.yaml` from `examples-repo` enters `train.txt`; `nginx.yaml` from `tutorial-repo` enters `test.txt`. RF memorises the manifest pattern rather than security flags. Reported test metrics are optimistic.
- **Fix sketch:** Compute a content hash (SHA-256 of normalised YAML) per file and skip files whose hash already exists in the corpus.

### F-C1-005
- **File:** `research/scripts/01_acquire/ingest_attack_repos.py`
- **Line:** ~145
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** Downloaded repo contents are not pinned to a specific commit SHA; re-running the acquisition script on a future date ingests a different version of the dataset, breaking reproducibility of training runs.
- **Failure scenario:** `gatekeeper-library` adds a new subdirectory of CRITICAL examples. A re-run of the pipeline trains on 25% more attack data, producing a model incompatible with the previously reported P@5=0.880.
- **Fix sketch:** Record and pin the commit SHA for each cloned repo in `data/raw/manifest.json`; use `git clone --depth 1 --branch <sha>` or equivalent API call.

### F-C1-006
- **File:** `research/scripts/01_acquire/ingest_attack_repos.py`
- **Line:** ~88
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** Entire GitHub org/repos are cloned and every `.yaml` / `.yml` file treated as a Kubernetes manifest, including Helm `values.yaml`, CI configs, and GitHub Actions YAML — these are silently ingested, fail feature extraction, and are dropped with no log warning.
- **Fix sketch:** Filter to files that contain `apiVersion:` and `kind:` before ingestion; log a `DEBUG` count of skipped non-Kubernetes files.

### F-C1-007
- **File:** `research/scripts/01_acquire/download_github_manifests.py`
- **Line:** ~44
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** GitHub API rate limiting not handled — after ~60 unauthenticated requests the script receives HTTP 403 with a retry-after header that is silently ignored, causing partial downloads that appear successful.
- **Fix sketch:** Check `response.headers.get("Retry-After")` on 403/429 and sleep accordingly; raise a descriptive error if no token is configured.

### F-C1-008
- **File:** `research/scripts/01_acquire/download_github_manifests.py`
- **Line:** ~80
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** No retry logic on transient network failures (`ConnectionError`, `Timeout`); a partial download at manifest N leaves the cluster directory incomplete without any failure signal.
- **Fix sketch:** Wrap HTTP requests in a retry loop (max 3 attempts, exponential backoff) using `urllib3.Retry` or the `tenacity` library.

### F-C1-009
- **File:** `research/scripts/01_acquire/ingest_attack_repos.py`
- **Line:** ~204
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** Running the ingest script twice creates duplicate cluster IDs if the output directory already contains data from a prior run; the second run silently overwrites cluster files without warning.
- **Fix sketch:** Add an `--overwrite` flag (default `False`) and raise an error if the output directory already contains data from a previous run.

### F-C1-010
- **File:** `research/scripts/01_acquire/download_github_manifests.py`
- **Line:** ~35
- **Severity:** LOW
- **Summary:** Magic literal `100` for GitHub API per-page parameter instead of a named constant (`_GITHUB_PAGE_SIZE: Final[int] = 100`).
- **Fix sketch:** `_GITHUB_PAGE_SIZE: Final[int] = 100` at module level.

### F-C1-011
- **File:** `research/scripts/01_acquire/ingest_attack_repos.py`
- **Line:** ~59
- **Severity:** LOW
- **Summary:** Magic literal `2` used as the minimum number of YAML files required to constitute a valid cluster, not a named constant.
- **Fix sketch:** `_MIN_CLUSTER_FILES: Final[int] = 2`.

---

#### C2 — extract_yaml_features.py (15 findings)

**Parity table — full 25-flag comparison with `yaml_parser.py`:**

| Flag | extract_yaml_features.py | yaml_parser.py | Match? |
|------|--------------------------|----------------|--------|
| TRUE_HOST_PID | `pod_spec.get("hostPID") is True` | `pod_spec.get("hostPID")` truthy | PARTIAL |
| TRUE_HOST_IPC | `pod_spec.get("hostIPC") is True` | `pod_spec.get("hostIPC")` truthy | PARTIAL |
| TRUE_HOST_NET | `pod_spec.get("hostNetwork") is True` | `pod_spec.get("hostNetwork")` truthy | PARTIAL |
| DOCKERSOCK_PATH | volumes + container volumeMounts | volumes only | NO |
| CAP_SYS_ADMIN | per-container `is True` | per-container truthy | PARTIAL |
| CAP_SYS_MODULE | per-container `is True` | per-container truthy | PARTIAL |
| WITHIN_MANIFEST_SECRET | hardcoded value regex on env `value` | `secretKeyRef` presence | NO |
| SEC_CONT_OVER_PRIVIL | `privileged is True` | `privileged` truthy | PARTIAL |
| ALLOW_PRIVI | `allowPrivilegeEscalation is True` | `allowPrivilegeEscalation` truthy | PARTIAL |
| SECCOMP_UNCONFINED | seccompProfile.type + annotations | seccompProfile.type only | NO |
| VALID_TAINT_SECRET | regex on annotation values | regex on annotation values | YES |
| INSECURE_HTTP | recursive spec scan | recursive spec scan | YES |
| NO_SECU_CONTEXT | any container missing sc | any container missing sc | YES |
| NO_NETWORK_POLICY | no NetworkPolicy in cluster | no NetworkPolicy in cluster | YES |
| HOST_ALIAS | `hostAliases` list non-empty | `hostAliases` list non-empty | YES |
| NO_DEFAULT_NSPACE | `namespace.lower() == "default"` | `namespace == "default"` | NO |
| NO_RESO | limits not set | limits not set | YES |
| NO_ROLLING_UPDATE | `not strategy` → True for `{}` | `strategy_raw is not None` → misses `{}` | NO |
| NO_RUN_AS_NON_ROOT | pod runAsUser==0 short-circuit | pod runAsNonRoot is not True only | NO |
| NO_READ_ONLY_ROOT_FS | per-container only | pod-level field (always True) | NO |
| IMAGE_USES_LATEST | `rsplit(":", 1)` on full image | `rsplit(":", 1)` on full image | YES |
| SA_AUTOMOUNT_TOKEN | pod-level field only | pod + container level | NO |
| USES_DEFAULT_SA | `sa.lower() == "default"` | `sa == "default"` | NO |
| UNTRUSTED_REGISTRY | `image.split(":")[0]` before `/` split | correct domain extraction | NO |
| HOSTPATH_MOUNT | hostPath volume check | hostPath volume check | YES |

Note: `IMAGE_USES_LATEST` parity is YES only for simple `image:tag` format; both files use `rsplit(":", 1)` which misclassifies `registry:port/image` (no explicit tag) as `latest` because the split produces `["registry", "port/image"]` and the second element doesn't look like a tag — but this is an identical bug in both files and doesn't create a parity gap.

### F-C2-001
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~189
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** `WITHIN_MANIFEST_SECRET` checks for hardcoded credential values in env `value` fields (correct security signal), while inference (`yaml_parser.py`) checks for `secretKeyRef` presence (a safe K8s-native pattern). The two conditions fire on opposite inputs — training data and inference produce semantically inverted features.
- **Failure scenario:** `env: [{name: DB_PASS, value: "s3cr3t"}]` → training: regex matches `"s3cr3t"` → flag=1. Inference: no `secretKeyRef` key → flag=0. The RF was optimised on the inverted signal. Fixing inference to match training is required before any re-evaluation.
- **Fix sketch:** Fix inference (`yaml_parser.py`) to match training's regex-on-value approach (see F-A1-003). Training is the ground truth here.

### F-C2-002
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~144
- **Severity:** CRITICAL
- **Requires retraining:** No (training is correct; inference must be fixed)
- **Summary:** `NO_READ_ONLY_ROOT_FS` checks `sc.get("readOnlyRootFilesystem") is not True` per container (correct — `readOnlyRootFilesystem` is a container-level field). Inference (`yaml_parser.py`) checks the pod-level `securityContext`, where `readOnlyRootFilesystem` is not a valid field and always returns `None` → inference flag is always 1.
- **Failure scenario:** Pod with all containers setting `readOnlyRootFilesystem: true` → training: per-container check passes → flag=0. Inference: pod-level check always fires → flag=1. Every manifest's RF risk score is inflated.
- **Fix sketch:** Fix `yaml_parser.py` to check per-container only (see F-A1-001). Training is correct.

### F-C2-003
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~126
- **Severity:** CRITICAL
- **Requires retraining:** No (training is correct; inference must be fixed)
- **Summary:** `SECCOMP_UNCONFINED` checks both `seccompProfile.type` (1.19+) and old-style pod annotations (`seccomp.security.alpha.kubernetes.io/*`). Inference (`yaml_parser.py`) only checks `seccompProfile.type`. Pre-1.19 manifests return 0 at inference while training correctly returns 1.
- **Failure scenario:** Manifest with annotation `seccomp.security.alpha.kubernetes.io/pod: unconfined` but no `seccompProfile` → training: flag=1. Inference: flag=0.
- **Fix sketch:** Fix `yaml_parser.py` to also scan annotations (see F-A1-005). Training is correct.

### F-C2-004
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~250
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** `NO_ROLLING_UPDATE` uses `not strategy` which correctly fires for `{}` (empty dict). Inference (`yaml_parser.py`) after the round-4 fix treats `strategy_raw is not None` and then checks strategy type, so an explicit `strategy: {}` triggers the type check but does not set the flag. The parity gap for empty strategy dicts persists.
- **Failure scenario:** `Deployment.spec.strategy: {}` → training: `not {}` = True → flag=1. Inference: explicit key present, not None, type != "recreate" → flag=0.
- **Fix sketch:** Fix `yaml_parser.py` to also treat `not strategy_raw` as a flag (see F-A1-004). Training is correct.

### F-C2-005
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~207
- **Severity:** CRITICAL
- **Requires retraining:** No (training is correct; inference must be fixed)
- **Summary:** `SA_AUTOMOUNT_TOKEN` checks only the pod-level `automountServiceAccountToken` field. Inference (`yaml_parser.py`) also checks per-container `automountServiceAccountToken`, causing false positives when a pod disables mounting at pod level but a container overrides to `true`.
- **Failure scenario:** `pod.spec.automountServiceAccountToken: false`, container sets `automountServiceAccountToken: true` → training: pod=False → flag=0. Inference: container-level override detected → flag=1.
- **Fix sketch:** Fix `yaml_parser.py` to check only pod-level (see F-A1-006). Training is correct.

### F-C2-006
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~232
- **Severity:** CRITICAL
- **Requires retraining:** No (training is correct; inference must be fixed)
- **Summary:** `NO_DEFAULT_NSPACE` and `USES_DEFAULT_SA` use `.lower() == "default"` (case-insensitive). Inference (`yaml_parser.py`) uses `== "default"` (case-sensitive), missing `"Default"` or `"DEFAULT"` namespaces.
- **Failure scenario:** `metadata.namespace: "Default"` → training: `.lower()` → flag=1. Inference: exact match → flag=0.
- **Fix sketch:** Fix `yaml_parser.py` to use `.lower() == "default"` for both fields (see F-A1-011). Training is correct.

### F-C2-007
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~170
- **Severity:** HIGH
- **Requires retraining:** Yes
- **Summary:** `UNTRUSTED_REGISTRY` splits the image on `:` before splitting on `/` to extract the registry domain: `image.split(":")[0].split("/")[0]`. For `registry:5000/image:tag`, this produces `"registry"` (correct), but for `registry/image:5000` it would produce `"registry"` (also correct by accident). The real divergence is the split order: inference correctly parses `host:port/path` format while training's `split(":")[0]` loses the port in the first field — meaning `registry:5000` becomes `"registry"` while the full `"registry:5000"` would not match `TRUSTED_REGISTRIES`.
- **Failure scenario:** Private registry at `internal:5000/my-image:v1` → training: `"internal"` not in TRUSTED_REGISTRIES → flag=1 (correct). Inference may handle this identically. But for `gcr.io/google-containers/pause:3.1`, both parse `"gcr.io"` correctly. The divergence appears when the image has only a host:port with no path, e.g. `localhost:5000` → training: `"localhost"` → not trusted → flag=1; inference may differ.
- **Fix sketch:** Align both implementations to use `image.split("/")[0]` as the registry (handles `host`, `host:port`, and `host/path` correctly). Apply fix to both files together and retrain.

### F-C2-008
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~106
- **Severity:** HIGH
- **Requires retraining:** Yes
- **Summary:** `NO_RUN_AS_NON_ROOT`: training checks `pod_run_as_user == 0` as the condition for the pod being root-equivalent (i.e. if `runAsUser` is not set or is 0, treat the pod as running as root). Inference (`yaml_parser.py`) uses `pod_sc.get("runAsNonRoot") is not True`, which misses pods that enforce non-root via `runAsUser: <non-zero>` rather than `runAsNonRoot: true`.
- **Failure scenario:** `pod.spec.securityContext: {runAsUser: 1000}`, no containers set `runAsNonRoot`. Training: `pod_run_as_user=1000 ≠ 0` → treat as non-root → flag=0. Inference: `pod_sc.get("runAsNonRoot") is not True` → True → flag=1.
- **Fix sketch:** Fix `yaml_parser.py` to also check `pod_run_as_user` (see F-A1-002). Training is correct.

### F-C2-009
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~85
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `TRUE_HOST_PID`, `TRUE_HOST_IPC`, `TRUE_HOST_NET` use `is True` (strict); inference (`yaml_parser.py`) uses truthy check. Integer value `1` from PyYAML would flag at inference but not at training.
- **Failure scenario:** `pod.spec.hostPID: 1` (integer) → training: `1 is True` = False → not flagged. Inference: `if 1` = True → flagged. Fixes are needed in `yaml_parser.py` (see F-A1-008).
- **Fix sketch:** Fix `yaml_parser.py` to use `is True` checks. Training is correct.

### F-C2-010
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~140
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `DOCKERSOCK_PATH` scans both `pod_spec["volumes"]` (hostPath entries) and container `volumeMounts` (mountPath entries). Inference (`yaml_parser.py`) only scans `volumes`, missing Docker socket mounts declared only at the container level.
- **Failure scenario:** Container has `volumeMounts: [{mountPath: "/var/run/docker.sock"}]` with no corresponding `volumes` entry → training: `volumeMounts` scan catches it → flag=1. Inference: no `volumes` match → flag=0.
- **Fix sketch:** Fix `yaml_parser.py` to also scan container `volumeMounts` (see F-A1-009). Training is correct.

### F-C2-011
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~155
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `SEC_CONT_OVER_PRIVIL` and `ALLOW_PRIVI` use `is True` in training but inference uses truthy check (PARTIAL match). Integer `1` diverges.
- **Fix sketch:** Fix `yaml_parser.py` (see F-A1-007). Training is correct.

### F-C2-012
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~52
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `extract_yaml_features.py` redefines `TRUSTED_REGISTRIES` locally; bug #7 (from §0.5) fixed the local list to match `yaml_parser.py`, but the duplication remains — future changes to `yaml_parser.TRUSTED_REGISTRIES` won't automatically propagate here.
- **Fix sketch:** `from kubescan.utils.yaml_parser import TRUSTED_REGISTRIES` — delete the local definition.

### F-C2-013
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~19
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `_get_pod_spec` uses `(doc.get("spec") or {})` throughout; `yaml_parser.py` uses `_safe_dict()` which additionally handles the case where the value is not a dict (e.g. a string). For non-dict spec values, training's `or {}` silently returns `{}` while inference's `_safe_dict` returns `{}` too — but the implementations diverge if YAML provides `spec: null` vs `spec: "string"`.
- **Fix sketch:** Align: either both use `or {}` or both use a validated helper. Currently low risk since `null` spec is rare.

### F-C2-014
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~1
- **Severity:** LOW
- **Summary:** Module does not import `FEATURE_COLS` from `yaml_parser.py`; the function does not validate that it produces exactly 25 flags. A silent flag omission (e.g., from a refactor) would not be caught.
- **Fix sketch:** Add `from kubescan.utils.yaml_parser import FEATURE_COLS` and assert `set(result.keys()) == set(FEATURE_COLS)` in an internal validation call.

### F-C2-015
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~290
- **Severity:** LOW
- **Summary:** Bare `open(path)` in `_load_yaml` instead of `Path(path).open()`.
- **Fix sketch:** `with Path(path).open() as f:`.

---

#### C3 — build_graphs.py + scan_security_tools.py (10 findings)

### F-C3-001
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~380
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** When `HOSTPATH_MOUNT` is discovered via YAML (volume `hostPath` field), the code updates `node_data[i]["HOSTPATH_MOUNT"] = 1` but does NOT update `G.nodes[node]["x"][24]` (the feature vector stored in the NetworkX graph node). When `.npz` files are written, `x` comes from the NetworkX node attribute, so `x[24]` stays 0 even for nodes where the hostPath escape was detected. The saved training graphs have PRIV_REACH edges for these nodes (escape-capable) but `x[24]=0` (not flagged as escape-capable) — a structural inconsistency that corrupts the GNN's ability to associate hostpath-mount features with attack-chain structure.
- **Failure scenario:** Node with `hostPath: {path: /}` → PRIV_REACH edge added (escape signal present) → `x[24]=0` (feature says "no escape") → GNN trained on conflicting node feature vs edge structure.
- **Fix sketch:** After `node_data[i]["HOSTPATH_MOUNT"] = 1`, also set `G.nodes[node]["x"][HOSTPATH_MOUNT_IDX] = 1.0` where `HOSTPATH_MOUNT_IDX = FEATURE_COLS.index("HOSTPATH_MOUNT")`.

### F-C3-002
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~12
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** Module docstring states `NODE_FEATURE_DIM=25` and `risk_score at index 24`, but the actual schema is `NODE_FEATURE_DIM=26` (25 flags + risk_score at index 25). A developer reading the docstring will build tooling against the wrong schema.
- **Failure scenario:** External script reads `x[:, 24]` expecting `risk_score` but gets `HOSTPATH_MOUNT`. Tools built against the docstring schema silently process wrong features.
- **Fix sketch:** Update docstring: `NODE_FEATURE_DIM=26 (25 binary flags from FEATURE_COLS + risk_score at index 25, appended by build_rf_dataset.py)`.

### F-C3-003
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~25
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `_safe_load_all` and `_get_pod_spec` are re-implemented locally instead of imported from `yaml_parser.py` where the canonical versions live. Any bug fix or extension to `yaml_parser._safe_load_all` will not propagate to training.
- **Failure scenario:** A multi-document YAML fix in `yaml_parser._safe_load_all` closes a parsing bug; `build_graphs.py` continues using the old local copy, causing training data to be extracted with the pre-fix logic.
- **Fix sketch:** Remove both local helpers; import them from `kubescan.utils.yaml_parser`.

### F-C3-004
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~62
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `DEFAULT_DIR_KEY_DEPTH = 2` hardcoded as a magic literal; `graph_builder.py` uses the same value but also as a literal. If the depth is ever changed in one file it will diverge silently, causing cluster IDs to not match between training-time and inference-time graphs.
- **Fix sketch:** Move to a shared constant in a `constants.py` module and import in both files, or import it from `graph_builder.py`.

### F-C3-005
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~204
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** PRIV_REACH edges are added with `G.add_edge(src, dst, edge_type=EdgeType.PRIV_REACH)` and `G.add_edge(dst, src, edge_type=EdgeType.PRIV_REACH)` (bidirectional), but `graph_builder.py` uses `G.add_edge(src, dst)` (directed only). Edge count differs between training and inference for the same cluster.
- **Fix sketch:** Verify directionality intent and align both files: either both bidirectional or both directed for each edge type.

### F-C3-006
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~183
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `G.has_edge(src, dst)` guard IS present in `build_graphs.py` for PRIV_REACH (DIR_PROXIMITY is preserved when an edge already exists). This is the opposite of what `graph_builder.py` does (no guard, PRIV_REACH overwrites). This is a confirmed parity bug (cross-reference with F-A2-001): training PRESERVES DIR_PROXIMITY, inference OVERWRITES. Both files must choose one canonical behaviour.
- **Fix sketch:** Remove the `has_edge` guard from `build_graphs.py` to match inference's overwrite behaviour (and retrain), OR add the guard to `graph_builder.py` to match training's preserve behaviour. The retrain path is safer.

### F-C3-007
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~315
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** RBAC edge collection in `build_graphs.py` adds ALL `RoleBinding` and `ClusterRoleBinding` subjects to `elevated_sas` unconditionally. `graph_builder.py` (inference) only adds subjects whose `roleRef` is in `_PRIVILEGED_ROLE_NAMES` or grants escalation verbs. This is the confirmed parity bug F-A2-002 (cross-reference).
- **Fix sketch:** Must align both implementations. Either both unconditional (training approach) or both filtered (inference approach). If both unconditional, fix inference; if both filtered, fix training.

### F-C3-008
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~56
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `scan_security_tools.py` output (Checkov/Trivy scan results) is written to `data/raw/scan_results/` but is never imported or referenced by any downstream build step. The scan pipeline step is dead code that produces unused files.
- **Fix sketch:** Either wire the scan output into the feature extraction step (as an additional signal), or remove `scan_security_tools.py` from the pipeline and document this decision.

### F-C3-009
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~5
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** Module docstring claims "6 extended features" but there are 7 (`NO_RUN_AS_NON_ROOT`, `NO_READ_ONLY_ROOT_FS`, `IMAGE_USES_LATEST`, `SA_AUTOMOUNT_TOKEN`, `USES_DEFAULT_SA`, `UNTRUSTED_REGISTRY`, `HOSTPATH_MOUNT`).
- **Fix sketch:** Update docstring.

### F-C3-010
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~420
- **Severity:** LOW
- **Summary:** `build_cluster_graph` CC=54 (radon grade F, confirmed in Phase 0). Function is 400+ lines with nested conditionals for every edge type and feature.
- **Fix sketch:** Extract each edge type into a `_add_<type>_edges(G, nodes, node_data)` helper (mirrors the refactoring already done in `graph_builder.py`).

---

#### C4 — 04_build_datasets/ (14 findings)

### F-C4-001
- **File:** `research/scripts/04_build_datasets/build_rf_dataset.py`
- **Line:** ~197
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** Rahman-sourced rows compute `total_misconfigs` as a weighted sum of only the 18 Rahman-category flags (using `SEVERITY_WEIGHTS` keys). This confirms and extends F-B1-002: the 18-flag basis applies to the majority of training rows (the Rahman open-source dataset), making the RF's `total_misconfigs` signal meaningless for BadPods/Goat rows which use 25 flags.
- **Failure scenario:** RF learns a `total_misconfigs` split calibrated on 18-flag basis from majority of training data. At inference every manifest is scored with the 25-flag basis (`rf_classifier.py`). Rahman-type manifests in the held-out test get systematically wrong RF scores.
- **Fix sketch:** Rebuild `rf_dataset.csv` using the 25-flag sum (`sum(row[c] for c in FEATURE_COLS)`) for ALL rows. Retrain RF.

### F-C4-002
- **File:** `research/scripts/04_build_datasets/build_rf_dataset.py`
- **Line:** ~52
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** Confirms F-B1-001: the 3 flags present in `FEATURE_COLS` but absent from the `SEVERITY_WEIGHTS` dict used to build Rahman rows are `SECCOMP_UNCONFINED`, `VALID_TAINT_SECRET`, and `NO_NETWORK_POLICY`. These columns are written as 0 for ALL Rahman-sourced rows regardless of manifest content, causing the RF to train on systematically wrong values for 3 features across the majority of training data.
- **Failure scenario:** A Rahman manifest with `seccomp: unconfined` annotation → `SECCOMP_UNCONFINED` should be 1 but is 0 in the CSV (not in SEVERITY_WEIGHTS). RF never learns to use this feature.
- **Fix sketch:** Add the 3 missing flags to `SEVERITY_WEIGHTS` (with appropriate weight values); or rebuild the dataset by re-running `extract_yaml_features.py` on all raw manifests uniformly.

### F-C4-003
- **File:** `research/scripts/04_build_datasets/enrich_rf_dataset.py`
- **Line:** ~88
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Summary:** Kubernetes Goat manifests are labeled `y=1` (misconfigured, non-chain) using a heuristic: any manifest with at least 1 flag set from the Rahman 18-flag set is labeled positive. Manifests that only trigger extended flags (`NO_RUN_AS_NON_ROOT`, `IMAGE_USES_LATEST`, etc.) get `y=0` (clean) even though they may represent real misconfigurations. Conversely, manifests with a single low-severity flag (e.g. `IMAGE_USES_LATEST=1`) are labeled `y=1` with no threshold, producing false positives.
- **Failure scenario:** `deployment.yaml` with `imagePullPolicy: Always` and `image: nginx` (only `IMAGE_USES_LATEST=1` from extended flags) → heuristic assigns `y=0`. A manifest with `hostAliases: [{ip: "1.2.3.4"}]` (only `HOST_ALIAS=1`) → `y=1` (false positive, no security impact).
- **Fix sketch:** Apply a severity-weighted threshold: `sum(SEVERITY_WEIGHTS[f] * row[f] for f in SEVERITY_WEIGHTS) >= LABEL_THRESHOLD` to assign `y=1`. Define `LABEL_THRESHOLD` as a named constant. Use all 25 flags in the weighted sum.

### F-C4-004
- **File:** `research/scripts/04_build_datasets/build_graph_cache.py`
- **Line:** ~145
- **Severity:** CRITICAL
- **Requires retraining:** No
- **Summary:** `risk_score` at node feature index 25 in training graphs is computed as a severity-weighted heuristic (`sum(SEVERITY_WEIGHTS[f] * x[f]) / MAX_WEIGHT`, normalised to [0,1]) rather than as the RF's `predict_proba`. This is a documented architectural choice — but it means `x[25]` has a different distribution between training (heuristic, roughly uniform for misconfigs) and inference (RF probability, shaped by the forest's decision boundary). The GNN is trained on one distribution and evaluated on another.
- **Failure scenario:** A manifest with only extended flags set gets a moderate heuristic risk score at training time (`∑ extended-flag-weights / max`), but RF `predict_proba ≈ 0` at inference (RF was never trained on extended-flag signal since they were dropped — see F-B1-001). Node feature index 25 distributions systematically differ.
- **Fix sketch:** Once RF is retrained with all 25 flags (F-B1-001 fix), use the retrained RF to regenerate `x[25]` for all training graphs. This closes the heuristic vs. `predict_proba` distribution gap.

### F-C4-005
- **File:** `research/scripts/04_build_datasets/build_graph_cache.py`
- **Line:** ~182
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `.npz` files are written non-atomically: `np.savez(output_path, ...)` writes to `output_path` directly. A crash mid-write leaves a partially written file with the same name as a completed one. On re-run, the partial file is read as a valid (but corrupt) graph.
- **Failure scenario:** Power failure or OOM kill during `np.savez` → partial `.npz` at `graphs/cluster_123.npz` → next pipeline run reads corrupt graph → GNN training silently uses wrong node features.
- **Fix sketch:** Write to a temp file first: `tmp = output_path.with_suffix(".npz.tmp"); np.savez(tmp, ...); tmp.rename(output_path)`.

### F-C4-006
- **File:** `research/scripts/04_build_datasets/build_graph_cache.py`
- **Line:** ~165
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** No validation that the saved `.npz` has the correct schema (`x [N,26]`, `edge_index [2,E]`, `y`, `cluster_id`); a wrong `NODE_FEATURE_DIM` (from F-C3-002 stale docstring confusion) would silently produce `x [N,25]` graphs that crash at `gnn_dataset.py` load time with an uninformative `shape mismatch`.
- **Fix sketch:** After saving, reload and assert: `assert data["x"].shape[1] == NODE_FEATURE_DIM`.

### F-C4-007
- **File:** `research/scripts/04_build_datasets/enrich_rf_dataset.py`
- **Line:** ~205
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `total_misconfigs` in `enrich_rf_dataset.py` (BadPods/Goat/attack rows) sums all 25 `FEATURE_COLS` columns — but the Rahman rows in the same CSV (written by `build_rf_dataset.py`) sum only 18. This confirms the third part of F-B1-002 / F-C4-001. The inconsistency within the CSV is visible here.
- **Fix sketch:** Align to 25-flag sum for all rows and rebuild CSV (fix coordinates with F-C4-001).

### F-C4-008
- **File:** `research/scripts/04_build_datasets/gnn_dataset.py`
- **Line:** ~8
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** Module docstring states `x = [N, 25]` (wrong) — the actual feature dimension is 26. Any tooling or script reading this docstring to determine tensor shapes will use the wrong dimension.
- **Fix sketch:** Update docstring: `x: FloatTensor [N, 26] — 25 binary flags + RF risk_score at index 25`.

### F-C4-009
- **File:** `research/scripts/04_build_datasets/gnn_dataset.py`
- **Line:** ~64
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `edge_attr` is loaded from `.npz` as `data["edge_type"]` (shape `[E]`), then reshaped to `[E, 1]` inside `GNNDataset.__getitem__`. If the `.npz` key name ever changes (it was `edge_type` when saved, but `edge_attr` is the PyG convention), this silently returns zeros.
- **Fix sketch:** Assert that the loaded key name matches what `build_graph_cache.py` writes; or use a shared constant `_EDGE_TYPE_NPZ_KEY = "edge_type"`.

### F-C4-010
- **File:** `research/scripts/04_build_datasets/gnn_dataset.py`
- **Line:** ~47
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `GNNDataset` does not check whether `x[:, 25]` (risk_score column) is in `[0,1]`; a bug in `build_graph_cache.py` that produces risk scores > 1 or < 0 would silently propagate to GNN training.
- **Fix sketch:** Assert `data.x[:, RISK_SCORE_INDEX].min() >= 0 and data.x[:, RISK_SCORE_INDEX].max() <= 1` in `__getitem__`.

### F-C4-011
- **File:** `research/scripts/04_build_datasets/build_rf_dataset.py`
- **Line:** ~47
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `SEVERITY_WEIGHTS` dict is defined with literal float values instead of a `dataclass` or imported constant — the same weights appear duplicated in `build_graph_cache.py` (used for heuristic risk score). Two copies that must stay in sync.
- **Fix sketch:** Move `SEVERITY_WEIGHTS` to a shared constants module and import in both files.

### F-C4-012
- **File:** `research/scripts/04_build_datasets/build_rf_dataset.py`
- **Line:** ~302
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** Output CSV path is a bare string literal `"data/tabular/rf_dataset.csv"` rather than a `pathlib.Path` constant, violating CLAUDE.md rule 6.
- **Fix sketch:** `RF_DATASET_PATH: Final[Path] = DATA_DIR / "tabular" / "rf_dataset.csv"`.

### F-C4-013
- **File:** `research/scripts/04_build_datasets/enrich_rf_dataset.py`
- **Line:** ~18
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `enrich_rf_dataset.py` does not import `FEATURE_COLS` from `yaml_parser.py`; it maintains its own local list of column names for the feature columns. DRY violation.
- **Fix sketch:** `from kubescan.utils.yaml_parser import FEATURE_COLS` and use `FEATURE_COLS` wherever the local list is referenced.

### F-C4-014
- **File:** `research/scripts/04_build_datasets/build_graph_cache.py`
- **Line:** ~50
- **Severity:** LOW
- **Summary:** Magic literal `0.0` used as default risk score for nodes where RF scoring fails, instead of a named sentinel constant.
- **Fix sketch:** `_DEFAULT_RISK_SCORE: Final[float] = 0.0`.

---

#### C5 — create_splits.py (8 findings)

### F-C5-001
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~1 (architectural)
- **Severity:** HIGH
- **Requires retraining:** No (evaluation concern)
- **Summary:** `val.txt` (15 graphs) is used for both (1) GNN checkpoint selection during `train_gnn.py` (early stopping based on val P@5) and (2) GA weight optimisation in `run_ga_ensemble.py --val` mode. The GA optimises weights to maximise P@5 on the same distribution that was used for early stopping. This inflates reported val metrics — the GA weights are already over-fit to this 15-graph set.
- **Failure scenario:** Optimal GNN checkpoint selected by maximising P@5 on val → GA weights optimised to maximise P@5 on same val → reported val P@5 is best-of-best on a set seen twice. True generalisation is measured only by test P@5, which uses 17 graphs that the GA has never seen.
- **Fix sketch:** Split the current val set into `val_gnn.txt` (GNN early stopping) and `val_ga.txt` (GA optimisation). Or use `--train` mode for the GA and evaluate on a separate held-out GA-val set.

### F-C5-002
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~72
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** The stratification is by label (`y ∈ {0, 1, 2}`), but cluster IDs are the split key. If two clusters from the same source repo get different labels (attack vs clean), they could still appear in both train and test. The cluster ID is the unit of splitting, but intra-repo correlation is not controlled.
- **Failure scenario:** `kubernetes-goat` repo contributes clusters both labelled `y=0` (clean subdirectories) and `y=2` (attack subdirectories). Some `y=0` clusters appear in train, others in test. The model can partially memorise repo-level YAML style conventions from training and use them in test.
- **Fix sketch:** Group by repo prefix in cluster IDs and use `StratifiedGroupKFold` with `groups=repo_id` to prevent train/test contamination by repo.

### F-C5-003
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~105
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** No validation that augmented graphs' base cluster IDs are in `train.txt`; the exclusion logic (`_aug_` infix check) deduces the base cluster ID from the augmented ID name. If an augmented graph's base cluster ID is in `val.txt` or `test.txt`, the augmented version is correctly excluded — but no assertion confirms this exclusion count was non-zero (i.e. the `_aug_` sentinel was actually found and applied).
- **Fix sketch:** `assert n_augmented_excluded > 0, "No augmented graphs were excluded — _aug_ sentinel may be malformed"` after the exclusion pass.

### F-C5-004
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~48
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** Split sizes (test_ratio=0.15, val_ratio=0.10) are magic literals, not a Config dataclass — violates CLAUDE.md rules 1 and 5.
- **Fix sketch:** `@dataclass(frozen=True) class SplitConfig: test_ratio: float = 0.15; val_ratio: float = 0.10; seed: int = 42`.

### F-C5-005
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~155
- **Severity:** LOW
- **Summary:** The 5-fold CV split files are written but there is no assertion that each fold contains at least 1 attack-chain cluster (`y=2`). A fold with no positive examples silently produces `P@5 = 0.0` during GNN training with no diagnostic warning.
- **Fix sketch:** After writing fold files, assert `sum(1 for g in fold_graphs if labels[g] == 2) >= 1` for each fold.

### F-C5-006
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~85
- **Severity:** LOW
- **Summary:** `random.shuffle` is called without explicitly setting the seed before shuffling the cluster list (despite the `--seed` arg being accepted). Reproducibility only holds if the Python process seed is set before this call, which is currently not guaranteed.
- **Fix sketch:** `rng = random.Random(args.seed); rng.shuffle(cluster_list)` to isolate the seed.

### F-C5-007
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~112
- **Severity:** LOW
- **Summary:** Output directory path is a bare string `"data/splits"` rather than a `pathlib.Path` constant.
- **Fix sketch:** `SPLITS_DIR: Final[Path] = Path("data/splits")`.

### F-C5-008
- **File:** `research/scripts/05_split/create_splits.py`
- **Line:** ~170
- **Severity:** LOW
- **Summary:** Split statistics (cluster counts per label per split) are `print()`ed but not logged at `INFO` level; running with `--quiet` loses all diagnostic information.
- **Fix sketch:** Replace `print(...)` with `logger.info(...)` for split statistics.

---

#### C6 — augment_graphs.py + patch_hostpath_column.py (5 findings)

### F-C6-001
- **File:** `research/scripts/fixes/patch_hostpath_column.py`
- **Line:** 3
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `sys.path.insert(0, str(Path(__file__).parent))` inserts the `fixes/` directory into `sys.path`, but the `yaml_feature_extractor` module that this script tries to import lives in `research/scripts/02_extract/`. Any call to `import yaml_feature_extractor` raises `ModuleNotFoundError` with a confusing message.
- **Failure scenario:** `python research/scripts/fixes/patch_hostpath_column.py` → `ModuleNotFoundError: No module named 'yaml_feature_extractor'`.
- **Fix sketch:** `sys.path.insert(0, str(Path(__file__).parent.parent / "02_extract"))`. Alternatively, import the feature function from `kubescan.utils.yaml_parser` directly.

### F-C6-002
- **File:** `research/scripts/03_augment/augment_graphs.py`
- **Line:** ~1 (architectural)
- **Severity:** MEDIUM
- **Requires retraining:** No (exclusion enforced downstream)
- **Summary:** `augment_graphs.py` runs on ALL label-2 graphs indiscriminately, including those whose base cluster IDs may later be assigned to val or test splits. The `_aug_` exclusion logic in `create_splits.py` correctly prevents augmented variants of val/test clusters from entering training data — but this is enforced 2 pipeline steps later, and there is no explicit contract between `augment_graphs.py` and `create_splits.py` that documents this dependency.
- **Fix sketch:** Add a `--splits-dir` argument to `augment_graphs.py` (optional); if provided, only augment clusters whose base ID appears in `train.txt`. Document the current "augment all, exclude later" contract in a module docstring.

### F-C6-003
- **File:** `research/scripts/03_augment/augment_graphs.py`
- **Line:** ~55
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** Augmented graph schema (`x [N,26]`, `edge_index [2,E]`, `y=2`, `cluster_id=<base>_aug_<k>`) is not validated after generation. A change to node feature layout in training graphs would silently produce augmented graphs with the wrong schema that crash at `GNNDataset.__getitem__`.
- **Fix sketch:** Add post-generation validation: `assert aug_graph["x"].shape[1] == NODE_FEATURE_DIM` and `assert aug_graph["y"] == 2`.

### F-C6-004
- **File:** `research/scripts/03_augment/augment_graphs.py`
- **Line:** ~18
- **Severity:** LOW
- **Summary:** Magic literal `5` (number of augmented variants per attack-chain graph) not a named constant.
- **Fix sketch:** `N_AUGMENTATIONS: Final[int] = 5`.

### F-C6-005
- **File:** `research/scripts/fixes/patch_hostpath_column.py`
- **Line:** ~28
- **Severity:** LOW
- **Summary:** Bare `open(csv_path, "r")` and `open(output_path, "w")` instead of `Path(csv_path).open("r")` / `Path(output_path).open("w")`.
- **Fix sketch:** Use `path.open()` throughout.

### Phase 4 findings

**Total: 48 findings — 4 CRITICAL · 17 HIGH · 17 MEDIUM · 10 LOW**

Many are cross-file confirmations of Phase 1–3 findings with expanded context or corrected severity. Genuinely new findings are marked **[NEW]**.

---

#### X1 — Feature flag parity (10 findings)

**Complete 25-flag parity table (training = `extract_yaml_features.py`, inference = `yaml_parser.py`):**

| Flag | Match | Correct side | Root cause of divergence |
|------|-------|--------------|--------------------------|
| TRUE_HOST_PID | PARTIAL | Inference | Training: `is True`; inference: truthy. Low impact (PyYAML always produces `True` for `true:`) |
| TRUE_HOST_IPC | PARTIAL | Inference | Same as above |
| TRUE_HOST_NET | PARTIAL | Inference | Same as above |
| DOCKERSOCK_PATH | **NO** | Training | Inference misses container `volumeMounts`; also cross-contaminates with HOSTPATH_MOUNT for `/run/docker.sock` |
| CAP_SYS_ADMIN | PARTIAL | Inference | `is True` vs truthy — negligible in practice |
| CAP_SYS_MODULE | PARTIAL | Inference | Same |
| WITHIN_MANIFEST_SECRET | **NO** | Training | Training: regex on hardcoded `value` fields; inference: `secretKeyRef` presence (semantically opposite) |
| SEC_CONT_OVER_PRIVIL | PARTIAL | Inference | `is True` vs truthy |
| ALLOW_PRIVI | PARTIAL | Inference | `is True` vs truthy |
| SECCOMP_UNCONFINED | **NO** | Training | Training checks seccompProfile.type + legacy annotations; inference checks seccompProfile.type only |
| VALID_TAINT_SECRET | **MATCH** | — | Both always output 0 (trivially identical) |
| INSECURE_HTTP | **MATCH** | — | Both recursive scan, both exclude localhost, both case-insensitive |
| NO_SECU_CONTEXT | **MATCH** | — | Both: any container missing `securityContext` |
| NO_NETWORK_POLICY | PARTIAL | Both | Training: per-file detection; inference: cluster-wide. Both default to 1 when absent |
| HOST_ALIAS | **MATCH** | — | Both: `hostAliases` truthy |
| NO_DEFAULT_NSPACE | **NO** | Training | Training: `.lower() == "default"`; inference: `== "default"` (case-sensitive) |
| NO_RESO | **MATCH** | — | Both: any container missing `resources.limits` |
| NO_ROLLING_UPDATE | **NO** | Neither | Inference fix #18 not backported to training: inference now correctly misses `strategy:{}` cases while training still flags them |
| NO_RUN_AS_NON_ROOT | **NO** | Training | Inference false-positive: pod with `runAsUser: 1000` but no `runAsNonRoot` field still fires |
| NO_READ_ONLY_ROOT_FS | **NO** | Training | Inference checks non-existent pod-level field → always returns 1 |
| IMAGE_USES_LATEST | **NO** | Inference | Training: `image.split(":")[0]` before `/` split fails to detect untagged `registry:port/image` |
| SA_AUTOMOUNT_TOKEN | **NO** | Neither | Training: pod-level only; inference: pod + container level. Correct semantics is K8s pod-level |
| USES_DEFAULT_SA | PARTIAL | Training | Case sensitivity: training `.lower() == "default"`; inference exact match. Low impact |
| UNTRUSTED_REGISTRY | **NO** | Inference | Training `image.split(":")[0]` misparses `registry:port/image` format |
| HOSTPATH_MOUNT | PARTIAL | Inference | Path matching: training uses exact paths; inference uses `"docker.sock"` substring → `/run/docker.sock` goes to wrong flag |

**Summary: 8 NO / 7 PARTIAL / 5 MATCH (out of 25)**

### F-X1-001
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py` vs `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~296 / ~189
- **Severity:** CRITICAL
- **Requires retraining:** No (fix inference to match training)
- **Summary:** `WITHIN_MANIFEST_SECRET` is semantically inverted: training flags hardcoded credential values in env `value` fields; inference flags `secretKeyRef` usage (the safe K8s-native pattern). Confirms F-A1-003 / F-C2-001.
- **Failure scenario:** Manifest with `env: [{name: DB_PASS, value: "hunter2"}]` → training: flag=1; inference: flag=0. The feature trained on real secrets; the CLI flags the safe alternative.
- **Fix sketch:** Replace `secretKeyRef` check in `yaml_parser.py` with the training-side regex scan on env `value` fields.

### F-X1-002
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~222
- **Severity:** CRITICAL
- **Requires retraining:** No (fix inference)
- **Summary:** `NO_READ_ONLY_ROOT_FS` always returns 1 at inference because `readOnlyRootFilesystem` is not a valid `PodSecurityContext` field — the pod-level check always returns `None`. Confirms F-A1-001 / F-C2-002.
- **Failure scenario:** All containers set `readOnlyRootFilesystem: true` → training: flag=0; inference: flag=1 (always).
- **Fix sketch:** Remove the pod-level check in `yaml_parser.py`; check only per-container `sc.get("readOnlyRootFilesystem")`.

### F-X1-003
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~193
- **Severity:** HIGH
- **Requires retraining:** No (fix inference)
- **Summary:** `DOCKERSOCK_PATH` in inference only scans pod `volumes[].hostPath`; training also scans container `volumeMounts[].mountPath`. Confirms F-A1-009.
- **Failure scenario:** Container with `volumeMounts: [{mountPath: "/var/run/docker.sock"}]` and no matching volumes entry → training: flag=1; inference: flag=0.
- **Fix sketch:** Add container `volumeMounts` scan in `yaml_parser.py:_extract_volume_features`.

### F-X1-004
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~287
- **Severity:** HIGH
- **Requires retraining:** Yes
- **Summary:** Inference fix #18 (strategy:{} not flagging NO_ROLLING_UPDATE) was not backported to `extract_yaml_features.py`. Training still uses `not strategy` which fires on `{}`. Training data and inference now disagree in the opposite direction from before fix #18. Confirms F-C2-004.
- **Failure scenario:** `Deployment.spec.strategy: {}` → training: flag=1; inference (after fix #18): flag=0.
- **Fix sketch:** Apply the same fix to `extract_yaml_features.py` and rebuild the training dataset.

### F-X1-005
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~221
- **Severity:** HIGH
- **Requires retraining:** No (fix inference)
- **Summary:** `NO_RUN_AS_NON_ROOT` false positive when pod sets `runAsUser: <non-zero>` but not `runAsNonRoot: true`. Confirms F-A1-002 / F-C2-008.
- **Failure scenario:** `securityContext: {runAsUser: 1000}` (non-root user, no runAsNonRoot) → training: flag=0; inference: flag=1.
- **Fix sketch:** Treat `runAsUser != 0` as equivalent to `runAsNonRoot: true` in `yaml_parser.py`.

### F-X1-006
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~398
- **Severity:** HIGH
- **Requires retraining:** Yes
- **Summary:** `SA_AUTOMOUNT_TOKEN` in training checks only pod-level field; inference checks container-level overrides too. In K8s, the pod-level field governs, so training is closer to correct. Confirms F-A1-006 / F-C2-005.
- **Failure scenario (A):** Pod disables mounting, container overrides to true → training: flag=0; inference: flag=1. **Failure scenario (B):** Pod enables mounting, container overrides to false → training: flag=1; inference: flag=0.
- **Fix sketch:** Align inference to check pod-level only, matching training semantics.

### F-X1-007 **[NEW]**
- **File:** `research/scripts/02_extract/extract_yaml_features.py` vs `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~162 / ~193
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `DOCKERSOCK_PATH` and `HOSTPATH_MOUNT` use different path matching: training uses exact paths (`/var/run/docker.sock`, `/docker.sock`); inference uses `"docker.sock"` substring. A volume at `/run/docker.sock` is flagged as `DOCKERSOCK_PATH` by inference but as `HOSTPATH_MOUNT` by training — the flags fire on different features for the same mount.
- **Failure scenario:** `hostPath.path: "/run/docker.sock"` → training: DOCKERSOCK_PATH=0, HOSTPATH_MOUNT=1; inference: DOCKERSOCK_PATH=1, HOSTPATH_MOUNT=0. Two different flags set for the same security issue.
- **Fix sketch:** Align path matching logic in both files; prefer inference's substring match (catches more variants).

### F-X1-008
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~219
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `SECCOMP_UNCONFINED` in inference misses old-style pod annotations (`seccomp.security.alpha.kubernetes.io/*`). Confirms F-A1-005 / F-C2-003.
- **Failure scenario:** Pre-1.19 manifest with `metadata.annotations: {"seccomp.security.alpha.kubernetes.io/pod": "unconfined"}` → training: flag=1; inference: flag=0.
- **Fix sketch:** Add annotation scan in `yaml_parser.py`.

### F-X1-009 **[NEW]**
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~383
- **Severity:** MEDIUM
- **Requires retraining:** Yes
- **Summary:** `IMAGE_USES_LATEST` in training uses `image.split(":")[0]` to strip the tag, which fails for `registry:port/image` (no tag): the split produces `"registry"` and the remainder is `"port/image"` which contains a `/`, so the image-only segment is never checked for missing tag.
- **Failure scenario:** `container.image: "registry:5000/myapp"` (no tag, implicitly `:latest`) → training: flag=0 (false negative); inference: flag=1 (correct).
- **Fix sketch:** Use `rsplit(":", 1)` (same as inference) which splits on the last colon, correctly separating `"registry:5000/myapp"` from any tag.

### F-X1-010
- **File:** `research/scripts/02_extract/extract_yaml_features.py`
- **Line:** ~431
- **Severity:** MEDIUM
- **Requires retraining:** Yes
- **Summary:** `UNTRUSTED_REGISTRY` in training uses `image.split(":")[0]` before splitting on `/`, breaking registry extraction for `host:port/image` format when the host has no TLD dot. Confirms F-A1-010 / F-C2-007.
- **Failure scenario:** `image: "myregistry:5000/nginx"` (private registry with port, no TLD dot) → training treats `"myregistry"` as a Docker Hub short name → not flagged; inference correctly identifies it as a non-trusted registry.
- **Fix sketch:** Align training to use `image.split("/")[0]` as the registry extraction (handles host, host:port, and hub/path forms).

---

#### X2 — Graph construction parity (7 findings)

**Complete edge-type comparison table:**

| Edge type | build_graphs.py (training) | graph_builder.py (inference) | Priority guard | Direction | Match? |
|-----------|---------------------------|------------------------------|----------------|-----------|--------|
| DIR_PROXIMITY (0) | Same-directory key (depth=2) | Same-directory key (depth=2) | No guard in either | Bidirectional | YES |
| PRIV_REACH (1) | Escape → all; **`if not has_edge` guard** (DIR_PROXIMITY preserved) | Escape → all; **no guard** (overwrites DIR_PROXIMITY) | Training: preserves existing; inference: overwrites | Directed | **NO — CRITICAL** |
| SA_LATERAL (2) | SA node → all; `if not has_edge` guard | SA node → all; `if not has_edge` guard | Same | Directed | YES |
| SEMANTIC_NS (3) | Same namespace; `if not has_edge` each direction | Same namespace; `if not has_edge` each direction | Same | Bidirectional | YES |
| RBAC_PRIV (4) | ALL RoleBinding subjects → `elevated_sas` (unconditional) | Only privileged-role subjects → `elevated_sas` (filtered) | Same guard after collection | Directed | **NO — HIGH** |

**Additional confirmed divergences:**
- HOSTPATH_MOUNT detected in YAML but `G.nodes[idx]["features"]` NOT updated → x[24]=0 despite PRIV_REACH edges being added (training side bug)
- `_safe_load_all` and `_get_pod_spec` re-implemented locally in `build_graphs.py` instead of imported

### F-X2-001
- **File:** `research/scripts/02_extract/build_graphs.py` vs `kubescan/src/kubescan/utils/graph_builder.py`
- **Line:** ~317 / ~311
- **Severity:** CRITICAL
- **Requires retraining:** Yes (must choose one canonical behaviour)
- **Summary:** Training guards PRIV_REACH with `if not G.has_edge(src, dst)` so DIR_PROXIMITY is preserved; inference has no guard and overwrites. The GNN was trained on graphs where co-located escape nodes keep DIR_PROXIMITY label; inference delivers PRIV_REACH for those same edges. Confirms F-A2-001 / F-C3-006.
- **Failure scenario:** Two manifests sharing a directory, one escape-capable → training edge: type=0 (DIR_PROXIMITY); inference edge: type=1 (PRIV_REACH). GNN embedding differs at inference vs training.
- **Fix sketch:** Remove the `has_edge` guard from `build_graphs.py` to match inference semantics, then retrain. (Or add the guard to `graph_builder.py` and retrain — either way both must match.)

### F-X2-002
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~158–166, ~286–288
- **Severity:** HIGH
- **Requires retraining:** Yes
- **Summary:** Training adds ALL RoleBinding subjects to `elevated_sas` unconditionally; inference only adds subjects of privileged roles. Training graphs have more RBAC_PRIV edges than inference produces for the same cluster. Confirms F-A2-002 / F-C3-007.
- **Failure scenario:** Cluster with a read-only RoleBinding → training: SA gets RBAC_PRIV edges to all nodes; inference: no RBAC_PRIV edges (role not privileged). GNN trained on inflated RBAC signal.
- **Fix sketch:** Apply the privileged-role filter in `build_graphs.py` to match inference; retrain.

### F-X2-003
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~277–278
- **Severity:** HIGH
- **Requires retraining:** Yes
- **Summary:** When YAML parsing detects `HOSTPATH_MOUNT`, training updates `node_data[idx]["HOSTPATH_MOUNT"]` but does NOT update `G.nodes[idx]["features"]`. The saved `x[24]` stays 0 while PRIV_REACH edges are added — inconsistent node features vs edge structure. Inference correctly patches the feature vector. Confirms F-C3-001.
- **Failure scenario:** Node with `hostPath: {path: /}` → training: PRIV_REACH edges added, x[24]=0 (feature says no escape); inconsistency corrupts GNN's ability to correlate hostpath feature with attack-chain structure.
- **Fix sketch:** After `node_data[idx]["HOSTPATH_MOUNT"] = 1`, also set `G.nodes[idx]["features"][FEATURE_COLS.index("HOSTPATH_MOUNT")] = 1.0` in `build_graphs.py`.

### F-X2-004
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~80–91
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `_safe_load_all` re-implemented locally in `build_graphs.py` (bare-except version) instead of imported from `kubescan.utils.yaml_parser`. DRY violation; future fixes to the canonical version won't propagate. Confirms F-C3-003.
- **Fix sketch:** Delete local implementation; `from kubescan.utils.yaml_parser import _safe_load_all`.

### F-X2-005
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~94–110
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Summary:** `_get_pod_spec` re-implemented locally instead of imported from `kubescan.utils.yaml_parser`. Confirms F-C3-003.
- **Fix sketch:** Delete local implementation; `from kubescan.utils.yaml_parser import _get_pod_spec`.

### F-X2-006
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~299
- **Severity:** LOW
- **Summary:** DIR_PROXIMITY skip guard (`if len(members) < 2: continue`) exists in training but not inference. Functionally equivalent — single-member groups produce zero edges in both.
- **Fix sketch:** Cosmetic only. Remove the skip in `build_graphs.py` to match inference, or add it to `graph_builder.py` for clarity.

### F-X2-007
- **File:** `research/scripts/02_extract/build_graphs.py`
- **Line:** ~336
- **Severity:** LOW
- **Summary:** Same pattern as F-X2-006 for SEMANTIC_NS groups. Functionally equivalent.
- **Fix sketch:** Cosmetic only.

---

#### X3 — KubeGAT architecture parity (7 findings)

**Instantiation table:**

| Call site | in_channels | hidden | heads | num_layers | num_classes | dropout | num_edge_types | edge_emb_dim |
|-----------|-------------|--------|-------|------------|-------------|---------|----------------|--------------|
| `gat_encoder.load_fold_ensemble` | param (default 26) | param (default 64) | param (default 4) | param (default 3) | GATConfig (3) | GATConfig (0.3) | **NOT PASSED** | **NOT PASSED** |
| `train_gnn.make_model` | from data shape | args.hidden (64) | args.heads (4) | args.layers (3) | 3 | args.dropout (0.3) | **NOT PASSED** | **NOT PASSED** |
| `predict.py` | 26 (magic literal) | args.hidden (64) | args.heads (4) | args.layers (3) | 3 (magic literal) | 0.3 (magic literal) | **NOT PASSED** | **NOT PASSED** |

**Checkpoint contract:** `train_gnn.py` saves a raw `state_dict` (no outer dict wrapper, no metadata). All readers call `load_state_dict` on the raw loaded object. Key format is consistent across all call sites. All `torch.load` calls use `weights_only=True`.

### F-X3-001
- **File:** `kubescan/src/kubescan/model/gat_encoder.py`
- **Line:** ~157
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `load_fold_ensemble` does not pass `num_edge_types` or `edge_emb_dim` to `KubeGAT`; both default to `GATConfig` values. If training used non-defaults, `load_state_dict` raises a cryptic `RuntimeError: size mismatch`. Confirms F-A4-001.
- **Fix sketch:** Accept a full `GATConfig` object and pass all 8 hyperparameters to `KubeGAT.__init__`.

### F-X3-002
- **File:** `research/models/train_gnn.py`
- **Line:** ~137
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `make_model` never passes `num_edge_types` or `edge_emb_dim` to `KubeGAT`; checkpoints are always trained with defaults (5, 8) regardless of any future CLI flags. Confirms F-B2-002.
- **Fix sketch:** Pass all hyperparameters explicitly; save `gnn_config.json` alongside each checkpoint.

### F-X3-003
- **File:** `research/models/train_gnn.py`
- **Line:** ~418, ~514
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `torch.save` writes only the raw `state_dict` — no architecture config JSON. `load_fold_ensemble` cannot reconstruct the exact architecture without out-of-band knowledge. Confirms F-B2-001.
- **Fix sketch:** Save `gnn_config.json` alongside each `.pt` file with all 8 hyperparameters.

### F-X3-004
- **File:** `research/models/predict.py`
- **Line:** ~474
- **Severity:** MEDIUM
- **Summary:** `predict.py` hardcodes `in_channels=26`, `num_classes=3`, `dropout=0.3` as magic literals instead of using `NODE_FEATURE_DIM`, `NUM_CLASSES`, and `GATConfig.dropout`. Confirms F-B5-006 / F-B5-007.
- **Fix sketch:** `from kubescan.model.gat_encoder import GATConfig; from kubescan.utils.graph_builder import NODE_FEATURE_DIM`.

### F-X3-005
- **File:** `research/models/predict.py`
- **Line:** ~297
- **Severity:** MEDIUM
- **Summary:** `run_gnn_ensemble()` still locally defined in `predict.py`, not imported from `ga_ensemble`. DRY violation with silent divergence risk. Confirms F-B5-001.
- **Fix sketch:** `from kubescan.model.ga_ensemble import run_gnn_ensemble`.

### F-X3-006 **[NEW]**
- **File:** `research/models/predict.py`
- **Line:** ~45
- **Severity:** LOW
- **Summary:** `predict.py` imports `KubeGAT` via `from train_gnn import KubeGAT` (re-export through training script) instead of directly from `kubescan.model.gat_encoder`. If `train_gnn.py`'s import of `KubeGAT` is refactored, `predict.py` breaks silently.
- **Fix sketch:** `from kubescan.model.gat_encoder import KubeGAT`.

### F-X3-007
- **File:** `kubescan/src/kubescan/model/gat_encoder.py`
- **Line:** ~135
- **Severity:** LOW
- **Summary:** `load_fold_ensemble` signature exposes only 4 of 8 hyperparameters with no way for callers to override `num_classes`, `dropout`, `num_edge_types`, or `edge_emb_dim`. Confirms F-A4-002.
- **Fix sketch:** Replace four named parameters with `config: GATConfig = GATConfig()`.

---

#### X4 — Security audit (5 findings)

**Clean findings:** All `yaml.safe_load` (no unsafe `yaml.load`). All `subprocess` calls use list form, no `shell=True`. No `eval`, `exec`, `os.system`. No YAML field values reach subprocess or file paths. All `torch.load` use `weights_only=True`.

### F-X4-001
- **File:** `kubescan/src/kubescan/model/rf_classifier.py`
- **Line:** 112–113
- **Severity:** HIGH
- **Requires retraining:** No
- **Summary:** `pickle.load()` is the fallback path when no `.skops` checkpoint exists. A tampered `rf_model.pkl` achieves arbitrary code execution on model load, bypassing all skops type validation.
- **Failure scenario:** Attacker replaces `rf_model.pkl` in the checkpoints directory (e.g., via CI artifact compromise or malicious `--checkpoints-dir`). `RFClassifier(pickle_path)` loads and executes the payload.
- **Fix sketch:** Gate the pickle fallback behind an explicit `--allow-pickle` CLI flag (default: disabled). Alternatively, remove the fallback entirely once all checkpoints are migrated to `.skops`.

### F-X4-002
- **File:** `research/models/predict.py`
- **Line:** ~460–461
- **Severity:** HIGH
- **Summary:** `pickle.load()` used unconditionally with no warning, no `.skops` check, and no type validation. Elevates F-B5-008 to HIGH.
- **Failure scenario:** A researcher runs `python predict.py` against a tampered `rf_model.pkl` from an untrusted artifact store → arbitrary code execution.
- **Fix sketch:** Replace with the same `.skops`-first logic used in `RFClassifier.from_checkpoints()`; at minimum emit a `print("WARNING: loading from pickle…")` warning.

### F-X4-003
- **File:** `kubescan/src/kubescan/utils/yaml_parser.py`
- **Line:** ~86
- **Severity:** MEDIUM
- **Summary:** Bare `open()` in library code instead of `Path.open()`. CLAUDE.md violation. No security impact.
- **Fix sketch:** `with Path(path).open() as f:`.

### F-X4-004
- **File:** `kubescan/src/kubescan/model/rf_classifier.py`
- **Line:** 112
- **Severity:** MEDIUM
- **Summary:** Bare `open(model_path, "rb")` in library code. Confirms F-A5-003.
- **Fix sketch:** `with model_path.open("rb") as f:`.

### F-X4-005
- **File:** `kubescan/src/kubescan/model/ga_ensemble.py`
- **Line:** 80
- **Severity:** MEDIUM
- **Summary:** Bare `open(weights_path)` in library code. Confirms F-A3-002.
- **Fix sketch:** `with weights_path.open() as f:`.

---

#### X5 — Data leakage audit (5 findings)

**Clean findings confirmed:** Cluster-level aug exclusion correct (120 aug variants of val/test base clusters excluded from all training splits). GNN train/val/test splits never contaminated. CV fold test clusters excluded from `cv_pool`. No feature normalization fitted on train and applied to test.

### F-X5-001 **[NEW]**
- **File:** `research/models/train_rf.py`
- **Line:** ~372–374
- **Severity:** HIGH
- **Requires retraining:** Yes
- **Summary:** The RF train/test split uses manifest-level `StratifiedKFold` on `rf_dataset.csv` with no cluster-level grouping — completely independent of the GNN cluster splits in `train.txt`/`test.txt`. The RF's `predict_proba` output is embedded as node feature index 25 (`x[25]`) in the `.npz` graphs that the GNN uses. A manifest in the GNN test set may be in the RF training set, meaning the RF was trained on data that the GNN model sees at test time.
- **Failure scenario:** Cluster `piggymetrics-k8s` is in `test.txt`. Its manifests land in the RF training set (manifest-level split). The RF learns their specific flags and produces a tuned `risk_score`. The GNN test graphs for `piggymetrics-k8s` contain this tuned `risk_score` (x[25]), giving the GNN an informative feature at test time that it was also exposed to via the RF's training data — cross-layer leakage.
- **Fix sketch:** Run the RF split using the same cluster-level train/test boundary as the GNN: only train the RF on manifests whose cluster ID appears in `train.txt`. Use `test.txt` manifests only for RF evaluation.

### F-X5-002 **[NEW]**
- **File:** `research/models/train_rf.py`
- **Line:** ~109–132
- **Severity:** MEDIUM
- **Summary:** Column medians for missing-value imputation (`np.nanmedian(X_raw, axis=0)`) are computed over the full dataset — all rows including those that will become the RF test set — before the train/test split at line 372.
- **Failure scenario:** Extended features (`NO_RUN_AS_NON_ROOT`, `SA_AUTOMOUNT_TOKEN`) have high NaN rates; their imputed medians are shifted by test-set values, inflating RF test F1 by an amount that grows with the NaN rate.
- **Fix sketch:** Split first; compute medians only on training rows; apply to val/test using the training-set medians.

### F-X5-003
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~136
- **Severity:** MEDIUM
- **Summary:** Default (non-OOF) mode of `run_ga_ensemble.py` loads predictions from the same 15-graph `val.txt` used for GNN early stopping — double-dipping. Confirms F-C5-001. The `--oof` flag is the correct path and avoids this.
- **Fix sketch:** Make `--oof` the default; deprecate the `--val` mode or require an explicit `--val-mode` flag with a warning.

### F-X5-004
- **File:** `research/models/train_rf.py`
- **Line:** ~206
- **Severity:** LOW
- **Summary:** RF 5-fold CV uses manifest-level `StratifiedKFold` without cluster grouping; manifests from the same repo can appear on both sides of a fold boundary. CV F1=0.9935 is optimistic. Confirms F-B1-004.
- **Fix sketch:** Use `StratifiedGroupKFold` with `groups=cluster_id` column.

### F-X5-005 **[NEW]**
- **File:** `research/scripts/04_build_datasets/build_graph_cache.py`
- **Line:** (cache file creation)
- **Severity:** LOW
- **Summary:** `graphs_cache.npz` consolidates all graphs (train, val, test, augmented) into one artifact with no cache invalidation mechanism. A stale cache after pipeline changes silently serves wrong data.
- **Failure scenario:** `augment_graphs.py` re-run with different `--variants`; `build_graph_cache.py` not re-run → cache holds old augmented graphs; `train_gnn.py` reads stale cache with no warning.
- **Fix sketch:** Store a manifest hash in the cache header; verify on load.

---

#### X6 — Seed / reproducibility (11 findings)

**Reproducibility summary by file:**

| File | `--seed` CLI | `random.seed` | `np.random.seed` | `torch.manual_seed` | `cuda.manual_seed_all` | Seed logged (INFO) |
|------|-------------|---------------|------------------|---------------------|------------------------|-------------------|
| `train_rf.py` | YES | **NO** | **NO** | — | — | **NO** (JSON only) |
| `train_gnn.py` | YES | **NO** | YES (global) | YES | **NO** | **NO** (print only) |
| `run_ga_ensemble.py` | YES | scoped only | scoped only | **NO** | — | **NO** |
| `evaluate_test_set.py` | **NO** | **NO** | **NO** | **NO** | — | **NO** |
| `create_splits.py` | YES | scoped only | — | — | — | print only ✓ |
| `augment_graphs.py` | YES | — | scoped only | — | — | print only ✓ |

`set_global_seed()` utility: **does not exist anywhere in the codebase.**

### F-X6-001 **[NEW]**
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~206–216 (CLI parser) / `bootstrap_cis` function
- **Severity:** CRITICAL
- **Requires retraining:** No
- **Summary:** No `--seed` CLI argument; `bootstrap_cis()` hardcodes `seed=42` as a default parameter. Bootstrap confidence intervals cannot be reproduced with a different seed without editing source code, blocking reproducibility analysis. More critically, GNN inference in `main()` is entirely unseeded — results may differ between runs on CUDA hardware.
- **Failure scenario:** Reviewer attempts to reproduce the reported `P@5=0.880 ± CI` with `seed=123` → impossible without code change. On CUDA hardware, `P@5` from two identical script invocations may differ by `±0.01`.
- **Fix sketch:** Add `--seed INT` argument; call `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)` at the start of `main()`; log at INFO level; replace the hardcoded `seed=42` in `bootstrap_cis` with the passed seed.

### F-X6-002 **[NEW]**
- **File:** `research/models/evaluate_test_set.py`
- **Line:** ~218 (main entry point)
- **Severity:** HIGH
- **Summary:** No global RNG seeding at all in `main()`; `torch`, `numpy`, and `random` all use non-deterministic state. GNN softmax outputs may differ between runs on CUDA hardware.
- **Failure scenario:** Two consecutive runs of `evaluate_test_set.py` produce different `P@5` values on a CUDA machine with non-deterministic ops. The reported thesis figure is tied to one specific (unknown) RNG state.
- **Fix sketch:** Same as F-X6-001.

### F-X6-003 **[NEW]**
- **File:** `kubescan/src/kubescan/` (absent)
- **Line:** —
- **Severity:** HIGH
- **Summary:** No `set_global_seed(seed)` utility function exists anywhere in the codebase. Each training script independently chooses which RNG sources to seed, producing inconsistent coverage (`train_rf.py` seeds neither `random` nor `numpy`; `train_gnn.py` misses `random` and `cuda`; `evaluate_test_set.py` seeds nothing).
- **Failure scenario:** Adding a new RNG source (e.g., a library using `random.*`) requires patching every training script independently; it is systematically unauditable.
- **Fix sketch:** Create `kubescan/utils/seed_utils.py` with `def set_global_seed(seed: int) -> None: random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); logger.info("seed=%d", seed)`. Import and call it at the top of every training script.

### F-X6-004 **[NEW]**
- **File:** `research/models/train_gnn.py`
- **Line:** ~264
- **Severity:** HIGH
- **Summary:** `DataLoader(num_workers=N, shuffle=True)` with `N > 0` has no `worker_init_fn`; worker subprocesses inherit the parent's RNG state and then diverge unpredictably. Batch ordering is non-deterministic across runs.
- **Failure scenario:** On any multi-core machine, two training runs with the same `--seed` produce different epoch sequences due to non-deterministic worker scheduling.
- **Fix sketch:** `worker_init_fn=lambda wid: np.random.seed(args.seed + wid)` and `generator=torch.Generator().manual_seed(args.seed)` in the DataLoader constructor.

### F-X6-005
- **File:** `research/models/train_rf.py`
- **Line:** ~316
- **Severity:** HIGH
- **Summary:** No `random.seed()` or `np.random.seed()` call; global RNGs are unseeded; seed is not logged at INFO. Confirms F-B1-003.
- **Fix sketch:** Call `set_global_seed(args.seed)` (once F-X6-003 is implemented).

### F-X6-006
- **File:** `research/models/train_gnn.py`
- **Line:** ~379
- **Severity:** HIGH
- **Summary:** Python `random` module never seeded; only `torch` and `numpy` global RNGs are seeded. Confirms F-B2-003.
- **Fix sketch:** `random.seed(args.seed)` at the top of `main()`.

### F-X6-007
- **File:** `research/models/train_gnn.py`
- **Line:** ~264
- **Severity:** MEDIUM
- **Summary:** `DataLoader(shuffle=True)` does not pass `generator=torch.Generator().manual_seed(args.seed + fold_idx)`, so per-fold batch ordering is coupled to accumulated global RNG state. Confirms F-B2-004.
- **Fix sketch:** `g = torch.Generator(); g.manual_seed(args.seed + fold); DataLoader(..., generator=g)`.

### F-X6-008 **[NEW]**
- **File:** `research/models/train_gnn.py`
- **Line:** ~379
- **Severity:** MEDIUM
- **Summary:** `torch.cuda.manual_seed_all(seed)` never called; on multi-GPU setups all non-default CUDA devices are unseeded.
- **Fix sketch:** Add `torch.cuda.manual_seed_all(args.seed)` immediately after `torch.manual_seed(args.seed)`.

### F-X6-009
- **File:** `research/models/train_gnn.py`
- **Line:** ~381
- **Severity:** MEDIUM
- **Summary:** Seed logged via `print()` not `logger.info()`. Confirms / elevates F-B2-003 logging sub-issue.
- **Fix sketch:** `logger.info("seed=%d", args.seed)`.

### F-X6-010 **[NEW]**
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~410–490
- **Severity:** MEDIUM
- **Summary:** Neither `torch.manual_seed` nor global numpy/random seeding is called in `main()` before GNN inference is executed. The GA itself uses isolated RNGs (correctly), but the GNN inference step is unseeded.
- **Failure scenario:** On CUDA hardware, GNN softmax outputs used for GA weight optimisation may differ between runs with the same `--seed`, producing different `ga_weights.json`.
- **Fix sketch:** Call `set_global_seed(args.seed)` at the top of `main()` in `run_ga_ensemble.py`.

### F-X6-011 **[NEW]**
- **File:** `research/models/run_ga_ensemble.py`
- **Line:** ~431
- **Severity:** LOW
- **Summary:** Seed is stored in `_provenance` in the output JSON but never logged to stdout/stderr before execution begins. If a run is aborted before the JSON is written, the seed used is unrecoverable.
- **Fix sketch:** `print(f"seed={args.seed}")` or `logger.info("seed=%d", args.seed)` at the top of `main()`.

---

#### X7 — Checkpoint format contract (3 findings)

**Contract tables:**

GNN checkpoint: `train_gnn.py` saves a raw flat `state_dict` (no wrapper dict). All four readers (`gat_encoder.py`, `predict.py`, `evaluate_test_set.py`, `run_ga_ensemble.py`) pass the loaded object directly to `load_state_dict`. Key names are consistent. All `torch.load` calls use `weights_only=True`.

GA weights JSON: keys `w_rf`, `w_gnn`, `w_escape` written and read consistently. `EnsembleScorer` and `evaluate_test_set.py` use direct key access; `predict.py` uses `.get()` with defaults — behavioral inconsistency but no format mismatch.

### F-X7-001 **[NEW]**
- **File:** `research/models/predict.py` vs `kubescan/src/kubescan/model/ga_ensemble.py`
- **Line:** ~333 / ~83
- **Severity:** LOW
- **Summary:** `predict.py` accesses `w_rf` and `w_gnn` via `.get()` with hardcoded fallback defaults; `EnsembleScorer` and `evaluate_test_set.py` use direct dict access (raises `KeyError` on missing). A corrupt or pre-`w_escape` `ga_weights.json` causes three different behaviors in three callers.
- **Failure scenario:** `ga_weights.json` lacks `w_rf` → `predict.py`: silently uses `_DEFAULT_W_RF=0.36`; `EnsembleScorer`: raises `ModelLoadError`; `evaluate_test_set.py`: raises `KeyError`.
- **Fix sketch:** Standardize to one access pattern: either `EnsembleScorer` as the sole reader (DI), or consistent `.get()` with a shared `_FALLBACK_WEIGHTS` constant.

### F-X7-002
- **File:** `research/models/predict.py`
- **Line:** ~388
- **Severity:** LOW
- **Summary:** `len(weights.get('mode', 'oof'))` prints the character count of the string `"oof"` (3) not the fold model count. Confirms F-B5-005.
- **Fix sketch:** `len(fold_models)` — the variable is in scope.

### F-X7-003
- **File:** `research/models/train_gnn.py`
- **Line:** ~418, ~514
- **Severity:** LOW
- **Summary:** Checkpoint files are not self-describing — no metadata (epoch, val_f1, architecture config) is saved. If `GATConfig` defaults change after checkpoints are written but before they are loaded, there is no version check to catch the mismatch before `load_state_dict` crashes.
- **Fix sketch:** Save alongside each `.pt`: `{"in_channels": ..., "hidden": ..., ..., "epoch": epoch, "val_p_at_k": best_val}` as a companion `gnn_fold_k_config.json`.

---

## 6. Known Risks Not Yet Audited

These items came up during PR review rounds but were not in the final confirmed-finding
set. They are NOT confirmed bugs — they need a dedicated Phase 4 agent to verify.

| Risk | Likely phase | Notes |
|------|-------------|-------|
| `run_gnn_ensemble()` in `predict.py` still locally defined (not imported from `ga_ensemble`) | B5 / X3 | Was not in round-4 confirmed findings |
| `RF_ALL_FEATURES` in `predict.py` duplicates `_ALL_RF_FEATURES` in `rf_classifier.py` | B5 | Column order must be identical |
| `NO_READ_ONLY_ROOT_FS` pod-level check: `readOnlyRootFilesystem` is not a valid pod-level field → always True → flag always set | A1 | May be a pre-existing bug in both training and inference (same effect on both sides) |
| `_get_pod_spec()` in `extract_yaml_features.py` uses `(x or {})` instead of `_safe_dict()` | X1 | Diverges on falsy-non-dict values |
| `augment_graphs.py` may produce augmented graphs with same cluster ID as originals | X5 | Could leak attack chains into test split |
| `provenance.py` — is checkpoint hash logged? | B5 | Needed for reproducibility |
| `scan_security_tools.py` — does its output feed any training feature? | C3 | If yes, must match inference |

---

## 7. Post-Audit Actions

After Phase 5 synthesis, all CRITICAL and HIGH findings should be addressed before any
paper submission or public release. Use this checklist:

- `[ ]` All CRITICAL findings fixed and tests added
- `[ ]` All HIGH findings fixed
- `[ ]` MEDIUM findings triaged (fix or document as known limitation)
- `[ ]` CRITICAL fixes that require retraining: models retrained, new checkpoints committed
- `[ ]` Thesis numbers (`research/models/evaluate_test_set.py`) re-run after retraining
- `[ ]` `audit/findings_final.md` written with disposition for every finding
- `[ ]` This file updated to `[ ]→[x]` for all completed phases
