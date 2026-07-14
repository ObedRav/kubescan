# kubescan — Audit Findings (Final)

**Generated:** 2026-06-24
**Phases completed:** 0–4
**Total unique findings:** 50
**Severity breakdown:** 17C · 14H · 13M · 6L
**Requires-retraining count:** 14 findings

---

## Critical Findings (must fix before thesis submission / public release)

### [FIN-001] WITHIN_MANIFEST_SECRET semantically inverted between training and inference
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 296), `research/scripts/02_extract/extract_yaml_features.py` (~line 189)
- **Severity:** CRITICAL
- **Requires retraining:** No (fix inference to match training ground truth)
- **Root cause:** The training extractor flags a manifest when env `value` fields contain hardcoded credential strings (regex match on suspicious key names). The inference extractor flags a manifest when env vars use `secretKeyRef` — the *safe* Kubernetes-native pattern. The two conditions fire on mutually opposite inputs: real secrets are missed and safe manifests are flagged.
- **Confirmed by:** F-A1-003, F-C2-001, F-X1-001
- **Fix:** Replace the `secretKeyRef` check in `yaml_parser.py` with a regex scan of env `value` fields identical to the pattern used in `extract_yaml_features.py`.

---

### [FIN-002] NO_READ_ONLY_ROOT_FS always returns 1 at inference (pod-level field does not exist in K8s API)
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 222), `research/scripts/02_extract/extract_yaml_features.py` (~line 144)
- **Severity:** CRITICAL
- **Requires retraining:** No (fix inference)
- **Root cause:** Inference checks `pod_spec["securityContext"].get("readOnlyRootFilesystem")`, but `readOnlyRootFilesystem` is not a valid `PodSecurityContext` field. This always returns `None`, so `pod_writable_fs` is always `True`, so the flag is always 1 for every manifest regardless of container settings. Training correctly checks the flag per-container. Every manifest's RF risk score is inflated by one unit.
- **Confirmed by:** F-A1-001, F-C2-002, F-X1-002
- **Fix:** Remove the pod-level `readOnlyRootFilesystem` check from `yaml_parser.py`; check only per-container `sc.get("readOnlyRootFilesystem")`.

---

### [FIN-003] NO_RUN_AS_NON_ROOT false positive when pod sets runAsUser instead of runAsNonRoot
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 221), `research/scripts/02_extract/extract_yaml_features.py` (~line 106)
- **Severity:** CRITICAL
- **Requires retraining:** No (fix inference; training is correct)
- **Root cause:** Inference uses `pod_sc.get("runAsNonRoot") is not True`. This fires even when the pod enforces non-root via `runAsUser: <non-zero>`. Training uses `pod_run_as_user == 0 or None`, which correctly treats a non-zero `runAsUser` as non-root enforcement. A pod with `{runAsUser: 1000}` and no `runAsNonRoot` field returns flag=0 at training and flag=1 at inference.
- **Confirmed by:** F-A1-002, F-C2-008, F-X1-005
- **Fix:** In `yaml_parser.py`, treat `pod_run_as_user is not None and pod_run_as_user != 0` as equivalent to `runAsNonRoot: true` for the container-loop short-circuit.

---

### [FIN-004] PRIV_REACH edge priority guard: training preserves DIR_PROXIMITY, inference overwrites it
- **Files:** `kubescan/src/kubescan/utils/graph_builder.py` (~line 311), `research/scripts/02_extract/build_graphs.py` (~line 317)
- **Severity:** CRITICAL
- **Requires retraining:** Yes (must pick one canonical behaviour and retrain)
- **Root cause:** `build_graphs.py` (training) guards PRIV_REACH addition with `if not G.has_edge(src, dst)`, preserving DIR_PROXIMITY when an edge already exists. `graph_builder.py` (inference) has no guard and always overwrites. For the same cluster, a pair of co-located escape-capable nodes gets edge_type=0 at training and edge_type=1 at inference. The GNN edge embedding differs, producing a wrong chain probability.
- **Confirmed by:** F-A2-001, F-C3-006, F-X2-001
- **Fix:** Remove the `has_edge` guard from `build_graphs.py` to match inference semantics, then retrain. (Or add the guard to `graph_builder.py` and retrain — both must match.)

---

### [FIN-005] HOSTPATH_MOUNT: feature vector x[24] not updated in training graphs despite PRIV_REACH edges being added
- **Files:** `research/scripts/02_extract/build_graphs.py` (~line 277)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** When YAML parsing detects a `hostPath` volume, `build_graphs.py` sets `node_data[idx]["HOSTPATH_MOUNT"] = 1` and adds PRIV_REACH edges but does NOT update `G.nodes[idx]["features"][HOSTPATH_MOUNT_IDX]`. The `.npz` file receives `x[24]=0` while PRIV_REACH edges imply escape capability. The GNN trains on structurally inconsistent data: edge structure says "escape-capable node", feature vector says "no escape flags set".
- **Confirmed by:** F-C3-001, F-X2-003
- **Fix:** After `node_data[idx]["HOSTPATH_MOUNT"] = 1`, also execute `G.nodes[idx]["features"][FEATURE_COLS.index("HOSTPATH_MOUNT")] = 1.0`.

---

### [FIN-006] total_misconfigs computed on 3 different bases across training pipeline and inference
- **Files:** `research/scripts/04_build_datasets/build_rf_dataset.py` (~line 197), `research/scripts/04_build_datasets/enrich_rf_dataset.py` (~line 205), `kubescan/src/kubescan/model/rf_classifier.py` (line 75)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** Rahman-sourced rows (majority of training data) sum only the 18 Rahman-category flags. BadPods/Goat rows sum all 25. Inference always uses all 25. A single CSV column encodes three different quantities for different rows. The RF learns split points calibrated on mixed-basis values; at inference every manifest is scored with the 25-flag basis, producing systematically wrong risk scores for Rahman-type manifests.
- **Confirmed by:** F-B1-002, F-C4-001, F-C4-007
- **Fix:** Rebuild `rf_dataset.csv` with `total_misconfigs = sum(row[c] for c in FEATURE_COLS)` for ALL rows uniformly. Retrain the RF.

---

### [FIN-007] 3 security flags silently dropped from RF input matrix: SECCOMP_UNCONFINED, VALID_TAINT_SECRET, NO_NETWORK_POLICY
- **Files:** `research/models/train_rf.py` (~line 52), `research/scripts/04_build_datasets/build_rf_dataset.py` (~line 52)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** `RAHMAN_FEATURES` (15 items) used to build the RF X matrix is missing 3 flags that exist in `FEATURE_COLS`. These 3 columns are present in `rf_dataset.csv` but not in the X matrix. The RF was structurally unable to learn from these features. For Rahman-sourced rows they are also written as 0 in the CSV regardless of manifest content (absent from `SEVERITY_WEIGHTS`).
- **Confirmed by:** F-B1-001, F-C4-002
- **Fix:** Audit `RAHMAN_FEATURES` against `FEATURE_COLS`; add the 3 missing flags. Rebuild the RF dataset and retrain.

---

### [FIN-008] P@5 formula divides by k=5 (fixed) instead of min(5, n_attack_chains)
- **Files:** `research/models/evaluate_test_set.py` (~line 138), `research/models/train_gnn.py` (~line 223), `research/models/run_ga_ensemble.py` (compute_objective)
- **Severity:** CRITICAL
- **Requires retraining:** No (evaluation scripts; but early stopping and GA optimisation used the same wrong metric)
- **Root cause:** All three files implement `precision_at_k` as `hits / k` where k=5 fixed. Under the thesis spec and standard definition ("divide by min(5, n_attack_chains)"), when the eval set has fewer than 5 attack chains the formula computes Recall@K, not Precision@K. The GA was also optimised with this non-standard metric. The reported `P@5=0.880` may differ from the standard-definition value.
- **Confirmed by:** F-B4-001, F-B4-002, F-B2-005
- **Fix:** `n_pos = sum(1 for l in labels if l == ATTACK_CHAIN_CLASS); return hits / min(k, n_pos) if n_pos > 0 else 0.0`. Apply in all three files and re-run `evaluate_test_set.py` to confirm or update the reported P@5 figure.

---

### [FIN-009] NO_ROLLING_UPDATE parity broken: inference fix #18 was NOT backported to training extractor
- **Files:** `research/scripts/02_extract/extract_yaml_features.py` (~line 250), `kubescan/src/kubescan/utils/yaml_parser.py` (~line 324)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** Bug fix #18 (commit `19da4ea`) corrected inference to not flag `NO_ROLLING_UPDATE` when `strategy` is an explicit empty dict `{}`. The training extractor still uses `not strategy` which fires for `{}`. For `Deployment.spec.strategy: {}`, training sets flag=1 and inference sets flag=0 — parity broken in the opposite direction from before.
- **Confirmed by:** F-A1-004, F-C2-004, F-X1-004
- **Fix:** Apply the same `strategy_raw is None or not strategy_raw` condition to `extract_yaml_features.py`, then rebuild the training dataset and retrain.

---

### [FIN-010] Fixture repos with contradictory labels: clean and attack manifests co-ingested under the same cluster label
- **Files:** `research/scripts/01_acquire/ingest_attack_repos.py` (~line 85)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** Three fixture repos (`gatekeeper-library`, `kubeaudit-fixtures`, `datree-tests`) contain a mix of intentionally clean and intentionally vulnerable manifests within the same directory tree. All manifests from a repo are ingested with a single repo-level label. Benign example manifests are labeled `y=2` (attack chain) and attack manifests may be labeled clean, giving the RF and GNN contradictory training signal.
- **Confirmed by:** F-C1-001, F-C1-002
- **Fix:** Filter by subdirectory or split repos into per-subfolder clusters with independent labels; or exclude ambiguous repos entirely.

---

### [FIN-011] Kubernetes Goat labeling uses wrong heuristic: single low-severity Rahman flag → y=1, extended flags ignored
- **Files:** `research/scripts/04_build_datasets/enrich_rf_dataset.py` (~line 88)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** Goat manifests are labeled `y=1` if any of the 18 Rahman flags is set (no severity threshold). `HOST_ALIAS=1` → `y=1`. Manifests that only trigger extended flags get `y=0` even when they represent real misconfigurations. Produces both false positives and false negatives.
- **Confirmed by:** F-C4-003
- **Fix:** Apply a severity-weighted threshold: label `y=1` only when `sum(SEVERITY_WEIGHTS[f] * row[f] for f in all_25_flags) >= LABEL_THRESHOLD` (named constant).

---

### [FIN-012] RF split manifest-level: RF risk_scores at x[25] in GNN test graphs trained on GNN-test manifests
- **Files:** `research/models/train_rf.py` (~line 372)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** The RF train/test split uses manifest-level `StratifiedKFold` on `rf_dataset.csv`, entirely independent of the GNN cluster-level splits. A manifest in a GNN test cluster can be in the RF training set. Node feature index 25 (`x[25]`) in GNN test graphs contains the RF `predict_proba` tuned on those same test manifests — cross-layer leakage.
- **Confirmed by:** F-X5-001
- **Fix:** Train the RF only on manifests whose cluster ID appears in `train.txt`; evaluate on `test.txt` manifests only.

---

### [FIN-013] evaluate_test_set.py has no --seed argument; GNN inference entirely unseeded; bootstrap seed hardcoded
- **Files:** `research/models/evaluate_test_set.py` (~line 206, `bootstrap_cis`)
- **Severity:** CRITICAL
- **Requires retraining:** No (evaluation script only)
- **Root cause:** No `--seed` CLI argument. `bootstrap_cis()` has `seed=42` hardcoded. GNN inference in `main()` is entirely unseeded. On CUDA hardware, two consecutive runs can produce different P@5 values. The reported thesis figure `P@5=0.880` is tied to one specific unknown RNG state and cannot be reproduced with a different seed without editing source code.
- **Confirmed by:** F-X6-001, F-X6-002
- **Fix:** Add `--seed INT`; call `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` at start of `main()`; log at INFO; thread seed to `bootstrap_cis`.

---

### [FIN-014] RBAC edge collection: training adds ALL RoleBinding subjects; inference adds only privileged-role subjects
- **Files:** `kubescan/src/kubescan/utils/graph_builder.py` (~line 264), `research/scripts/02_extract/build_graphs.py` (~line 315)
- **Severity:** CRITICAL
- **Requires retraining:** Yes
- **Root cause:** Training adds every `RoleBinding`/`ClusterRoleBinding` subject to `elevated_sas` unconditionally. Inference filters by `roleRef` — only SAs bound to `cluster-admin` or escalation-granting roles get RBAC_PRIV edges. Clusters with custom admin roles have training graphs with RBAC_PRIV edges that inference never produces for the same cluster. The GNN was trained on inflated RBAC signal.
- **Confirmed by:** F-A2-002, F-C3-007, F-X2-002
- **Fix:** Apply the privileged-role filter to `build_graphs.py` to match inference semantics, then retrain.

---

### [FIN-015] SA_AUTOMOUNT_TOKEN: inference checks pod + container level; training and K8s semantics say pod level only
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 284), `research/scripts/02_extract/extract_yaml_features.py` (~line 207)
- **Severity:** CRITICAL
- **Requires retraining:** No (fix inference)
- **Root cause:** Inference checks both pod-level and per-container `automountServiceAccountToken`. Training checks only pod-level. In Kubernetes, the pod-level field governs. A pod with `automountServiceAccountToken: false` and a container override of `true` fires the flag at inference but not at training.
- **Confirmed by:** F-A1-006, F-C2-005, F-X1-006
- **Fix:** Remove the per-container check from `yaml_parser.py`; check only `pod_spec.get("automountServiceAccountToken") is not False`.

---

### [FIN-016] NO_DEFAULT_NSPACE and USES_DEFAULT_SA case-insensitive in training, case-sensitive at inference
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 312), `research/scripts/02_extract/extract_yaml_features.py` (~line 232)
- **Severity:** CRITICAL
- **Requires retraining:** No (fix inference)
- **Root cause:** Training uses `.lower() == "default"` for both namespace and SA name comparisons. Inference uses exact `== "default"`. `metadata.namespace: "Default"` is flagged by training and missed at inference.
- **Confirmed by:** F-A1-011, F-C2-006
- **Fix:** Add `.lower()` before `== "default"` comparisons in `yaml_parser.py` for both fields.

---

### [FIN-017] Heuristic risk_score at x[25] in training graphs has different distribution than RF predict_proba at inference
- **Files:** `research/scripts/04_build_datasets/build_graph_cache.py` (~line 145)
- **Severity:** CRITICAL
- **Requires retraining:** Yes (after FIN-006 and FIN-007 are fixed and RF is retrained)
- **Root cause:** Training graphs store `x[25]` as a severity-weighted heuristic score. At inference, `x[25]` is the RF's `predict_proba`. These have fundamentally different distributions — heuristic is roughly linear in flag count; RF output is shaped by decision tree boundaries. The GNN was trained on heuristic node feature distributions and receives RF probability distributions at inference.
- **Confirmed by:** F-C4-004
- **Fix:** After retraining the RF with the full 25-flag feature set, use the retrained RF to regenerate `x[25]` for all training graphs.

---

## High Findings

### [FIN-018] DOCKERSOCK_PATH: inference misses container volumeMounts; training scans both volumes and volumeMounts
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 193), `research/scripts/02_extract/extract_yaml_features.py` (~line 162)
- **Severity:** HIGH
- **Requires retraining:** No (fix inference)
- **Root cause:** Inference scans only `pod_spec["volumes"]` for hostPath entries. Training also scans container `volumeMounts[].mountPath` for docker socket paths. A container declaring a docker socket mount without a corresponding `volumes` entry gets flag=1 from training and flag=0 from inference.
- **Confirmed by:** F-A1-009, F-C2-010, F-X1-003
- **Fix:** Add container `volumeMounts` scan in `yaml_parser.py:_extract_volume_features`, checking `mountPath` for `/docker.sock` or `/var/run/docker.sock`.

---

### [FIN-019] val.txt double-dipping: used for both GNN early stopping and GA weight optimisation
- **Files:** `research/scripts/05_split/create_splits.py` (~line 1), `research/models/run_ga_ensemble.py` (~line 136)
- **Severity:** HIGH
- **Requires retraining:** No (pipeline restructuring needed)
- **Root cause:** The 15-graph `val.txt` is used both for GNN checkpoint selection (early stopping on val P@5) and as the GA optimisation target. The GA maximises P@5 on the same 15 graphs that selected the best GNN checkpoint. The GA weights are over-fit to this set. True generalisation is measured only by the 17-graph test set.
- **Confirmed by:** F-C5-001, F-X5-003
- **Fix:** Split val into `val_gnn.txt` (early stopping) and `val_ga.txt` (GA optimisation). Or make `--oof` mode the default and default GA evaluation target.

---

### [FIN-020] load_fold_ensemble does not pass num_edge_types or edge_emb_dim; non-default training → cryptic RuntimeError
- **Files:** `kubescan/src/kubescan/model/gat_encoder.py` (~line 157), `research/models/train_gnn.py` (~line 137, 418)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `load_fold_ensemble` and `make_model` never pass `num_edge_types` or `edge_emb_dim` to `KubeGAT`. If training uses non-default hyperparameters that affect tensor shapes, the checkpoint has different-dimensional tensors than what `load_fold_ensemble` builds, causing `RuntimeError: size mismatch`. No `gnn_config.json` is saved alongside checkpoints to allow reconstruction of the exact architecture.
- **Confirmed by:** F-A4-001, F-A4-002, F-B2-001, F-B2-002, F-X3-001, F-X3-002, F-X3-003, F-X3-007
- **Fix:** Save `gnn_config.json` alongside every `.pt` checkpoint with all 8 hyperparameters. `load_fold_ensemble` should accept a `GATConfig` object and read the config from JSON if not provided.

---

### [FIN-021] set_global_seed() utility does not exist; training scripts seed inconsistently or not at all
- **Files:** `research/models/train_rf.py` (~line 316), `research/models/train_gnn.py` (~line 379), `research/models/run_ga_ensemble.py` (~line 410), `research/models/evaluate_test_set.py`
- **Severity:** HIGH
- **Requires retraining:** No (but trained checkpoints cannot be reproduced)
- **Root cause:** `set_global_seed()` does not exist anywhere in the codebase. `train_rf.py` seeds neither `random` nor `numpy`. `train_gnn.py` misses `random` and `torch.cuda`. `run_ga_ensemble.py` uses only scoped RNGs. `evaluate_test_set.py` seeds nothing. Results cannot be confirmed as reproducible.
- **Confirmed by:** F-X6-003, F-B1-003, F-B2-003, F-X6-005, F-X6-006
- **Fix:** Create `kubescan/utils/seed_utils.py` with `def set_global_seed(seed: int) -> None: random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); logger.info("seed=%d", seed)`. Call it at the top of every training script.

---

### [FIN-022] pickle.load used as RF fallback without access control; enables arbitrary code execution
- **Files:** `kubescan/src/kubescan/model/rf_classifier.py` (line 112), `research/models/predict.py` (~line 460)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `rf_classifier.py` uses `pickle.load()` as a fallback when no `.skops` file exists, bypassing all type validation. `predict.py` uses `pickle.load()` unconditionally with no validation and no warning. A tampered `rf_model.pkl` achieves arbitrary code execution at model load time.
- **Confirmed by:** F-X4-001, F-X4-002
- **Fix:** Gate the pickle fallback in `rf_classifier.py` behind an explicit `--allow-pickle` CLI flag (default: disabled). In `predict.py`, replace `pickle.load` with the `.skops`-first logic from `RFClassifier.from_checkpoints()`.

---

### [FIN-023] patch_hostpath_column.py crashes on import: sys.path points to fixes/ instead of 02_extract/
- **Files:** `research/scripts/fixes/patch_hostpath_column.py` (line 3)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `sys.path.insert(0, str(Path(__file__).parent))` inserts `fixes/` into sys.path, but `yaml_feature_extractor` lives in `02_extract/`. Any invocation immediately raises `ModuleNotFoundError: No module named 'yaml_feature_extractor'`.
- **Confirmed by:** F-C6-001
- **Fix:** Change to `sys.path.insert(0, str(Path(__file__).parent.parent / "02_extract"))`. Better: import directly from `kubescan.utils.yaml_parser`.

---

### [FIN-024] KubescanError not exported from package __all__; external callers cannot catch it
- **Files:** `kubescan/src/kubescan/__init__.py` (~line 14)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `KubescanError` is defined in `kubescan/exceptions.py` but not re-exported from the package's `__all__`. `from kubescan import KubescanError` raises `ImportError`, making the CLI-boundary catch contract unusable for external integrators.
- **Confirmed by:** F-A7-001
- **Fix:** Add `from .exceptions import KubescanError` to `__init__.py` and include in `__all__`.

---

### [FIN-025] _run_inference_pipeline unwrapped in CLI: raw Python tracebacks exposed to users
- **Files:** `kubescan/src/kubescan/cli.py` (~lines 418, 490)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `_run_inference_pipeline` is called in both `scan` and `live` commands without a `try/except KubescanError` wrapper. Any `KubescanError` or `RuntimeError` from a corrupt checkpoint propagates as a raw Python traceback exposing internal paths and implementation details.
- **Confirmed by:** F-A6-001, F-A6-002
- **Fix:** Wrap both call sites with `try: ... except KubescanError as exc: raise click.ClickException(str(exc)) from exc`.

---

### [FIN-026] torch import unconditional in device_utils.py: ModuleNotFoundError escapes KubescanError hierarchy
- **Files:** `kubescan/src/kubescan/utils/device_utils.py` (line 8)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `import torch` at module level causes importing `device_utils` in a torch-free environment to raise `ModuleNotFoundError`, which does not inherit `KubescanError` and escapes the CLI-boundary exception handler.
- **Confirmed by:** F-A7-002
- **Fix:** Lazy-import torch inside `resolve_device()` and wrap with `try/except ImportError: raise KubescanDependencyError("torch is required") from exc`.

---

### [FIN-027] test_rf_classifier.py and test_gat_encoder.py are missing entirely
- **Files:** `kubescan/tests/unit/test_rf_classifier.py` (absent), `kubescan/tests/unit/test_gat_encoder.py` (absent)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `RFClassifier`, `_validate_skops_types`, `predict_risk_scores`, `from_checkpoints`, `KubeGAT.forward`, `load_fold_ensemble`, and `GATConfig` have zero unit tests. The security-critical `_validate_skops_types` (which prevents loading unsafe pickle types) is completely untested.
- **Confirmed by:** F-A8-006, F-A8-007
- **Fix:** Create both test files. `test_rf_classifier.py`: skops loading, type validation rejection, risk score shape, pickle fallback. `test_gat_encoder.py`: forward-pass shape, `load_fold_ensemble` error path.

---

### [FIN-028] DataLoader shuffle non-deterministic: no worker_init_fn and no generator seed
- **Files:** `research/models/train_gnn.py` (~line 264)
- **Severity:** HIGH
- **Requires retraining:** No (but training results are not reproducible)
- **Root cause:** `DataLoader(num_workers=N, shuffle=True)` with no `worker_init_fn` and no `generator` argument. Worker subprocesses diverge unpredictably. Batch ordering is non-deterministic across runs even with the same `--seed`.
- **Confirmed by:** F-X6-004, F-B2-004
- **Fix:** `g = torch.Generator(); g.manual_seed(args.seed + fold_idx); DataLoader(..., shuffle=True, generator=g, worker_init_fn=lambda wid: np.random.seed(args.seed + wid))`.

---

### [FIN-029] RF cross-validation uses manifest-level StratifiedKFold; CV F1=0.9935 is optimistic
- **Files:** `research/models/train_rf.py` (~line 195)
- **Severity:** HIGH
- **Requires retraining:** No (but reported CV F1 is not trustworthy)
- **Root cause:** `run_cv()` uses manifest-level `StratifiedKFold` with no grouping by cluster or repo. Manifests from the same repo appear on both sides of fold boundaries. The RF can memorise repo-level YAML conventions, inflating CV F1 above what holds for truly unseen repos.
- **Confirmed by:** F-B1-004, F-X5-004
- **Fix:** Use `GroupKFold` or `StratifiedGroupKFold` with `groups=cluster_id` column.

---

### [FIN-030] build_graph_cache.py writes .npz non-atomically: crash leaves corrupt file with valid name
- **Files:** `research/scripts/04_build_datasets/build_graph_cache.py` (~line 182)
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** `np.savez(output_path, ...)` writes directly to the final path. A crash mid-write leaves a partially-written `.npz` file that has the same name as a completed file. On re-run the corrupt graph is read as valid, silently corrupting GNN training.
- **Confirmed by:** F-C4-005
- **Fix:** `tmp = output_path.with_suffix(".npz.tmp"); np.savez(tmp, ...); tmp.rename(output_path)`.

---

### [FIN-031] Missing critical yaml_parser tests: strategy:{}, CronJob guard, runAsNonRoot inheritance, INSECURE_HTTP in env
- **Files:** `kubescan/tests/unit/test_yaml_parser.py`
- **Severity:** HIGH
- **Requires retraining:** No
- **Root cause:** No tests cover: `strategy: {}` triggering `NO_ROLLING_UPDATE`; `NO_ROLLING_UPDATE` NOT firing on `CronJob`/`Job`/`Pod`; pod-level `runAsNonRoot: true` suppressing the flag for containers without their own setting; `INSECURE_HTTP` in container env var values. These edge cases had confirmed bugs that were recently fixed.
- **Confirmed by:** F-A8-001, F-A8-002, F-A8-003, F-A8-004
- **Fix:** Add four named tests following the `test_<function>_<condition>_<expected>` convention.

---

## Medium Findings

### [FIN-032] SECCOMP_UNCONFINED misses legacy pre-1.19 pod annotations at inference
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 219)
- **Severity:** MEDIUM
- **Requires retraining:** No (inference-only fix; training already checks both)
- **Root cause:** Inference checks only `seccompProfile.type` (K8s 1.19+). Training also scans `seccomp.security.alpha.kubernetes.io/*` annotations. Pre-1.19 manifests with the annotation return flag=1 from training and flag=0 from inference.
- **Confirmed by:** F-A1-005, F-C2-003, F-X1-008
- **Fix:** Add annotation scan in `yaml_parser.py`: iterate annotations for any key containing `"seccomp"` whose value contains `"unconfined"`.

---

### [FIN-033] TRUE_HOST_* and CAP_SYS_*: truthy vs is-True divergence for integer values
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 179)
- **Severity:** MEDIUM
- **Requires retraining:** No (PyYAML always produces Python bool True for YAML `true:`)
- **Root cause:** Five flags use truthy checks at inference while training uses `is True`. The divergence only manifests for non-standard manifests using integer `1` for boolean fields. Impact is very low in practice but creates semantic inconsistency.
- **Confirmed by:** F-A1-007, F-A1-008, F-C2-009, F-C2-011
- **Fix:** Change inference checks to `is True` for all five flags to match training.

---

### [FIN-034] UNTRUSTED_REGISTRY: training uses split(":")[0] before split("/"), misparses host:port/image format
- **Files:** `research/scripts/02_extract/extract_yaml_features.py` (~line 170)
- **Severity:** MEDIUM
- **Requires retraining:** Yes (training-side fix)
- **Root cause:** Training does `image.split(":")[0]` first, losing the port component for `registry:5000/image:tag`. Inference uses `image.split("/")[0]` which correctly extracts `"registry:5000"` as the registry. Private registries with non-standard ports may be misclassified.
- **Confirmed by:** F-A1-010, F-C2-007, F-X1-010
- **Fix:** Align training to use `image.split("/")[0]` as registry extraction. Retrain after applying.

---

### [FIN-035] DOCKERSOCK_PATH and HOSTPATH_MOUNT use different path matching: substring vs exact
- **Files:** `kubescan/src/kubescan/utils/yaml_parser.py` (~line 193), `research/scripts/02_extract/extract_yaml_features.py` (~line 162)
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Root cause:** Training uses exact paths for DOCKERSOCK_PATH; inference uses `"docker.sock"` substring. A volume at `/run/docker.sock` is flagged as `DOCKERSOCK_PATH=1` by inference but `HOSTPATH_MOUNT=1` by training — different flags for the same security issue.
- **Confirmed by:** F-X1-007
- **Fix:** Align path matching in both files; prefer inference's substring match.

---

### [FIN-036] IMAGE_USES_LATEST: training uses split(":")[0] which misses untagged registry:port/image
- **Files:** `research/scripts/02_extract/extract_yaml_features.py` (~line 383)
- **Severity:** MEDIUM
- **Requires retraining:** Yes (training-side false negative)
- **Root cause:** Training splits on `:` first to strip the tag, which fails for `registry:5000/myapp` (no explicit tag). Inference uses `rsplit(":", 1)` and correctly detects the missing tag.
- **Confirmed by:** F-X1-009
- **Fix:** Use `rsplit(":", 1)` in `extract_yaml_features.py`. Retrain after applying.

---

### [FIN-037] extract_yaml_features.py redefines TRUSTED_REGISTRIES locally (DRY violation, future divergence risk)
- **Files:** `research/scripts/02_extract/extract_yaml_features.py` (~line 52)
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Root cause:** Bug #7 fixed a content mismatch between the local definition and the canonical one in `yaml_parser.py`. The duplication remains — future changes to `yaml_parser.TRUSTED_REGISTRIES` will not propagate to the training extractor.
- **Confirmed by:** F-C2-012
- **Fix:** `from kubescan.utils.yaml_parser import TRUSTED_REGISTRIES` — delete the local definition.

---

### [FIN-038] _safe_load_all and _get_pod_spec re-implemented locally in build_graphs.py
- **Files:** `research/scripts/02_extract/build_graphs.py` (~lines 80–110)
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Root cause:** Both have canonical implementations in `kubescan.utils.yaml_parser`. The local versions diverge (e.g., bare-except) and will not receive upstream fixes.
- **Confirmed by:** F-C3-003, F-X2-004, F-X2-005
- **Fix:** Delete both local implementations; import from `kubescan.utils.yaml_parser`.

---

### [FIN-039] run_gnn_ensemble() locally re-defined in predict.py (DRY violation)
- **Files:** `research/models/predict.py` (~line 297)
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Root cause:** `run_gnn_ensemble()` is fully defined again in `predict.py`, currently identical to the canonical version in `ga_ensemble.py`. If the canonical version is updated, `predict.py` silently uses the old formula.
- **Confirmed by:** F-B5-001, F-X3-005
- **Fix:** Delete the local definition; `from kubescan.model.ga_ensemble import run_gnn_ensemble`.

---

### [FIN-040] Imputation medians computed over full dataset before RF train/test split (mild data leakage)
- **Files:** `research/models/train_rf.py` (~line 109)
- **Severity:** MEDIUM
- **Requires retraining:** No (but RF test F1 is slightly inflated)
- **Root cause:** `np.nanmedian(X_raw, axis=0)` is computed over all rows before the train/test split. Imputed medians for extended features with high NaN rates are shifted by test-set values, creating mild data leakage.
- **Confirmed by:** F-X5-002
- **Fix:** Split first; compute medians only on training rows; apply to val/test using training-set medians.

---

### [FIN-041] GNNDataset .npz key "edge_type" not validated; silent zero-fill if key name changes
- **Files:** `research/scripts/04_build_datasets/gnn_dataset.py` (~line 64)
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Root cause:** `edge_attr` is loaded as `data["edge_type"]`. If the key name changes, the lookup silently fails and edge attributes default to zeros, training the GNN with no edge type signal.
- **Confirmed by:** F-C4-009
- **Fix:** Define `_EDGE_TYPE_NPZ_KEY = "edge_type"` as a shared constant; assert its presence before accessing.

---

### [FIN-042] build_graph_cache.py does not validate output .npz schema after saving
- **Files:** `research/scripts/04_build_datasets/build_graph_cache.py` (~line 165)
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Root cause:** No post-write validation that saved `.npz` has `x.shape[1] == NODE_FEATURE_DIM`. A wrong feature dimension would produce wrong-shaped graphs that crash at `GNNDataset.__getitem__` with an uninformative `shape mismatch` error.
- **Confirmed by:** F-C4-006
- **Fix:** `assert np.load(output_path)["x"].shape[1] == NODE_FEATURE_DIM` after saving.

---

### [FIN-043] create_splits.py uses stratification without repo-level grouping; train/test contamination by repo style
- **Files:** `research/scripts/05_split/create_splits.py` (~line 72)
- **Severity:** MEDIUM
- **Requires retraining:** No (but val/test metrics are optimistic)
- **Root cause:** Split uses label stratification on cluster IDs with no grouping by repo. Clusters from the same source repo can appear in both train and test. The GNN can partially memorise repo-level YAML style conventions.
- **Confirmed by:** F-C5-002
- **Fix:** Use `StratifiedGroupKFold` with `groups=repo_id` (extracted from cluster ID prefix).

---

### [FIN-044] escape_signal computed inline in run_ga_ensemble.py instead of calling compute_escape_signal()
- **Files:** `research/models/run_ga_ensemble.py` (~line 111)
- **Severity:** MEDIUM
- **Requires retraining:** No
- **Root cause:** `_infer_dataset` re-implements escape signal computation inline. If `compute_escape_signal` is updated in `ga_ensemble.py`, the GA will optimise with the old definition while inference uses the new one — GA weights become wrong for the updated signal.
- **Confirmed by:** F-B3-001
- **Fix:** `from kubescan.model.ga_ensemble import compute_escape_signal` and call it on each cluster's feature matrix slice.

---

## Low Findings (deferred / optional)

### [FIN-045] Bare open() calls in library code (7 occurrences across 6 files)
- **Files:** `kubescan/src/kubescan/model/ga_ensemble.py` (line 80), `kubescan/src/kubescan/model/rf_classifier.py` (line 112), `kubescan/src/kubescan/utils/yaml_parser.py` (~line 86), `research/models/run_ga_ensemble.py` (~lines 562, 572), `research/scripts/02_extract/extract_yaml_features.py` (~line 290), `research/scripts/fixes/patch_hostpath_column.py` (~line 28)
- **Severity:** LOW
- **Root cause:** Multiple files use bare `open()` instead of `path.open()` — violates CLAUDE.md rule 6. No runtime impact.
- **Confirmed by:** F-A3-002, F-A5-003, F-X4-003, F-X4-004, F-X4-005, F-B3-002, F-B3-003
- **Fix:** Mechanically replace all bare `open()` calls with `path.open()`.

---

### [FIN-046] Magic literals in training hyperparameters not extracted to Config dataclasses
- **Files:** `research/models/train_rf.py` (~lines 354, 507), `research/scripts/05_split/create_splits.py` (~line 48), `research/scripts/03_augment/augment_graphs.py` (~line 18), `research/scripts/04_build_datasets/build_graph_cache.py` (~line 50)
- **Severity:** LOW
- **Root cause:** RF hyperparameters (`n_estimators=500`, `test_size=0.20`, `n_splits=5`, `F1_TARGET=0.85`), split ratios, augmentation count, and default risk score are bare literals not in Config dataclasses.
- **Confirmed by:** F-B1-005, F-B1-006, F-C5-004, F-C6-004, F-C4-014
- **Fix:** Create `RFConfig`, `SplitConfig` dataclasses; extract all literals to `Final` constants.

---

### [FIN-047] Stale docstrings claim NODE_FEATURE_DIM=25 and risk_score at index 24 (should be 26 and 25)
- **Files:** `research/scripts/02_extract/build_graphs.py` (~lines 5, 12), `research/scripts/04_build_datasets/gnn_dataset.py` (~line 8), `research/models/train_gnn.py` (line 12)
- **Severity:** LOW
- **Root cause:** Multiple module docstrings state `NODE_FEATURE_DIM=25` and `risk_score at index 24`. Correct values are 26 and 25. `train_gnn.py` also says "18 Rahman flags + 6 extended = 25" when there are 7 extended flags.
- **Confirmed by:** F-C3-002, F-C4-008, F-A4-004, F-A4-005
- **Fix:** Update all four docstrings to `NODE_FEATURE_DIM=26 (25 binary flags from FEATURE_COLS + risk_score at index 25)`.

---

### [FIN-048] Property test silently returns on None from _extract_file, masking extractor regressions
- **Files:** `kubescan/tests/unit/test_properties.py` (~line 77)
- **Severity:** LOW
- **Root cause:** `test_extractor_features_are_binary_and_complete` has `if result is None: return` for a well-formed Pod manifest. If the extractor starts returning `None` due to a bug, the test silently passes.
- **Confirmed by:** F-A8-009
- **Fix:** `assert result is not None, "extractor returned None for a valid Pod manifest"`.

---

### [FIN-049] Test fixture creates only 2 of 5 GNN fold checkpoints; degraded ensemble warning fires on every CI run
- **Files:** `kubescan/tests/fixtures/make_fixtures.py` (~line 58)
- **Severity:** LOW
- **Root cause:** The fixture creation loop only creates 2 fold checkpoints while `NUM_FOLDS=5`. The full 5-fold path is never exercised by tests.
- **Confirmed by:** F-A8-010
- **Fix:** Change loop to `for fold in range(NUM_FOLDS):`.

---

### [FIN-050] SEVERITY_WEIGHTS dict duplicated in build_rf_dataset.py and build_graph_cache.py
- **Files:** `research/scripts/04_build_datasets/build_rf_dataset.py` (~line 47), `research/scripts/04_build_datasets/build_graph_cache.py`
- **Severity:** LOW
- **Root cause:** The same `SEVERITY_WEIGHTS` dict with identical float values is defined in two files. A change to one silently diverges from the other.
- **Confirmed by:** F-C4-011
- **Fix:** Move `SEVERITY_WEIGHTS` to a shared constants module and import in both files.

---

## Findings requiring model retraining

**RF retraining required (fix training data first, then retrain):**
- FIN-006 — `total_misconfigs` inconsistent basis across rows; rebuild `rf_dataset.csv`
- FIN-007 — 3 flags dropped from RF X matrix; rebuild `rf_dataset.csv`
- FIN-012 — RF split manifest-level; change to cluster-level split protocol

**GNN retraining required (fix training graphs first, then retrain):**
- FIN-004 — PRIV_REACH `has_edge` guard mismatch; fix `build_graphs.py` and retrain
- FIN-005 — HOSTPATH_MOUNT `x[24]` not updated in training graphs; fix and retrain
- FIN-009 — NO_ROLLING_UPDATE not backported to training extractor; fix, rebuild, retrain
- FIN-010 — Contradictory cluster labels from fixture repos; fix ingestion, rebuild, retrain
- FIN-011 — Goat labeling heuristic wrong; fix `enrich_rf_dataset.py`, rebuild, retrain
- FIN-014 — RBAC edge collection mismatch; fix `build_graphs.py`, retrain

**GA re-optimisation required (after RF and GNN are retrained):**
- FIN-008 — P@5 formula wrong; fix all three files, re-run GA, re-evaluate and update thesis numbers
- FIN-017 — x[25] heuristic vs predict_proba distribution gap; regenerate graphs with retrained RF, retrain GNN, re-run GA
- FIN-034 — UNTRUSTED_REGISTRY training-side fix; rebuild dataset, retrain
- FIN-036 — IMAGE_USES_LATEST training-side fix; rebuild dataset, retrain

**Recommended retraining order:** Fix FIN-007 → FIN-006 → FIN-009 → FIN-010 → FIN-011 → FIN-012 (rebuild RF dataset and retrain RF) → FIN-004 → FIN-005 → FIN-014 → FIN-034 → FIN-036 (rebuild GNN graphs and retrain GNN) → FIN-017 (regenerate x[25]) → retrain GNN again → FIN-008 (re-run GA and evaluate).

---

## Confirmed-clean items (Phase 4 cleared)

- **Checkpoint key format consistent across all readers/writers** (X7): `train_gnn.py` saves a raw flat `state_dict`; all four readers pass it directly to `load_state_dict`. Key names match.
- **All `torch.load` calls use `weights_only=True`** (X4/X7): Confirmed across all files.
- **No `yaml.load` (unsafe)** (X4): All YAML loading uses `yaml.safe_load` or `yaml.safe_load_all`.
- **No `subprocess` with `shell=True`** (X4): All subprocess calls use list form.
- **No `eval` or `exec` on untrusted input** (X4): Confirmed absent across all audited files.
- **Augmented graph exclusion from val/test is correct** (X5): The `_aug_` infix sentinel correctly prevents augmented variants of val/test clusters from entering any training split. 120 aug variants confirmed excluded.
- **GNN train/val/test splits not contaminated at cluster level** (X5): No cluster ID appears in more than one of train/val/test.
- **No feature normalisation fitted on train and applied to test** (X5): Confirmed absent.
- **`ga_weights.json` keys consistent** (X7): `w_rf`, `w_gnn`, `w_escape` are written by `run_ga_ensemble.py` and read correctly by `EnsembleScorer.__init__()`.
- **INSECURE_HTTP parity: MATCH** (X1): Both training and inference implement recursive spec scan excluding localhost, case-insensitive.
- **NO_SECU_CONTEXT parity: MATCH** (X1): Both check any container missing `securityContext`.
- **HOST_ALIAS parity: MATCH** (X1): Both check `hostAliases` truthy.
- **NO_RESO parity: MATCH** (X1): Both check any container missing `resources.limits`.
- **scan_security_tools.py output NOT used in any training feature** (C3): Confirmed dead code — output never imported by downstream build steps.
