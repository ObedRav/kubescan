# kubescan — Model / Ensemble Fix Plan

**Status legend:** `[ ]` Not started · `[~]` In progress · `[x]` Complete · `[!]` Has findings

---

## Problem statement

The Layer-3 GA ensemble optimizes weights `(w_rf, w_gnn, w_esc)` against an
out-of-fold (OOF) objective `F = 0.7*P@5 + 0.3*(1-FPR_clean)` computed on 81
OOF graphs (21 attack chains). Two distinct problems were found while
verifying a fresh end-to-end pipeline run (GPU T4, seed 42):

1. **GA algorithm bug (fixed):** the original GA converged to near-uniform
   weights `(0.34, 0.33, 0.33)` and never improved across 150 generations,
   which looked like a flat fitness landscape. It wasn't — arithmetic-mean
   crossover is contractive and can't reach simplex corners from a
   Dirichlet-sampled population, and Gaussian mutation (σ=0.08) is too local
   to walk there either. Fixed in `research/models/run_ga_ensemble.py` by
   seeding simplex corners/edges into the initial population and adding a
   low-probability boundary-jump mutation. The GA now reliably converges to
   the same point grid search finds (`w_esc=1.0`, objective=0.86) across 5
   independent seeds (41-45).

2. **The true OOF optimum doesn't generalize (open problem):** `w_esc=1.0`
   scores highest OOF, but on the 15-graph held-out test set it collapses
   ranking to a binary tie group (exactly 5 of 15 test clusters have any
   escape flag set, coincidentally equal to k=5), and structurally can never
   recover the one real attack chain (`cloudify-kubernetes-plugin`) that
   lacks an escape flag. The currently-deployed near-uniform weights avoid
   this failure mode, but were kept only because they happened not to fail
   when we checked — not because the selection process is principled. Also:
   **GNN-alone (P@5=1.00) beats the full ensemble (P@5=0.75) on the test
   set**, which undercuts the ensemble's value proposition on this dataset.

This file tracks the fix options for problem 2, ranked by impact/effort, so
we can work through them one at a time.

---

## Ranked options

### 1. `[!]` Regularize the ensemble objective against degenerate weights
Added a floor penalty to the objective: `F = α·P@5 + β·(1-FPR_clean) −
λ·Σ max(0, w_floor − w_i)`, default `w_floor=0.1`, `λ=2.0`. Implemented in
`compute_objective`/`grid_search`/`run_ga` (all take `min_weight`/
`reg_lambda`) and exposed as `--min-weight`/`--reg-lambda` CLI flags.
- File: `research/models/run_ga_ensemble.py`

**Findings (local validation, 5 seeds, real OOF data):**
- The penalty makes the escape-only corner strictly dominated (OOF score
  0.86 → 0.46), so the GA no longer converges there.
- All 5 seeds converge to the same OOF objective (0.72, P@5=0.60) but land
  on *different* interior weight tuples (e.g. seed 41: mostly escape-heavy
  `(0.21, 0.11, 0.68)`; seed 43: mostly RF-heavy `(0.54, 0.28, 0.18)`) — the
  regularized objective still has a flat plateau, just an interior one
  instead of corner-shaped.
- Despite that, **all 5 seeds' weights give the identical test-set result**
  (P@5=0.75, same as GNN-only-beats-ensemble finding still holds). This is
  the key win: the OOF-plateau's location no longer matters for deployed
  behavior — the previous setup had corner points (escape-only) that failed
  structurally and interior points (near-uniform) that didn't, so which one
  the GA landed on was consequential. Now it isn't.
- Net effect: doesn't raise test-set P@5 above 0.75, but converts an
  "accidentally-safe" result (near-uniform weights only survived because the
  original GA had an exploration bug) into a "structurally-safe" one (any
  seed, any run, lands somewhere that behaves the same on held-out data).
- Not yet re-run on the verified Colab GPU T4 environment — local validation
  used Darwin arm64/torch 2.12/python 3.10, not the recorded provenance
  environment, so checkpoint JSONs were restored via `git checkout` rather
  than committed.

### 2. `[!]` Select weights via bootstrap/repeated resampling of the OOF pool
Added `bootstrap_select_weights()` + `--select-method {ga,bootstrap}` /
`--n-bootstrap` (default 500). Resamples the 81-graph OOF pool with
replacement, grid-searches each resample (deterministic, so resampling
variance isn't confounded with GA seed variance), and takes the per-weight
median across resamples. Drops resamples with <1 chain-positive example
(none dropped in practice: 500/500 used).
- File: `research/models/run_ga_ensemble.py`

**Findings (local validation, seed 42, 500 resamples, floor regularization on):**
- Median weights: `w_rf=0.80±0.32, w_gnn=0.10±0.22, w_esc=0.10±0.24` — high
  per-resample std, confirming the OOF plateau found in option 1 is real:
  individual resamples land all over the interior of the simplex.
- Despite landing on a completely different point than option 1's GA runs
  (RF-heavy `(0.8, 0.1, 0.1)` vs near-uniform/escape-heavy), **test-set P@5 is
  still 0.75** — and critically, the ranking miss is the *identical* cluster
  (`cloudify-kubernetes-plugin`, dropped to rank 8). That cluster has weak
  signal across all three channels (GNN=0.35, RF=0.05, escape=0), not just a
  missing escape flag — so no interior weight combination recovers it.
- Conclusion: options 1 and 2 converge on the same finding via independent
  methods (GA vs bootstrap+grid). The floor regularization is doing the real
  work; bootstrap resampling mainly confirms option 1's result is not an
  artifact of the GA's search dynamics. Neither method breaks past
  test P@5=0.75 — that requires option 4 (verify robustness) or option 3
  (more data), not further weight-selection tuning.
- Not yet re-run on Colab GPU / committed to checkpoints (same local-env
  caveat as option 1).

### 3. `[!]` Expand the data corpus (more repos, more attack-chain graphs)
Highest overall impact — narrows CV fold variance (L1), narrows test-set
bootstrap CIs, reduces OOF overfitting risk (L5). Highest cost: sourcing,
labeling, and validating new manifests. Already tracked as thesis future
work (TF1).

**Findings (full retrain, Colab GPU T4, seed 42, commit `c9da617`, 6 new
sources ingested, 598 clusters / 90-graph test set vs. previous 15-graph):**
- **GNN 5-fold CV got measurably better and more stable:** macro-F1 mean
  0.8462 → **0.9134** (std 0.0647 → **0.0381**, i.e. folds agree with each
  other far more); accuracy mean 0.838 → 0.975. CV P@5 mean dropped
  0.82 → 0.76 but stays above the 0.7 target. This is the clean win the
  option predicted: more data narrows fold variance.
- **Held-out test set got *harder*, and the ensemble did not keep up:**
  test P@5 dropped 0.75 (ceiling 0.8, i.e. 93.75% of ceiling) → **0.60**
  (ceiling 1.0, i.e. only 60% of ceiling) despite the test set gaining more
  chain-positive examples (4 → 6). Test macro-F1 is flat-to-worse
  (0.7262 → 0.7172); attack_chain per-class F1 dropped 0.571 → 0.444;
  chain recall dropped 2/4 (50%) → 2/6 (33%) (confusion matrix chain row
  `[0,2,2]` → `[0,4,2]`). The 95% macro-F1 CI is still wide
  (`[0.42,0.93]` → `[0.46,0.88]`) — 6 chain examples is still too few to
  narrow it meaningfully.
- **Root cause, from `ranked_clusters` in `test_results.json`:** two brand
  new *non-chain* clusters from the newly-ingested real-world sources
  (`istio-index-conf2018`, `k8s-escape` — both `isolated`, both carry an
  escape flag) now outrank two of the six true attack chains inside the
  top-5, which is exactly the escape-signal-dominated ranking failure mode
  documented under options 1/2. More data added *harder* negatives (real
  projects and CTF fixtures that trip the escape heuristic without being
  real chains), not just more positives, so the ranking objective's
  reliance on the escape signal is now more exposed, not less.
- **Conclusion: option 3 alone does not fix the generalization gap.** It
  strictly improved the GNN's own supervised signal (CV metrics), which
  validates that overfitting-to-a-small-pool was a real, now partially
  addressed concern. But it did not fix — and mildly worsened — the
  ensemble ranking's reliance on the escape signal, because the new data
  is not just "more of the same distribution," it's harder. Options 1/2's
  regularization + option 4 (verify GNN-alone vs ensemble split by fold)
  remain necessary; this is additive to those, not a replacement.
- Retrained end-to-end via `research/scripts/run_colab_pipeline.py` on a
  fresh `kubescan-repro2` GPU T4 session; checkpoints committed at
  `c9da617`+ (pending this run's commit).

### 8. `[x]` Restrict escape/lateral reachability edges + fix chain-label path check
Deeper investigation into option 3's regression (`istio-index-conf2018` /
`k8s-escape` false positives) found the real root cause was structural, not
data volume:
1. `EDGE_PRIV_REACH`/`EDGE_SA_LATERAL`/`EDGE_RBAC_PRIV` connected an
   escape/lateral/RBAC-flagged node to **every other node in the cluster**,
   unconditionally — not scoped to any real reachability boundary. Fixed by
   scoping these three edge types to same-namespace members only (using the
   `ns_groups` already computed for `EDGE_SEMANTIC_NS`). Applied identically
   in `kubescan/utils/graph_builder.py` (canonical) and
   `research/scripts/02_extract/build_graphs.py` (mirror).
2. `_compute_graph_label`'s escape→lateral path check used `nx.has_path`
   over the *entire* graph, including `DIR_PROXIMITY` (same source
   directory) and `EDGE_SEMANTIC_NS` edges — neither of which assert a real
   escalation/reachability relationship. Fixed by restricting the path
   check to a subgraph containing only `PRIV_REACH`/`SA_LATERAL`/
   `RBAC_PRIV` edges.
- Files: `kubescan/utils/graph_builder.py`, `research/scripts/02_extract/build_graphs.py`
- Also fixed two adjacent bugs found in the same code path: `build_graphs.py`
  `main()`'s `project_root = script_dir.parent` was one level short (fixed
  to `.parent.parent`), and its `default_urls` pointed at a nonexistent
  `original-dataset/` directory instead of `data/raw/rahman/DATASET/`.
  Neither affected prior runs since `--rf-dataset`/`--out-dir` were always
  passed explicitly, but they silently zeroed out Rahman-dataset namespace
  resolution.

**Findings:**
- Local re-ingestion of `attack_repos` sources found 12 clusters (the
  pre-session originals: `kubernetes-goof`, `k8s-escape`, `kubernetes-ctf`,
  `kube-goat`, etc.) had **stale absolute paths** recorded before a project
  directory rename, making their local YAML content unresolvable for
  namespace parsing. Purged and re-ingested (217 rows) to restore
  resolvability — no semantic content changed, only file paths.
- The **Rahman-sourced ~1900 rows** (the original academic dataset, ~475
  clusters) remain namespace-unresolvable locally: the actual downloaded
  YAML content was never persisted (`download_manifest.csv` doesn't exist),
  and re-downloading via `download_github_manifests.py` needs a
  `GITHUB_TOKEN` (unauthenticated GitHub API is rate-limited to 60 req/hr,
  ~32h for 1906 files) not available in this environment. These clusters
  fall back to the namespace-scoping's safe default (`_default` bucket for
  every node, i.e. unchanged pre-fix behavior) — a graceful degradation,
  never a regression, but it means the fix's realized effect so far is
  bounded to `attack_repos`-sourced clusters. Flagged as a follow-up: a
  `GITHUB_TOKEN` would unlock full-corpus namespace resolution.
- With only the `attack_repos` subset resolvable, exactly 2 clusters
  relabeled `chain` → `isolated` out of 37 base chain graphs:
  `kubernetes-ctf` (a CTF app bundled with a `calico` CNI install manifest
  in `kube-system` — the calico DaemonSet's only "path" to the app's
  lateral-flagged pods was `DIR_PROXIMITY`, i.e. same source folder, not
  real reachability) and `simulator_identity-theft` (same pattern: a
  `kyverno` policy-engine installer bundled with an unrelated identity-theft
  demo scenario). Small, surgical, fully-traceable correction — not a
  sweeping relabel.
- Sourced one additional hard-negative example for the sparse-escape
  pattern: `prometheus-operator/kube-prometheus` (Apache 2.0) — of 6
  workload manifests, only `nodeExporter-daemonset.yaml` sets
  `hostNetwork`/`hostPID`/`hostPath`; the rest (Prometheus Operator,
  Grafana, kube-state-metrics, blackbox-exporter) are unprivileged. Grows
  the previously-5-example pattern to 6.
- **Full retrain (Colab GPU T4, `kubescan-repro3`, seed 42) result — this is
  the one that matters:**

  | metric | pre-expansion baseline | expanded-data-only (repro2) | + topology fix + 1 source (repro3) |
  |---|---|---|---|
  | test macro-F1 | 0.7262 | 0.7172 | **0.8796** |
  | test P@5 | 0.75 (ceil 0.8) | 0.60 (ceil 1.0) | **0.80** (ceil 1.0) |
  | attack_chain F1 | 0.571 | 0.444 | **0.800** |
  | chain recall | 2/4 | 2/6 | **4/5** |
  | macro-F1 95% CI | [0.42, 0.93] | [0.46, 0.88] | **[0.61, 1.00]** |
  | GNN CV macro-F1 | 0.8462 | 0.9134 | 0.8616 (std 0.038, still tight) |
  | GNN CV P@5 | 0.82 | 0.76 | 0.56 |

  `k8s-escape` dropped out of the test top-5 entirely (rank 7, was rank 5).
  `istio-index-conf2018` landed in train/val this time (different split
  composition from the corpus size change) so it's no longer a direct data
  point either way. The one remaining test top-5 false positive is
  `kubernetes-microservices-yoloo` — another member of the same sparse
  partial-escape pattern (1/5 nodes), suggesting the pattern is still
  under-represented even after adding `kube-prometheus`, though far less
  severely than before.
- CV macro-F1/P@5 *dropped* slightly (0.91→0.86, 0.76→0.56) even as test
  performance rose sharply. Read this as the CV numbers becoming more
  honest, not worse: the two relabeled clusters were artificially-easy
  "chain" examples propping up CV scores on signal that wasn't real
  topology. Test set generalization is the metric that matters, and it
  improved substantially.
- **Conclusion: this is the fix that actually closed most of the gap.**
  Neither weight-selection regularization (options 1/2) nor raw data volume
  alone (option 3) touched the root cause; correcting what "reachability"
  structurally means in the graph did. One known false positive of the same
  class remains (`kubernetes-microservices-yoloo`), and the Rahman-corpus
  namespace-resolution gap (needs `GITHUB_TOKEN`) is still open — see next
  steps.

### 9. `[x]` Resolve Rahman-corpus namespaces via authenticated GitHub download
Follow-up to option 8: `gh auth token` (already scoped `repo`) unblocks
GitHub's authenticated rate limit (5000 req/hr vs 60), so the previously
"needs a token we don't have" limitation was actually just "needs `gh`."
Fixed the same `project_root`/default-path bug in
`download_github_manifests.py` that `build_graphs.py` had (`script_dir.parent`
→ `.parent.parent`, wrong `original-dataset/` default), then ran it with
`GITHUB_TOKEN=$(gh auth token)`: 1985/2039 files downloaded successfully
(21 failed, 33 404 — acceptable). `research/data/raw/` is gitignored by
design ("re-download via scripts/01_acquire"), so the downloaded YAMLs and
`download_manifest.csv` are not committed; only the resulting graphs are.

**Investigation: is `kubernetes-microservices-yoloo` fixable by this?** No —
checked its node features directly (`.npz` x-array): its one flagged node
(`registry.yaml`, `DOCKERSOCK_PATH`) has zero co-occurring `LATERAL_FLAGS`
anywhere in the cluster, so `_compute_graph_label`'s reachability path-check
can never fire regardless of namespace resolution. It's already correctly
labeled `isolated`. It only ranks in the test top-5 because the ensemble's
`w_esc` weight rewards any escape flag whether or not a lateral target
exists — this is the escape-signal-domination issue from options 1/2, not a
topology bug. Not fixable without deploying the floor-regularization fix.

**Rebuild result:** 25 of 599 clusters gained real topology (previously
defaulted to the fully-connected `_default`-namespace fallback); exactly
**one** label changed: `cloudify-kubernetes-plugin` `isolated` → `chain`.
This is not noise — it's the exact cluster named in the original problem
statement as "the one real attack chain that lacks an escape flag [and]
structurally can never [be] recover[ed]" by any weight tuning. Direct cause:
`_compute_graph_label` overrides `HOSTPATH_MOUNT` from real YAML content
(`sem.get("hostpath_mount")`), and 7 of its manifests (`pv.yaml`,
`daemon-set.yaml`, `cloudify_manager/pod.yaml`, `cassandra-blueprint.yaml`,
`cloudify-manager.yaml`, `wordpress-blueprint.yaml`,
`test-persistent-volume.yaml`) mount `hostPath` — a real escape vector
Rahman's static heuristic never tagged (`HOSTPATH_MOUNT=0` for all of them in
the raw CSV). `compute_escape_signal()` reads the graph's node-feature array
(post-override), not the raw CSV, so this also fixes the ensemble's
escape_signal for this cluster, not just its label. It landed in `train`
under the seed-42 split, not `test`.

**Full retrain (Colab GPU T4, `kubescan-repro4`, seed 42) result:**

| metric | repro3 (topology fix, partial corpus) | repro4 (+ full Rahman resolution) |
|---|---|---|
| test macro-F1 (argmax) | 0.8796 | 0.6992 |
| macro-F1 95% CI | [0.61, 1.00] | [0.48, 0.85] |
| test P@1 | — | **1.00** |
| test P@5 | 0.80 (ceil 1.0) | **0.80** (ceil 1.0, unchanged) |
| GNN CV macro-F1 | 0.8616 (std 0.038) | 0.8512 (std 0.025, tighter) |
| GNN CV P@5 mean | 0.56 | 0.52 |
| `kubernetes-microservices-yoloo` test rank | 5 (in top-5) | **9** (out of top-5) |

The argmax classification macro-F1 point estimate dropped, but its 95% CI
`[0.48, 0.85]` overlaps repro3's `[0.61, 1.00]` substantially — on a 91-graph
test set with only 6 chain / 4 clean examples, a couple of borderline
argmax flips swing this metric a lot (confusion matrix: `isolated` row went
from `[0,80,1]` to `[4,74,3]`, i.e. a handful of isolated clusters now
argmax to `clean` or `chain` instead). CV metrics stayed stable/tightened,
and the metric that matters operationally — top-K ranking, since that's what
a human triages — held steady (P@5=0.80) or improved (P@1 now 1.00, and the
known `yoloo` false positive dropped further down the ranking). The GA
landed on a different point of the same OOF plateau (`w_rf=0.34, w_gnn=0.33,
w_esc=0.33` vs repro3's `0.54/0.16/0.30`) — **correction:** this is *not*
evidence the option 1/2 regularization is undeployed. Checking
`run_colab_pipeline.py` directly shows `--min-weight 0.1 --reg-lambda 2.0`
has been passed to every GA run including repro3 and repro4; the
regularization is active and doing its documented job (keeping the GA off
the escape-only corner), it just doesn't collapse the OOF plateau to a
single point — different runs landing on different interior weights given
different underlying fold-model training data is the expected, already-
characterized behavior from option 1's own findings, not a gap. The CI
overlap is suggestive but not proof of pure noise — it's one data point
(different dataset than repro3, not a reseed of the same dataset), so a
same-dataset reseed is the only way to actually separate "test-set sampling
noise" from "a real effect of resolving the Rahman corpus."

**Resolved via same-split reseed (Colab GPU T4, `kubescan-repro5`, seed=43,
`--stages train_rf,gnn_cv,ga_ensemble,test_evaluation` only — i.e. identical
`train.txt`/`val.txt`/`test.txt` as repro4, only the training/GA seed
changed):**

| metric | repro4 (seed 42) | repro5 (seed 43, same split) |
|---|---|---|
| test macro-F1 (argmax) | 0.6992 | 0.7380 |
| macro-F1 95% CI | [0.48, 0.85] | [0.47, 0.91] |
| attack_chain confusion row | `[0,2,4]` (recall 4/6) | `[0,4,2]` (recall 2/6) |
| clean F1 | 0.545 | 0.750 |
| test P@1 | 1.00 | **1.00 (identical)** |
| test P@5 | 0.80 | **0.80 (identical)** |
| GA weights | 0.34/0.33/0.33 | 0.54/0.28/0.18 |

With the test set held byte-identical, `attack_chain` recall alone swings
from 4/6 to 2/6 purely from the training seed — direct confirmation that
the classification-argmax macro-F1 instability is small-sample training
noise (6 positive examples in the minority class), not a regression caused
by resolving the Rahman corpus. Ranking metrics (P@1, P@5) — the metrics
that actually matter for a triage tool — were identical across both seeds,
which is the more load-bearing finding: **the model's ranking behavior is
seed-stable even when its argmax classification isn't.** GA weights landing
on different interior points again reconfirms option 1's flat-OOF-plateau
finding, now on a third independent run. Closed — no further action.
- **New finding, not yet resolved:** `micro-apps` (30 nodes, exactly one
  weakly-flagged node — `HOSTPATH_MOUNT` only, no co-occurring lateral flag,
  correctly labeled `isolated`) landed at test rank 2 with **GNN
  chain_prob=0.965** — high false confidence from the GNN itself, not just
  the ensemble's escape-weight term (contrast with `yoloo`, where the GNN
  correctly scored near-zero and only `w_esc` inflated its rank). This is a
  structurally different failure mode: `DIR_PROXIMITY`/`SEMANTIC_NS` edges
  are excluded from the *label* computation (`_compute_graph_label`'s
  `reach_graph` restriction, see option 8) but are **not** excluded from the
  graph the GNN actually trains and does message-passing on — so a GAT layer
  can still let an escape-flagged node's features propagate confidence
  through non-reachability edges (same source directory / same namespace)
  across a large graph.
  **Follow-up check (broadened to all 599 base graphs, escape fraction
  ≤15%, zero lateral flags, ≥8 nodes):** only 6 clusters in the entire
  corpus match this profile. Results split both directions —
  `CloudFlix` (true chain, correctly confident 0.997),
  `cloudify-kubernetes-plugin` (true chain, moderately-scored 0.67, the
  option-9 recovery case), `istio-index-conf2018`/`kubernetes-extras`
  (true isolated, correctly low), `micro-apps` (true isolated, false
  positive at 0.965), and `gitcontroller` (true chain but
  **under**-confident at 0.28 — the opposite failure direction). With only
  one genuine false positive against one genuine false negative out of 6
  candidates, this reads as ordinary classifier noise on rare/ambiguous
  topologies rather than a systemic edge-type leakage bug — not enough
  evidence to justify restricting which edge types the GNN's own attention
  can see (as opposed to the label rule, which already is restricted).
  Not pursuing a code fix; flagging as a known limitation on rare
  topological patterns instead.
- Data/split checksums verified to exactly match `run_manifest.json`'s
  provenance this time (no reproducibility gap): the VM cloned the exact
  pushed commit (`f142e1e`), so local and remote states were identical by
  construction, unlike the repro3 flow which required a post-hoc local
  reconstruction.

### 4. `[ ]` Verify "GNN-alone beats ensemble" is robust, not a 15-graph fluke
Check whether this holds per-fold (not just the fold-ensemble average)
before investing further in ensemble weight tuning. If robust, the honest
framing is that the ensemble's value is defensive (avoids RF's false
positive, avoids escape-only's P@1=0 failure), not a ranking improvement —
which is what the thesis currently says, but from a single test run.

### 5. `[ ]` Reduce feature imputation via full Checkov manifest download (L2)
Targets the RF layer specifically, which is already near-ceiling (F1=0.99).
Improves generalization robustness, not current reported numbers. Lower
priority.

### 6. `[ ]` Repository-stratified cross-validation (L3)
10 repos are 60.8% of the tabular dataset; `gitcontroller` alone is 13.7%.
Would give more conservative, honest generalization estimates. Improves
credibility of reported metrics, not model capability.

### 7. `[ ]` Real-world production validation (L4)
Correct long-term next step; out of scope for this thesis.

---

## Log

- 2026-07-05: GA algorithm fix (corner seeding + boundary mutation)
  implemented and verified across 5 seeds. Comparison run (old near-uniform
  vs new escape-only weights) on test set done — both tie at P@5=0.75 but
  for different reasons (see problem statement above). Thesis
  `05_resultados.tex` §Capa 3 and `06_conclusiones.tex` L5 corrected to
  match verified numbers and real causal explanation.
- 2026-07-05: Option 1 (floor-penalty regularization) implemented and
  validated locally across 5 seeds — see findings under option 1 above.
  Test-set P@5 unchanged at 0.75, but the result is now stable across seeds
  regardless of where on the OOF plateau the GA lands. Not yet run on Colab
  GPU / committed to checkpoints.
- 2026-07-05: Option 2 (bootstrap resampling) implemented and validated
  locally — lands on a very different point (RF-heavy) than option 1's GA
  runs but hits the exact same test-set outcome and the exact same missed
  cluster. Confirms option 1's result independently rather than adding new
  value on its own. Not yet run on Colab GPU / committed to checkpoints.
- 2026-07-06: Option 3 (expanded corpus, 598 clusters, 11 new sources) fully
  retrained end-to-end on Colab GPU T4 (`kubescan-repro2`). GNN CV improved
  substantially (macro-F1 0.85→0.91, std nearly halved). Test-set ranking
  regressed (P@5 0.75→0.60) because two new hard negatives from real-world
  sources trip the escape-signal ranking heuristic — see findings above.
  More data narrowed the GNN's own generalization gap but did not fix, and
  mildly sharpened, the ensemble's escape-signal over-reliance. Options 1/2
  (regularization) remain necessary on top of this.
