# Dataset datasheet — kubescan corpus

Provenance and composition of the data behind every number in the thesis.
Format loosely follows *Datasheets for Datasets* (Gebru et al., 2021).

## Composition

| Artifact | Count | Location |
|---|---|---|
| Labeled manifests (RF layer) | 2,827 rows | `tabular/rf_dataset.csv` |
| Original cluster graphs (GNN layer) | 599 (25 clean / 537 isolated / 37 chain) | `graphs/*.npz` |
| Augmented graphs (chain class only, 15 variants each) | 555 | `graphs/*_aug_*.npz` |
| Consolidated graph cache | 1 file, 1,154 graphs | `graphs/graphs_cache.npz` |

One original graph exists per unique `repo_name` in `rf_dataset.csv` (599 repos),
so the row corpus and the graph corpus cover the same clusters.

RF label distribution: 1,591 clean / 1,236 misconfigured.
RF severity distribution: 1,591 class 0 / 755 class 1 / 481 class 2.

## Sources

Row counts in `tabular/rf_dataset.csv` by `source` column:

1. **Rahman et al. SLI-KUBE corpus** (GitHub/GitLab repos with security-smell
   annotations) — the backbone of `rf_dataset.csv`, contributing the
   `github` (1,510) and `gitlab` (396) rows, 1,906 in total. Rows retain the
   original `yaml_path` values from the source distribution (paths from the
   original authors' machine; resolve through the download manifest, not
   literally).
2. **attack_repos** — public repositories with deliberately vulnerable or
   attack-oriented Kubernetes manifests, ingested by
   `scripts/01_acquire/ingest_attack_repos.py` (779 rows). This source supplies
   most of the chain-class signal at the graph layer.
3. **badpods** (BishopFox) — intentionally insecure pod specs (128 rows);
   ground-truth attack examples.
4. **kubernetes-goat / kube-goat / kubernetes_goat scenarios** — deliberately
   vulnerable lab environments (14 rows + several cluster graphs).

Total: 1,510 + 779 + 396 + 128 + 14 = 2,827 rows.

Raw inputs live under `data/raw/` and are NOT redistributed in this
repository (third-party licenses); the acquisition scripts re-download them.

## Labels

### Manifest level (RF)

`label` has **two different provenances depending on the source**, and the
distinction matters when interpreting Layer 1 results.

**Rahman rows (`source` in {github, gitlab}, 1,906 rows — 67 % of the corpus).**
The label comes from the source corpus's own `INSECURE` ground truth, not from
our rule. Verified: the `compute_label` disjunction below reproduces the shipped
label for only 245/1,510 github and 87/396 gitlab rows, which would be
impossible if the rule had generated them.

**attack_repos rows (779 rows).** These are labelled by
`scripts/01_acquire/ingest_attack_repos.py::compute_label`, which reproduces the
shipped label for **779/779** rows exactly:

```python
MISCONFIG_COLS = frozenset(FEATURE_COLS) - {"VALID_TAINT_SECRET"}

def compute_label(row: dict) -> int:
    """0=clean, 1=misconfig (any flag set)."""
    return 1 if any(int(row.get(c, 0)) for c in MISCONFIG_COLS) else 0
```

That is, for these rows `label = 1` iff **any** of the 24 misconfiguration flags
is set — all 25 `FEATURE_COLS` from `kubescan/utils/yaml_parser.py` except
`VALID_TAINT_SECRET`, which is excluded because the extractor never sets it.
The remaining fixture sources (badpods 112/128, kubernetes_goat 13/14) largely
follow the same rule.

`severity_class` is derived from the same row by `compute_severity`: class 2
if any `ESCAPE_COLS` flag is set, otherwise `compute_label(row)`.

### Known label/feature circularity

The circularity does not come from our labelling rule — it survives even on the
Rahman rows, whose labels we did not generate. Rahman et al. define `INSECURE`
as "the manifest exhibits at least one of the catalogued misconfiguration
categories", and our extractor reproduces those same categories as features.
The independently-sourced label is therefore still a deterministic function of
our own feature space. Two measured consequences on the shipped corpus:

- **Rahman rows (`source` in {github, gitlab}, 1,906 rows):** the binary label
  is *exactly* the disjunction of eight input features —
  `INSECURE_HTTP`, `NO_SECU_CONTEXT`, `WITHIN_MANIFEST_SECRET`, `NO_RESO`,
  `CAP_SYS_ADMIN`, `TRUE_HOST_NET`, `ALLOW_PRIVI`, `DOCKERSOCK_PATH`.
  All 332/332 positives are covered with zero false positives
  (TP=332, FP=0, FN=0, TN=1,574) — the eight-literal disjunction reproduces the
  label exactly on this subset.
- **Full corpus (2,827 rows):** the same eight-feature rule gives TP=1,195,
  FP=17, FN=41 — still near-deterministic, because the extra sources introduce
  flags outside those eight rather than genuinely independent labels.

Near-perfect RF scores must therefore be read as a measure of how well the
model recovers a rule it was implicitly given, not as evidence of
generalisation to human-judged insecurity. This is discussed in the thesis
limitations section; any external validation must use a corpus whose labels
come from outside `FEATURE_COLS`.

### Cluster level (GNN)

Rule-based, computed by
`scripts/02_extract/build_graphs.py::_compute_graph_label`:

- `0 clean` — no node has any flag set
- `1 isolated` — misconfigured nodes but no compounding chain
- `2 attack_chain` — ≥2 escape-capable nodes, or an escape-capable node
  coexisting with a lateral-movement-capable node

Labels are functions of the same feature space the models observe; the models
learn a graph-structural approximation of this rule under noise, partial
observability, and augmentation (see thesis §Limitations).

## Known biases

- Top-10 repositories account for 47.6 % of `rf_dataset.csv`
  (`gitcontroller` alone 281 rows, 9.9 %). Row-level RF splits therefore share
  repository context between train and test.
- The 6 extended features (`NO_RUN_AS_NON_ROOT`, `NO_READ_ONLY_ROOT_FS`,
  `IMAGE_USES_LATEST`, `SA_AUTOMOUNT_TOKEN`, `USES_DEFAULT_SA`,
  `UNTRUSTED_REGISTRY`) are empty for the 1,906 Rahman rows (67.4 % of the
  corpus) and must be Checkov-filled or median-imputed. The `checkov_*`
  columns are empty for 1,652 rows (58.4 %).
- All 555 augmented graphs belong to the chain class, so the chain class is the
  only one whose training signal is synthetic (37 real chain graphs, 555
  variants).

## Split protocol (leakage rules)

Generated by `scripts/05_split/create_splits.py` (seed 42); the audit record is
`splits/splits_config.json`:

- Stratified 70/15/15 over the 599 originals → 425 train / 88 val / 86 test
  original clusters.
- Augmented variants join a training partition only when their **base cluster**
  is in that partition; variants of val/test clusters are excluded from
  training entirely. 390 of the 555 augmented graphs enter the global training
  set, 165 are excluded. Total training partition: 815 graphs.
- Original clusters sharing a template family (e.g. `badpods_*`, `datadog_*`)
  are kept together in one partition and never split across CV folds.
- The 86 test clusters are held out of the 5 CV folds (CV pool = 513
  originals); out-of-fold predictions used for GA weight tuning therefore never
  contain a test cluster.
- Verify with the audit in `splits_config.json` / re-run the base-cluster
  overlap check before reporting any number.

## Unused / legacy artefacts

- **`data/raw/sever_dogan/` (91 MB, includes an 18 MB nested `.git`)** —
  CVE-labelled network-flow CSVs derived from `.pcap` captures (CICFlowMeter
  output, from "A Kubernetes dataset for misuse detection", ITU J. 4(2), 2023).
  This is network-traffic data, unrelated to the Kubernetes *manifest* corpus
  that every layer of kubescan consumes. It is referenced by **no** script, no
  config, and no thesis chapter. It is already excluded from version control by
  the `research/data/raw/` rule in the repository `.gitignore` and is untracked.
  Retained locally only as a leftover from an abandoned exploration; **slated
  for removal** — deleting the directory has no effect on any pipeline stage or
  reported result.

## Regeneration

Deterministic end-to-end (seeds fixed, no `PYTHONHASHSEED` dependence):

```bash
python scripts/03_augment/augment_graphs.py        # idempotent
python scripts/04_build_datasets/build_graph_cache.py
python scripts/05_split/create_splits.py
```

After any regeneration, fingerprint the state:

```bash
python scripts/snapshot_run_manifest.py            # → checkpoints/run_manifest.json
```
