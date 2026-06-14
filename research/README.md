# Research Pipeline

Reproducible pipeline for training the GNN + Random Forest ensemble.
Run these steps in order to reproduce all trained artifacts from scratch.

## Directory layout

```
research/
├── data/
│   ├── raw/                      ← cloned attack repos + Rahman dataset
│   ├── tabular/                  ← rf_dataset.csv (2,479 rows × 57 cols)
│   ├── graphs/                   ← 96 original + 375 augmented .npz cluster graphs
│   └── splits/                   ← train/val/test + 5-fold CV split .txt files
├── scripts/
│   ├── 01_acquire/               ← download + ingest raw manifests
│   ├── 02_extract/               ← YAML feature extraction + graph construction
│   ├── 03_augment/               ← attack-chain graph augmentation
│   ├── 04_build_datasets/        ← assemble RF and GNN dataset files
│   ├── 05_split/                 ← stratified train/val/test + 5-fold CV splits
│   └── fixes/                    ← one-off data patches (never in main pipeline)
└── models/
    ├── train_rf.py               ← Layer 1: Random Forest
    ├── train_gnn.py              ← Layer 2: Graph Attention Network
    ├── run_ga_ensemble.py        ← Layer 3: GA ensemble weight optimisation
    ├── evaluate_test_set.py      ← held-out test evaluation
    ├── predict.py                ← end-to-end inference (research-side CLI)
    └── checkpoints/              ← .pt, .pkl, .json artifacts used by kubescan
```

## Quickstart (full pipeline from scratch)

```bash
cd research/

# 1. Build tabular dataset from raw YAML sources
python scripts/04_build_datasets/build_rf_dataset.py
python scripts/01_acquire/ingest_attack_repos.py
python scripts/04_build_datasets/enrich_rf_dataset.py

# 2. Train Layer 1 — Random Forest
python models/train_rf.py

# 3. Build cluster graphs (uses RF risk scores from step 2)
python scripts/02_extract/build_graphs.py

# 4. Augment attack-chain graphs (15 variants per chain, deterministic seeds)
python scripts/03_augment/augment_graphs.py

# 5. Consolidate graphs into a single cache file (fast split loading)
python scripts/04_build_datasets/build_graph_cache.py

# 6. Create group-aware splits + 5-fold CV splits
#    (augmented variants follow their base cluster; test clusters held out of folds)
python scripts/05_split/create_splits.py

# 7. Train Layer 2 — GNN (5-fold CV, ~60 min)
python models/train_gnn.py
#    Ablations: --layers 2 | --conv gcn (use --out-dir to keep checkpoints apart)

# 8. Optimise Layer 3 ensemble weights (OOF mode, ~5 min)
python models/run_ga_ensemble.py --oof

# 9. Final test-set evaluation (ablation table + bootstrap CIs)
python models/evaluate_test_set.py --show-rankings
```

Steps 4–6 must be re-run together: the splits are derived from the augmented
manifest, and the cache must be rebuilt whenever graphs change.

## Trained model metrics (final, 2026-06-11 — group-aware leak-free protocol)

| Layer | Model | Metric | Value | Target |
|-------|-------|--------|-------|--------|
| L1 | Random Forest | Binary F1 (test) | **0.9935** | > 0.85 ✓ |
| L1 | Random Forest | 5-fold CV F1 | **0.9908 ± 0.006** | — |
| L2 | GAT-emb (5-fold, group-aware) | P@5 | **0.720 ± 0.098** | > 0.70 ✓ |
| L2 | GAT-emb (5-fold, group-aware) | Macro-F1 | **0.870 ± 0.075** | — |
| L3 | Ensemble (held-out test) | P@1 | **1.00** | — |
| L3 | Ensemble (held-out test) | P@5 | **0.60** (ceiling 0.80) | — |
| L3 | Ensemble (held-out test) | FPR_clean | **0.00** | < 0.10 ✓ |

Architecture ablation (same protocol): GAT-3L P@5 0.72 > GAT-2L 0.68 > GCN-3L 0.64.
Test set: 15 graphs, 4 chains — 3 of 4 chains ranked in top-5 (4th at rank 8);
the test clusters are excluded from CV folds, GA tuning and augmentation.
95% bootstrap CIs in `models/checkpoints/test_results.json`; full provenance in
`models/checkpoints/run_manifest.json`. Earlier (pre-2026-06) numbers used a
protocol with augmentation leakage and are not comparable.

## Dependencies

```bash
pip install torch torch-geometric scikit-learn pandas numpy pyyaml networkx matplotlib
```
