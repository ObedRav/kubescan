# kubescan

Kubernetes attack-chain risk scanner — Master's thesis project (UNIR, 2026).

Predicts whether a Kubernetes cluster's YAML manifests form an exploitable
multi-hop attack chain (pod-escape → lateral movement → impact) using a
three-layer ensemble: Random Forest + Graph Attention Network + GA-optimised scorer.

## Results

| Layer | Model | Metric | Result | Target |
|-------|-------|--------|--------|--------|
| 1 | Random Forest | Macro-F1 (test) | **0.9935** | > 0.85 ✅ |
| 2 | GAT (5-fold CV, group-aware) | Precision@5 | **0.720 ± 0.098** | > 0.70 ✅ |
| 3 | GA Ensemble (held-out test) | P@1 / FPR\_clean | **1.00 / 0.00** | — |

Evaluation uses group-aware splits (augmented graph variants never cross
train/eval boundaries; the 15 test clusters are excluded from CV folds and GA
tuning). See `research/data/DATASET.md` for the full protocol.

## Repository layout

```
TFE/
├── kubescan/          # Installable Python package — pip install -e kubescan/
│   ├── src/kubescan/
│   │   ├── cli.py                 # kubescan scan <dir>  |  kubescan live
│   │   ├── model/                 # Inference: GAT encoder, RF classifier, GA ensemble
│   │   └── utils/                 # YAML feature extractor, cluster graph builder
│   └── tests/
├── research/          # Reproducible training pipeline (not a package)
│   ├── scripts/       # 01_acquire → 02_extract → 03_augment → 04_build → 05_split
│   ├── models/        # train_rf.py, train_gnn.py, run_ga_ensemble.py + checkpoints/
│   └── data/          # raw/, tabular/, graphs/, splits/
└── thesis/            # LaTeX source — compiled PDF at thesis/latex/plantilla.pdf
```

## Quick start

```bash
# Install the package
pip install -e kubescan/

# Scan a directory of Kubernetes manifests
kubescan scan ./my-cluster/

# JSON output (CI/CD)
kubescan scan ./my-cluster/ --format json

# Live mode — scan the running cluster via kubectl
kubescan live --namespace default
```

See [kubescan/README.md](kubescan/README.md) for full CLI reference.

## Reproduce the training pipeline

```bash
# 1. Acquire raw manifests
python research/scripts/01_acquire/download_github_manifests.py
python research/scripts/01_acquire/ingest_attack_repos.py

# 2–6. Extract, augment, build graph cache, create group-aware splits
# (see research/README.md for the full 9-step sequence)

# 7. Train all three layers
python research/models/train_rf.py
python research/models/train_gnn.py --epochs 300 --hidden 64 --heads 4 --layers 3
python research/models/run_ga_ensemble.py --oof

# 8. Evaluate on the held-out test set (excluded from CV folds and GA tuning)
python research/models/evaluate_test_set.py
```

See [research/README.md](research/README.md) for details, expected outputs, and metrics.

## Build the thesis

```bash
make thesis        # full 3-pass compile → thesis/latex/plantilla.pdf
make thesis-check  # fast syntax check (no PDF written, ~5 s)
```

## License

MIT — see [LICENSE](LICENSE).
