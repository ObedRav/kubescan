# kubescan

Kubernetes attack-chain risk scanner.

Predicts whether a Kubernetes cluster's YAML manifests form an exploitable
multi-hop attack chain (pod-escape → lateral movement → impact) using a
three-layer ensemble: Random Forest + Graph Attention Network + GA-optimised scorer.

## Results

| Layer | Model | Metric | Result | Target |
|-------|-------|--------|--------|--------|
| 1 | Random Forest | Macro-F1 (test) | **0.9946** | > 0.85 ✅ |
| 2 | GAT (5-fold CV, family-aware) | Precision@5 | **0.720 ± 0.160** | > 0.70 ✅ |
| 3 | GA Ensemble (held-out test) | P@1 / P@5 / FPR\_clean | **1.00 / 0.80 / 0.00** | — |
| — | Usability (SUS, n=3 experts) | SUS score | **88.3** ("excellent") | — |

The Layer-3 ranking applies a structural feasibility gate (a single-manifest
cluster cannot host a multi-hop chain). Evaluation uses group- and
template-family-aware splits: augmented graph variants and near-duplicate
template families never cross train/eval boundaries, and the 86 held-out test
graphs — including all 5 real attack chains — are excluded from the CV folds and
GA tuning. See `research/data/DATASET.md` for the full protocol.

## System requirements

| Resource | Minimum | Notes |
|----------|---------|-------|
| Python | 3.10+ | |
| RAM | 8 GB | 16 GB recommended for MPS/CUDA training |
| Disk | **40 GB free** | PyTorch + PyG wheels (~2.5 GB site-packages), Metal shader cache built on first MPS run (~12 GB in `/var/folders/`), raw manifest data (~4 GB), training artefacts |
| GPU | optional | MPS (Apple Silicon) or CUDA auto-detected; CPU fallback always works |

> **macOS / MPS note:** the first `train_gnn.py` run compiles Metal shader libraries into
> `/var/folders/` — this cache alone can reach **~12 GB** and grows with each new model
> configuration. Ensure at least **40 GB free** on the boot volume before training.

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
