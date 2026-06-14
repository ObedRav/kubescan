## Makefile — TFE / kubescan development tasks
## Run from the TFE root directory.

PYTHON      := python3
RUFF        := ruff
PYTEST      := pytest
KUBESCAN    := kubescan/
RESEARCH    := research
SEED        := 42
GNN_EPOCHS  := 300
PDFLATEX    := /Library/TeX/texbin/pdflatex
BIBTEX      := /Library/TeX/texbin/bibtex
THESIS_DIR  := thesis/latex
THESIS_MAIN := plantilla

.PHONY: lint lint-fix test test-unit test-integration test-fast format check \
        fixtures data reproduce reproduce-ablations thesis thesis-check thesis-clean

## ── Linting ──────────────────────────────────────────────────────────────────

lint:
	$(RUFF) check kubescan/src kubescan/tests research/models research/scripts

lint-fix:
	$(RUFF) check --fix kubescan/src kubescan/tests research/models research/scripts

format:
	$(RUFF) format kubescan/src kubescan/tests research/models research/scripts

## ── Testing ──────────────────────────────────────────────────────────────────

test:
	cd $(KUBESCAN) && $(PYTEST) tests/ -v

test-unit:
	cd $(KUBESCAN) && $(PYTEST) tests/unit/ -v

test-integration:
	cd $(KUBESCAN) && $(PYTEST) tests/integration/ -v

test-fast:   ## unit tests only, parallel
	cd $(KUBESCAN) && $(PYTEST) tests/unit/ -q -n auto

fixtures:    ## regenerate tiny CI checkpoints for integration tests
	$(PYTHON) $(KUBESCAN)/tests/fixtures/make_fixtures.py

## ── Reproducibility ──────────────────────────────────────────────────────────
## Regenerate every artifact and metric from the existing graph .npz files.
## Assumes steps 1–3 (acquire → extract → build_graphs) have already produced
## research/data/graphs/*.npz; those are network/heavy and run separately.

data:        ## augment → consolidate cache → group-aware splits
	$(PYTHON) $(RESEARCH)/scripts/03_augment/augment_graphs.py --seed $(SEED)
	$(PYTHON) $(RESEARCH)/scripts/04_build_datasets/build_graph_cache.py
	$(PYTHON) $(RESEARCH)/scripts/05_split/create_splits.py --seed $(SEED)

reproduce: data   ## full pipeline: data → RF → GNN(5-fold) → GA → test eval → provenance
	$(PYTHON) $(RESEARCH)/models/train_rf.py --seed $(SEED)
	cd $(RESEARCH)/models && $(PYTHON) train_gnn.py --cv-folds 5 --epochs $(GNN_EPOCHS) --seed $(SEED)
	cd $(RESEARCH)/models && $(PYTHON) train_gnn.py --cv-folds 0 --epochs $(GNN_EPOCHS) --seed $(SEED)
	cd $(RESEARCH)/models && $(PYTHON) run_ga_ensemble.py --oof --seed $(SEED)
	cd $(RESEARCH)/models && $(PYTHON) evaluate_test_set.py --show-rankings
	$(PYTHON) $(RESEARCH)/scripts/snapshot_run_manifest.py
	@echo "reproduce: done — see research/models/checkpoints/run_manifest.json"

reproduce-ablations:   ## architecture ablations (GAT 2-layer, GCN 3-layer)
	cd $(RESEARCH)/models && $(PYTHON) train_gnn.py --cv-folds 5 --layers 2 \
	  --seed $(SEED) --out-dir checkpoints_ablation/gat_2layer
	cd $(RESEARCH)/models && $(PYTHON) train_gnn.py --cv-folds 5 --conv gcn \
	  --seed $(SEED) --out-dir checkpoints_ablation/gcn_3layer

## ── Thesis ───────────────────────────────────────────────────────────────────

thesis-check:
	cd $(THESIS_DIR) && \
	  $(PDFLATEX) -draftmode -interaction=nonstopmode $(THESIS_MAIN).tex | \
	  grep -E "^!" || true
	@echo "thesis-check: OK (no LaTeX errors)"

thesis:
	cd $(THESIS_DIR) && \
	  $(PDFLATEX) -interaction=nonstopmode $(THESIS_MAIN).tex && \
	  $(BIBTEX) $(THESIS_MAIN) && \
	  $(PDFLATEX) -interaction=nonstopmode $(THESIS_MAIN).tex && \
	  $(PDFLATEX) -interaction=nonstopmode $(THESIS_MAIN).tex
	@echo "PDF: $(THESIS_DIR)/$(THESIS_MAIN).pdf"

thesis-clean:
	cd $(THESIS_DIR) && \
	  rm -f $(THESIS_MAIN).aux $(THESIS_MAIN).bbl $(THESIS_MAIN).blg \
	        $(THESIS_MAIN).log $(THESIS_MAIN).out $(THESIS_MAIN).toc \
	        $(THESIS_MAIN).lof $(THESIS_MAIN).lot $(THESIS_MAIN).synctex.gz \
	        chapters/*.aux appendix/*.aux

## ── Combined check (CI entry point) ──────────────────────────────────────────

check: lint test   ## lint + full test suite (CI entry point)
