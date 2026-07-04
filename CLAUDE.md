# CLAUDE.md — TFE / kubescan

Read automatically by Claude and Cursor on every interaction.
All generated or modified code must follow every rule below without exception.
If a rule conflicts with a request, surface the conflict and propose a compliant
alternative instead of silently breaking the rule.

---

## Project overview

**kubescan** is a Kubernetes attack-chain risk scanner using a GNN + Random Forest
ensemble. The repository is split into three pillars:

```
TFE/
├── kubescan/          # distributable Python package  (pip install -e kubescan/)
│   ├── src/kubescan/
│   │   ├── cli.py                    # entry point: kubescan scan <dir>
│   │   ├── model/
│   │   │   ├── gat_encoder.py        # inference-only GAT architecture
│   │   │   ├── rf_classifier.py      # inference-only RF wrapper
│   │   │   └── ga_ensemble.py        # ensemble scorer (GA-optimised weights)
│   │   └── utils/
│   │       ├── yaml_parser.py        # YAML feature extraction
│   │       └── graph_builder.py      # in-memory cluster graph builder
│   └── tests/
│       ├── unit/                     # yaml_parser, graph_builder, ga_ensemble
│       └── integration/              # full CLI scan against trained checkpoints
├── research/          # reproducible training pipeline  (not importable as package)
│   ├── data/
│   │   ├── raw/                      # cloned repos + Rahman dataset
│   │   ├── tabular/                  # rf_dataset.csv
│   │   ├── graphs/                   # .npz cluster graphs
│   │   └── splits/                   # train/val/test + 5-fold CV .txt files
│   ├── scripts/
│   │   ├── 01_acquire/               # download + ingest raw manifests
│   │   ├── 02_extract/               # YAML features + graph construction
│   │   ├── 03_augment/               # attack-chain graph augmentation
│   │   ├── 04_build_datasets/        # assemble RF and GNN dataset files
│   │   ├── 05_split/                 # stratified splits + 5-fold CV
│   │   └── fixes/                    # one-off data patches (never in main pipeline)
│   └── models/
│       ├── train_rf.py               # Layer 1: Random Forest
│       ├── train_gnn.py              # Layer 2: Graph Attention Network (5-fold CV)
│       ├── run_ga_ensemble.py        # Layer 3: GA ensemble weight optimisation
│       ├── evaluate_test_set.py      # held-out test evaluation
│       ├── predict.py                # end-to-end research-side inference
│       └── checkpoints/              # .pt, .skops, .json — consumed by kubescan
└── thesis/            # LaTeX source, PDFs, figures — zero Python
```

**Model architecture:**
- **Layer 1** — Random Forest: 25 binary flags per manifest → `risk_score ∈ [0,1]`
- **Layer 2** — GAT (5-fold ensemble, edge-type embeddings): cluster graph → `chain_probability ∈ [0,1]`
- **Layer 3** — GA-optimised ensemble: `score = w_rf·risk + w_gnn·chain + w_escape·escape_signal`
  (`escape_signal` is BINARY — 1.0 if any node has an escape flag set; never score with the fraction)
- The KubeGAT architecture lives ONLY in `kubescan/model/gat_encoder.py`;
  `research/models/train_gnn.py` imports it (single source of truth).

**Separation contract:**
- `kubescan/model/` contains inference code only. No training imports (`torch.optim`,
  training DataLoaders, etc.) are allowed here.
- `research/` may import from `kubescan/` but the reverse is forbidden.
- `kubescan/utils/yaml_parser.py` is the canonical YAML feature extractor.
  Research scripts that need the same logic must import from it, never re-implement it.

---

## Naming conventions

### Files

| Context | Pattern | Example |
|---------|---------|---------|
| Pipeline step (research/scripts/) | `verb_noun.py` | `build_graphs.py` |
| Module / class container (kubescan/) | `noun_noun.py` | `gat_encoder.py` |
| CLI entry point | `verb.py` | `cli.py` |
| One-off fix | `patch_*.py` in `fixes/` | `patch_hostpath_column.py` |

**Core rule:** files that *are* things use nouns; files that *do* things use verb_object.

### Python identifiers

| Kind | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `GraphDataset`, `RFClassifier` |
| Functions / methods | verb_noun | `build_graph()`, `load_weights()` |
| Variables | noun or adj_noun | `graph_list`, `escape_frac` |
| Constants | UPPER_SNAKE | `MAX_NODES`, `NODE_FEATURE_DIM` |
| Private helpers | `_verb_noun` | `_normalize_resource()` |
| Directories (Python) | snake_case | `graph_data/` |
| Directories (pipeline) | `NN_noun/` | `01_acquire/` |
| Data files | snake_case.ext | `rf_dataset.csv` |
| LaTeX files | kebab-case.tex | `chapter-3.tex` |

---

## 1. No magic strings or magic numbers

Every literal string or number with domain meaning must be a named constant.

```python
# WRONG
if node_type == "Pod":
    ...
if score > 0.75:
    ...

# RIGHT
class NodeType(str, Enum):
    POD = "Pod"
    DEPLOYMENT = "Deployment"

class Thresholds:
    ATTACK_CHAIN_CONFIDENCE: Final[float] = 0.75

if node_type == NodeType.POD:
    ...
if score > Thresholds.ATTACK_CHAIN_CONFIDENCE:
    ...
```

Rules:
- Strings used as identifiers (node types, labels, feature names) → `Enum`.
- Numeric thresholds, layer sizes, seeds → `dataclass` or module-level `Final` constant.
- File paths → `pathlib.Path` constants, never bare strings.
- Config values that vary per run → `Config` dataclass loaded from YAML/JSON, not hardcoded.
- Allowed literals in logic: `0`, `1`, `True`, `False`, `""`, `[]`, `{}` in their
  structural sense only. Everything else gets a name.

---

## 2. DRY — Don't Repeat Yourself

If the same logic appears twice, it is in the wrong place.

```python
# WRONG — graph loading duplicated across files
def load_graphs_train():
    return [np.load(f) for f in Path("data/graphs").glob("*.npz")]

def load_graphs_eval():
    return [np.load(f) for f in Path("data/graphs").glob("*.npz")]

# RIGHT — one place, imported everywhere
def load_graphs(directory: Path, pattern: str = "*.npz") -> list[np.ndarray]:
    return [np.load(f) for f in sorted(directory.glob(pattern))]
```

Rules:
- Extraction threshold: if a block of logic appears in two files, extract it immediately.
- Feature extraction logic is owned by `kubescan/utils/yaml_parser.py`; research scripts
  import from it — they never re-implement it.
- Column name lists, feature lists, and label maps are defined once (in a constants module
  or config file) and imported everywhere else.

---

## 3. SOLID principles

### 3.1 Single Responsibility

Each class/module does exactly one thing. The name describes that one thing completely.

```python
# WRONG — GraphBuilder knows about files, parsing, AND ML
class GraphBuilder:
    def load_yaml(self, path): ...
    def parse_features(self, doc): ...
    def to_pyg_data(self, features): ...
    def save(self, data, path): ...

# RIGHT — each class owns one concern
class YamlParser:
    def parse(self, path: Path) -> dict[str, object]: ...

class FeatureExtractor:
    def extract(self, manifest: dict[str, object]) -> NodeFeatures: ...

class GraphConverter:
    def convert(self, features: NodeFeatures) -> Data: ...
```

### 3.2 Open/Closed

Extend behaviour by adding new classes, not by editing existing if-chains.

```python
# WRONG — every new scanner requires editing this function
def run_scanner(tool: str, path: Path) -> ScanResult:
    if tool == "checkov": ...
    elif tool == "trivy": ...  # <-- edit required for every new tool

# RIGHT — new tools are new classes
class Scanner(Protocol):
    def scan(self, path: Path) -> ScanResult: ...

class CheckovScanner:
    def scan(self, path: Path) -> ScanResult: ...

class ScanOrchestrator:
    def __init__(self, scanners: list[Scanner]) -> None:
        self._scanners = scanners

    def run_all(self, path: Path) -> list[ScanResult]:
        return [s.scan(path) for s in self._scanners]
```

### 3.3 Liskov Substitution

Subtypes must be substitutable for their base type. Never override a method to raise
`NotImplementedError` — that means the hierarchy is wrong.

```python
# WRONG
class BaseModel:
    def predict(self, x): raise NotImplementedError

# RIGHT — Protocol for structural subtyping (preferred)
class Predictor(Protocol):
    def predict(self, x: np.ndarray) -> np.ndarray: ...
```

### 3.4 Interface Segregation

Protocols are small and focused. Never create a fat protocol that only some implementors
can satisfy.

```python
# WRONG — not all models support explain()
class Model(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...
    def explain(self, X): ...  # GNN can't do this the same way RF does

# RIGHT — composable protocols
class Fittable(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

class Predictable(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...

class Explainable(Protocol):
    def explain(self, X: np.ndarray) -> Explanation: ...
```

### 3.5 Dependency Inversion

High-level modules depend on abstractions, not concrete implementations.
Pass dependencies in; never instantiate collaborators inside a class.

```python
# WRONG — hardwired concrete dependency
class EnsemblePredictor:
    def __init__(self, weights_path: Path) -> None:
        self._gnn = GATEncoder()            # concrete — untestable
        self._rf = RandomForestClassifier()

# RIGHT — inject abstractions
class EnsemblePredictor:
    def __init__(
        self,
        gnn: Predictable,
        rf: Predictable,
        ga_weights: GAWeights,
    ) -> None:
        self._gnn = gnn
        self._rf = rf
        self._weights = ga_weights
```

---

## 4. Type system — use it fully

All function signatures carry complete type hints. No `Any` unless wrapping a
third-party API that genuinely returns `Any`, and that must be bounded immediately.

```python
# WRONG
def build_graph(manifest, label):
    ...

# RIGHT
from __future__ import annotations
from pathlib import Path
from torch_geometric.data import Data

def build_graph(manifest: dict[str, object], label: int) -> Data:
    ...
```

Rules:
- `from __future__ import annotations` at the top of every file.
- Prefer `X | None` over `Optional[X]`.
- `TypeAlias` for complex repeated types: `Adjacency: TypeAlias = list[tuple[int, int]]`.
- `dataclass(frozen=True)` for value objects (features, results, configs).
- Return types are mandatory — including `-> None`.

---

## 5. Configuration — no hardcoded run parameters

```python
# WRONG — in train_gnn.py
LR = 0.001
HIDDEN_DIM = 64
EPOCHS = 100

# RIGHT
@dataclass(frozen=True)
class GNNConfig:
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_layers: int = 3
    epochs: int = 100
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> GNNConfig:
        with path.open() as f:
            return cls(**yaml.safe_load(f))
```

Rules:
- Every configurable value lives in a `Config` dataclass.
- Training scripts accept `--config path/to/config.yaml`. Default configs live in
  `research/configs/`.
- Seeds are always logged at `INFO` level at the start of every run.

---

## 6. Paths — always `pathlib.Path`, never strings

```python
# WRONG
path = "data/graphs/" + filename
os.path.join(base, "splits", "train.txt")

# RIGHT
GRAPHS_DIR = Path("data/graphs")
path = GRAPHS_DIR / filename
train_path = Path("data/splits") / "train.txt"
```

---

## 7. Error handling — explicit and typed

```python
# WRONG
try:
    data = load(path)
except:
    print("error")
    return None   # silent failure propagates corruption

# RIGHT
class KubescanError(Exception):
    """Base for all kubescan errors — catch this at the CLI boundary."""

class ManifestParseError(KubescanError):
    def __init__(self, path: Path, reason: str) -> None:
        super().__init__(f"Cannot parse {path}: {reason}")
        self.path = path

try:
    data = load(path)
except yaml.YAMLError as exc:
    raise ManifestParseError(path, str(exc)) from exc
```

Rules:
- Define `KubescanError` base in `kubescan/exceptions.py`; all package errors inherit it.
- Never catch bare `except:` or `except Exception:` in library code. Only at CLI boundary.
- Never return `None` to signal failure — raise a typed exception.
- Always chain with `raise X from original_exc`.
- Research scripts may use broader catches at the top-level `main()` only, with logging.

---

## 8. Logging — never `print()` in library code

```python
# WRONG (in any kubescan/ module)
print(f"Processing {path}")

# RIGHT
import logging
logger = logging.getLogger(__name__)

logger.debug("Processing %s", path)
logger.info("Graph built: nodes=%d edges=%d", n_nodes, n_edges)
logger.warning("Empty manifest at %s — skipping", path)
logger.error("Scanner failed for %s", path, exc_info=True)
```

Rules:
- `print()` is allowed only in `research/scripts/` pipeline entry points and `research/models/`
  training scripts, for progress reporting only.
- All `kubescan/` library code uses `logging.getLogger(__name__)`.
- `cli.py` configures root logger level from a `--verbose` flag.
- Log messages use `%s` formatting (lazy), not f-strings.

---

## 9. Module structure — imports and `__all__`

Every public module declares `__all__`. Nothing is implicitly public.

```python
# kubescan/model/ga_ensemble.py
from __future__ import annotations

__all__ = ["EnsembleScorer", "GAWeights", "run_gnn_ensemble", "compute_escape_fraction"]

# Import order:
# 1. from __future__ import annotations
# 2. stdlib
# 3. third-party (torch, numpy, click, …)
# 4. local (kubescan.*)
```

---

## 10. Functions — small, verb-named, one abstraction level

```python
# WRONG — one function doing everything
def process_manifest(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    # ... 50 lines of mixed parsing, feature extraction, graph building ...

# RIGHT — each function at one abstraction level, ≤ 30 lines
def parse_manifest(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    return [_normalize_resource(r) for r in (raw.get("items", [raw]) or []) if r]

def _normalize_resource(resource: dict[str, object]) -> dict[str, object]:
    return {
        "kind":       resource.get("kind", ""),
        "privileged": _is_privileged(resource),
    }

def _is_privileged(resource: dict[str, object]) -> bool:
    spec = resource.get("spec") or {}
    return bool((spec.get("securityContext") or {}).get("privileged", False))
```

Rules:
- Max ~30 lines per function. If it exceeds that, it does more than one thing.
- Private helpers prefixed `_`, defined in the same module.
- No boolean flag parameters that change behaviour: split into two functions or use an enum.
  `load(path, is_train=True)` → `load_train(path)` / `load_eval(path)`.

---

## 11. Tests

```
kubescan/tests/
  unit/
    test_yaml_parser.py       # mirrors kubescan/utils/yaml_parser.py
    test_graph_builder.py     # mirrors kubescan/utils/graph_builder.py
    test_gat_encoder.py       # mirrors kubescan/model/gat_encoder.py
    test_rf_classifier.py
    test_ga_ensemble.py
  integration/
    test_cli_scan.py          # end-to-end: kubescan scan ./fixtures/
  fixtures/
    valid_pod.yaml
    privileged_pod.yaml
    empty_configmap.yaml
```

Rules:
- Test filename mirrors module: `utils/yaml_parser.py` → `tests/unit/test_yaml_parser.py`.
- Test function name: `test_<function>_<condition>_<expected>`.
  Example: `test_extract_features_privileged_pod_sets_sec_cont_flag`.
- AAA pattern (Arrange / Act / Assert). One logical assertion per test.
- Fixtures use `pytest.fixture`, never module-level globals.
- Research scripts are not unit-tested; integration smoke tests in `research/tests/` verify
  the pipeline produces expected file shapes/sizes.

---

## 12. ML-specific rules

### Reproducibility
- Every training script accepts `--seed INT` and calls a shared `set_global_seed(seed: int)`
  utility that sets `random.seed`, `np.random.seed`, and `torch.manual_seed` in one place.
- The seed is logged at `INFO` level at the start of every run.

### Feature contracts
- The canonical node feature list is defined once in `kubescan/utils/yaml_parser.py:FEATURE_COLS`.
- `NODE_FEATURE_DIM = len(FEATURE_COLS) + 1` (the `+1` is the RF `risk_score` appended at
  index 25 by `graph_builder.py`).
- Training scripts must import and use `FEATURE_COLS` — never a hardcoded list.
- Changing `FEATURE_COLS` requires: (1) retraining all models, (2) bumping a schema version
  constant, (3) adding a migration entry in `research/scripts/fixes/`.

### Graph file format
- `.npz` files always contain exactly: `x` (node features `[N, 26]`), `edge_index` (`[2, E]`),
  `y` (integer label), `cluster_id` (str). Any schema change requires a migration script
  in `research/scripts/fixes/`.

### Checkpoint resolution (kubescan CLI)
- `--checkpoints-dir` flag → `KUBESCAN_CHECKPOINTS` env var →
  `kubescan/checkpoints/trained/` symlink → `FileNotFoundError`.
- The symlink target is `../../research/models/checkpoints/` (relative).

### Train vs inference boundary
- `kubescan/model/` is inference-only. Allowed imports: `torch`, `torch_geometric.data.Data`,
  `torch_geometric.nn`. Forbidden: `torch.optim`, training DataLoaders, `sklearn` fit methods.
- This boundary is enforced by convention and code review — not a runtime check.

---

## Summary cheat-sheet

| Category | Rule |
|----------|------|
| Magic values | Named constant, Enum, or Config dataclass |
| Duplication | Extract on second occurrence — no exceptions |
| Class size | One responsibility, one reason to change |
| Extension | New class, never an edited if-chain |
| Dependencies | Injected, never instantiated inside a class |
| Types | Full hints everywhere, no `Any` |
| Paths | `pathlib.Path` only |
| Errors | Typed hierarchy, never silent `None` return |
| Logging | `logging` in library, `print` only in scripts |
| Functions | ≤ 30 lines, one abstraction level, no bool flags |
| Tests | One assert, AAA, name encodes condition + expectation |
| Features | `FEATURE_COLS` from `yaml_parser.py` — single source of truth |
| Inference | `kubescan/model/` — no training code, ever |
