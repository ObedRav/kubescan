# kubescan

Kubernetes attack-chain risk scanner using a GNN + Random Forest ensemble.

Scans a directory of Kubernetes YAML manifests and predicts whether the cluster
contains an exploitable multi-hop attack chain (pod-escape → lateral movement).

## Install

```bash
pip install -e /path/to/TFE/kubescan
```

## Usage

```bash
# Point to your cluster manifests
kubescan scan ./my-cluster/

# JSON output (for CI/CD integration)
kubescan scan ./configs/ --format json

# Show per-manifest breakdown
kubescan scan ./configs/ --show-nodes

# Specify checkpoints explicitly
kubescan scan ./configs/ \
  --checkpoints-dir /path/to/TFE/research/models/checkpoints
```

### Checkpoint resolution order

1. `--checkpoints-dir` flag
2. `KUBESCAN_CHECKPOINTS` environment variable
3. `kubescan/checkpoints/trained/` (symlink to `research/models/checkpoints/`)

## Output example (text)

```
==================================================================
  KUBESCAN  ·  Attack-Chain Risk Report
  Cluster : my-cluster
  Path    : /home/user/my-cluster
==================================================================

  VERDICT  ·  ATTACK_CHAIN   ✗  HIGH RISK — review immediately

  Ensemble score    : 0.7197
  Chain probability : 0.9898   (5-fold GNN ensemble)
  Clean probability : 0.0032
  Mean RF risk      : 0.1791
  Escape fraction   : 1.0000   (3/3 manifests have escape flags)
  Lateral fraction  : 2/3 manifests have lateral flags

  Weights: w_rf=0.337  w_gnn=0.328  w_escape=0.335
==================================================================
```

## Architecture

```
YAML files
    │
    ▼
yaml_parser.py   ─── 25 binary security flags per manifest
    │
    ▼
rf.py            ─── risk_score ∈ [0,1] per manifest  (Random Forest)
    │
    ▼
graph_builder.py ─── cluster graph (nodes=manifests, edges=attack paths)
    │
    ▼
gat.py           ─── chain_probability ∈ [0,1]  (5-fold GAT ensemble)
    │
    ▼
ensemble.py      ─── final_score = w_rf·risk + w_gnn·chain + w_esc·escape
```

## Dependencies

```
torch>=2.0
torch-geometric>=2.4
scikit-learn>=1.3
pyyaml>=6.0
networkx>=3.0
numpy>=1.24
click>=8.1
```
