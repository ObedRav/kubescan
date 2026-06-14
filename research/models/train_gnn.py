"""
train_gnn.py
=============
Train a Graph Neural Network (Layer 2) to predict multi-hop attack chains
in Kubernetes cluster graphs.

Architecture: GAT (Graph Attention Network) with global mean pooling.
Task        : Graph-level 3-class classification (clean / isolated / chain).
Evaluation  : Macro-F1 across 5-fold cross-validation.

Input graphs (from graph_builder.py):
  Node features : 25-dim vector (18 Rahman flags + 6 extended + risk_score)
  Edge types    : 0-3 (directory proximity, privilege reach, SA lateral, namespace)
  Graph labels  : 0=clean, 1=isolated_misconfig, 2=attack_chain

Usage:
  python models/train_gnn.py
  python models/train_gnn.py --epochs 200 --hidden 64 --heads 4 --layers 3
  python models/train_gnn.py --binary  # binary classification (clean vs misconfigured)

Requirements:
  pip install torch torch-geometric scikit-learn
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n"
        "Install with: pip install torch torch-geometric scikit-learn"
    )

# Add dataset/scripts to path for gnn_dataset import
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "04_build_datasets"))
from gnn_dataset import LABEL_NAMES, KubeClusterDataset, load_split

# Single source of truth for the production architecture: the kubescan package.
# Research may import from kubescan (never the reverse).
try:
    from kubescan.model.gat_encoder import KubeGAT
    from kubescan.utils.device_utils import resolve_device
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT.parent / "kubescan" / "src"))
    from kubescan.model.gat_encoder import KubeGAT
    from kubescan.utils.device_utils import resolve_device

from provenance import provenance

__all__ = ["KubeGAT", "KubeGCN"]

# ---------------------------------------------------------------------------
# GCN ablation baseline
# ---------------------------------------------------------------------------

class KubeGCN(nn.Module):
    """
    GCN ablation baseline: same skeleton as KubeGAT (input projection,
    LayerNorm + residual, mean+max pooling, MLP head) with GCNConv instead of
    attention. GCNConv has no edge-feature support, so edge types are ignored —
    that is the point of the ablation: does attention over typed edges help?
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.input_proj = nn.Linear(in_channels, hidden)
        self.input_norm = nn.LayerNorm(hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch):  # edge_attr unused by design
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms, strict=True):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = x + residual
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        return self.classifier(torch.cat([x_mean, x_max], dim=-1))


def make_model(args: argparse.Namespace, in_channels: int, num_classes: int) -> nn.Module:
    """Instantiate the model selected by --conv (gat = production, gcn = ablation)."""
    if args.conv == "gcn":
        return KubeGCN(
            in_channels=in_channels,
            hidden=args.hidden,
            num_layers=args.layers,
            num_classes=num_classes,
            dropout=args.dropout,
        )
    return KubeGAT(
        in_channels=in_channels,
        hidden=args.hidden,
        heads=args.heads,
        num_layers=args.layers,
        num_classes=num_classes,
        dropout=args.dropout,
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: KubeClusterDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to handle imbalance."""
    counts = torch.zeros(num_classes)
    for i in range(len(dataset)):
        label = dataset[i].y.item()
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    return weights / weights.sum() * num_classes


def train_epoch(
    model: KubeGAT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        # Gradient clipping for stability on small graphs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: KubeGAT,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    all_preds  = []
    all_true   = []
    all_probs  = []    # softmax probabilities for ranking
    for batch in loader:
        batch = batch.to(device)
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = F.softmax(out, dim=-1)
        preds = out.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(batch.y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    labels = list(range(num_classes))
    chain_class = min(2, num_classes - 1)   # label index for attack_chain

    # Precision@K: rank by predicted chain probability, check top-K are truly chains
    chain_scores = [p[chain_class] for p in all_probs]
    ranked_true  = [y for _, y in sorted(
        zip(chain_scores, all_true), reverse=True
    )]

    def precision_at_k(ranked: list[int], k: int, positive_label: int) -> float:
        top_k = ranked[:k]
        return sum(1 for y in top_k if y == positive_label) / k if k else 0.0

    p_at_1  = precision_at_k(ranked_true, 1,  chain_class)
    p_at_3  = precision_at_k(ranked_true, 3,  chain_class)
    p_at_5  = precision_at_k(ranked_true, 5,  chain_class)
    p_at_10 = precision_at_k(ranked_true, 10, chain_class)

    return {
        "accuracy":      accuracy_score(all_true, all_preds),
        "macro_f1":      f1_score(all_true, all_preds, labels=labels,
                                  average="macro", zero_division=0),
        "per_class_f1":  f1_score(all_true, all_preds, labels=labels,
                                   average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(all_true, all_preds,
                                              labels=labels).tolist(),
        "precision_at_1":  p_at_1,
        "precision_at_3":  p_at_3,
        "precision_at_5":  p_at_5,   # primary metric from project spec
        "precision_at_10": p_at_10,
        "predictions":   all_preds,
        "true_labels":   all_true,
        "chain_scores":  [float(s) for s in chain_scores],
    }


# ---------------------------------------------------------------------------
# Single fold training loop
# ---------------------------------------------------------------------------

def train_fold(
    train_set: KubeClusterDataset,
    val_set: KubeClusterDataset,
    args: argparse.Namespace,
    device: torch.device,
    fold_idx: int = 0,
) -> dict:

    num_classes = 2 if args.binary else 3
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)

    in_channels = train_set[0].x.shape[1]
    model = make_model(args, in_channels, num_classes).to(device)

    class_weights = compute_class_weights(train_set, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    best_f1     = 0.0
    best_state  = None
    patience    = args.patience
    no_improve  = 0
    history     = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, num_classes)
        scheduler.step()

        val_f1 = val_metrics["macro_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_f1": val_f1,
                         "val_acc": val_metrics["accuracy"]})

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"    [fold {fold_idx}] epoch {epoch:4d}  "
                  f"loss={train_loss:.4f}  val_F1={val_f1:.3f}  "
                  f"P@5={val_metrics['precision_at_5']:.2f}  "
                  f"val_acc={val_metrics['accuracy']:.3f}"
                  + ("  ← best" if no_improve == 0 else ""))

        if no_improve >= patience:
            print(f"    [fold {fold_idx}] Early stop at epoch {epoch} (no improve for {patience} epochs)")
            break

    # Load best weights for final evaluation
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_metrics = evaluate(model, val_loader, device, num_classes)

    return {
        "fold":          fold_idx,
        "best_val_f1":   best_f1,
        "final_metrics": final_metrics,
        "history":       history,
        "model_state":   best_state,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent

    default_graphs = project_root / "data" / "graphs"
    default_splits = project_root / "data" / "splits"
    default_out    = project_root / "models" / "checkpoints"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Data
    parser.add_argument("--graphs-dir",  type=Path, default=default_graphs)
    parser.add_argument("--splits-dir",  type=Path, default=default_splits)
    parser.add_argument("--out-dir",     type=Path, default=default_out)
    # Model
    parser.add_argument("--conv",        type=str,   default="gat", choices=["gat", "gcn"],
                        help="gat = production KubeGAT; gcn = ablation baseline")
    parser.add_argument("--hidden",      type=int,   default=64)
    parser.add_argument("--heads",       type=int,   default=4)
    parser.add_argument("--layers",      type=int,   default=3)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--binary",      action="store_true",
                        help="Binary classification: clean vs any-misconfig")
    # Training
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=5e-4)
    parser.add_argument("--weight-decay",type=float, default=1e-4)
    parser.add_argument("--patience",    type=int,   default=50,
                        help="Early stopping patience (epochs)")
    # Cross-validation
    parser.add_argument("--cv-folds",    type=int,   default=5,
                        help="Number of CV folds (0 = use train/val/test split)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Device: {device}")
    print(f"Mode: {'binary' if args.binary else '3-class'}  "
          f"| hidden={args.hidden}, heads={args.heads}, layers={args.layers}")

    num_classes = 2 if args.binary else 3
    label_names = {0: "clean", 1: "misconfigured"} if args.binary else LABEL_NAMES

    # ------------------------------------------------------------------
    # Cross-validation mode
    # ------------------------------------------------------------------
    if args.cv_folds > 0:
        print(f"\n{args.cv_folds}-fold cross-validation")
        fold_results = []

        for fold_idx in range(args.cv_folds):
            train_file = args.splits_dir / f"fold_{fold_idx}_train.txt"
            val_file   = args.splits_dir / f"fold_{fold_idx}_val.txt"
            if not train_file.exists():
                print(f"  [!] Missing split file: {train_file}. Run create_splits.py first.")
                continue

            train_set = load_split(args.graphs_dir, train_file, binary=args.binary)
            val_set   = load_split(args.graphs_dir, val_file,   binary=args.binary)
            print(f"\n  Fold {fold_idx}: train={len(train_set)}, val={len(val_set)}")

            result = train_fold(train_set, val_set, args, device, fold_idx)
            fold_results.append(result)

            fm = result["final_metrics"]
            print(f"  Fold {fold_idx} result: macro-F1={fm['macro_f1']:.4f}  "
                  f"P@5={fm['precision_at_5']:.2f}  acc={fm['accuracy']:.4f}")
            print("  Per-class F1: "
                  + " | ".join(f"{label_names[i]}={fm['per_class_f1'][i]:.3f}"
                                for i in range(num_classes) if i < len(fm['per_class_f1'])))

            # Save fold model
            ckpt_path = args.out_dir / f"gnn_fold_{fold_idx}.pt"
            torch.save(result["model_state"], ckpt_path)

        # ------------------------------------------------------------------
        # Aggregate CV results
        # ------------------------------------------------------------------
        if fold_results:
            all_f1s = [r["final_metrics"]["macro_f1"] for r in fold_results]
            all_acc = [r["final_metrics"]["accuracy"]  for r in fold_results]

            all_p5  = [r["final_metrics"]["precision_at_5"] for r in fold_results]
            all_p10 = [r["final_metrics"]["precision_at_10"] for r in fold_results]

            print(f"\n{'='*60}")
            print("CROSS-VALIDATION SUMMARY")
            print(f"{'='*60}")
            print(f"  Macro-F1        : {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
            print(f"  Precision@5     : {np.mean(all_p5):.4f} ± {np.std(all_p5):.4f}  (target: > 0.70)")
            print(f"  Precision@10    : {np.mean(all_p10):.4f} ± {np.std(all_p10):.4f}")
            print(f"  Accuracy        : {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")

            # Per-fold breakdown
            per_class_f1s = [r["final_metrics"]["per_class_f1"] for r in fold_results]
            print("\n  Per-class macro-F1 (mean ± std):")
            for i in range(num_classes):
                vals = [fold[i] for fold in per_class_f1s if i < len(fold)]
                print(f"    {label_names.get(i, i):20s}: "
                      f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")

            # Save aggregate results
            cv_summary = {
                "num_folds":              args.cv_folds,
                "binary":                 args.binary,
                "macro_f1_mean":          float(np.mean(all_f1s)),
                "macro_f1_std":           float(np.std(all_f1s)),
                "precision_at_5_mean":    float(np.mean(all_p5)),
                "precision_at_5_std":     float(np.std(all_p5)),
                "precision_at_5_target":  0.70,
                "accuracy_mean":          float(np.mean(all_acc)),
                "accuracy_std":           float(np.std(all_acc)),
                "fold_f1s":               [float(f) for f in all_f1s],
                "fold_p5s":               [float(p) for p in all_p5],
                "model_config": {
                    "conv": args.conv,
                    "hidden": args.hidden, "heads": args.heads,
                    "layers": args.layers, "dropout": args.dropout,
                    "lr": args.lr, "epochs": args.epochs,
                },
                "_provenance": provenance(
                    seed=args.seed, conv=args.conv, layers=args.layers,
                ),
            }
            summary_path = args.out_dir / "cv_results.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(cv_summary, f, indent=2)
            print(f"\n  Results saved to {summary_path}")

    # ------------------------------------------------------------------
    # Fixed split mode (--cv-folds 0)
    # ------------------------------------------------------------------
    else:
        print("\nFixed train/val/test split")
        train_set = load_split(args.graphs_dir, args.splits_dir / "train.txt", binary=args.binary)
        val_set   = load_split(args.graphs_dir, args.splits_dir / "val.txt",   binary=args.binary)
        test_set  = load_split(args.graphs_dir, args.splits_dir / "test.txt",  binary=args.binary)

        print(f"  train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

        result = train_fold(train_set, val_set, args, device, fold_idx=0)

        # Evaluate on test set
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        test_in_channels = test_set[0].x.shape[1]
        test_model = make_model(args, test_in_channels, num_classes).to(device)
        test_model.load_state_dict(
            {k: v.to(device) for k, v in result["model_state"].items()}
        )
        test_metrics = evaluate(test_model, test_loader, device, num_classes)

        print(f"\n{'='*60}")
        print("TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"  Macro-F1       : {test_metrics['macro_f1']:.4f}")
        print(f"  Precision@5    : {test_metrics['precision_at_5']:.4f}  (target: > 0.70)")
        print(f"  Precision@10   : {test_metrics['precision_at_10']:.4f}")
        print(f"  Accuracy       : {test_metrics['accuracy']:.4f}")
        print("\n  Per-class F1:")
        for i in range(num_classes):
            if i < len(test_metrics["per_class_f1"]):
                print(f"    {label_names.get(i, i):20s}: {test_metrics['per_class_f1'][i]:.4f}")
        print("\n  Confusion matrix (rows=true, cols=pred):")
        for row in test_metrics["confusion_matrix"]:
            print(f"    {row}")

        # Save model
        ckpt_path = args.out_dir / "gnn_best.pt"
        torch.save(result["model_state"], ckpt_path)
        print(f"\n  Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
