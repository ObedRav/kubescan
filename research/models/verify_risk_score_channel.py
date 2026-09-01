#!/usr/bin/env python3
"""
Characterise and ablate the Layer-1 -> Layer-2 bridge (node feature index 25).

Answers two register items:
  - L10  is the risk_score channel actually populated?
  - L10  does removing it change the deployed test-set result?

Part A is static: it reports how index 25 is produced and how many graphs carry
a non-zero value, read straight from the shipped .npz files.

Part B re-runs the deployed fold ensemble on the test split with index 25 zeroed.
It never writes to research/models/checkpoints/ -- evaluate_test_set.main()
persists test_results.json, so this module reimplements only the forward pass and
the ranking metrics it needs, leaving every checkpoint artefact untouched.

Usage:
    python research/models/verify_risk_score_channel.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "research" / "models"))
sys.path.insert(0, str(REPO / "research" / "scripts" / "04_build_datasets"))
sys.path.insert(0, str(REPO / "kubescan" / "src"))

GRAPHS = REPO / "research" / "data" / "graphs"
SPLITS = REPO / "research" / "data" / "splits"
CKPT = REPO / "research" / "models" / "checkpoints"
CSV_PATH = REPO / "research" / "data" / "tabular" / "rf_dataset.csv"

RISK_SCORE_INDEX = 25
LABEL_NAMES = {0: "clean", 1: "isolated", 2: "attack_chain"}


# ---------------------------------------------------------------------------
# Part A — static characterisation
# ---------------------------------------------------------------------------
def characterise_channel() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    empty = [r for r in rows if (r.get("risk_score") or "") == ""]
    print("=" * 68)
    print("A. Provenance of node attribute 25 (risk_score)")
    print("=" * 68)
    print(f"  tabular rows                    : {len(rows)}")
    print(f"  rows with empty risk_score      : {len(empty)}")
    print(f"    by source                     : {dict(Counter(r['source'] for r in empty))}")
    print("  computed by                     : compute_risk_score() — weighted sum")
    print("                                    of FEATURE_COLS, NOT an RF prediction")

    manifest = [
        x for x in csv.DictReader((GRAPHS / "graph_manifest.csv").open())
        if "_aug_" not in x["cluster"]
    ]
    nonzero = zero = 0
    by_label: dict[str, Counter] = {"zero": Counter(), "nonzero": Counter()}
    for m in manifest:
        path = GRAPHS / f"{m['safe_name']}.npz"
        if not path.exists():
            continue
        x = np.load(path, allow_pickle=True)["x"]
        has = bool(np.abs(x[:, RISK_SCORE_INDEX]).sum() > 0)
        bucket = "nonzero" if has else "zero"
        by_label[bucket][LABEL_NAMES[int(m["label"])]] += 1
        nonzero += has
        zero += not has

    total = nonzero + zero
    print(f"\n  original graphs inspected       : {total}")
    print(f"    index 25 all-zero             : {zero}  ({zero / total:.1%})")
    print(f"    index 25 populated            : {nonzero}  ({nonzero / total:.1%})")
    for bucket in ("zero", "nonzero"):
        c = by_label[bucket]
        n = sum(c.values())
        if n:
            print(f"      [{bucket:7s}] clean={c['clean']:>3} "
                  f"isolated={c['isolated']:>3} attack_chain={c['attack_chain']:>3}")


# ---------------------------------------------------------------------------
# Part B — ablation on the deployed fold ensemble (read-only)
# ---------------------------------------------------------------------------
def ablate_channel() -> None:
    import torch
    import torch.nn.functional as F
    from gnn_dataset import load_split
    from sklearn.metrics import confusion_matrix, f1_score

    from kubescan.model.gat_encoder import KubeGAT
    from kubescan.model.ga_ensemble import chain_rank_key
    from kubescan.utils.device_utils import resolve_device
    from kubescan.utils.seed_utils import set_global_seed

    set_global_seed(42)
    device = resolve_device()

    def run(zero_idx25: bool) -> dict:
        dataset = load_split(GRAPHS, SPLITS / "test.txt")
        if zero_idx25:
            for i in range(len(dataset)):
                dataset[i].x[:, RISK_SCORE_INDEX] = 0.0

        models = []
        for fold in range(5):
            p = CKPT / f"gnn_fold_{fold}.pt"
            if not p.exists():
                continue
            m = KubeGAT(in_channels=dataset[0].x.shape[1], hidden=64, heads=4,
                        num_layers=3, num_classes=3, dropout=0.3).to(device)
            m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            m.eval()
            models.append(m)

        y_true, y_pred, chain_p, n_nodes = [], [], [], []
        with torch.no_grad():
            for i in range(len(dataset)):
                d = dataset[i].to(device)
                batch = torch.zeros(d.x.shape[0], dtype=torch.long, device=device)
                probs = torch.stack([
                    F.softmax(m(d.x, d.edge_index, d.edge_attr, batch), dim=-1)
                    for m in models
                ]).mean(0).squeeze(0)
                y_true.append(int(d.y.item()))
                y_pred.append(int(probs.argmax().item()))
                chain_p.append(float(probs[2].item()))
                n_nodes.append(int(d.x.shape[0]))

        order = sorted(range(len(y_true)),
                       key=lambda i: chain_rank_key(chain_p[i], n_nodes[i]), reverse=True)
        out = {}
        for k in (1, 3, 5):
            top = order[:k]
            n_pos = sum(1 for t in y_true if t == 2)
            hits = sum(1 for i in top if y_true[i] == 2)
            out[f"P@{k}"] = hits / min(k, n_pos) if n_pos else 0.0
        out["FPR_clean"] = sum(1 for i in order[:5] if y_true[i] == 0) / 5
        out["macro_f1"] = f1_score(y_true, y_pred, average="macro")
        out["per_class"] = f1_score(y_true, y_pred, average=None).tolist()
        out["cm"] = confusion_matrix(y_true, y_pred).tolist()
        return out

    print("\n" + "=" * 68)
    print("B. Ablation of node attribute 25 on the test split")
    print("=" * 68)
    base = run(zero_idx25=False)
    abl = run(zero_idx25=True)
    for key in ("P@1", "P@3", "P@5", "FPR_clean", "macro_f1"):
        d = abl[key] - base[key]
        print(f"  {key:11s} deployed={base[key]:.4f}   idx25 zeroed={abl[key]:.4f}   Δ={d:+.4f}")
    print(f"  per-class F1 deployed={[round(v, 4) for v in base['per_class']]}")
    print(f"  per-class F1 zeroed  ={[round(v, 4) for v in abl['per_class']]}")
    print(f"  confusion identical  ={base['cm'] == abl['cm']}")
    print("\n  (no checkpoint artefact was written by this script)")


if __name__ == "__main__":
    characterise_channel()
    try:
        ablate_channel()
    except ImportError as exc:
        print(f"\n[skip] torch / torch_geometric unavailable: {exc}")
