"""
create_splits.py
=================
Create stratified train/val/test splits and 5-fold cross-validation splits
for the GNN graph dataset.

Leakage rules (group-aware by origin cluster):
  1. Augmented variants ('_aug_' in the name) are derived near-duplicates of
     their base cluster. A variant joins a TRAINING partition only when its
     base cluster is in that same partition — variants of val/test clusters
     are excluded from training entirely, never just relabelled.
  2. The test clusters are held out of the CV folds altogether: folds
     partition only the train+val originals. Fold models therefore never see
     a test cluster (raw or augmented), and out-of-fold predictions used for
     GA ensemble-weight tuning contain no test cluster either.

Split logic:
  1. Separate original and augmented graphs from manifest
  2. Stratify-split original graphs into train/val/test
  3. Build k folds over the train+val originals only
  4. Per partition, append only the augmented variants whose base is in
     that partition's training originals

Outputs (dataset/splits/):
  train.txt, val.txt, test.txt   – cluster names for the default 70/15/15 split
  fold_<k>_train.txt             – cross-validation folds (k=0..4)
  fold_<k>_val.txt
  splits_config.json             – metadata on the splits

Usage:
  python scripts/create_splits.py
  python scripts/create_splits.py --seed 123 --val-frac 0.15 --test-frac 0.15
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def base_cluster(name: str) -> str:
    """Origin cluster of a graph name: 'foo_aug_03' → 'foo', 'foo' → 'foo'."""
    return name.split("_aug_")[0]


def augmented_for(
    augmented: list[tuple[str, int]],
    train_originals: list[str],
) -> list[str]:
    """Augmented variants whose base cluster is inside train_originals."""
    allowed = set(train_originals)
    return [name for name, _ in augmented if base_cluster(name) in allowed]


def stratified_split(
    clusters: list[tuple[str, int]],  # (name, label)
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Stratified train/val/test split.
    Maintains label distribution across splits.
    """
    rng = random.Random(seed)

    # Group by label
    by_label: dict[int, list[str]] = defaultdict(list)
    for name, lbl in clusters:
        by_label[lbl].append(name)

    train_names, val_names, test_names = [], [], []

    for _lbl, names in sorted(by_label.items()):
        names = names.copy()
        rng.shuffle(names)
        n = len(names)
        n_test = max(1, round(n * test_frac))
        n_val  = max(1, round(n * val_frac))
        n_train = n - n_val - n_test

        # Guard: ensure at least 1 in train when group is tiny
        if n_train <= 0:
            n_train = 1
            if n_val + n_test + n_train > n:
                n_val = max(0, n - n_train - n_test)

        test_names  += names[:n_test]
        val_names   += names[n_test:n_test + n_val]
        train_names += names[n_test + n_val:]

    rng.shuffle(train_names)
    rng.shuffle(val_names)
    rng.shuffle(test_names)

    return train_names, val_names, test_names


def k_fold_splits(
    clusters: list[tuple[str, int]],
    k: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    """
    Stratified k-fold cross-validation.
    Returns list of (train_names, val_names) for each fold.
    """
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for name, lbl in clusters:
        by_label[lbl].append(name)

    # Assign fold indices per stratum
    label_folds: dict[int, list[list[str]]] = {}
    for lbl, names in sorted(by_label.items()):
        names = names.copy()
        rng.shuffle(names)
        # Distribute into k buckets
        buckets: list[list[str]] = [[] for _ in range(k)]
        for i, name in enumerate(names):
            buckets[i % k].append(name)
        label_folds[lbl] = buckets

    folds = []
    for fold_idx in range(k):
        val   = []
        train = []
        for _lbl, buckets in label_folds.items():
            val   += buckets[fold_idx]
            for j, bucket in enumerate(buckets):
                if j != fold_idx:
                    train += bucket
        rng.shuffle(train)
        rng.shuffle(val)
        folds.append((train, val))

    return folds


def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent.parent  # research/ (scripts live in scripts/05_split/)
    default_manifest = project_root / "data" / "graphs" / "graph_manifest.csv"
    default_out      = project_root / "data" / "splits"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest",  type=Path, default=default_manifest)
    parser.add_argument("--out-dir",   type=Path, default=default_out)
    parser.add_argument("--val-frac",  type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--folds",     type=int,   default=5)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load graph manifest
    print(f"Loading {args.manifest}...")
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Separate originals from augmented (augmented have '_aug_' in cluster name)
    original_rows  = [r for r in rows if "_aug_" not in r["cluster"]]
    augmented_rows = [r for r in rows if "_aug_" in r["cluster"]]

    originals  = [(r["cluster"], int(r["label"])) for r in original_rows]
    augmented  = [(r["cluster"], int(r["label"])) for r in augmented_rows]

    print(f"  {len(rows)} total graphs: {len(originals)} original + {len(augmented)} augmented")

    label_dist = defaultdict(int)
    for _, lbl in [(r["cluster"], int(r["label"])) for r in rows]:
        label_dist[lbl] += 1
    orig_dist = defaultdict(int)
    for _, lbl in originals:
        orig_dist[lbl] += 1
    print(f"  Original label distribution: {dict(sorted(orig_dist.items()))}")
    print(f"  Total label distribution:    {dict(sorted(label_dist.items()))}")

    # ------------------------------------------------------------------
    # Train / Val / Test split  (originals only → then add augmented to train)
    # ------------------------------------------------------------------
    train_orig, val, test = stratified_split(originals, args.val_frac, args.test_frac, args.seed)

    # Group-aware: only variants of TRAIN originals may join training.
    # Variants of val/test clusters are dropped — training on them would leak.
    aug_train_names = augmented_for(augmented, train_orig)
    n_aug_dropped   = len(augmented) - len(aug_train_names)
    train = train_orig + aug_train_names
    random.Random(args.seed).shuffle(train)
    print(f"  Augmented: {len(aug_train_names)} join train, "
          f"{n_aug_dropped} excluded (base cluster in val/test)")

    def write_list(path: Path, names: list[str]):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")

    write_list(args.out_dir / "train.txt", train)
    write_list(args.out_dir / "val.txt",   val)
    write_list(args.out_dir / "test.txt",  test)

    lbl_map = dict(originals + augmented)
    print(f"\nTrain/Val/Test split (seed={args.seed}):")
    for split_name, names in [("train", train), ("val", val), ("test", test)]:
        dist = defaultdict(int)
        for n in names:
            dist[lbl_map[n]] += 1
        dist_str = " | ".join(f"label{k}={v}" for k, v in sorted(dist.items()))
        print(f"  {split_name:5s}: {len(names):3d} graphs  [{dist_str}]")

    # ------------------------------------------------------------------
    # K-fold cross-validation splits — test clusters held out entirely.
    # Folds partition train+val originals; OOF predictions (used for GA
    # weight tuning) therefore never include a test cluster.
    # ------------------------------------------------------------------
    test_set  = set(test)
    cv_pool   = [(name, lbl) for name, lbl in originals if name not in test_set]
    folds = k_fold_splits(cv_pool, args.folds, args.seed)
    print(f"\n{args.folds}-fold cross-validation splits "
          f"({len(cv_pool)} originals, {len(test)} test clusters held out):")
    lbl_map = dict(originals + augmented)
    for fold_idx, (fold_train_orig, fold_val) in enumerate(folds):
        # Only variants of this fold's TRAIN originals join its training set
        fold_aug   = augmented_for(augmented, fold_train_orig)
        fold_train = fold_train_orig + fold_aug
        random.Random(args.seed + fold_idx).shuffle(fold_train)

        write_list(args.out_dir / f"fold_{fold_idx}_train.txt", fold_train)
        write_list(args.out_dir / f"fold_{fold_idx}_val.txt",   fold_val)
        dist = defaultdict(int)
        for n in fold_val:
            dist[lbl_map[n]] += 1
        dist_str = " | ".join(f"label{k}={v}" for k, v in sorted(dist.items()))
        print(f"  fold {fold_idx}: train={len(fold_train)} (orig={len(fold_train_orig)}, aug={len(fold_aug)}), val={len(fold_val)}  val_dist=[{dist_str}]")

    # ------------------------------------------------------------------
    # Save splits_config.json
    # ------------------------------------------------------------------
    config = {
        "seed": args.seed,
        "total_graphs":     len(rows),
        "original_graphs":  len(originals),
        "augmented_graphs": len(augmented),
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "train_count": len(train),
        "val_count":   len(val),
        "test_count":  len(test),
        "k_folds":     args.folds,
        "augmentation_note": (
            "Group-aware splits: an augmented graph (_aug_ suffix) joins a training "
            "partition only when its base cluster is in that partition. Variants of "
            "val/test clusters are excluded from training entirely. Test clusters are "
            "held out of the CV folds, so OOF predictions used for GA weight tuning "
            "contain no test cluster."
        ),
        "label_names": {"0": "clean", "1": "isolated_misconfig", "2": "attack_chain"},
        "label_distribution_total":    dict(sorted(label_dist.items())),
        "label_distribution_originals": dict(sorted(orig_dist.items())),
        "cv_pool_originals": len(cv_pool),
        "augmented_in_global_train": len(aug_train_names),
        "augmented_excluded_from_train": n_aug_dropped,
    }

    config_path = args.out_dir / "splits_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSplits written to {args.out_dir}/")
    print("  train.txt, val.txt, test.txt")
    print(f"  fold_{{0..{args.folds-1}}}_train.txt, fold_{{0..{args.folds-1}}}_val.txt")
    print("  splits_config.json")


if __name__ == "__main__":
    main()
