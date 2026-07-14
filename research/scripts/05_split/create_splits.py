"""
create_splits.py
=================
Create stratified train/val/test splits and 5-fold cross-validation splits
for the GNN graph dataset.

Leakage rules (group-aware by template family):
  1. Augmented variants ('_aug_' in the name) are derived near-duplicates of
     their base cluster. A variant joins a TRAINING partition only when its
     base cluster is in that same partition — variants of val/test clusters
     are excluded from training entirely, never just relabelled.
  2. Base clusters that share a template family (e.g. 'badpods_priv' and
     'badpods_hostpid', both derived from the same BadPods fixture and
     differing only in which escape flag is toggled) are near-duplicates of
     each other, not independent samples. They are kept together in a single
     partition — never scattered across train/val/test or across CV folds —
     otherwise a model trained on five siblings of a family can trivially
     "solve" the one held out, or conversely a family's only rare-flag
     variant can land alone in test with no in-distribution training signal.
  3. The test clusters are held out of the CV folds altogether: folds
     partition only the train+val originals. Fold models therefore never see
     a test cluster (raw or augmented), and out-of-fold predictions used for
     GA ensemble-weight tuning contain no test cluster either.

Split logic:
  1. Separate original and augmented graphs from manifest
  2. Group original graphs into template families (family_of)
  3. Stratify-split families (by each family's dominant label) into
     train/val/test, then expand back into member cluster names
  4. Build k folds over the train+val families only
  5. Per partition, append only the augmented variants whose base is in
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

# Attack-chain is the rarest, highest-value class: if a family contains any
# member of a class, that class should win the family's stratification label
# (e.g. a family with one 'chain' and five 'isolated' members is a chain
# family for splitting purposes, not an isolated one).
LABEL_PRIORITY: list[int] = [2, 0, 1]  # attack_chain > clean > isolated


def base_cluster(name: str) -> str:
    """Origin cluster of a graph name: 'foo_aug_03' → 'foo', 'foo' → 'foo'."""
    return name.split("_aug_", maxsplit=1)[0]


def family_of(name: str) -> str:
    """Template family of a graph's origin cluster.

    'badpods_priv' -> 'badpods', 'datadog_cluster-agent' -> 'datadog',
    'longhorn' -> 'longhorn' (single-repo clusters are their own family).
    Groups near-duplicate template variants that differ only in which
    flag/rule is toggled, so splitting treats the whole family as one
    leakage-prone unit instead of scattering its variants across partitions.
    """
    base = base_cluster(name)
    return base.split("_", maxsplit=1)[0] if "_" in base else base


def family_label(members: list[int]) -> int:
    """Dominant label for a family, preferring the rarest class present."""
    present = set(members)
    for lbl in LABEL_PRIORITY:
        if lbl in present:
            return lbl
    return members[0]


def augmented_for(
    augmented: list[tuple[str, int]],
    train_originals: list[str],
) -> list[str]:
    """Augmented variants whose base cluster is inside train_originals."""
    allowed = set(train_originals)
    return [name for name, _ in augmented if base_cluster(name) in allowed]


def group_by_family(
    clusters: list[tuple[str, int]],
) -> dict[str, list[tuple[str, int]]]:
    """Group (name, label) pairs into leakage units for splitting.

    A template family becomes one atomic unit only when it is
    label-mixed (e.g. 'badpods_hostpid' is isolated-adjacent chain
    scaffolding while 'badpods_priv' is chain — near-duplicate variants
    that differ across the chain/non-chain boundary must not be scattered
    across partitions). Label-pure families (e.g. the 245 'checkov_*'
    fixtures, all isolated) don't exhibit that risk — grouping them
    atomically would instead badly distort the stratified proportions
    given their size — so each of their members is kept independently
    splittable.
    """
    by_family: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for name, lbl in clusters:
        by_family[family_of(name)].append((name, lbl))

    units: dict[str, list[tuple[str, int]]] = {}
    for family_id, members in by_family.items():
        if len({lbl for _, lbl in members}) > 1:
            units[family_id] = members
        else:
            for name, lbl in members:
                units[name] = [(name, lbl)]
    return units


def _stratum_weight(members: list[int], stratum_label: int) -> int:
    """Count of a unit's members that actually carry the stratum's label.

    A mixed-label family (e.g. 12-member kubernetes_goat, 2 of them chain)
    is one atomic unit, but its *weight* inside the chain stratum should be
    2, not 12 — otherwise one large family swings a bucket's true chain
    count unpredictably.
    """
    return sum(1 for lbl in members if lbl == stratum_label)


def _greedy_allocate(
    unit_ids: list[str],
    weights: dict[str, int],
    frac_by_bucket: dict[str, float],
    rng: random.Random,
) -> dict[str, list[str]]:
    """Assign units to buckets so each bucket's total weight tracks its
    target fraction, largest-weight unit first (longest-processing-time
    bin balancing). Plain per-unit *count* splitting would let one large
    family's presence or absence swing a bucket's true class
    representation unpredictably; balancing by weight keeps the number of
    stratum-label graphs per bucket close to the requested fraction
    regardless of which units happen to land where.
    """
    ids = unit_ids.copy()
    rng.shuffle(ids)
    ids.sort(key=lambda fid: -weights[fid])
    total = sum(weights[fid] for fid in ids)
    totals = dict.fromkeys(frac_by_bucket, 0)
    assigned: dict[str, list[str]] = {b: [] for b in frac_by_bucket}
    for fid in ids:
        deficits = {b: frac_by_bucket[b] * total - totals[b] for b in frac_by_bucket}
        best = max(deficits, key=lambda b: deficits[b])
        assigned[best].append(fid)
        totals[best] += weights[fid]
    return assigned


def stratified_split(
    clusters: list[tuple[str, int]],  # (name, label)
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Stratified train/val/test split, grouped by template family so that
    near-duplicate variants of the same family never land in different
    partitions, and weighted so each family's true stratum-label count
    (not its unit count) tracks the requested val/test fractions.
    """
    rng = random.Random(seed)
    families = group_by_family(clusters)
    fracs = {"train": 1 - val_frac - test_frac, "val": val_frac, "test": test_frac}

    by_label: dict[int, list[str]] = defaultdict(list)
    for family_id, members in families.items():
        lbl = family_label([m_lbl for _, m_lbl in members])
        by_label[lbl].append(family_id)

    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for lbl, family_ids in sorted(by_label.items()):
        weights = {
            fid: _stratum_weight([m_lbl for _, m_lbl in families[fid]], lbl)
            for fid in family_ids
        }
        allocation = _greedy_allocate(family_ids, weights, fracs, rng)
        for bucket, ids in allocation.items():
            buckets[bucket] += ids

    def expand(ids: list[str]) -> list[str]:
        names = [name for fid in ids for name, _ in families[fid]]
        rng.shuffle(names)
        return names

    return expand(buckets["train"]), expand(buckets["val"]), expand(buckets["test"])


def k_fold_splits(
    clusters: list[tuple[str, int]],
    k: int,
    seed: int,
) -> list[tuple[list[str], list[str]]]:
    """
    Stratified k-fold cross-validation, grouped by template family so that
    no family's variants are split between a fold's train and validation
    sets, and weighted so each fold's true stratum-label count tracks 1/k.
    Returns list of (train_names, val_names) for each fold.
    """
    rng = random.Random(seed)
    families = group_by_family(clusters)

    by_label: dict[int, list[str]] = defaultdict(list)
    for family_id, members in families.items():
        lbl = family_label([m_lbl for _, m_lbl in members])
        by_label[lbl].append(family_id)

    fold_keys = [str(i) for i in range(k)]
    fracs = dict.fromkeys(fold_keys, 1.0 / k)
    fold_ids: dict[str, list[str]] = {key: [] for key in fold_keys}
    for lbl, family_ids in sorted(by_label.items()):
        weights = {
            fid: _stratum_weight([m_lbl for _, m_lbl in families[fid]], lbl)
            for fid in family_ids
        }
        allocation = _greedy_allocate(family_ids, weights, fracs, rng)
        for key, ids in allocation.items():
            fold_ids[key] += ids

    def expand(ids: list[str]) -> list[str]:
        return [name for fid in ids for name, _ in families[fid]]

    folds = []
    for fold_idx, val_key in enumerate(fold_keys):
        val_ids   = fold_ids[val_key]
        train_ids = [fid for key in fold_keys if key != val_key for fid in fold_ids[key]]
        train = expand(train_ids)
        val   = expand(val_ids)
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest",   type=Path, default=default_manifest)
    parser.add_argument("--out-dir",    type=Path, default=default_out)
    parser.add_argument("--val-frac",   type=float, default=0.15)
    parser.add_argument("--test-frac",  type=float, default=0.15)
    parser.add_argument("--folds",      type=int,   default=5)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument(
        "--split-val",
        action="store_true",
        help=(
            "Split val.txt into val_gnn.txt (first 8) and val_ga.txt (rest). "
            "Redundant when run_ga_ensemble uses --oof (the default); add this "
            "flag only if you need a separate held-out GA validation set."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load graph manifest
    print(f"Loading {args.manifest}...")
    with args.manifest.open(newline="", encoding="utf-8") as f:
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

    def write_list(path: Path, names: list[str]) -> None:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")

    write_list(args.out_dir / "train.txt", train)
    write_list(args.out_dir / "val.txt",   val)
    write_list(args.out_dir / "test.txt",  test)

    if args.split_val:
        # Split val into GNN-validation and GA-validation halves.
        # Only useful when NOT using --oof in run_ga_ensemble (which is the default).
        _val_gnn_size = min(8, len(val))
        write_list(args.out_dir / "val_gnn.txt", val[:_val_gnn_size])
        write_list(args.out_dir / "val_ga.txt",  val[_val_gnn_size:])
        print(f"  --split-val: val_gnn={_val_gnn_size}, val_ga={len(val) - _val_gnn_size}")

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
            "val/test clusters are excluded from training entirely. Original clusters "
            "sharing a template family (e.g. badpods_*, datadog_*) are kept together "
            "in one partition and never split across CV folds. Test clusters are "
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
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\nSplits written to {args.out_dir}/")
    print("  train.txt, val.txt, test.txt")
    print(f"  fold_{{0..{args.folds-1}}}_train.txt, fold_{{0..{args.folds-1}}}_val.txt")
    print("  splits_config.json")


if __name__ == "__main__":
    main()
