"""Train/val/test split creation and loading.

`build_splits()` is called ONLY from `scripts/00_prepare_data.py`. Every other
script (training, evaluation) must use `load_splits()`, which reads the
already-persisted split from disk and validates it against the committed
`split_manifest.json`. This guarantees every consumer sees byte-identical
partitions and that the test set is never touched before evaluation time.
"""
from __future__ import annotations

import json

from datasets import Dataset, DatasetDict

from imdb_sentiment.config import AppConfig
from imdb_sentiment.data.load_raw import load_and_normalize, sha256_of_file


def _label_counts(ds: Dataset) -> dict[str, int]:
    counts = {"negative": 0, "positive": 0}
    for label in ds["label"]:
        counts["negative" if label == 0 else "positive"] += 1
    return counts


def build_splits(cfg: AppConfig) -> DatasetDict:
    """Load raw CSV, dedupe, stratified-split 70/15/15 (seeded), persist to disk
    and write the committed split manifest. Returns the resulting DatasetDict.
    """
    full_ds, n_duplicates_removed = load_and_normalize(cfg.paths.raw_csv)

    temp_size = cfg.split.val_ratio + cfg.split.test_ratio
    split1 = full_ds.train_test_split(
        test_size=temp_size, stratify_by_column="label", seed=cfg.split.seed
    )
    train_ds = split1["train"]
    temp_ds = split1["test"]

    rel_test_size = cfg.split.test_ratio / temp_size
    split2 = temp_ds.train_test_split(
        test_size=rel_test_size, stratify_by_column="label", seed=cfg.split.seed
    )
    val_ds = split2["train"]
    test_ds = split2["test"]

    dsd = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    cfg.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    full_ds.save_to_disk(str(cfg.paths.full_dataset_dir))
    dsd.save_to_disk(str(cfg.paths.splits_dir))

    manifest = {
        "seed": cfg.split.seed,
        "train_ratio": cfg.split.train_ratio,
        "val_ratio": cfg.split.val_ratio,
        "test_ratio": cfg.split.test_ratio,
        "raw_csv_sha256": sha256_of_file(cfg.paths.raw_csv),
        "num_duplicates_removed": n_duplicates_removed,
        "num_rows_total_deduped": len(full_ds),
        "splits": {
            name: {"size": len(ds), "label_counts": _label_counts(ds)}
            for name, ds in dsd.items()
        },
    }
    cfg.paths.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.split_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    return dsd


def load_splits(cfg: AppConfig) -> DatasetDict:
    """Load the persisted train/val/test split, validated against the manifest.

    Raises if the on-disk split is missing (run `scripts/00_prepare_data.py`
    first) or if its sizes disagree with the committed manifest (data drift).
    """
    if not cfg.paths.splits_dir.exists():
        raise FileNotFoundError(
            f"No processed splits found at {cfg.paths.splits_dir}. "
            "Run `scripts/00_prepare_data.py` first."
        )
    if not cfg.paths.split_manifest.exists():
        raise FileNotFoundError(
            f"No split manifest found at {cfg.paths.split_manifest}."
        )

    dsd = DatasetDict.load_from_disk(str(cfg.paths.splits_dir))

    with open(cfg.paths.split_manifest) as f:
        manifest = json.load(f)

    for name in ("train", "val", "test"):
        expected = manifest["splits"][name]["size"]
        actual = len(dsd[name])
        if actual != expected:
            raise ValueError(
                f"Split '{name}' size mismatch: manifest expects {expected}, "
                f"on-disk has {actual}. Re-run scripts/00_prepare_data.py."
            )

    if cfg.paths.raw_csv.exists():
        current_hash = sha256_of_file(cfg.paths.raw_csv)
        if current_hash != manifest["raw_csv_sha256"]:
            raise ValueError(
                "Raw CSV has changed since the split was built "
                f"(manifest sha256={manifest['raw_csv_sha256'][:12]}..., "
                f"current sha256={current_hash[:12]}...). "
                "Re-run scripts/00_prepare_data.py to rebuild the split."
            )

    return dsd
