import csv
from pathlib import Path

from imdb_sentiment.config import AppConfig
from imdb_sentiment.data.splits import build_splits, load_splits


def make_config(tmp_path: Path) -> AppConfig:
    csv_path = tmp_path / "dataset" / "IMDB Dataset.csv"
    csv_path.parent.mkdir(parents=True)
    rows = []
    for i in range(20):
        rows.append((f"positive review number {i}", "positive"))
        rows.append((f"negative review number {i}", "negative"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "sentiment"])
        writer.writerows(rows)

    cfg = AppConfig()
    cfg.paths.raw_csv = csv_path
    cfg.paths.processed_dir = tmp_path / "data" / "processed"
    cfg.paths.full_dataset_dir = cfg.paths.processed_dir / "imdb_full"
    cfg.paths.splits_dir = cfg.paths.processed_dir / "imdb_splits"
    cfg.paths.split_manifest = tmp_path / "data" / "splits" / "split_manifest.json"
    return cfg


def test_splits_have_no_text_overlap(tmp_path: Path):
    cfg = make_config(tmp_path)
    dsd = build_splits(cfg)

    train_texts = set(dsd["train"]["text"])
    val_texts = set(dsd["val"]["text"])
    test_texts = set(dsd["test"]["text"])

    assert not (train_texts & val_texts)
    assert not (train_texts & test_texts)
    assert not (val_texts & test_texts)


def test_splits_partition_full_deduped_set(tmp_path: Path):
    cfg = make_config(tmp_path)
    dsd = build_splits(cfg)

    total = len(dsd["train"]) + len(dsd["val"]) + len(dsd["test"])
    assert total == 40  # no duplicates in this synthetic dataset


def test_build_splits_is_reproducible_given_seed(tmp_path: Path):
    cfg = make_config(tmp_path)
    train1 = list(build_splits(cfg)["train"]["text"])
    train2 = list(build_splits(cfg)["train"]["text"])

    assert train1 == train2


def test_load_splits_matches_build_splits(tmp_path: Path):
    cfg = make_config(tmp_path)
    build_splits(cfg)

    loaded = load_splits(cfg)
    assert len(loaded["train"]) + len(loaded["val"]) + len(loaded["test"]) == 40
