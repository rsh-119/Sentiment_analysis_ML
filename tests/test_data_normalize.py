import csv
from pathlib import Path

from imdb_sentiment.data.load_raw import load_and_normalize


def make_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "sentiment"])
        writer.writerows(rows)


def test_normalize_maps_labels_and_columns(tmp_path: Path):
    csv_path = tmp_path / "imdb.csv"
    make_csv(csv_path, [("great movie", "positive"), ("terrible movie", "negative")])

    ds, n_removed = load_and_normalize(csv_path)

    assert n_removed == 0
    assert set(ds.column_names) == {"text", "label"}
    assert set(ds["label"]) == {0, 1}


def test_normalize_dedupes_exact_duplicates(tmp_path: Path):
    csv_path = tmp_path / "imdb.csv"
    make_csv(
        csv_path,
        [
            ("great movie", "positive"),
            ("great movie", "positive"),
            ("terrible movie", "negative"),
        ],
    )

    ds, n_removed = load_and_normalize(csv_path)

    assert n_removed == 1
    assert len(ds) == 2
