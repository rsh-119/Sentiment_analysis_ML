"""Load the raw IMDB CSV via the Hugging Face `datasets` library and normalize it
into the project's common `text`/`label` schema (label: 0=negative, 1=positive),
deduplicating exact-text-duplicate rows before any split is created.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from datasets import Dataset, load_dataset, ClassLabel

LABEL_MAP = {"negative": 0, "positive": 1}


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_normalize(raw_csv: Path) -> tuple[Dataset, int]:
    """Return (deduped_normalized_dataset, num_duplicates_removed)."""
    # "all" is a reserved split keyword in `datasets`, so the loading key must be something else.
    ds = load_dataset("csv", data_files={"train": str(raw_csv)})["train"]

    ds = ds.rename_column("review", "text")
    ds = ds.map(lambda ex: {"label": LABEL_MAP[ex["sentiment"]]}, remove_columns=["sentiment"])
    ds = ds.cast_column("label", ClassLabel(names=["negative", "positive"]))

    n_before = len(ds)
    seen: set[str] = set()
    keep_indices = []
    for i, text in enumerate(ds["text"]):
        if text not in seen:
            seen.add(text)
            keep_indices.append(i)
    ds = ds.select(keep_indices)
    n_removed = n_before - len(ds)

    return ds, n_removed
