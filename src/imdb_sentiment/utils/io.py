"""Small JSON I/O helpers shared across scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def dir_size_bytes(path: Path) -> int:
    """Total size in bytes of all files under `path` (or a single file's size)."""
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
