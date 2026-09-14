"""Simple whitespace/regex tokenizer and vocabulary for the BiLSTM baseline.

Deliberately simpler than a HF subword tokenizer: it keeps the classical
baseline self-contained and gives a clean tokenization contrast against
Llama's BPE tokenizer for the final comparison report. The vocab is built
from the TRAIN split only (never val/test) to avoid leakage.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    text = _HTML_TAG_RE.sub(" ", text.lower())
    return _TOKEN_RE.findall(text)


def build_vocab(texts: list[str], min_freq: int = 2, max_size: int = 30_000) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # Deterministic ordering: frequency desc, ties broken alphabetically.
    items = sorted(
        (w for w, c in counter.items() if c >= min_freq),
        key=lambda w: (-counter[w], w),
    )
    items = items[: max_size - 2]  # reserve slots for <pad>/<unk>

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word in items:
        vocab[word] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(vocab, f)


def load_vocab(path: Path) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    unk_id = vocab[UNK_TOKEN]
    ids = [vocab.get(tok, unk_id) for tok in tokenize(text)]
    return ids[:max_len]
