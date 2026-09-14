"""PyTorch Dataset + collate_fn for the BiLSTM baseline, with dynamic per-batch padding."""
from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from imdb_sentiment.data.vocab import UNK_TOKEN, encode


class BiLSTMDataset(Dataset):
    def __init__(self, hf_dataset, vocab: dict[str, int], max_len: int):
        self.texts = hf_dataset["text"]
        self.labels = hf_dataset["label"]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        if not ids:
            ids = [self.vocab[UNK_TOKEN]]
        return torch.tensor(ids, dtype=torch.long), self.labels[idx]


def collate_fn(batch: list[tuple[torch.Tensor, int]]):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.float)
    return padded, lengths, labels
