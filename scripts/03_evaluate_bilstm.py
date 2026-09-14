#!/usr/bin/env python
"""Evaluate the trained BiLSTM once on the held-out test split."""
import argparse

import torch
from torch.nn.utils.rnn import pad_sequence

from imdb_sentiment.config import apply_cli_overrides, load_config
from imdb_sentiment.data.splits import load_splits
from imdb_sentiment.data.vocab import UNK_TOKEN, encode, load_vocab
from imdb_sentiment.eval.efficiency import count_parameters, model_disk_size_bytes
from imdb_sentiment.eval.evaluate import run_full_evaluation
from imdb_sentiment.models.bilstm import BiLSTMClassifier
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def build_predict_fn(model, vocab, max_len, device):
    def predict_fn(texts: list[str]) -> tuple[list[int], list[float]]:
        seqs = []
        for text in texts:
            ids = encode(text, vocab, max_len)
            if not ids:
                ids = [vocab[UNK_TOKEN]]
            seqs.append(torch.tensor(ids, dtype=torch.long))
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        padded = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(padded, lengths))
        return (probs >= 0.5).long().cpu().tolist(), probs.cpu().tolist()

    return predict_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(cfg.paths.vocab_path)

    checkpoint_path = cfg.paths.checkpoints_dir / "bilstm" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    model = BiLSTMClassifier(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=model_cfg["embedding_dim"],
        hidden_size=model_cfg["hidden_size"],
        num_layers=model_cfg["num_layers"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = load_splits(cfg)["test"]
    predict_fn = build_predict_fn(model, vocab, cfg.bilstm.max_seq_len, device)

    result = run_full_evaluation(
        model_name="bilstm",
        predict_fn=predict_fn,
        texts=test_ds["text"],
        labels=test_ds["label"],
        param_counts=count_parameters(model),
        disk_size_bytes=model_disk_size_bytes(checkpoint_path),
        eval_cfg=cfg.eval,
        paths_cfg=cfg.paths,
    )
    logger.info(f"BiLSTM test accuracy={result['accuracy']:.4f} f1={result['f1']:.4f}")


if __name__ == "__main__":
    main()
