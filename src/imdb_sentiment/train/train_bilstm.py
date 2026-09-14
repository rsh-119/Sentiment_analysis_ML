"""BiLSTM training loop. Uses only the train/val splits — the test split is
never imported in this module.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import partial
from statistics import mean

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from imdb_sentiment.config import AppConfig
from imdb_sentiment.data.bilstm_dataset import BiLSTMDataset, collate_fn
from imdb_sentiment.data.splits import load_splits
from imdb_sentiment.data.vocab import build_vocab, save_vocab
from imdb_sentiment.models.bilstm import BiLSTMClassifier
from imdb_sentiment.seeding import make_generator, set_seed, worker_init_fn
from imdb_sentiment.utils.io import save_json
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def train_bilstm(cfg: AppConfig) -> dict:
    set_seed(cfg.seed.seed, cfg.seed.deterministic)

    dsd = load_splits(cfg)
    train_ds, val_ds = dsd["train"], dsd["val"]

    vocab = build_vocab(train_ds["text"], cfg.bilstm.min_freq, cfg.bilstm.vocab_size)
    save_vocab(vocab, cfg.paths.vocab_path)
    logger.info(f"Built vocab of size {len(vocab)} from {len(train_ds)} train examples")

    train_dataset = BiLSTMDataset(train_ds, vocab, cfg.bilstm.max_seq_len)
    val_dataset = BiLSTMDataset(val_ds, vocab, cfg.bilstm.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.bilstm.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=make_generator(cfg.seed.seed),
        worker_init_fn=partial(worker_init_fn, base_seed=cfg.seed.seed),
        num_workers=cfg.bilstm.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.bilstm.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.bilstm.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=cfg.bilstm.embedding_dim,
        hidden_size=cfg.bilstm.hidden_size,
        num_layers=cfg.bilstm.num_layers,
        dropout=cfg.bilstm.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.bilstm.lr, weight_decay=cfg.bilstm.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = cfg.paths.checkpoints_dir / "bilstm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(cfg.bilstm.epochs):
        model.train()
        train_losses = []
        for input_ids, lengths, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, all_preds, all_labels = [], [], []
        with torch.no_grad():
            for input_ids, lengths, labels in val_loader:
                input_ids_dev, labels_dev = input_ids.to(device), labels.to(device)
                logits = model(input_ids_dev, lengths)
                loss = criterion(logits, labels_dev)
                val_losses.append(loss.item())
                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.long().tolist())

        val_f1 = f1_score(all_labels, all_preds)
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": mean(train_losses),
            "val_loss": mean(val_losses),
            "val_f1": val_f1,
        }
        history.append(epoch_log)
        logger.info(
            f"Epoch {epoch + 1}/{cfg.bilstm.epochs} "
            f"train_loss={epoch_log['train_loss']:.4f} "
            f"val_loss={epoch_log['val_loss']:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(vocab),
                    "model_config": asdict(cfg.bilstm),
                },
                checkpoint_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.bilstm.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (best val_f1={best_val_f1:.4f})")
                break

    save_json(history, cfg.paths.metrics_dir / "bilstm_train_log.json")
    return {"best_val_f1": best_val_f1, "history": history}
