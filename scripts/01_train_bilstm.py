#!/usr/bin/env python
"""Train the BiLSTM baseline."""
import argparse

from imdb_sentiment.config import apply_cli_overrides, load_config
from imdb_sentiment.train.train_bilstm import train_bilstm
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bilstm.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.override)

    result = train_bilstm(cfg)
    logger.info(f"Best val F1: {result['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
