#!/usr/bin/env python
"""Fine-tune Llama-3.1-8B-Instruct with QLoRA."""
import argparse

from imdb_sentiment.config import apply_cli_overrides, load_config
from imdb_sentiment.train.train_llama import train_llama
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llama.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.override)

    result = train_llama(cfg)
    logger.info(f"Saved LoRA adapter to {result['checkpoint_dir']}")


if __name__ == "__main__":
    main()
