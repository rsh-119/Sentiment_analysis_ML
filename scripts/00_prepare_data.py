#!/usr/bin/env python
"""Build the train/val/test split (the ONLY place this happens) and write the
committed split manifest. Run this once before any training/evaluation script.
"""
import argparse

from imdb_sentiment.config import apply_cli_overrides, load_config
from imdb_sentiment.data.splits import build_splits
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.override)

    dsd = build_splits(cfg)

    logger.info(f"Wrote splits to {cfg.paths.splits_dir}")
    logger.info(f"Wrote manifest to {cfg.paths.split_manifest}")
    for name in ("train", "val", "test"):
        logger.info(f"  {name}: {len(dsd[name])} examples")


if __name__ == "__main__":
    main()
