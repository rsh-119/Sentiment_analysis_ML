#!/usr/bin/env python
"""Evaluate the fine-tuned Llama model once on the held-out test split."""
import os

# Set the environment variable to allow PyTorch to use expandable segments for CUDA memory allocation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

from imdb_sentiment.config import apply_cli_overrides, load_config
from imdb_sentiment.data.splits import load_splits
from imdb_sentiment.eval.efficiency import count_parameters, model_disk_size_bytes
from imdb_sentiment.eval.evaluate import run_full_evaluation
from imdb_sentiment.eval.llama_predictor import LlamaPredictor
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llama.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args.override)

    checkpoint_dir = cfg.paths.checkpoints_dir / "llama_lora"
    predictor = LlamaPredictor(str(checkpoint_dir), cfg.llama)
    param_counts = count_parameters(predictor.model)

    test_ds = load_splits(cfg)["test"]

    result = run_full_evaluation(
        model_name="llama",
        predict_fn=predictor.predict,
        texts=test_ds["text"],
        labels=test_ds["label"],
        param_counts=param_counts,
        disk_size_bytes=model_disk_size_bytes(checkpoint_dir),
        eval_cfg=cfg.eval,
        paths_cfg=cfg.paths,
    )
    logger.info(f"Llama test accuracy={result['accuracy']:.4f} f1={result['f1']:.4f}")


if __name__ == "__main__":
    main()
