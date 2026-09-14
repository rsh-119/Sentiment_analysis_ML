"""Full evaluation function for running tests and measuring efficiency."""
from __future__ import annotations

from typing import Callable

import pandas as pd

from imdb_sentiment.config import EvalConfig, PathsConfig
from imdb_sentiment.eval.efficiency import measure_inference_efficiency
from imdb_sentiment.eval.metrics import compute_classification_metrics
from imdb_sentiment.utils.io import save_json


def run_full_evaluation(
    model_name: str,
    predict_fn: Callable[[list[str]], tuple[list[int], list[float]]],
    texts: list[str],
    labels: list[int],
    param_counts: dict,
    disk_size_bytes: int,
    eval_cfg: EvalConfig,
    paths_cfg: PathsConfig,
) -> dict:
    """`predict_fn(batch_texts) -> (pred_labels, pred_probs)`. Runs over the
    full `texts`/`labels` once (this is the ONE test-set pass), then times
    `predict_fn` separately for the efficiency numbers.
    """
    bs = eval_cfg.batch_size
    pred_labels: list[int] = []
    pred_probs: list[float] = []
    for i in range(0, len(texts), bs):
        batch_preds, batch_probs = predict_fn(texts[i : i + bs])
        pred_labels.extend(batch_preds)
        pred_probs.extend(batch_probs)

    metrics = compute_classification_metrics(labels, pred_labels, pred_probs)
    efficiency = measure_inference_efficiency(
        predict_fn, texts, bs, eval_cfg.latency_num_warmup, eval_cfg.latency_num_runs
    )

    result = {
        "model_name": model_name,
        **metrics,
        "efficiency": efficiency,
        "param_count": param_counts,
        "disk_size_bytes": disk_size_bytes,
    }
    save_json(result, paths_cfg.metrics_dir / f"{model_name}_test_metrics.json")

    df = pd.DataFrame(
        {
            "idx": range(len(texts)),
            "text": texts,
            "true_label": labels,
            "pred_label": pred_labels,
            "pred_prob": pred_probs,
        }
    )
    predictions_path = paths_cfg.metrics_dir / f"{model_name}_test_predictions.csv"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(predictions_path, index=False)

    return result
