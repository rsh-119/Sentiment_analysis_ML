#!/usr/bin/env python
"""Compare the results of the BiLSTM and fine-tuned Llama-3.1-8B-Instruct models on the IMDB sentiment dataset."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from imdb_sentiment.config import PathsConfig, load_config
from imdb_sentiment.utils.io import load_json
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def load_results(paths_cfg: PathsConfig, model_name: str) -> tuple[dict, pd.DataFrame]:
    metrics = load_json(paths_cfg.metrics_dir / f"{model_name}_test_metrics.json")
    predictions = pd.read_csv(paths_cfg.metrics_dir / f"{model_name}_test_predictions.csv")
    return metrics, predictions


def plot_confusion_matrix(cm: list, title: str, out_path) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["neg", "pos"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["neg", "pos"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_curves(bilstm_preds: pd.DataFrame, llama_preds: pd.DataFrame, out_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, preds in [("BiLSTM", bilstm_preds), ("Llama-3.1", llama_preds)]:
        fpr, tpr, _ = roc_curve(preds["true_label"], preds["pred_prob"])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_efficiency(bilstm_metrics: dict, llama_metrics: dict, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ["BiLSTM", "Llama-3.1"]
    for ax, field in zip(axes, ["latency_ms_per_example", "throughput_examples_per_sec"]):
        values = [bilstm_metrics["efficiency"][field], llama_metrics["efficiency"][field]]
        ax.bar(labels, values)
        ax.set_title(field)
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_disagreements(bilstm_preds: pd.DataFrame, llama_preds: pd.DataFrame) -> pd.DataFrame:
    merged = bilstm_preds.merge(llama_preds, on="idx", suffixes=("_bilstm", "_llama"))
    disagree = merged[merged["pred_label_bilstm"] != merged["pred_label_llama"]].copy()

    def categorize(row) -> str:
        bilstm_correct = row["pred_label_bilstm"] == row["true_label_bilstm"]
        llama_correct = row["pred_label_llama"] == row["true_label_bilstm"]
        if bilstm_correct and not llama_correct:
            return "bilstm_correct_llama_wrong"
        if llama_correct and not bilstm_correct:
            return "llama_correct_bilstm_wrong"
        return "both_wrong"

    disagree["category"] = disagree.apply(categorize, axis=1)
    return disagree


def render_report(paths_cfg: PathsConfig, bilstm_metrics: dict, llama_metrics: dict, disagree: pd.DataFrame):
    lines = [
        "# BiLSTM vs Fine-Tuned Llama-3.1-8B-Instruct — IMDB Sentiment Comparison",
        "",
        "## Metrics (test split)",
        "",
        "| Metric | BiLSTM | Llama-3.1 |",
        "|---|---|---|",
    ]
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        lines.append(f"| {key} | {bilstm_metrics.get(key):.4f} | {llama_metrics.get(key):.4f} |")

    lines += [
        "",
        "## Efficiency",
        "",
        "| Metric | BiLSTM | Llama-3.1 |",
        "|---|---|---|",
        f"| latency (ms/example) | {bilstm_metrics['efficiency']['latency_ms_per_example']:.3f} | "
        f"{llama_metrics['efficiency']['latency_ms_per_example']:.3f} |",
        f"| throughput (examples/sec) | {bilstm_metrics['efficiency']['throughput_examples_per_sec']:.2f} | "
        f"{llama_metrics['efficiency']['throughput_examples_per_sec']:.2f} |",
        f"| total params | {bilstm_metrics['param_count']['total']:,} | {llama_metrics['param_count']['total']:,} |",
        f"| trainable params | {bilstm_metrics['param_count']['trainable']:,} | "
        f"{llama_metrics['param_count']['trainable']:,} |",
        f"| disk size (MB) | {bilstm_metrics['disk_size_bytes'] / 1e6:.2f} | "
        f"{llama_metrics['disk_size_bytes'] / 1e6:.2f} (LoRA adapter only, base 4-bit weights not included) |",
        "",
        f"## Disagreements ({len(disagree)} examples)",
        "",
    ]
    for category in ["bilstm_correct_llama_wrong", "llama_correct_bilstm_wrong", "both_wrong"]:
        count = int((disagree["category"] == category).sum())
        lines.append(f"- {category}: {count}")

    lines += [
        "",
        "## Figures",
        "",
        "![Confusion matrix - BiLSTM](../outputs/figures/confusion_matrix_bilstm.png)",
        "![Confusion matrix - Llama](../outputs/figures/confusion_matrix_llama.png)",
        "![ROC curves](../outputs/figures/roc_curves.png)",
        "![Efficiency comparison](../outputs/figures/efficiency_comparison.png)",
        "",
        "## Analysis",
        "",
        "_Fill in: false positive/negative patterns, generalization, disagreement themes, "
        "classical-vs-LLM tradeoffs. See `outputs/metrics/disagreements.csv` for the full "
        "disagreement set with review text._",
        "",
    ]

    report_path = paths_cfg.reports_dir / "comparison.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    bilstm_metrics, bilstm_preds = load_results(cfg.paths, "bilstm")
    llama_metrics, llama_preds = load_results(cfg.paths, "llama")

    cfg.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        bilstm_metrics["confusion_matrix"], "BiLSTM confusion matrix",
        cfg.paths.figures_dir / "confusion_matrix_bilstm.png",
    )
    plot_confusion_matrix(
        llama_metrics["confusion_matrix"], "Llama-3.1 confusion matrix",
        cfg.paths.figures_dir / "confusion_matrix_llama.png",
    )
    plot_roc_curves(bilstm_preds, llama_preds, cfg.paths.figures_dir / "roc_curves.png")
    plot_efficiency(bilstm_metrics, llama_metrics, cfg.paths.figures_dir / "efficiency_comparison.png")

    disagree = build_disagreements(bilstm_preds, llama_preds)
    disagree.to_csv(cfg.paths.metrics_dir / "disagreements.csv", index=False)

    report_path = render_report(cfg.paths, bilstm_metrics, llama_metrics, disagree)
    logger.info(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
