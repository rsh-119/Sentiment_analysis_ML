from imdb_sentiment.eval.metrics import compute_classification_metrics


def test_metrics_match_known_confusion_matrix():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_prob = [0.1, 0.6, 0.9, 0.8]

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert 0.0 <= metrics["roc_auc"] <= 1.0
