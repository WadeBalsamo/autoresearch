import numpy as np

from common import metrics


def test_classification_metrics_perfect():
    y = [0, 1, 2, 3, 4, 2, 1]
    m = metrics.classification_metrics(y, y, num_classes=5)
    assert abs(m["macro_f1"] - 1.0) < 1e-9
    assert m["min_per_class_f1"] == 1.0


def test_classification_metrics_partial():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    m = metrics.classification_metrics(y_true, y_pred, num_classes=5)
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert "f1_class3" in m  # all 5 classes reported even if absent


def test_pair_metrics_separable():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    m = metrics.pair_metrics(y, scores)
    assert m["roc_auc"] == 1.0
    assert m["pair_acc"] == 1.0


def test_ranking_metrics_topranked_relevant():
    # two queries; in each, the progressing cue gets the highest score
    groups = ["q1", "q1", "q1", "q2", "q2"]
    y = [1, 0, 0, 0, 1]
    scores = [0.9, 0.5, 0.1, 0.2, 0.8]
    r = metrics.ranking_metrics(groups, y, scores, ks=(1, 2))
    assert r["n_queries"] == 2
    assert r["mrr"] == 1.0
    assert r["recall@1"] == 1.0


def test_ece_bounds():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]])
    y = [0, 0, 1]
    ece = metrics.expected_calibration_error(probs, y, n_bins=5)
    assert 0.0 <= ece <= 1.0
