"""Tests for Module C, Milestone 3, Section E:
src/credit_risk/evaluation.py -- classification metrics at a named,
non-0.5 threshold."""
import numpy as np
import pytest

from src.credit_risk.evaluation import EvaluationError, compute_classification_metrics


def test_confusion_matrix_at_named_non_0p5_threshold_is_correct():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.20, 0.35, 0.40, 0.60, 0.90])
    # threshold 0.3 (not 0.5): predictions = [0,0,1,1,1,1]
    metrics = compute_classification_metrics(
        y_true, y_prob, threshold_value=0.3, threshold_name="custom_threshold",
        split_name="toy_split", model_name="toy_model",
    )
    assert metrics.true_negatives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.true_positives == 3
    assert metrics.threshold_name == "custom_threshold"
    assert metrics.threshold_value == 0.3


def test_precision_recall_f1_hand_computed():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.6, 0.9])
    # threshold 0.5 -> y_pred = [0, 1, 1, 1]; TP=2, FP=1, FN=0, TN=1
    metrics = compute_classification_metrics(
        y_true, y_prob, threshold_value=0.5, threshold_name="t",
        split_name="s", model_name="m",
    )
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(2 * (2 / 3) / (2 / 3 + 1))


def test_prevalence_pct_matches_observed_mean():
    y_true = np.array([0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8])
    metrics = compute_classification_metrics(
        y_true, y_prob, threshold_value=0.5, threshold_name="t", split_name="s", model_name="m"
    )
    assert metrics.prevalence_pct == pytest.approx(25.0)


def test_single_class_raises_rather_than_returning_undefined_auc():
    y_true = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(EvaluationError):
        compute_classification_metrics(
            y_true, y_prob, threshold_value=0.5, threshold_name="t", split_name="s", model_name="m"
        )


def test_mismatched_lengths_raises():
    with pytest.raises(EvaluationError):
        compute_classification_metrics(
            np.array([0, 1]), np.array([0.1, 0.2, 0.3]),
            threshold_value=0.5, threshold_name="t", split_name="s", model_name="m",
        )


def test_as_dict_rounds_and_nests_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.6, 0.6, 0.9])
    metrics = compute_classification_metrics(
        y_true, y_prob, threshold_value=0.5, threshold_name="t", split_name="s", model_name="m"
    )
    d = metrics.as_dict()
    assert set(d["confusion_matrix"].keys()) == {
        "true_negatives", "false_positives", "false_negatives", "true_positives",
    }
    assert d["threshold_value"] == 0.5
