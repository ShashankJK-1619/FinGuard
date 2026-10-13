"""Tests for Module C, Milestone 3, Section H:
src/credit_risk/ablation.py -- Model A -> Model B lift, framed strictly as
signal quantification, never a deployment recommendation."""
import pytest

from src.credit_risk.ablation import FRAMING_NOTE, compute_ablation
from src.credit_risk.evaluation import compute_classification_metrics
import numpy as np


def _metrics(split_name, model_name, roc_auc_bias):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = np.clip(y_true * 0.3 + roc_auc_bias + rng.normal(scale=0.1, size=200), 0, 1)
    return compute_classification_metrics(y_true, y_prob, 0.5, "t", split_name, model_name)


def test_compute_ablation_lift_is_model_b_minus_model_a():
    model_a_metrics = _metrics("validation", "model_a", roc_auc_bias=0.3)
    model_b_metrics = _metrics("validation", "model_b", roc_auc_bias=0.4)
    result = compute_ablation("validation", model_a_metrics, model_b_metrics)
    assert result.roc_auc_lift == pytest.approx(model_b_metrics.roc_auc - model_a_metrics.roc_auc)
    assert result.pr_auc_lift == pytest.approx(model_b_metrics.pr_auc - model_a_metrics.pr_auc)


def test_compute_ablation_brier_improvement_sign_convention():
    """brier_improvement positive means Model B's Brier is LOWER (better)."""
    model_a_metrics = _metrics("validation", "model_a", roc_auc_bias=0.3)
    model_b_metrics = _metrics("validation", "model_b", roc_auc_bias=0.3)
    result = compute_ablation("validation", model_a_metrics, model_b_metrics)
    assert result.brier_improvement == pytest.approx(model_a_metrics.brier_score - model_b_metrics.brier_score)


def test_compute_ablation_rejects_mismatched_splits():
    model_a_metrics = _metrics("validation", "model_a", roc_auc_bias=0.3)
    model_b_metrics = _metrics("benchmark", "model_b", roc_auc_bias=0.3)
    with pytest.raises(ValueError):
        compute_ablation("validation", model_a_metrics, model_b_metrics)


def test_framing_note_present_on_every_result_and_as_dict():
    model_a_metrics = _metrics("validation", "model_a", roc_auc_bias=0.3)
    model_b_metrics = _metrics("validation", "model_b", roc_auc_bias=0.4)
    result = compute_ablation("validation", model_a_metrics, model_b_metrics)
    assert result.framing_note == FRAMING_NOTE
    d = result.as_dict()
    assert d["framing_note"] == FRAMING_NOTE
    # The framing note must never be silently droppable from the serialized form.
    assert "framing_note" in d


def test_framing_note_never_recommends_deployment():
    assert "never a recommendation" in FRAMING_NOTE or "not a recommendation" in FRAMING_NOTE
