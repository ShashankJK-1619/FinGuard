"""Tests for Module C, Milestone 4, Section E: src/credit_risk/decisioning.py
-- illustrative three-way (approve_like/manual_review/decline_like)
decisioning with ordered-threshold validation."""
import numpy as np
import pytest

from src.credit_risk.decisioning import (
    DECISION_LABELS,
    DECISIONING_DISCLAIMER,
    DecisionThresholdError,
    classify_decisions,
)


def test_classify_decisions_assigns_three_bands_correctly():
    y_prob = np.array([0.05, 0.2, 0.3, 0.5, 0.7, 0.95])
    decisions = classify_decisions(y_prob, review_threshold=0.3, decline_threshold=0.7)
    assert list(decisions) == [
        "approve_like",
        "approve_like",
        "manual_review",
        "manual_review",
        "decline_like",
        "decline_like",
    ]


def test_all_returned_labels_are_within_decision_labels():
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(size=500)
    decisions = classify_decisions(y_prob, review_threshold=0.4, decline_threshold=0.8)
    assert set(decisions.tolist()) <= set(DECISION_LABELS)


def test_rejects_inverted_thresholds():
    with pytest.raises(DecisionThresholdError):
        classify_decisions(np.array([0.1, 0.9]), review_threshold=0.8, decline_threshold=0.2)


def test_rejects_equal_thresholds_collapsed_review_band():
    with pytest.raises(DecisionThresholdError):
        classify_decisions(np.array([0.1, 0.9]), review_threshold=0.5, decline_threshold=0.5)


def test_rejects_thresholds_outside_unit_interval():
    with pytest.raises(DecisionThresholdError):
        classify_decisions(np.array([0.1, 0.9]), review_threshold=-0.1, decline_threshold=0.5)
    with pytest.raises(DecisionThresholdError):
        classify_decisions(np.array([0.1, 0.9]), review_threshold=0.5, decline_threshold=1.1)


def test_boundary_values_are_inclusive_lower_exclusive_upper():
    # score == review_threshold -> manual_review (lower bound inclusive);
    # score == decline_threshold -> decline_like (upper bound of review is
    # exclusive, i.e. decline_threshold itself belongs to decline_like).
    y_prob = np.array([0.3, 0.7])
    decisions = classify_decisions(y_prob, review_threshold=0.3, decline_threshold=0.7)
    assert list(decisions) == ["manual_review", "decline_like"]


def test_disclaimer_is_a_non_empty_string_and_uses_neutral_labels():
    assert isinstance(DECISIONING_DISCLAIMER, str) and DECISIONING_DISCLAIMER
    assert DECISION_LABELS == ("approve_like", "manual_review", "decline_like")
