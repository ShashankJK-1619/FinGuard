from __future__ import annotations

import numpy as np
import pytest

from src.models.cost_model import (
    CostAssumptionError,
    CostAssumptions,
    ThresholdError,
    classify_actions,
    evaluate_decision_policy,
    evaluate_policy_grid,
    load_cost_assumptions,
    sensitivity_analysis,
    threshold_pair_grid,
)

COSTS = CostAssumptions(
    cost_per_missed_fraud=500.0,
    cost_per_false_positive_decline=25.0,
    cost_per_manual_review=3.0,
)


def test_classify_actions_three_buckets():
    y_prob = np.array([0.9, 0.2, 0.95, 0.4, 0.05])
    actions = classify_actions(y_prob, review_threshold=0.3, decline_threshold=0.8)
    assert list(actions) == ["decline", "approve", "decline", "review", "approve"]


def test_classify_actions_rejects_inverted_thresholds():
    y_prob = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ThresholdError):
        classify_actions(y_prob, review_threshold=0.8, decline_threshold=0.3)


def test_classify_actions_rejects_equal_thresholds():
    y_prob = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ThresholdError):
        classify_actions(y_prob, review_threshold=0.5, decline_threshold=0.5)


def test_classify_actions_rejects_out_of_range_thresholds():
    y_prob = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ThresholdError):
        classify_actions(y_prob, review_threshold=-0.1, decline_threshold=0.5)
    with pytest.raises(ThresholdError):
        classify_actions(y_prob, review_threshold=0.1, decline_threshold=1.1)


def test_evaluate_decision_policy_matches_hand_computation():
    """Hand-computed on a 5-row example:
    y_true=[1,1,0,0,0], y_prob=[0.9,0.2,0.95,0.4,0.05],
    review_threshold=0.3, decline_threshold=0.8
    -> actions = [decline, approve, decline, review, approve]
    fraud caught = row0 only (row1 is fraud but approved) -> recall 1/2
    flagged = rows 0,2,3 (2 fraud? only row0 is fraud among flagged) -> precision 1/3
    false_positives (legit declined) = row2 -> 1
    false_negatives (fraud approved) = row1 -> 1
    review_volume = 1 (row3), decline_volume = 2 (rows 0,2)
    cost = 1*500 (missed fraud) + 1*25 (fp decline) + 1*3 (review) = 528
    """
    y_true = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.95, 0.4, 0.05])

    result = evaluate_decision_policy(y_true, y_prob, review_threshold=0.3, decline_threshold=0.8, costs=COSTS)

    assert result["fraud_recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["review_volume"] == 1
    assert result["decline_volume"] == 2
    assert result["pct_reviewed"] == pytest.approx(0.2)
    assert result["pct_declined"] == pytest.approx(0.4)
    assert result["pct_affected"] == pytest.approx(0.6)
    assert result["estimated_illustrative_total_cost"] == pytest.approx(528.0)


def test_evaluate_decision_policy_zero_cost_when_everything_approved_and_no_fraud():
    y_true = np.zeros(10, dtype=int)
    y_prob = np.full(10, 0.01)
    result = evaluate_decision_policy(y_true, y_prob, review_threshold=0.5, decline_threshold=0.9, costs=COSTS)
    assert result["estimated_illustrative_total_cost"] == pytest.approx(0.0)
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_threshold_pair_grid_only_contains_ordered_pairs():
    pairs = threshold_pair_grid([0.1, 0.5, 0.9], [0.2, 0.5, 0.8])
    assert all(r < d for r, d in pairs)
    assert (0.9, 0.8) not in pairs
    assert (0.5, 0.5) not in pairs
    assert (0.1, 0.2) in pairs


def test_evaluate_policy_grid_is_sorted_ascending_by_cost():
    rng = np.random.default_rng(0)
    y_true = rng.choice([0, 1], size=200, p=[0.95, 0.05])
    y_prob = rng.uniform(0, 1, size=200)
    grid = evaluate_policy_grid(y_true, y_prob, [0.1, 0.3], [0.5, 0.7, 0.9], COSTS)
    costs_col = grid["estimated_illustrative_total_cost"].to_numpy()
    assert (np.diff(costs_col) >= 0).all()


def test_sensitivity_analysis_fraud_loss_increase_favors_more_aggressive_policy():
    """Raising the cost of a missed fraud enough should never make the
    chosen policy's fraud_recall go down relative to the base policy
    chosen for the unmodified cost — more expensive missed fraud should
    push the argmin-cost policy toward catching more fraud, not less."""
    rng = np.random.default_rng(1)
    y_true = rng.choice([0, 1], size=1000, p=[0.95, 0.05])
    y_prob = np.clip(y_true * 0.5 + rng.uniform(0, 0.5, size=1000), 0, 1)

    review_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    decline_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    variations = [{"label": "fraud_loss_100x", "cost_per_missed_fraud": COSTS.cost_per_missed_fraud * 100}]
    result = sensitivity_analysis(y_true, y_prob, review_thresholds, decline_thresholds, COSTS, variations)

    base_grid = evaluate_policy_grid(y_true, y_prob, review_thresholds, decline_thresholds, COSTS)
    base_recall = base_grid.iloc[0]["fraud_recall"]
    varied_recall = result.iloc[0]["fraud_recall"]
    assert varied_recall >= base_recall


def test_load_cost_assumptions_reads_configs_file():
    costs = load_cost_assumptions()
    assert costs.cost_per_missed_fraud > 0
    assert costs.cost_per_false_positive_decline > 0
    assert costs.cost_per_manual_review > 0


def test_default_review_params_reproduce_original_unlimited_perfect_review_behavior():
    """Methodology-correction regression check: the three new review
    parameters all default to values that exactly reproduce the original
    (unlimited capacity, perfect catch, no review error) numbers, so any
    caller built against the pre-correction API is unaffected."""
    y_true = np.array([1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.2, 0.95, 0.4, 0.05])
    result = evaluate_decision_policy(y_true, y_prob, review_threshold=0.3, decline_threshold=0.8, costs=COSTS)
    assert result["fraud_recall"] == pytest.approx(0.5)
    assert result["estimated_illustrative_total_cost"] == pytest.approx(528.0)
    assert result["review_capacity_limit"] is None
    assert result["review_overflow_to_approve"] == 0


def test_cost_assumptions_rejects_out_of_range_review_params():
    with pytest.raises(CostAssumptionError):
        CostAssumptions(500.0, 25.0, 3.0, review_capacity_frac=1.5)
    with pytest.raises(CostAssumptionError):
        CostAssumptions(500.0, 25.0, 3.0, review_fraud_catch_rate=-0.1)
    with pytest.raises(CostAssumptionError):
        CostAssumptions(500.0, 25.0, 3.0, review_false_positive_rate=1.1)


def test_review_capacity_overflow_falls_back_to_approve_not_decline():
    """10 rows, review band would claim 4 of them (prob in [0.3, 0.8)), but
    capacity is set to allow only 2. The 2 highest-scoring should be
    reviewed; the other 2 must fall back to approve, never decline."""
    y_true = np.zeros(10, dtype=int)
    y_prob = np.array([0.35, 0.40, 0.45, 0.50, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    costs = CostAssumptions(
        cost_per_missed_fraud=500.0,
        cost_per_false_positive_decline=25.0,
        cost_per_manual_review=3.0,
        review_capacity_frac=0.2,  # floor(0.2 * 10) = 2
    )
    result = evaluate_decision_policy(y_true, y_prob, review_threshold=0.3, decline_threshold=0.8, costs=costs)
    assert result["review_capacity_limit"] == 2
    assert result["review_volume"] == 2
    assert result["review_overflow_to_approve"] == 2
    assert result["decline_volume"] == 0


def test_review_capacity_prioritizes_highest_risk_transactions():
    """Among four review-band candidates, capacity=2 should keep the two
    with the HIGHEST y_prob (0.50 and 0.45), confirmed by checking which
    fraud case gets caught: the highest-scoring row is fraud, the
    lowest-scoring review-band row is also fraud — only the former should
    be reviewed (and caught) under capacity=2."""
    y_true = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.35, 0.40, 0.45, 0.50, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    costs = CostAssumptions(
        cost_per_missed_fraud=500.0,
        cost_per_false_positive_decline=25.0,
        cost_per_manual_review=3.0,
        review_capacity_frac=0.2,  # floor(0.2 * 10) = 2
    )
    result = evaluate_decision_policy(y_true, y_prob, review_threshold=0.3, decline_threshold=0.8, costs=costs)
    # Both fraud rows (index 2: prob 0.45, index 3: prob 0.50) are among the
    # two highest-scoring review candidates, so both should be reviewed and
    # (with default perfect catch rate) both caught.
    assert result["fraud_recall"] == pytest.approx(1.0)


def test_imperfect_review_catch_rate_reduces_recall_in_expectation():
    """4 fraud cases all routed to review (no capacity limit); with a 50%
    review catch rate, expected fraud caught should be exactly half, not
    all of them — the imperfect-review correction's core behavior."""
    y_true = np.array([1, 1, 1, 1, 0, 0])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    costs = CostAssumptions(
        cost_per_missed_fraud=500.0,
        cost_per_false_positive_decline=25.0,
        cost_per_manual_review=3.0,
        review_fraud_catch_rate=0.5,
    )
    result = evaluate_decision_policy(y_true, y_prob, review_threshold=0.3, decline_threshold=0.8, costs=costs)
    assert result["fraud_recall"] == pytest.approx(0.5)
    assert result["reviewed_fraud_expected_caught"] == pytest.approx(2.0)
    assert result["reviewed_fraud_expected_missed"] == pytest.approx(2.0)
    assert result["false_negatives"] == pytest.approx(2.0)


def test_review_false_positive_rate_adds_expected_false_positives():
    """6 legitimate transactions all routed to review; with a 25% review
    error rate, expected false positives from review error should be
    exactly 1.5, added on top of any outright-decline false positives
    (zero here, since nothing is declined)."""
    y_true = np.zeros(6, dtype=int)
    y_prob = np.full(6, 0.5)
    costs = CostAssumptions(
        cost_per_missed_fraud=500.0,
        cost_per_false_positive_decline=25.0,
        cost_per_manual_review=3.0,
        review_false_positive_rate=0.25,
    )
    result = evaluate_decision_policy(y_true, y_prob, review_threshold=0.3, decline_threshold=0.8, costs=costs)
    assert result["review_error_false_positives"] == pytest.approx(1.5)
    assert result["false_positives"] == pytest.approx(1.5)
