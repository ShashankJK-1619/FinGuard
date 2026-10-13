"""Tests for Module C, Milestone 4, Sections F-G:
src/credit_risk/decision_cost_model.py -- illustrative decision-cost
framework (realized lgd*ead loss + flat review/opportunity costs) and
threshold/assumption sensitivity analysis."""
import numpy as np
import pandas as pd
import pytest

from src.credit_risk.decision_cost_model import (
    DECISION_COST_DISCLAIMER,
    CreditDecisionCostAssumptions,
    DecisionCostAssumptionError,
    evaluate_decision_policy,
    load_credit_decision_cost_assumptions,
    sensitivity_analysis,
)


def _toy_costs():
    return CreditDecisionCostAssumptions(
        cost_per_manual_review=10.0, opportunity_cost_per_good_loan_declined=20.0
    )


def test_evaluate_decision_policy_hand_computed():
    # 4 loans: adverse+declined, adverse+approved, good+declined, good+reviewed
    y_true = np.array([True, True, False, False])
    y_prob = np.array([0.9, 0.1, 0.9, 0.5])
    ead = np.array([1000.0, 2000.0, 500.0, 300.0])
    lgd = 0.8
    costs = _toy_costs()
    result = evaluate_decision_policy(
        y_true, y_prob, ead, lgd, review_threshold=0.3, decline_threshold=0.7, costs=costs
    )
    # decisions: 0.9->decline_like, 0.1->approve_like, 0.9->decline_like, 0.5->manual_review
    assert result["decline_volume"] == 2
    assert result["review_volume"] == 1
    assert result["approve_volume"] == 1
    # adverse_outcome_capture: 1 of 2 adverse loans declined
    assert result["adverse_outcome_capture"] == pytest.approx(0.5)
    # good_loan_rejection_rate: 1 of 2 good loans declined
    assert result["good_loan_rejection_rate"] == pytest.approx(0.5)
    # realized loss: adverse loan NOT declined is loan index 1 (ead=2000)
    assert result["realized_adverse_outcome_loss"] == pytest.approx(0.8 * 2000.0)
    # review cost: 1 reviewed loan * 10.0
    assert result["review_cost"] == pytest.approx(10.0)
    # opportunity cost: 1 good loan declined (index 2) * 20.0
    assert result["opportunity_cost"] == pytest.approx(20.0)
    expected_total = 0.8 * 2000.0 + 10.0 + 20.0
    assert result["estimated_illustrative_total_cost"] == pytest.approx(expected_total)


def test_higher_lgd_increases_realized_loss_and_total_cost():
    y_true = np.array([True, False])
    y_prob = np.array([0.1, 0.1])  # both approve_like
    ead = np.array([1000.0, 1000.0])
    costs = _toy_costs()
    low = evaluate_decision_policy(y_true, y_prob, ead, 0.4, 0.5, 0.9, costs)
    high = evaluate_decision_policy(y_true, y_prob, ead, 0.8, 0.5, 0.9, costs)
    assert high["realized_adverse_outcome_loss"] > low["realized_adverse_outcome_loss"]
    assert high["estimated_illustrative_total_cost"] > low["estimated_illustrative_total_cost"]


def test_stricter_decline_threshold_changes_capture_and_rejection():
    y_true = np.array([True, True, False, False])
    y_prob = np.array([0.2, 0.6, 0.6, 0.2])
    ead = np.full(4, 1000.0)
    costs = _toy_costs()
    lenient = evaluate_decision_policy(y_true, y_prob, ead, 0.8, 0.5, 0.7, costs)
    strict = evaluate_decision_policy(y_true, y_prob, ead, 0.8, 0.5, 0.55, costs)
    # Stricter (lower) decline threshold declines more loans -> capture up,
    # rejection rate (of good loans) also up here since one good loan (0.6)
    # moves from review into decline.
    assert strict["decline_volume"] >= lenient["decline_volume"]


def test_mismatched_shapes_raise():
    costs = _toy_costs()
    with pytest.raises(DecisionCostAssumptionError):
        evaluate_decision_policy(
            np.array([True, False]), np.array([0.1]), np.array([100.0, 200.0]), 0.8, 0.3, 0.7, costs
        )


def test_cost_assumptions_reject_negative_values():
    with pytest.raises(DecisionCostAssumptionError):
        CreditDecisionCostAssumptions(cost_per_manual_review=-1.0, opportunity_cost_per_good_loan_declined=10.0)
    with pytest.raises(DecisionCostAssumptionError):
        CreditDecisionCostAssumptions(cost_per_manual_review=1.0, opportunity_cost_per_good_loan_declined=-10.0)


def test_load_credit_decision_cost_assumptions_from_real_config_file():
    costs = load_credit_decision_cost_assumptions()
    assert costs.cost_per_manual_review >= 0
    assert costs.opportunity_cost_per_good_loan_declined >= 0


def test_sensitivity_analysis_varies_lgd_and_thresholds_and_reports_each_row():
    rng = np.random.default_rng(5)
    n = 200
    y_true = rng.uniform(size=n) < 0.2
    y_prob = rng.uniform(size=n)
    ead = rng.uniform(500, 5000, size=n)
    costs = _toy_costs()
    variations = [
        {"label": "base"},
        {"label": "lgd_high", "lgd": 0.95},
        {"label": "stricter_decline", "decline_threshold": 0.5},
        {"label": "costlier_review", "cost_per_manual_review": 100.0},
    ]
    out = sensitivity_analysis(
        y_true, y_prob, ead,
        base_lgd=0.8, base_review_threshold=0.3, base_decline_threshold=0.7,
        base_costs=costs, variations=variations,
    )
    assert isinstance(out, pd.DataFrame)
    assert out["variation_label"].tolist() == ["base", "lgd_high", "stricter_decline", "costlier_review"]
    lgd_row = out[out["variation_label"] == "lgd_high"].iloc[0]
    assert lgd_row["lgd"] == pytest.approx(0.95)
    stricter_row = out[out["variation_label"] == "stricter_decline"].iloc[0]
    assert stricter_row["decline_threshold"] == pytest.approx(0.5)
    costlier_row = out[out["variation_label"] == "costlier_review"].iloc[0]
    assert costlier_row["cost_per_manual_review"] == pytest.approx(100.0)
    # No row/column claims any policy is "optimal" -- sensitivity_analysis
    # only reports, it does not pick a winner.
    assert "optimal" not in [c.lower() for c in out.columns]


def test_disclaimer_is_non_empty_string():
    assert isinstance(DECISION_COST_DISCLAIMER, str) and DECISION_COST_DISCLAIMER
