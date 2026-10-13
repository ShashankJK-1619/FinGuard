"""Tests for Module C, Milestone 4, Section I:
src/credit_risk/score_stability.py -- PSI drift heuristic, score
distribution by vintage, calibration by vintage, and adverse-outcome
rate by risk band over time."""
import numpy as np
import pytest

from src.credit_risk.score_stability import (
    PSI_MODERATE_CHANGE_THRESHOLD,
    PSI_SIGNIFICANT_CHANGE_THRESHOLD,
    ScoreStabilityError,
    adverse_outcome_rate_by_band_over_time,
    calibration_by_vintage,
    compute_psi,
    psi_drift_category,
    score_distribution_by_vintage,
)


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.uniform(size=5000)
    comparison = rng.uniform(size=5000)
    psi = compute_psi(reference, comparison, n_bins=10)
    assert abs(psi) < 0.02


def test_psi_is_large_for_a_shifted_distribution():
    rng = np.random.default_rng(1)
    reference = rng.uniform(0.0, 0.5, size=5000)
    comparison = rng.uniform(0.5, 1.0, size=5000)  # fully shifted, no overlap
    psi = compute_psi(reference, comparison, n_bins=10)
    assert psi > PSI_SIGNIFICANT_CHANGE_THRESHOLD


def test_psi_drift_category_bands():
    assert psi_drift_category(0.01) == "no_significant_change"
    assert psi_drift_category(PSI_MODERATE_CHANGE_THRESHOLD) == "moderate_change"
    assert psi_drift_category(0.15) == "moderate_change"
    assert psi_drift_category(PSI_SIGNIFICANT_CHANGE_THRESHOLD) == "significant_change"
    assert psi_drift_category(0.5) == "significant_change"


def test_psi_rejects_empty_inputs_and_too_few_bins():
    with pytest.raises(ScoreStabilityError):
        compute_psi(np.array([]), np.array([0.1, 0.2]))
    with pytest.raises(ScoreStabilityError):
        compute_psi(np.array([0.1, 0.2]), np.array([]))
    with pytest.raises(ScoreStabilityError):
        compute_psi(np.random.default_rng(0).uniform(size=100), np.random.default_rng(1).uniform(size=100), n_bins=1)


def test_psi_rejects_degenerate_reference_with_too_many_ties():
    reference = np.array([0.5] * 100)
    comparison = np.random.default_rng(0).uniform(size=100)
    with pytest.raises(ScoreStabilityError):
        compute_psi(reference, comparison, n_bins=10)


def test_score_distribution_by_vintage_reports_per_year_stats():
    origination_year = np.array([2012, 2012, 2013, 2013, 2013])
    y_prob = np.array([0.1, 0.3, 0.4, 0.5, 0.6])
    out = score_distribution_by_vintage(origination_year, y_prob)
    assert out["origination_year"].tolist() == [2012, 2013]
    row_2012 = out[out["origination_year"] == 2012].iloc[0]
    assert row_2012["n"] == 2
    assert row_2012["mean_predicted_probability"] == pytest.approx(0.2)
    row_2013 = out[out["origination_year"] == 2013].iloc[0]
    assert row_2013["n"] == 3
    assert row_2013["mean_predicted_probability"] == pytest.approx(0.5)


def test_score_distribution_by_vintage_mismatched_shapes_raise():
    with pytest.raises(ScoreStabilityError):
        score_distribution_by_vintage(np.array([2012, 2013]), np.array([0.1]))


def test_calibration_by_vintage_reports_gap_per_year():
    origination_year = np.array([2012, 2012, 2013, 2013])
    y_true = np.array([0, 1, 1, 1])
    y_prob = np.array([0.2, 0.8, 0.9, 0.7])
    out = calibration_by_vintage(origination_year, y_true, y_prob)
    row_2012 = out[out["origination_year"] == 2012].iloc[0]
    assert row_2012["observed_adverse_outcome_rate"] == pytest.approx(0.5)
    assert row_2012["mean_predicted_probability"] == pytest.approx(0.5)
    assert row_2012["calibration_gap"] == pytest.approx(0.0)
    row_2013 = out[out["origination_year"] == 2013].iloc[0]
    assert row_2013["observed_adverse_outcome_rate"] == pytest.approx(1.0)
    assert row_2013["mean_predicted_probability"] == pytest.approx(0.8)
    assert row_2013["calibration_gap"] == pytest.approx(-0.2)


def test_calibration_by_vintage_mismatched_shapes_raise():
    with pytest.raises(ScoreStabilityError):
        calibration_by_vintage(np.array([2012, 2013]), np.array([0, 1]), np.array([0.1]))


def test_adverse_outcome_rate_by_band_over_time_groups_correctly():
    origination_year = np.array([2012, 2012, 2012, 2013, 2013])
    risk_band = np.array(["low", "high", "high", "low", "low"], dtype=object)
    y_true = np.array([0, 1, 0, 0, 1])
    out = adverse_outcome_rate_by_band_over_time(origination_year, risk_band, y_true)
    row_2012_high = out[(out["origination_year"] == 2012) & (out["risk_band"] == "high")].iloc[0]
    assert row_2012_high["n"] == 2
    assert row_2012_high["observed_adverse_outcome_rate"] == pytest.approx(0.5)
    row_2013_low = out[(out["origination_year"] == 2013) & (out["risk_band"] == "low")].iloc[0]
    assert row_2013_low["n"] == 2
    assert row_2013_low["observed_adverse_outcome_rate"] == pytest.approx(0.5)
    # No cell for 2013/high in this input -- absent, not fabricated as zero.
    assert out[(out["origination_year"] == 2013) & (out["risk_band"] == "high")].empty


def test_adverse_outcome_rate_by_band_over_time_mismatched_shapes_raise():
    with pytest.raises(ScoreStabilityError):
        adverse_outcome_rate_by_band_over_time(
            np.array([2012, 2013]), np.array(["low", "high"], dtype=object), np.array([0])
        )
