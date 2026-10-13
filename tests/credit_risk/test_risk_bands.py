"""Tests for Module C, Milestone 4, Section B: src/credit_risk/risk_bands.py
-- illustrative low/medium/high risk bands fit on train (development data)
only, then applied to other splits without refitting."""
import numpy as np
import pandas as pd
import pytest

from src.credit_risk.risk_bands import (
    RISK_BAND_LABELS,
    RiskBandBoundaries,
    RiskBandError,
    assign_risk_band,
    fit_risk_band_boundaries,
    summarize_risk_bands,
)


def test_fit_risk_band_boundaries_default_tertiles():
    y_prob_train = np.arange(1, 10) / 10.0  # 0.1 .. 0.9, evenly spaced
    boundaries = fit_risk_band_boundaries("model_a", y_prob_train)
    assert boundaries.model_name == "model_a"
    assert boundaries.n_train == 9
    assert 0.0 < boundaries.low_medium_boundary < boundaries.medium_high_boundary < 1.0


def test_boundaries_reject_inverted_or_equal_cuts():
    with pytest.raises(RiskBandError):
        RiskBandBoundaries(model_name="m", low_medium_boundary=0.5, medium_high_boundary=0.5, n_train=10)
    with pytest.raises(RiskBandError):
        RiskBandBoundaries(model_name="m", low_medium_boundary=0.6, medium_high_boundary=0.4, n_train=10)


def test_fit_rejects_empty_train_array():
    with pytest.raises(RiskBandError):
        fit_risk_band_boundaries("model_a", np.array([]))


def test_fit_rejects_invalid_quantiles():
    y_prob_train = np.linspace(0.01, 0.99, 50)
    with pytest.raises(RiskBandError):
        fit_risk_band_boundaries("model_a", y_prob_train, low_quantile=0.7, high_quantile=0.3)
    with pytest.raises(RiskBandError):
        fit_risk_band_boundaries("model_a", y_prob_train, low_quantile=0.0, high_quantile=0.5)


def test_fit_rejects_degenerate_boundaries_from_ties():
    # Enough duplicate values at the tertile cut points to collapse
    # low_medium_boundary and medium_high_boundary onto the same value.
    y_prob_train = np.array([0.5] * 20 + [0.9])
    with pytest.raises(RiskBandError):
        fit_risk_band_boundaries("model_a", y_prob_train)


def test_assign_risk_band_uses_fitted_boundaries_not_the_new_splits_own_quantiles():
    boundaries = RiskBandBoundaries(
        model_name="model_a", low_medium_boundary=0.3, medium_high_boundary=0.7, n_train=100
    )
    y_prob_other_split = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    bands = assign_risk_band(y_prob_other_split, boundaries)
    assert list(bands) == ["low", "medium", "medium", "high", "high"]


def test_assign_risk_band_never_refits_quantiles_on_new_data():
    # A split whose OWN tertiles would differ wildly from the fitted
    # boundaries must still be bucketed by the fitted boundaries, not its
    # own distribution.
    boundaries = RiskBandBoundaries(
        model_name="model_a", low_medium_boundary=0.01, medium_high_boundary=0.02, n_train=100
    )
    y_prob_other_split = np.array([0.5, 0.6, 0.7])  # all far above both boundaries
    bands = assign_risk_band(y_prob_other_split, boundaries)
    assert list(bands) == ["high", "high", "high"]


def test_summarize_risk_bands_reports_n_rate_mean_prob_and_gap():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    bands = np.array(["low", "low", "medium", "medium", "high", "high"], dtype=object)
    out = summarize_risk_bands(y_true, y_prob, bands)
    assert out["risk_band"].tolist() == list(RISK_BAND_LABELS)

    low_row = out[out["risk_band"] == "low"].iloc[0]
    assert low_row["n"] == 2
    assert low_row["observed_adverse_outcome_rate"] == pytest.approx(0.0)
    assert low_row["mean_predicted_probability"] == pytest.approx(0.15)
    assert low_row["calibration_gap"] == pytest.approx(0.15)

    high_row = out[out["risk_band"] == "high"].iloc[0]
    assert high_row["n"] == 2
    assert high_row["observed_adverse_outcome_rate"] == pytest.approx(0.5)
    assert high_row["mean_predicted_probability"] == pytest.approx(0.85)
    assert high_row["calibration_gap"] == pytest.approx(0.35)


def test_summarize_risk_bands_reports_zero_row_band_explicitly():
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    bands = np.array(["low", "high"], dtype=object)
    out = summarize_risk_bands(y_true, y_prob, bands)
    medium_row = out[out["risk_band"] == "medium"].iloc[0]
    assert medium_row["n"] == 0
    # A DataFrame column holding a mix of floats and None coerces None to
    # NaN -- pandas's standard behavior, not a bug in summarize_risk_bands
    # (which itself returns a literal None per the module's docstring).
    assert pd.isna(medium_row["observed_adverse_outcome_rate"])
    assert pd.isna(medium_row["mean_predicted_probability"])
    assert pd.isna(medium_row["calibration_gap"])


def test_summarize_risk_bands_mismatched_shapes_raise():
    with pytest.raises(RiskBandError):
        summarize_risk_bands(np.array([0, 1]), np.array([0.1]), np.array(["low", "high"], dtype=object))
