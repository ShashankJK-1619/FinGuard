"""Tests for Module C, Milestone 3, Section F: src/credit_risk/calibration.py
-- leakage-safe calibration fit (train-only CV) vs. quality assessment
(validation-only), and the predeclared isotonic-eligibility/preference
rules."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.credit_risk.calibration import (
    ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT,
    MIN_TRAIN_ROWS_FOR_ISOTONIC,
    compare_calibration,
    fit_calibrated_variants,
    reliability_curve,
)


def _toy_train(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 1.2 * X[:, 0] - 0.7 * X[:, 1] + 0.3 * X[:, 2]
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def test_fit_calibrated_variants_fits_both_methods_on_train_only():
    X_train, y_train = _toy_train()
    base = LogisticRegression().fit(X_train, y_train)
    sigmoid, isotonic = fit_calibrated_variants(base, X_train, y_train)
    # Both are fitted CalibratedClassifierCV instances usable for prediction.
    assert sigmoid.predict_proba(X_train).shape == (len(y_train), 2)
    assert isotonic.predict_proba(X_train).shape == (len(y_train), 2)
    # fit_calibrated_variants must not have mutated/reused the exact `base`
    # instance's fitted state as its own estimator (it builds fresh clones).
    assert sigmoid.estimator is not base
    assert isotonic.estimator is not base


def test_isotonic_ineligible_below_min_train_rows_forces_sigmoid():
    n_train = MIN_TRAIN_ROWS_FOR_ISOTONIC - 1
    comparison = compare_calibration(
        "toy_model",
        n_train=n_train,
        y_validation=np.array([0, 1, 0, 1]),
        uncalibrated_prob_validation=np.array([0.1, 0.9, 0.2, 0.8]),
        sigmoid_prob_validation=np.array([0.1, 0.9, 0.2, 0.8]),
        isotonic_prob_validation=np.array([0.0, 1.0, 0.0, 1.0]),  # "perfect" but must still be rejected
    )
    assert comparison.isotonic_is_eligible_by_sample_size is False
    assert comparison.chosen_method == "sigmoid"


def test_isotonic_chosen_only_if_eligible_and_materially_better():
    y_validation = np.array([0, 1] * 50)
    uncalibrated = np.full(100, 0.5)
    sigmoid_prob = np.full(100, 0.5)
    # isotonic improves Brier by more than the predeclared margin.
    isotonic_prob = np.where(y_validation == 1, 0.5 + 2 * ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT, 0.5 - 2 * ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT)
    comparison = compare_calibration(
        "toy_model",
        n_train=MIN_TRAIN_ROWS_FOR_ISOTONIC,
        y_validation=y_validation,
        uncalibrated_prob_validation=uncalibrated,
        sigmoid_prob_validation=sigmoid_prob,
        isotonic_prob_validation=isotonic_prob,
    )
    assert comparison.isotonic_is_eligible_by_sample_size is True
    assert comparison.chosen_method == "isotonic"


def test_sigmoid_chosen_when_isotonic_improvement_below_predeclared_margin():
    y_validation = np.array([0, 1] * 50)
    uncalibrated = np.full(100, 0.5)
    sigmoid_prob = np.full(100, 0.5)
    isotonic_prob = np.full(100, 0.5001)  # trivially different, not materially better
    comparison = compare_calibration(
        "toy_model",
        n_train=MIN_TRAIN_ROWS_FOR_ISOTONIC,
        y_validation=y_validation,
        uncalibrated_prob_validation=uncalibrated,
        sigmoid_prob_validation=sigmoid_prob,
        isotonic_prob_validation=isotonic_prob,
    )
    assert comparison.chosen_method == "sigmoid"
    assert "did not improve" in comparison.chosen_method_reason


def test_reliability_curve_bins_sum_to_total_n_and_frequencies_in_range():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.uniform(size=500)
    bins = reliability_curve(y_true, y_prob, n_bins=10)
    assert sum(b.n for b in bins) == 500
    for b in bins:
        assert 0.0 <= b.mean_predicted_probability <= 1.0
        assert 0.0 <= b.observed_frequency <= 1.0


def test_reliability_curve_skips_empty_bins_rather_than_fabricating_them():
    y_true = np.array([0, 1])
    y_prob = np.array([0.05, 0.06])  # both fall in the same low bin
    bins = reliability_curve(y_true, y_prob, n_bins=10)
    assert len(bins) == 1
    assert bins[0].n == 2
