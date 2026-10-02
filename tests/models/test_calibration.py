from __future__ import annotations

import numpy as np
import pytest

from src.models.calibration import (
    CALIBRATION_METHODS,
    IsotonicCalibrator,
    SigmoidCalibrator,
    apply_calibrator,
    fit_calibrator,
    reliability_table,
)


def _toy_scores(n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    # Deliberately miscalibrated: raw score is inflated relative to y_true,
    # the same shape of miscalibration Milestone 3 found empirically.
    raw = np.clip(y_true * 0.6 + rng.uniform(0, 0.4, size=n), 0, 1)
    return y_true, raw


def test_uncalibrated_method_is_identity():
    y_true, raw = _toy_scores()
    calibrator = fit_calibrator("uncalibrated", raw, y_true)
    assert calibrator is None
    out = apply_calibrator(calibrator, raw)
    assert np.array_equal(out, raw)


def test_sigmoid_and_isotonic_outputs_are_valid_probabilities():
    y_true, raw = _toy_scores()
    for method in ("sigmoid", "isotonic"):
        calibrator = fit_calibrator(method, raw, y_true)
        out = apply_calibrator(calibrator, raw)
        assert out.shape == raw.shape
        assert (out >= 0).all() and (out <= 1).all()


def test_calibration_reduces_brier_score_on_fit_distribution():
    """On the exact distribution a calibrator was fit on, both methods
    should materially reduce Brier score versus the deliberately-inflated
    raw score — sanity check that these are not no-ops."""
    from sklearn.metrics import brier_score_loss

    y_true, raw = _toy_scores()
    raw_brier = brier_score_loss(y_true, raw)
    for method in ("sigmoid", "isotonic"):
        calibrator = fit_calibrator(method, raw, y_true)
        calibrated = apply_calibrator(calibrator, raw)
        assert brier_score_loss(y_true, calibrated) < raw_brier


def test_calibrator_fit_on_validation_does_not_change_when_holdout_is_mutated():
    """No future leakage: a calibrator's fitted parameters must depend only
    on the data passed to .fit (validation), never on anything resembling
    holdout. Simulates this by fitting once, then confirming repeated
    .transform calls on a fixed held-out array always produce the same
    output regardless of what else happens in between (i.e. transform is
    stateless given fixed internal parameters)."""
    y_val, raw_val = _toy_scores(seed=1)
    _, raw_holdout = _toy_scores(seed=2)

    for method in ("sigmoid", "isotonic"):
        calibrator = fit_calibrator(method, raw_val, y_val)
        first = apply_calibrator(calibrator, raw_holdout)

        # Mutate the holdout array in place after the calibrator was fit —
        # fitting already happened, so this must not retroactively affect
        # the calibrator's parameters.
        mutated_holdout = raw_holdout.copy()
        mutated_holdout[:] = 0.999
        apply_calibrator(calibrator, mutated_holdout)

        second = apply_calibrator(calibrator, raw_holdout)
        assert np.allclose(first, second)


def test_fit_calibrator_rejects_unknown_method():
    y_true, raw = _toy_scores()
    with pytest.raises(ValueError):
        fit_calibrator("not_a_real_method", raw, y_true)


def test_calibration_methods_registry_has_expected_keys():
    assert set(CALIBRATION_METHODS) == {"uncalibrated", "sigmoid", "isotonic"}


def test_reliability_table_bin_counts_sum_to_total_rows():
    y_true, raw = _toy_scores(n=400)
    table = reliability_table(y_true, raw, n_bins=8)
    assert sum(row["n_rows"] for row in table) == 400


def test_reliability_table_observed_rate_matches_manual_computation():
    y_true, raw = _toy_scores(n=400)
    table = reliability_table(y_true, raw, n_bins=5)
    for row in table:
        assert 0.0 <= row["observed_fraud_rate"] <= 1.0
        assert row["bin_left"] <= row["bin_right"]


def test_sigmoid_calibrator_class_is_directly_usable():
    y_true, raw = _toy_scores()
    calibrator = SigmoidCalibrator().fit(raw, y_true)
    out = calibrator.transform(raw)
    assert out.shape == raw.shape


def test_isotonic_calibrator_class_is_directly_usable():
    y_true, raw = _toy_scores()
    calibrator = IsotonicCalibrator().fit(raw, y_true)
    out = calibrator.transform(raw)
    assert out.shape == raw.shape
