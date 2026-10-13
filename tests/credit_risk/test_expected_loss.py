"""Tests for Module C, Milestone 4, Section C: src/credit_risk/expected_loss.py
-- illustrative LGD/EAD expected-loss framework (EL = p * LGD * EAD)."""
import numpy as np
import pytest

from src.credit_risk.expected_loss import (
    EAD_FIELD,
    EXPECTED_LOSS_DISCLAIMER,
    ExpectedLossAssumptions,
    ExpectedLossError,
    compute_expected_loss,
    load_expected_loss_assumptions,
)


def test_compute_expected_loss_hand_computed_example():
    y_prob = np.array([0.1, 0.5, 0.2])
    ead = np.array([1000.0, 2000.0, 500.0])
    lgd = 0.8
    out = compute_expected_loss(y_prob, ead, lgd)
    expected = np.array([0.1 * 0.8 * 1000.0, 0.5 * 0.8 * 2000.0, 0.2 * 0.8 * 500.0])
    np.testing.assert_allclose(out, expected)


def test_expected_loss_scales_proportionally_with_lgd():
    y_prob = np.array([0.3, 0.6])
    ead = np.array([1000.0, 1000.0])
    low = compute_expected_loss(y_prob, ead, lgd=0.4)
    high = compute_expected_loss(y_prob, ead, lgd=0.8)
    np.testing.assert_allclose(high, low * 2.0)


def test_expected_loss_scales_proportionally_with_ead():
    y_prob = np.array([0.3, 0.6])
    small_ead = np.array([1000.0, 1000.0])
    big_ead = small_ead * 3.0
    small = compute_expected_loss(y_prob, small_ead, lgd=0.8)
    big = compute_expected_loss(y_prob, big_ead, lgd=0.8)
    np.testing.assert_allclose(big, small * 3.0)


def test_mismatched_shapes_raise():
    with pytest.raises(ExpectedLossError):
        compute_expected_loss(np.array([0.1, 0.2]), np.array([1000.0]), lgd=0.8)


def test_lgd_out_of_range_raises():
    y_prob = np.array([0.1])
    ead = np.array([100.0])
    with pytest.raises(ExpectedLossError):
        compute_expected_loss(y_prob, ead, lgd=1.5)
    with pytest.raises(ExpectedLossError):
        compute_expected_loss(y_prob, ead, lgd=-0.1)


def test_y_prob_out_of_range_raises():
    with pytest.raises(ExpectedLossError):
        compute_expected_loss(np.array([1.1]), np.array([100.0]), lgd=0.8)
    with pytest.raises(ExpectedLossError):
        compute_expected_loss(np.array([-0.1]), np.array([100.0]), lgd=0.8)


def test_negative_ead_raises():
    with pytest.raises(ExpectedLossError):
        compute_expected_loss(np.array([0.2]), np.array([-5.0]), lgd=0.8)


def test_empty_arrays_do_not_raise_and_return_empty():
    out = compute_expected_loss(np.array([]), np.array([]), lgd=0.8)
    assert out.shape == (0,)


def test_expected_loss_assumptions_validates_lgd_range():
    ExpectedLossAssumptions(lgd=0.0)
    ExpectedLossAssumptions(lgd=1.0)
    with pytest.raises(ExpectedLossError):
        ExpectedLossAssumptions(lgd=1.01)
    with pytest.raises(ExpectedLossError):
        ExpectedLossAssumptions(lgd=-0.01)


def test_load_expected_loss_assumptions_from_real_config_file():
    assumptions = load_expected_loss_assumptions()
    assert 0.0 <= assumptions.lgd <= 1.0


def test_ead_field_is_loan_amnt_and_disclaimer_is_non_empty_string():
    # EAD proxy choice (Section C) is documented and pinned by this test so
    # it cannot silently drift to a different column.
    assert EAD_FIELD == "loan_amnt"
    assert isinstance(EXPECTED_LOSS_DISCLAIMER, str) and EXPECTED_LOSS_DISCLAIMER
