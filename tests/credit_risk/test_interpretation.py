"""Tests for Module C, Milestone 3, Section G:
src/credit_risk/interpretation.py -- odds-ratio reporting (never causal
language) with explicit one-hot reference levels."""
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder

from src.credit_risk.interpretation import (
    extract_coefficients,
    reference_levels_from_onehot,
    top_coefficients,
)
from src.credit_risk.modeling import fit_logistic_regression


def _toy_fitted_model():
    rng = np.random.default_rng(0)
    n = 300
    x_numeric = rng.normal(size=n)
    cat = rng.choice(["RENT", "OWN", "MORTGAGE"], size=n)
    onehot = OneHotEncoder(drop="first").fit(pd.DataFrame({"home_ownership": cat}))
    X_cat = onehot.transform(pd.DataFrame({"home_ownership": cat})).toarray()
    X = np.column_stack([x_numeric, X_cat])
    logits = 0.8 * x_numeric + 1.2 * X_cat[:, 0]
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    feature_names = ("x_numeric",) + tuple(onehot.get_feature_names_out(["home_ownership"]))
    fitted = fit_logistic_regression(X, y, feature_names, name="toy")
    return fitted, onehot


def test_reference_levels_from_onehot_reports_dropped_category():
    _, onehot = _toy_fitted_model()
    levels = reference_levels_from_onehot(["home_ownership"], onehot)
    # drop="first" drops the alphabetically-first category seen.
    assert levels["home_ownership"] == sorted(onehot.categories_[0])[0]


def test_extract_coefficients_odds_ratio_is_exp_of_coefficient():
    fitted, onehot = _toy_fitted_model()
    ref_levels = reference_levels_from_onehot(["home_ownership"], onehot)
    table = extract_coefficients(fitted, ref_levels)
    for _, row in table.table.iterrows():
        assert row["odds_ratio"] == pytest.approx(np.exp(row["coefficient"]))


def test_extract_coefficients_mismatched_length_raises():
    fitted, onehot = _toy_fitted_model()
    # Tamper with feature_names to create a count mismatch.
    bad_fitted = fitted.__class__(
        name=fitted.name, model=fitted.model, feature_names=("only_one",), c_regularization=fitted.c_regularization
    )
    with pytest.raises(ValueError):
        extract_coefficients(bad_fitted, {})


def test_top_coefficients_direction_labels_match_coefficient_sign():
    fitted, onehot = _toy_fitted_model()
    ref_levels = reference_levels_from_onehot(["home_ownership"], onehot)
    table = extract_coefficients(fitted, ref_levels)
    top = top_coefficients(table, n=5)
    for _, row in top.iterrows():
        if row["coefficient"] > 0:
            assert row["direction"] == "associated with HIGHER default odds"
        else:
            assert row["direction"] == "associated with LOWER default odds"
        # Never causal language.
        assert "causes" not in row["direction"].lower()
        assert "caused by" not in row["direction"].lower()


def test_top_coefficients_sorted_by_absolute_magnitude_descending():
    fitted, onehot = _toy_fitted_model()
    ref_levels = reference_levels_from_onehot(["home_ownership"], onehot)
    table = extract_coefficients(fitted, ref_levels)
    top = top_coefficients(table, n=3)
    abs_coefs = top["abs_coefficient"].tolist()
    assert abs_coefs == sorted(abs_coefs, reverse=True)
