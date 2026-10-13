"""Tests for Module C, Milestone 3, Section D: src/credit_risk/modeling.py
-- fitting logistic regression with a modest, predeclared, non-tuned
regularization strength."""
import numpy as np
import pytest

from src.credit_risk.modeling import DEFAULT_C, fit_logistic_regression, predict_probability


def _toy_xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    # Make the label genuinely a function of X so the fit isn't degenerate.
    logits = 1.5 * X[:, 0] - 0.5 * X[:, 1]
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y


def test_fit_logistic_regression_uses_predeclared_default_c_not_tuned():
    X, y = _toy_xy()
    fitted = fit_logistic_regression(X, y, ("a", "b", "c"), name="toy_model")
    assert fitted.c_regularization == DEFAULT_C
    assert fitted.model.C == DEFAULT_C
    assert fitted.model.class_weight is None


def test_fit_logistic_regression_mismatched_feature_names_raises():
    X, y = _toy_xy()
    with pytest.raises(ValueError):
        fit_logistic_regression(X, y, ("only_one_name",), name="toy_model")


def test_fit_logistic_regression_records_feature_names_and_name():
    X, y = _toy_xy()
    fitted = fit_logistic_regression(X, y, ("a", "b", "c"), name="toy_model")
    assert fitted.name == "toy_model"
    assert fitted.feature_names == ("a", "b", "c")
    assert len(fitted.model.coef_.ravel()) == 3


def test_predict_probability_returns_values_in_unit_interval():
    X, y = _toy_xy()
    fitted = fit_logistic_regression(X, y, ("a", "b", "c"), name="toy_model")
    probs = predict_probability(fitted, X)
    assert probs.shape == (len(y),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_custom_c_regularization_is_honored_not_silently_overridden():
    X, y = _toy_xy()
    fitted = fit_logistic_regression(X, y, ("a", "b", "c"), name="toy_model", c_regularization=0.1)
    assert fitted.c_regularization == 0.1
    assert fitted.model.C == 0.1
