"""Module C, Milestone 3, Section D: fitting logistic regression for
Model A and Model B.

Discipline:
  - A single, modest, PREDECLARED L2 regularization strength
    (`DEFAULT_C`) is used for both models. It is not tuned via a grid
    search against validation or benchmark -- doing so would leak
    evaluation-split information into model selection. `DEFAULT_C=1.0`
    is scikit-learn's own un-tuned default, chosen here precisely because
    it is a neutral, off-the-shelf baseline value, not a value selected
    because it happened to score well.
  - `class_weight=None` (the default) is used deliberately, not
    `"balanced"`: Section F needs predicted probabilities that are
    comparable to the real, observed train-split default prevalence
    (~15.1%) for calibration assessment to mean anything. Reweighting
    classes during fitting would shift predicted probabilities away from
    that real prevalence and make the calibration section's own
    "sigmoid vs isotonic" comparison meaningless.
  - `fit_logistic_regression` takes an already preprocessing-transformed
    train matrix -- it has no knowledge of raw columns, dates, or the
    chronological split, keeping it a pure, reusable "fit this matrix"
    function independent of how the features were produced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

DEFAULT_C = 1.0
DEFAULT_MAX_ITER = 1000


@dataclass(frozen=True)
class FittedModel:
    name: str
    model: LogisticRegression
    feature_names: tuple[str, ...]
    c_regularization: float


def fit_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: tuple[str, ...],
    name: str,
    c_regularization: float = DEFAULT_C,
) -> FittedModel:
    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"X_train has {X_train.shape[1]} columns but {len(feature_names)} feature "
            "names were provided -- these must match exactly for Section G's coefficient "
            "interpretation to attribute coefficients to the right feature."
        )
    model = LogisticRegression(
        penalty="l2",
        C=c_regularization,
        solver="lbfgs",
        max_iter=DEFAULT_MAX_ITER,
        class_weight=None,
    )
    model.fit(X_train, y_train)
    return FittedModel(
        name=name, model=model, feature_names=tuple(feature_names), c_regularization=c_regularization
    )


def predict_probability(fitted: FittedModel, X: np.ndarray) -> np.ndarray:
    return fitted.model.predict_proba(X)[:, 1]
