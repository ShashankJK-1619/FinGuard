"""Module C, Milestone 3, Section F: calibration assessment with
leakage-safe fit/eval separation.

Discipline:
  - The base logistic regression (`modeling.py`) is already fit on train.
  - Both candidate recalibration maps -- sigmoid/Platt and isotonic -- are
    fit via `CalibratedClassifierCV(cv=5, ...)`, using ONLY train data
    internally (5-fold cross-validation within train: each fold's
    calibration map is fit on the other folds' held-out predictions, never
    on validation/benchmark). This is the standard leakage-safe way to fit
    a calibration map without reusing the exact same train predictions the
    base model was already fit on.
  - Calibration QUALITY is assessed only on validation and benchmark --
    data neither the base model nor the calibration map ever saw.
  - Isotonic is only reported as preferred over sigmoid if it both (a)
    achieves a materially lower Brier score on validation AND (b) train
    has enough rows for isotonic's non-parametric step function to be
    trustworthy (`MIN_TRAIN_ROWS_FOR_ISOTONIC`) -- isotonic regression can
    overfit badly on small samples, so on a small split this module
    prefers sigmoid regardless of validation Brier.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

MIN_TRAIN_ROWS_FOR_ISOTONIC = 1000
ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT = 0.0005  # material improvement floor, predeclared

# Module C, Milestone 4, Section A addition: Cox calibration regression
# (logit-space slope/intercept). Clipping floor for probabilities before
# taking logit(), since logit(0) and logit(1) are undefined.
CALIBRATION_REGRESSION_PROBABILITY_CLIP_EPS = 1e-6


class CalibrationRegressionError(ValueError):
    pass


@dataclass(frozen=True)
class CalibrationSlopeIntercept:
    """Cox calibration regression result: fits
    `y_true ~ intercept + slope * logit(y_prob)` via unpenalized logistic
    regression. Perfect calibration is `slope == 1` and `intercept == 0`.
    `slope < 1` means predictions are too extreme (overconfident at the
    tails); `slope > 1` means predictions are too mild (underconfident).
    A nonzero `intercept` indicates systematic over-/under-prediction of
    the overall rate, independent of spread. `converged` reports whether
    statsmodels' MLE fit converged -- a non-converged fit's slope/intercept
    should be treated as unreliable, not silently trusted."""

    model_name: str
    split_name: str
    n: int
    intercept: float
    slope: float
    converged: bool


@dataclass(frozen=True)
class CalibrationComparison:
    model_name: str
    n_train: int
    uncalibrated_brier_validation: float
    sigmoid_brier_validation: float
    isotonic_brier_validation: float
    isotonic_is_eligible_by_sample_size: bool
    chosen_method: str
    chosen_method_reason: str


@dataclass(frozen=True)
class ReliabilityBin:
    bin_index: int
    mean_predicted_probability: float
    observed_frequency: float
    n: int


def fit_calibrated_variants(
    base_model: LogisticRegression, X_train: np.ndarray, y_train: np.ndarray
) -> tuple[CalibratedClassifierCV, CalibratedClassifierCV]:
    """Fits sigmoid and isotonic recalibration maps via 5-fold CV entirely
    within train. `base_model` is treated as the estimator template --
    CalibratedClassifierCV refits its own copy per fold, it does not reuse
    the already-trained `base_model` instance directly (avoiding the
    "calibrating a model against the exact same rows it memorized"
    leakage)."""
    sigmoid = CalibratedClassifierCV(estimator=LogisticRegression(**base_model.get_params()), method="sigmoid", cv=5)
    isotonic = CalibratedClassifierCV(
        estimator=LogisticRegression(**base_model.get_params()), method="isotonic", cv=5
    )
    sigmoid.fit(X_train, y_train)
    isotonic.fit(X_train, y_train)
    return sigmoid, isotonic


def compare_calibration(
    model_name: str,
    n_train: int,
    y_validation: np.ndarray,
    uncalibrated_prob_validation: np.ndarray,
    sigmoid_prob_validation: np.ndarray,
    isotonic_prob_validation: np.ndarray,
) -> CalibrationComparison:
    uncalibrated_brier = float(brier_score_loss(y_validation, uncalibrated_prob_validation))
    sigmoid_brier = float(brier_score_loss(y_validation, sigmoid_prob_validation))
    isotonic_brier = float(brier_score_loss(y_validation, isotonic_prob_validation))

    isotonic_eligible = n_train >= MIN_TRAIN_ROWS_FOR_ISOTONIC

    if (
        isotonic_eligible
        and (sigmoid_brier - isotonic_brier) >= ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT
    ):
        chosen = "isotonic"
        reason = (
            f"isotonic Brier ({isotonic_brier:.5f}) improves on sigmoid Brier "
            f"({sigmoid_brier:.5f}) by >= {ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT} on validation, "
            f"and train has {n_train} rows (>= {MIN_TRAIN_ROWS_FOR_ISOTONIC} eligibility floor)."
        )
    else:
        chosen = "sigmoid"
        if not isotonic_eligible:
            reason = (
                f"train has only {n_train} rows (< {MIN_TRAIN_ROWS_FOR_ISOTONIC}); isotonic's "
                "non-parametric step function is not trusted at this sample size regardless of "
                "its validation Brier."
            )
        else:
            reason = (
                f"isotonic did not improve validation Brier by the predeclared "
                f"{ISOTONIC_PREFERENCE_BRIER_IMPROVEMENT} margin "
                f"(sigmoid={sigmoid_brier:.5f}, isotonic={isotonic_brier:.5f}); sigmoid's simpler, "
                "monotonic parametric form is preferred by default."
            )

    return CalibrationComparison(
        model_name=model_name,
        n_train=n_train,
        uncalibrated_brier_validation=uncalibrated_brier,
        sigmoid_brier_validation=sigmoid_brier,
        isotonic_brier_validation=isotonic_brier,
        isotonic_is_eligible_by_sample_size=isotonic_eligible,
        chosen_method=chosen,
        chosen_method_reason=reason,
    )


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[ReliabilityBin]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)

    bins = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append(
            ReliabilityBin(
                bin_index=b,
                mean_predicted_probability=float(y_prob[mask].mean()),
                observed_frequency=float(y_true[mask].mean()),
                n=n,
            )
        )
    return bins


def calibration_slope_intercept(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = CALIBRATION_REGRESSION_PROBABILITY_CLIP_EPS,
) -> CalibrationSlopeIntercept:
    """Module C, Milestone 4, Section A addition (additive -- does not
    modify any Milestone 3 function above). Fits the Cox calibration
    regression `y_true ~ intercept + slope * logit(y_prob)` via an
    unpenalized logistic regression (`statsmodels.api.Logit`) and returns
    the fitted intercept/slope. `y_prob` is clipped to `[eps, 1 - eps]`
    before taking `logit()`, since `logit(0)`/`logit(1)` are undefined --
    predicted probabilities of exactly 0 or 1 are treated as "extremely
    close to the boundary," not literally certain.

    Raises `CalibrationRegressionError` on structurally invalid input
    (mismatched shapes, empty arrays, or `y_true` containing only one
    class -- a calibration regression is undefined with no contrast to
    fit against). Does NOT raise on MLE non-convergence -- that is
    reported via the `converged` field instead, since a non-converged fit
    is a data-quality signal worth surfacing, not necessarily a caller
    bug."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise CalibrationRegressionError(
            f"y_true and y_prob must have the same shape; got {y_true.shape} vs {y_prob.shape}"
        )
    if y_true.size == 0:
        raise CalibrationRegressionError("y_true/y_prob must be non-empty")
    if len(np.unique(y_true)) < 2:
        raise CalibrationRegressionError(
            "y_true must contain both classes (0 and 1) to fit a calibration regression"
        )

    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(y_prob_clipped / (1.0 - y_prob_clipped))
    design = sm.add_constant(logit_p)
    result = sm.Logit(y_true, design).fit(disp=0)
    converged = bool(result.mle_retvals.get("converged", True))

    return CalibrationSlopeIntercept(
        model_name=model_name,
        split_name=split_name,
        n=int(y_true.size),
        intercept=float(result.params[0]),
        slope=float(result.params[1]),
        converged=converged,
    )
