"""Milestone 4: recalibrating Milestone 3's class-weighted logistic scores.

Milestone 3 found that `LogisticRegression(class_weight="balanced")`
produces well-ordered scores (ROC-AUC 0.765 validation, 0.732 holdout) but
badly miscalibrated probabilities (Brier 0.260 validation, 0.316 holdout;
predicted probabilities several multiples above observed fraud rates in
every decile bin). Milestone 4 fixes the probability-quality problem
without touching the model or its ranking: a calibrator is a small,
separate, monotonic (sigmoid) or non-decreasing (isotonic) remapping
`raw_score -> calibrated_probability`, fit AFTER the classifier, on data
the classifier itself never saw.

Both calibrators here are implemented directly with
`sklearn.linear_model.LogisticRegression` (Platt/sigmoid scaling is exactly
a 1-D logistic regression of the true label on the raw score) and
`sklearn.isotonic.IsotonicRegression`, rather than
`sklearn.calibration.CalibratedClassifierCV`. Reasons: (1) CalibratedClassifierCV's
`cv="prefit"` option is deprecated/removed across the sklearn versions this
project's `scikit-learn>=1.4,<2` pin spans, so pinning to it would be
version-fragile; (2) a hand-rolled calibrator makes the "fit on validation
labels only, never refit, never see holdout" property trivial to state and
test directly (see tests/models/test_calibration.py), rather than relying
on correctly threading `cv="prefit"` through a wrapper class.

No future leakage: every calibrator here is fit once, on validation raw
probabilities + validation labels, via `.fit(...)`. `.transform(...)` never
mutates internal state. Holdout raw probabilities are only ever passed to
`.transform(...)`, never `.fit(...)` — see src/models/train_calibrated.py.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator(Protocol):
    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "Calibrator": ...

    def transform(self, raw_probs: np.ndarray) -> np.ndarray: ...


class SigmoidCalibrator:
    """Platt scaling: a 1-D logistic regression of the true label on the
    classifier's raw predicted probability. Assumes the miscalibration is
    a smooth, sigmoid-shaped distortion of the raw score — a reasonable fit
    for class-weighting-induced miscalibration, which uniformly inflates
    scores rather than distorting them non-monotonically.
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression()

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "SigmoidCalibrator":
        self._lr.fit(np.asarray(raw_probs).reshape(-1, 1), np.asarray(y_true))
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(np.asarray(raw_probs).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression: fits the most flexible non-decreasing step
    function from raw score to observed fraud rate. Makes no assumption
    about the shape of the miscalibration beyond monotonicity (preserving
    ranking), at the cost of needing more data per bin to avoid overfitting
    than the single-parameter-pair sigmoid fit.
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(np.asarray(raw_probs), np.asarray(y_true))
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        return self._iso.transform(np.asarray(raw_probs))


# "uncalibrated" maps to None: the identity transform (raw score passed
# through unchanged), included so every comparison table in Milestone 4
# can loop over CALIBRATION_METHODS uniformly.
CALIBRATION_METHODS = {
    "uncalibrated": None,
    "sigmoid": SigmoidCalibrator,
    "isotonic": IsotonicCalibrator,
}


def fit_calibrator(method: str, raw_probs_val: np.ndarray, y_val: np.ndarray) -> Calibrator | None:
    """Fits the named calibrator on VALIDATION raw probabilities and labels
    only. Never pass holdout data here — see module docstring.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"Unknown calibration method {method!r}; choose from {list(CALIBRATION_METHODS)}")
    cls = CALIBRATION_METHODS[method]
    if cls is None:
        return None
    return cls().fit(raw_probs_val, y_val)


def apply_calibrator(calibrator: Calibrator | None, raw_probs: np.ndarray) -> np.ndarray:
    """Applies a fitted calibrator's transform. `calibrator=None` means
    "uncalibrated" and returns `raw_probs` unchanged.
    """
    if calibrator is None:
        return np.asarray(raw_probs)
    return calibrator.transform(raw_probs)


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Quantile-binned reliability table: mean predicted probability vs.
    observed fraud rate per bin, PLUS bin size and bin edges. More detailed
    than `sklearn.calibration.calibration_curve` (used in Milestone 3's
    `evaluate`/`evaluate_probs`), which returns only the two parallel
    arrays without bin populations — Milestone 4 needs bin size to judge
    how much each reliability-curve point should be trusted (a bin with 5
    rows and a bin with 5,000 rows are not equally informative).

    Falls back from `pd.qcut` to `pd.cut` if the probability distribution
    has too many ties at common values (e.g. many isotonic-calibrated
    probabilities collapsing onto the same step) for `n_bins` distinct
    quantile edges to exist.
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)})
    try:
        df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["y_prob"], bins=n_bins)

    rows = []
    for interval, group in df.groupby("bin", observed=True):
        rows.append(
            {
                "bin_left": float(interval.left),
                "bin_right": float(interval.right),
                "n_rows": int(len(group)),
                "mean_predicted_prob": float(group["y_prob"].mean()),
                "observed_fraud_rate": float(group["y_true"].mean()),
            }
        )
    return rows
