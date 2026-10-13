"""Module C, Milestone 4, Section B: transparent illustrative risk bands.

Three bands (`low` / `medium` / `high`) are cut from a model's own predicted
probability of `terminal_adverse_outcome` using quantile boundaries FIT ON
TRAIN (development data) ONLY -- the same anti-threshold-snooping discipline
Milestone 3 already applied to its decision threshold
(`train_prevalence_matched`). Boundaries are never refit on
validation/benchmark; those splits only ever have the already-fit
boundaries applied to them.

These bands are NOT real LendingClub underwriting cutoffs and carry no
claim of being a production risk-segmentation policy -- they exist so
Sections C-J have a concrete, reproducible risk segmentation to report
expected loss, calibration, and decisioning behavior against.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RISK_BAND_LABELS = ("low", "medium", "high")

DEFAULT_LOW_QUANTILE = 1.0 / 3.0
DEFAULT_HIGH_QUANTILE = 2.0 / 3.0


class RiskBandError(ValueError):
    pass


@dataclass(frozen=True)
class RiskBandBoundaries:
    """`low_medium_boundary` / `medium_high_boundary` are predicted-probability
    cut points: `p < low_medium_boundary` -> low, `low_medium_boundary <= p <
    medium_high_boundary` -> medium, `p >= medium_high_boundary` -> high.
    Fit on `n_train` development-split rows only -- see module docstring.
    """

    model_name: str
    low_medium_boundary: float
    medium_high_boundary: float
    n_train: int

    def __post_init__(self) -> None:
        if not (0.0 <= self.low_medium_boundary < self.medium_high_boundary <= 1.0):
            raise RiskBandError(
                "Require 0 <= low_medium_boundary < medium_high_boundary <= 1; got "
                f"low_medium_boundary={self.low_medium_boundary}, "
                f"medium_high_boundary={self.medium_high_boundary}"
            )
        if self.n_train < 1:
            raise RiskBandError(f"n_train must be >= 1; got {self.n_train}")


def fit_risk_band_boundaries(
    model_name: str,
    y_prob_train: np.ndarray,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
    high_quantile: float = DEFAULT_HIGH_QUANTILE,
) -> RiskBandBoundaries:
    """Fits tertile-by-default boundaries on TRAIN predicted probabilities
    only. Raises `RiskBandError` (rather than silently producing a
    degenerate band) if the two quantiles collapse to the same cut point --
    this can genuinely happen with enough duplicate probability values and
    should be surfaced, not hidden.
    """
    if not (0.0 < low_quantile < high_quantile < 1.0):
        raise RiskBandError(
            f"Require 0 < low_quantile < high_quantile < 1; got low_quantile={low_quantile}, "
            f"high_quantile={high_quantile}"
        )
    y_prob_train = np.asarray(y_prob_train, dtype=float)
    if y_prob_train.size == 0:
        raise RiskBandError("y_prob_train must be non-empty")

    low_medium_boundary = float(np.quantile(y_prob_train, low_quantile))
    medium_high_boundary = float(np.quantile(y_prob_train, high_quantile))
    if low_medium_boundary >= medium_high_boundary:
        raise RiskBandError(
            "Fitted boundaries are degenerate (low_medium_boundary >= medium_high_boundary); "
            "train predicted probabilities have too many ties at the requested quantiles "
            f"(low_medium_boundary={low_medium_boundary}, medium_high_boundary={medium_high_boundary})"
        )

    return RiskBandBoundaries(
        model_name=model_name,
        low_medium_boundary=low_medium_boundary,
        medium_high_boundary=medium_high_boundary,
        n_train=int(y_prob_train.size),
    )


def assign_risk_band(y_prob: np.ndarray, boundaries: RiskBandBoundaries) -> np.ndarray:
    """Applies already-fit boundaries to any split (train, validation,
    benchmark). Boundaries are never refit here -- this function only ever
    reads `boundaries`, it never looks at quantiles of `y_prob` itself."""
    y_prob = np.asarray(y_prob, dtype=float)
    bands = np.full(y_prob.shape, "low", dtype=object)
    bands[y_prob >= boundaries.low_medium_boundary] = "medium"
    bands[y_prob >= boundaries.medium_high_boundary] = "high"
    return bands


def summarize_risk_bands(y_true: np.ndarray, y_prob: np.ndarray, bands: np.ndarray) -> pd.DataFrame:
    """Per-band loan count, observed adverse-outcome rate, average predicted
    probability, and calibration gap (`mean_predicted_probability -
    observed_adverse_outcome_rate`; positive = the model overestimates risk
    in that band, negative = underestimates). Rows are always returned in
    `RISK_BAND_LABELS` order, including bands with zero rows in this split
    (reported with `n=0` and `None` rate/probability/gap fields, rather than
    silently omitted -- a 0-row band is informative, not an error)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    bands = np.asarray(bands, dtype=object)
    if y_true.shape != y_prob.shape or y_true.shape != bands.shape:
        raise RiskBandError(
            "y_true, y_prob, and bands must all have the same shape; got "
            f"{y_true.shape}, {y_prob.shape}, {bands.shape}"
        )

    rows = []
    for band in RISK_BAND_LABELS:
        mask = bands == band
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "risk_band": band,
                    "n": 0,
                    "observed_adverse_outcome_rate": None,
                    "mean_predicted_probability": None,
                    "calibration_gap": None,
                }
            )
            continue
        observed_rate = float(y_true[mask].mean())
        mean_pred_prob = float(y_prob[mask].mean())
        rows.append(
            {
                "risk_band": band,
                "n": n,
                "observed_adverse_outcome_rate": observed_rate,
                "mean_predicted_probability": mean_pred_prob,
                "calibration_gap": mean_pred_prob - observed_rate,
            }
        )
    return pd.DataFrame(rows)
