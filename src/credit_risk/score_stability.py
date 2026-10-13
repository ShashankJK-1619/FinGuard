"""Module C, Milestone 4, Section I: governance/stability diagnostics --
score distribution by vintage, calibration by vintage, adverse-outcome
rate by risk band over time, and a Population Stability Index (PSI)
drift heuristic.

Kept deliberately SEPARATE from the existing, frozen
`src/credit_risk/vintage_analysis.py` (Module C, Milestone 3, Section I),
which already computes per-vintage ROC-AUC/Brier/evaluation-eligibility
tagging and must not be modified by this milestone. This module adds new,
complementary diagnostics rather than extending that one.

PSI drift-magnitude heuristic (`PSI_SIGNIFICANT_CHANGE_THRESHOLD`,
`PSI_MODERATE_CHANGE_THRESHOLD`): the commonly-cited 0.10/0.25 cut points
are an industry rule of thumb, NOT a calibrated hypothesis test with a
known false-positive rate on this dataset. `psi_drift_category` labels a
PSI value against these heuristic bands and says so explicitly in
`PSI_HEURISTIC_CAVEAT` -- nothing downstream should treat a PSI value
crossing one of these lines as a formal statistical finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE_CHANGE_THRESHOLD = 0.10
PSI_SIGNIFICANT_CHANGE_THRESHOLD = 0.25

PSI_HEURISTIC_CAVEAT = (
    "PSI drift-magnitude bands (0.10 / 0.25) are a commonly-cited industry "
    "rule of thumb, not a calibrated statistical hypothesis test on this "
    "dataset. A PSI value crossing one of these lines is a heuristic "
    "prompt to look closer, not a formal significance result."
)


class ScoreStabilityError(ValueError):
    pass


def compute_psi(reference: np.ndarray, comparison: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between a `reference` score distribution
    (e.g. train, or an earlier vintage year) and a `comparison` score
    distribution (e.g. validation/benchmark, or a later vintage year).

    Bin edges are quantiles of `reference` ONLY (mirroring
    `risk_bands.py`'s train-only-fit discipline -- the reference
    distribution defines the bins, `comparison` is only ever bucketed into
    already-fit edges, never used to redraw them). Edge bins are extended
    to +/-inf so every `comparison` value lands in some bin even if it
    falls outside `reference`'s observed range.

    `PSI = sum((comparison_pct - reference_pct) * ln(comparison_pct / reference_pct))`
    over bins, using a small `eps` floor on both percentages to avoid
    `log(0)`/division-by-zero when a bin is empty in one distribution.
    """
    reference = np.asarray(reference, dtype=float)
    comparison = np.asarray(comparison, dtype=float)
    if reference.size == 0 or comparison.size == 0:
        raise ScoreStabilityError("reference and comparison must both be non-empty")
    if n_bins < 2:
        raise ScoreStabilityError(f"n_bins must be >= 2; got {n_bins}")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        raise ScoreStabilityError(
            "reference distribution has too few distinct quantile edges to form "
            f"{n_bins} bins (likely too many ties); reduce n_bins or inspect reference"
        )
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    comp_counts, _ = np.histogram(comparison, bins=edges)

    eps = 1e-6
    ref_pct = np.clip(ref_counts / reference.size, eps, None)
    comp_pct = np.clip(comp_counts / comparison.size, eps, None)

    psi = float(np.sum((comp_pct - ref_pct) * np.log(comp_pct / ref_pct)))
    return psi


def psi_drift_category(psi: float) -> str:
    """Labels a PSI value against the heuristic bands documented in
    `PSI_HEURISTIC_CAVEAT`. Returns one of `"no_significant_change"`,
    `"moderate_change"`, `"significant_change"`."""
    if psi >= PSI_SIGNIFICANT_CHANGE_THRESHOLD:
        return "significant_change"
    if psi >= PSI_MODERATE_CHANGE_THRESHOLD:
        return "moderate_change"
    return "no_significant_change"


def score_distribution_by_vintage(origination_year: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Per-origination-year descriptive summary of a model's predicted
    probabilities: n, mean, std, and the 10th/50th/90th percentiles.
    Purely descriptive -- no significance testing."""
    origination_year = np.asarray(origination_year)
    y_prob = np.asarray(y_prob, dtype=float)
    if origination_year.shape != y_prob.shape:
        raise ScoreStabilityError(
            f"origination_year and y_prob must have the same shape; got "
            f"{origination_year.shape} vs {y_prob.shape}"
        )
    df = pd.DataFrame({"origination_year": origination_year, "y_prob": y_prob})
    rows = []
    for year, group in df.groupby("origination_year"):
        rows.append(
            {
                "origination_year": year,
                "n": int(len(group)),
                "mean_predicted_probability": float(group["y_prob"].mean()),
                "std_predicted_probability": float(group["y_prob"].std(ddof=0)),
                "p10_predicted_probability": float(group["y_prob"].quantile(0.10)),
                "p50_predicted_probability": float(group["y_prob"].quantile(0.50)),
                "p90_predicted_probability": float(group["y_prob"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("origination_year").reset_index(drop=True)


def calibration_by_vintage(
    origination_year: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray
) -> pd.DataFrame:
    """Per-origination-year n, observed adverse-outcome rate, mean
    predicted probability, and calibration gap (`mean_predicted_probability
    - observed_adverse_outcome_rate`), mirroring `risk_bands.py`'s
    calibration-gap convention but grouped by vintage year instead of risk
    band."""
    origination_year = np.asarray(origination_year)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if not (origination_year.shape == y_true.shape == y_prob.shape):
        raise ScoreStabilityError(
            "origination_year, y_true, and y_prob must all have the same shape; got "
            f"{origination_year.shape}, {y_true.shape}, {y_prob.shape}"
        )
    df = pd.DataFrame({"origination_year": origination_year, "y_true": y_true, "y_prob": y_prob})
    rows = []
    for year, group in df.groupby("origination_year"):
        observed_rate = float(group["y_true"].mean())
        mean_pred_prob = float(group["y_prob"].mean())
        rows.append(
            {
                "origination_year": year,
                "n": int(len(group)),
                "observed_adverse_outcome_rate": observed_rate,
                "mean_predicted_probability": mean_pred_prob,
                "calibration_gap": mean_pred_prob - observed_rate,
            }
        )
    return pd.DataFrame(rows).sort_values("origination_year").reset_index(drop=True)


def adverse_outcome_rate_by_band_over_time(
    origination_year: np.ndarray, risk_band: np.ndarray, y_true: np.ndarray
) -> pd.DataFrame:
    """Per (origination_year, risk_band) cell: n and observed
    adverse-outcome rate. A cell with zero rows for a given
    year/band combination is simply absent from the output (unlike
    `risk_bands.summarize_risk_bands`, which always reports all three
    bands for a single split) -- the number of distinct years is not
    predeclared, so there is no fixed set of "expected" cells to fill in
    with explicit zeros."""
    origination_year = np.asarray(origination_year)
    risk_band = np.asarray(risk_band, dtype=object)
    y_true = np.asarray(y_true).astype(int)
    if not (origination_year.shape == risk_band.shape == y_true.shape):
        raise ScoreStabilityError(
            "origination_year, risk_band, and y_true must all have the same shape; got "
            f"{origination_year.shape}, {risk_band.shape}, {y_true.shape}"
        )
    df = pd.DataFrame({"origination_year": origination_year, "risk_band": risk_band, "y_true": y_true})
    grouped = df.groupby(["origination_year", "risk_band"])["y_true"].agg(["count", "mean"])
    grouped = grouped.rename(columns={"count": "n", "mean": "observed_adverse_outcome_rate"}).reset_index()
    return grouped.sort_values(["origination_year", "risk_band"]).reset_index(drop=True)
