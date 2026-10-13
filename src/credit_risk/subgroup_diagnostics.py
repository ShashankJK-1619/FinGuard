"""Module C, Milestone 3, Section J: subgroup/governance diagnostics.

Explicitly NOT a demographic-lending-optimization exercise -- no race,
sex, age, or other protected-class field exists anywhere in the real
LendingClub file to begin with, so none is used here. This module
instead reports missingness, score distribution, calibration, and error
rates across a small, predeclared set of non-protected, loan/application-
structural fields:

  - `home_ownership` (RENT/MORTGAGE/OWN/...) -- a loan-structural fact
    about the borrower's housing arrangement, directly relevant to
    underwriting, not a protected class.
  - `verification_status` (Verified/Source Verified/Not Verified) -- a
    fact about the application process itself.
  - `application_type` (Individual/Joint App) -- a fact about how the
    loan was applied for.
  - `purpose` (debt_consolidation/credit_card/...) -- the borrower's
    stated reason for the loan.

`addr_state`/`zip_code` are deliberately EXCLUDED from this diagnostic
even though they are otherwise-eligible modeling features (Model A/B) --
geography is a well-documented proxy for protected classes (race/
ethnicity), and a defensible non-demographic governance diagnostic must
not reach for a geographic grouping just because it is convenient. This
module's purpose is purely descriptive monitoring (are any of these
structural segments poorly served by missingness, miscalibration, or
lopsided error rates?), never a recommendation to adjust underwriting
by segment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

NON_PROTECTED_SUBGROUP_FIELDS = (
    "home_ownership",
    "verification_status",
    "application_type",
    "purpose",
)


class SubgroupDiagnosticsError(ValueError):
    pass


def row_missingness_rate(df: pd.DataFrame, numeric_columns: list[str]) -> pd.Series:
    """Fraction of `numeric_columns` that are NULL for each row -- a
    single, simple per-row informativeness proxy used only for this
    descriptive diagnostic, never fed to a model."""
    if not numeric_columns:
        raise SubgroupDiagnosticsError("numeric_columns must not be empty")
    return df[numeric_columns].isna().mean(axis=1)


def compute_subgroup_diagnostics(
    df: pd.DataFrame,
    group_column: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_value: float,
    missingness_rate: pd.Series,
) -> pd.DataFrame:
    if group_column not in NON_PROTECTED_SUBGROUP_FIELDS:
        raise SubgroupDiagnosticsError(
            f"group_column={group_column!r} is not in the predeclared, non-protected "
            f"NON_PROTECTED_SUBGROUP_FIELDS={NON_PROTECTED_SUBGROUP_FIELDS} -- add it there "
            "explicitly, with a documented rationale, before using it as a diagnostic grouping."
        )
    work = pd.DataFrame(
        {
            "group": df[group_column].fillna("__missing__"),
            "y_true": np.asarray(y_true).astype(int),
            "y_prob": np.asarray(y_prob).astype(float),
            "missingness_rate": missingness_rate.to_numpy(),
        }
    )
    work["y_pred"] = (work["y_prob"] >= threshold_value).astype(int)

    rows = []
    for group_value, sub in work.groupby("group"):
        n = len(sub)
        actual_pos = sub["y_true"] == 1
        actual_neg = sub["y_true"] == 0
        n_pos = int(actual_pos.sum())
        n_neg = int(actual_neg.sum())
        fpr = float((sub.loc[actual_neg, "y_pred"] == 1).mean()) if n_neg else None
        fnr = float((sub.loc[actual_pos, "y_pred"] == 0).mean()) if n_pos else None
        rows.append(
            {
                "group_column": group_column,
                "group_value": group_value,
                "n": n,
                "mean_feature_missingness_pct": round(float(sub["missingness_rate"].mean() * 100), 3),
                "observed_default_rate_pct": round(float(sub["y_true"].mean() * 100), 3),
                "mean_predicted_probability_pct": round(float(sub["y_prob"].mean() * 100), 3),
                "calibration_gap_pct_points": round(
                    float((sub["y_prob"].mean() - sub["y_true"].mean()) * 100), 3
                ),
                "brier_score": round(float(brier_score_loss(sub["y_true"], sub["y_prob"])), 5)
                if sub["y_true"].nunique() > 0
                else None,
                "false_positive_rate": round(fpr, 4) if fpr is not None else None,
                "false_negative_rate": round(fnr, 4) if fnr is not None else None,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
