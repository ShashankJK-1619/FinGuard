"""Module C, Milestone 3, Section G: logistic regression coefficient
interpretation.

Discipline:
  - Coefficients are reported alongside their odds ratio (`exp(coef)`),
    never presented as a causal effect -- `extract_coefficients`'s output
    and every doc that consumes it must describe them as "associated
    with," never "causes."
  - Every one-hot-encoded categorical feature has an explicit, named
    reference level (`OneHotEncoder(drop="first")` in preprocessing.py);
    `reference_levels` surfaces which level was dropped per categorical
    field, so a coefficient like `purpose_small_business` is always read
    relative to its actual reference (`purpose`'s first category,
    alphabetically -- see `categorical_fields` order in
    configs/credit_risk_model_feature_sets.json), not silently assumed.
  - Numeric coefficients are reported on the STANDARDIZED scale (the
    `StandardScaler` in preprocessing.py) -- this module does not attempt
    to back-transform them to raw units, since doing so correctly would
    require propagating the scaler's per-feature mean/std, and silently
    skipping that step would misstate the coefficient. Any consumer of
    `numeric_feature_scale` can do that back-transform explicitly if
    needed; it is never done implicitly here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.credit_risk.modeling import FittedModel


@dataclass(frozen=True)
class CoefficientTable:
    model_name: str
    table: pd.DataFrame  # columns: feature, coefficient, odds_ratio, abs_coefficient
    reference_levels: dict[str, str]
    intercept: float


def reference_levels_from_onehot(categorical_columns: list[str], onehot: OneHotEncoder) -> dict[str, str]:
    """`OneHotEncoder(drop='first')` drops `categories_[i][0]` for the i-th
    input column -- this returns {categorical_column: dropped_reference_level}."""
    return {
        col: str(categories[0]) for col, categories in zip(categorical_columns, onehot.categories_)
    }


def extract_coefficients(fitted: FittedModel, reference_levels: dict[str, str]) -> CoefficientTable:
    coefs = fitted.model.coef_.ravel()
    if len(coefs) != len(fitted.feature_names):
        raise ValueError(
            f"Model '{fitted.name}' has {len(coefs)} coefficients but "
            f"{len(fitted.feature_names)} feature names were recorded -- cannot attribute "
            "coefficients to features."
        )
    table = pd.DataFrame(
        {
            "feature": fitted.feature_names,
            "coefficient": coefs,
        }
    )
    table["odds_ratio"] = np.exp(table["coefficient"])
    table["abs_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return CoefficientTable(
        model_name=fitted.name,
        table=table,
        reference_levels=reference_levels,
        intercept=float(fitted.model.intercept_[0]),
    )


def top_coefficients(table: CoefficientTable, n: int = 10) -> pd.DataFrame:
    """Returns the n strongest-magnitude coefficients, annotated with sign
    direction in plain language -- "higher default odds" / "lower default
    odds" -- for direct use in the Section G writeup."""
    out = table.table.head(n).copy()
    out["direction"] = np.where(
        out["coefficient"] > 0, "associated with HIGHER default odds", "associated with LOWER default odds"
    )
    return out
