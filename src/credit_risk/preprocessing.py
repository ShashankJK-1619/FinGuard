"""Module C, Milestone 3, Section C: a reusable, leakage-safe preprocessing
pipeline shared by Model A and Model B.

Leakage-safety discipline enforced here:
  - The returned `ColumnTransformer` is a *specification*; it must be
    `.fit()` on the TRAIN split only (`fit_preprocessing_pipeline`), then
    `.transform()`-ed (never re-fit) on validation/benchmark. This mirrors
    the project's existing train-only-fit discipline
    (`chronological_split.py`'s whole reason for existing).
  - Numeric columns: `SimpleImputer(strategy="median", add_indicator=True)`
    then `StandardScaler`. The median is computed on train only;
    `add_indicator=True` emits one extra binary "was this value missing"
    column per numeric feature with any missingness in train, so columns
    like `mths_since_last_delinq` (NULL means "no delinquency ever
    recorded", not random missingness -- see
    configs/credit_risk_model_feature_sets.json) keep that information
    instead of having it silently erased by imputation. This is a single,
    uniform rule applied to every numeric column, not a special case
    hand-picked for any one field.
  - Categorical columns: `SimpleImputer(strategy="constant",
    fill_value="__missing__")` then `OneHotEncoder(handle_unknown="ignore",
    drop="first")`. `handle_unknown="ignore"` is the graceful-unseen-
    category handling the milestone spec calls for -- a category value
    present in validation/benchmark but never seen in train is encoded as
    all-zeros for that field's dummies rather than raising or silently
    crashing. `drop="first"` gives every categorical feature an explicit
    reference level (needed for Section G's coefficient interpretation).
  - No post-origination field and no identifier is ever threaded through
    this module -- it only ever receives the column lists
    `feature_loader.feature_matrix_columns` derives from an already
    eligibility-checked `FeatureSet` (feature_selection.py); this module
    does not re-validate that itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_MISSING_INDICATOR_INFIX = "missingindicator_"


@dataclass(frozen=True)
class FittedPreprocessor:
    """A fitted ColumnTransformer plus the resolved output feature names,
    in the same order as the transformed array's columns -- sklearn's
    ColumnTransformer does not hand these back in one simple call across
    all the scikit-learn versions this project may run under, so this
    wrapper resolves them once, explicitly, at fit time."""

    column_transformer: ColumnTransformer
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    output_feature_names: tuple[str, ...]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.column_transformer.transform(df)


def build_preprocessing_pipeline(
    numeric_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def _numeric_output_names(numeric_columns: list[str], imputer: SimpleImputer) -> list[str]:
    names = list(numeric_columns)
    # SimpleImputer(add_indicator=True) appends one missing-indicator column
    # per input column that had >=1 missing value IN THE FIT DATA (train),
    # in the order those columns appear in `indicator_.features_`.
    indicator = imputer.indicator_
    if indicator is not None:
        for idx in indicator.features_:
            names.append(f"{NUMERIC_MISSING_INDICATOR_INFIX}{numeric_columns[idx]}")
    return names


def _categorical_output_names(categorical_columns: list[str], onehot: OneHotEncoder) -> list[str]:
    return list(onehot.get_feature_names_out(categorical_columns))


def fit_preprocessing_pipeline(
    df_train: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]
) -> FittedPreprocessor:
    """Fits the pipeline on `df_train` ONLY. Callers must never pass
    validation/benchmark data here -- use `FittedPreprocessor.transform`
    for those."""
    ct = build_preprocessing_pipeline(numeric_columns, categorical_columns)
    ct.fit(df_train[numeric_columns + categorical_columns])

    numeric_imputer: SimpleImputer = ct.named_transformers_["numeric"].named_steps["impute"]
    categorical_onehot: OneHotEncoder = ct.named_transformers_["categorical"].named_steps["onehot"]

    numeric_names = _numeric_output_names(numeric_columns, numeric_imputer)
    categorical_names = _categorical_output_names(categorical_columns, categorical_onehot)

    return FittedPreprocessor(
        column_transformer=ct,
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        output_feature_names=tuple(numeric_names + categorical_names),
    )
