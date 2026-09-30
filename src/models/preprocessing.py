"""Milestone 3: preprocessing pipeline for the interpretable baseline.

Builds a single sklearn ColumnTransformer implementing the column-role
policy documented in src/models/schema.py:

- Numeric (masked codes, continuous, drift/trend, frequency/recency):
  median-impute, then standard-scale. `add_indicator=True` gives
  seconds_since_last_card1/addr1 an explicit missing-indicator column
  rather than silently treating "no prior transaction" as an averaged-out
  gap.
- Categorical: impute missing as an explicit "missing" level (not
  dropped, not silently merged into an existing category), then one-hot
  with `handle_unknown="ignore"` so a category never seen in training
  (e.g. a rare card6 value appearing only in validation/holdout) produces
  an all-zero row instead of raising or leaking information about the
  unseen category's identity.
- Binary 0/1 flags: passed through unscaled, so each coefficient is
  directly a log-odds effect of "flag on vs. off" rather than an
  arbitrary standardized unit.

`remainder="drop"` is explicit and load-bearing: it guarantees columns not
named in NUMERIC_FEATURES/CATEGORICAL_FEATURES/BINARY_FEATURES — including
TransactionID, isFraud, TransactionDT, and any masked column — can never
reach the model even if accidentally left in the input frame.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.schema import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    binary_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="most_frequent"))]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
    )
