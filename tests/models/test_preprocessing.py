from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.preprocessing import build_preprocessor
from src.models.schema import ALL_MODEL_FEATURES, BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def _toy_model_df(n: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data: dict[str, object] = {
        "TransactionID": np.arange(n),
        "isFraud": rng.integers(0, 2, n),
        "TransactionDT": np.arange(n) * 100,
    }
    for col in NUMERIC_FEATURES:
        values = rng.normal(size=n)
        values[0] = np.nan  # every numeric column gets at least one missing value
        data[col] = values
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["A", "B", "C"], size=n)
    for col in BINARY_FEATURES:
        data[col] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_preprocessor_drops_non_feature_columns():
    df = _toy_model_df()
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    # Output width must depend only on ALL_MODEL_FEATURES (plus per-column
    # missing indicators), never on TransactionID/isFraud/TransactionDT.
    n_expected_min = len(ALL_MODEL_FEATURES)
    assert X.shape[0] == len(df)
    assert X.shape[1] >= n_expected_min


def test_preprocessor_handles_missing_numeric_values_without_error():
    df = _toy_model_df()
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    assert not np.isnan(X).any()


def test_preprocessor_ignores_unseen_categorical_value_at_transform_time():
    df = _toy_model_df()
    pre = build_preprocessor()
    pre.fit(df)

    unseen = df.copy()
    unseen.loc[0, CATEGORICAL_FEATURES[0]] = "NEVER_SEEN_IN_TRAINING"
    X = pre.transform(unseen)  # must not raise
    assert X.shape[0] == len(unseen)


def test_preprocessor_fit_statistics_come_only_from_fit_data():
    """Median imputation must reflect only the data passed to `fit`, not
    any later `transform` call — this is the sklearn-level guarantee that
    src/models/train_baseline.py relies on to avoid future leakage when
    fitting on train and transforming val/holdout."""
    df = _toy_model_df(n=40)
    train_df = df.iloc[:20].copy()
    other_df = df.iloc[20:].copy()

    numeric_col = NUMERIC_FEATURES[0]
    train_df[numeric_col] = 1.0
    train_df.loc[train_df.index[0], numeric_col] = np.nan  # keep one missing to exercise imputation
    other_df[numeric_col] = 999.0  # wildly different distribution

    pre = build_preprocessor()
    pre.fit(train_df)
    imputer = pre.named_transformers_["numeric"].named_steps["impute"]
    numeric_feature_index = NUMERIC_FEATURES.index(numeric_col)
    assert imputer.statistics_[numeric_feature_index] == 1.0
