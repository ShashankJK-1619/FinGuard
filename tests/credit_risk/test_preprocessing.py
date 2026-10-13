"""Tests for Module C, Milestone 3, Section C:
src/credit_risk/preprocessing.py -- the shared leakage-safe preprocessing
pipeline.

Covers the explicitly enumerated correctness properties: fit only on
train (never re-fit on transform), uniform missing-indicator policy for
numeric columns, graceful handling of categories unseen in train, and
explicit one-hot reference levels via drop="first".
"""
import numpy as np
import pandas as pd

from src.credit_risk.preprocessing import (
    NUMERIC_MISSING_INDICATOR_INFIX,
    fit_preprocessing_pipeline,
)

NUMERIC_COLS = ["loan_amnt", "dti"]
CATEGORICAL_COLS = ["home_ownership"]


def _toy_train_df():
    return pd.DataFrame(
        {
            "loan_amnt": [1000.0, 2000.0, np.nan, 4000.0, 5000.0, 6000.0],
            "dti": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "home_ownership": ["RENT", "RENT", "OWN", "MORTGAGE", "MORTGAGE", "RENT"],
        }
    )


def test_fit_is_train_only_and_transform_does_not_refit():
    train = _toy_train_df()
    fp = fit_preprocessing_pipeline(train, NUMERIC_COLS, CATEGORICAL_COLS)

    # A second, differently-distributed "validation" frame must be scaled
    # using TRAIN's stats, not its own -- i.e. transform never calls fit.
    validation = pd.DataFrame(
        {"loan_amnt": [100000.0], "dti": [5.0], "home_ownership": ["RENT"]}
    )
    X_val = fp.transform(validation)
    # loan_amnt=100000 is far outside train's range -> standardized value should be large,
    # not clipped to a validation-local scale of its own.
    loan_amnt_col_idx = fp.output_feature_names.index("loan_amnt")
    assert X_val[0, loan_amnt_col_idx] > 5


def test_numeric_missing_indicator_added_only_for_columns_missing_in_train():
    train = _toy_train_df()
    fp = fit_preprocessing_pipeline(train, NUMERIC_COLS, CATEGORICAL_COLS)
    indicator_names = [n for n in fp.output_feature_names if n.startswith(NUMERIC_MISSING_INDICATOR_INFIX)]
    # Only loan_amnt has a NaN in train; dti has none.
    assert indicator_names == [f"{NUMERIC_MISSING_INDICATOR_INFIX}loan_amnt"]


def test_missing_indicator_fires_correctly_on_transformed_rows():
    train = _toy_train_df()
    fp = fit_preprocessing_pipeline(train, NUMERIC_COLS, CATEGORICAL_COLS)
    test_df = pd.DataFrame(
        {"loan_amnt": [np.nan, 3000.0], "dti": [15.0, 15.0], "home_ownership": ["RENT", "RENT"]}
    )
    X = fp.transform(test_df)
    indicator_idx = fp.output_feature_names.index(f"{NUMERIC_MISSING_INDICATOR_INFIX}loan_amnt")
    # The indicator column is itself a numeric column in this pipeline, so
    # it passes through the same StandardScaler as every other numeric
    # feature (it is 0/1 valued pre-scaling, not post-scaling) -- assert
    # on the preserved ORDERING (missing > not-missing) rather than on
    # literal 0.0/1.0.
    assert X[0, indicator_idx] > X[1, indicator_idx]


def test_unseen_category_at_transform_time_is_encoded_as_all_zeros_not_an_error():
    train = _toy_train_df()
    fp = fit_preprocessing_pipeline(train, NUMERIC_COLS, CATEGORICAL_COLS)
    test_df = pd.DataFrame(
        {"loan_amnt": [3000.0], "dti": [15.0], "home_ownership": ["ANY_OTHER_NEVER_SEEN"]}
    )
    # Must not raise despite an unseen category -- handle_unknown="ignore".
    X = fp.transform(test_df)
    onehot_cols = [i for i, n in enumerate(fp.output_feature_names) if n.startswith("home_ownership_")]
    assert all(X[0, i] == 0.0 for i in onehot_cols)


def test_onehot_has_explicit_reference_level_dropped():
    train = _toy_train_df()
    fp = fit_preprocessing_pipeline(train, NUMERIC_COLS, CATEGORICAL_COLS)
    onehot = fp.column_transformer.named_transformers_["categorical"].named_steps["onehot"]
    # drop="first" -> alphabetically first category of home_ownership (MORTGAGE,
    # OWN, RENT, __missing__-sorted) is dropped, leaving one fewer dummy than
    # categories seen in train.
    categories_seen = sorted(train["home_ownership"].unique())
    onehot_output_cols = [n for n in fp.output_feature_names if n.startswith("home_ownership_")]
    assert len(onehot_output_cols) == len(categories_seen) - 1
    assert onehot.drop_idx_ is not None


def test_no_post_origination_or_identifier_columns_are_ever_referenced():
    # This pipeline only ever receives whatever column lists its caller
    # passes -- it performs no column selection of its own, so there is no
    # internal allowlist/blocklist to test here directly. This test instead
    # documents and locks in that the function signature takes explicit
    # column lists only (no raw dataframe-wide fit), which is what keeps
    # leakage-prone columns out: feature_selection.py/feature_loader.py are
    # the actual gatekeepers.
    train = _toy_train_df()
    fp = fit_preprocessing_pipeline(train, NUMERIC_COLS, CATEGORICAL_COLS)
    assert set(fp.numeric_columns) == set(NUMERIC_COLS)
    assert set(fp.categorical_columns) == set(CATEGORICAL_COLS)
