"""Leakage-focused tests for the chronological split.

The core property under test: no row in validation has a TransactionDT
earlier than any row in train, and no row in holdout has a TransactionDT
earlier than any row in validation — and no single day is split across
two segments.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.models.split import SECONDS_PER_DAY, SplitError, chronological_split, chronological_split_two_way


def _toy_df(n_days: int = 20, rows_per_day: int = 10, fraud_every: int = 25):
    rows = []
    txn_id = 0
    for day in range(n_days):
        for _ in range(rows_per_day):
            txn_id += 1
            rows.append(
                {
                    "TransactionID": txn_id,
                    "TransactionDT": day * SECONDS_PER_DAY + (txn_id % SECONDS_PER_DAY),
                    "isFraud": 1 if txn_id % fraud_every == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_split_covers_every_row_exactly_once():
    df = _toy_df()
    train_df, val_df, holdout_df, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    assert len(train_df) + len(val_df) + len(holdout_df) == len(df)
    all_ids = set(train_df["TransactionID"]) | set(val_df["TransactionID"]) | set(holdout_df["TransactionID"])
    assert all_ids == set(df["TransactionID"])


def test_split_is_strictly_chronological_across_segment_boundaries():
    df = _toy_df()
    train_df, val_df, holdout_df, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    assert train_df["TransactionDT"].max() < val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() < holdout_df["TransactionDT"].min()


def test_no_calendar_day_is_split_across_two_segments():
    df = _toy_df()
    train_df, val_df, holdout_df, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    train_days = set(train_df["TransactionDT"] // SECONDS_PER_DAY)
    val_days = set(val_df["TransactionDT"] // SECONDS_PER_DAY)
    holdout_days = set(holdout_df["TransactionDT"] // SECONDS_PER_DAY)
    assert train_days & val_days == set()
    assert val_days & holdout_days == set()
    assert train_days & holdout_days == set()


def test_report_fraud_rate_matches_manual_computation():
    df = _toy_df()
    _, _, _, report = chronological_split(df, train_frac=0.6, val_frac=0.2)
    for segment in ("train", "val", "holdout"):
        n = report[segment]["n_rows"]
        n_fraud = report[segment]["n_fraud"]
        assert report[segment]["fraud_rate"] == pytest.approx(n_fraud / n)


def test_report_dt_ranges_are_consistent_with_actual_rows():
    df = _toy_df()
    train_df, val_df, holdout_df, report = chronological_split(df, train_frac=0.6, val_frac=0.2)
    assert report["train"]["dt_min"] == train_df["TransactionDT"].min()
    assert report["train"]["dt_max"] == train_df["TransactionDT"].max()
    assert report["holdout"]["dt_max"] == holdout_df["TransactionDT"].max()


def test_invalid_fractions_raise():
    df = _toy_df()
    with pytest.raises(SplitError):
        chronological_split(df, train_frac=0.9, val_frac=0.3)
    with pytest.raises(SplitError):
        chronological_split(df, train_frac=0.0, val_frac=0.2)


def test_shuffled_input_row_order_produces_the_same_split():
    df = _toy_df()
    shuffled = df.sample(frac=1.0, random_state=3).reset_index(drop=True)

    a_train, a_val, a_holdout, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    b_train, b_val, b_holdout, _ = chronological_split(shuffled, train_frac=0.6, val_frac=0.2)

    assert set(a_train["TransactionID"]) == set(b_train["TransactionID"])
    assert set(a_val["TransactionID"]) == set(b_val["TransactionID"])
    assert set(a_holdout["TransactionID"]) == set(b_holdout["TransactionID"])


def test_two_way_split_covers_every_row_exactly_once_and_is_chronological():
    df = _toy_df()
    _, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    first_df, second_df, _ = chronological_split_two_way(val_df, first_frac=0.5)

    assert len(first_df) + len(second_df) == len(val_df)
    assert set(first_df["TransactionID"]) | set(second_df["TransactionID"]) == set(val_df["TransactionID"])
    assert first_df["TransactionDT"].max() < second_df["TransactionDT"].min()


def test_two_way_split_never_splits_a_calendar_day():
    df = _toy_df()
    _, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    first_df, second_df, _ = chronological_split_two_way(val_df, first_frac=0.5)

    first_days = set(first_df["TransactionDT"] // SECONDS_PER_DAY)
    second_days = set(second_df["TransactionDT"] // SECONDS_PER_DAY)
    assert first_days & second_days == set()


def test_two_way_split_rejects_out_of_range_frac():
    df = _toy_df()
    _, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    with pytest.raises(SplitError):
        chronological_split_two_way(val_df, first_frac=0.0)
    with pytest.raises(SplitError):
        chronological_split_two_way(val_df, first_frac=1.0)


def test_two_way_split_report_matches_actual_rows():
    df = _toy_df()
    _, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    first_df, second_df, report = chronological_split_two_way(val_df, first_frac=0.5)

    assert report["first"]["n_rows"] == len(first_df)
    assert report["second"]["n_rows"] == len(second_df)
    assert report["first"]["n_fraud"] == int(first_df["isFraud"].sum())
