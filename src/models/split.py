"""Milestone 3: chronological train/validation/holdout split.

Fraud behavior drifts over time (docs/time_structure.md). A random split
lets the model see transactions from the same time window it's evaluated
on, inflating offline metrics relative to real deployment, where the model
only ever sees the future. This module builds the split Milestone 1
flagged as necessary but deferred.

Split boundaries fall on whole-day boundaries (day = TransactionDT //
86400), not an arbitrary row index, so no single calendar day has some
transactions in train and others in validation/holdout — a day belongs
entirely to one split. Within-day ordering ties are irrelevant here (no
row-level assignment happens inside a day), unlike the row-level tie-break
in src/features/velocity_features.py.

Uses only train_transaction (+ train_identity via the Milestone 2 baseline
table). test_transaction/test_identity are never loaded anywhere in this
project and are never used as labeled evaluation data — the holdout below
is a held-out slice of the same labeled training data, not the Kaggle
test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.schema import TARGET_COL, TIME_COL

SECONDS_PER_DAY = 86_400


class SplitError(ValueError):
    pass


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    time_col: str = TIME_COL,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Splits df into (train, val, holdout, report) by whole-day boundaries.

    train_frac/val_frac are targets, not guarantees: the actual boundary is
    snapped to the nearest whole day, so realized fractions will differ
    slightly (reported exactly in `report`). The remaining
    1 - train_frac - val_frac goes to holdout.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise SplitError(f"Invalid fractions: train_frac={train_frac}, val_frac={val_frac}")

    day = df[time_col] // SECONDS_PER_DAY
    counts = day.value_counts().sort_index()
    days_sorted = counts.index.to_numpy()
    cum = counts.cumsum().to_numpy()
    total = len(df)

    train_target = total * train_frac
    val_target = total * (train_frac + val_frac)

    train_idx = int(np.searchsorted(cum, train_target, side="left"))
    val_idx = int(np.searchsorted(cum, val_target, side="left"))
    train_idx = min(train_idx, len(days_sorted) - 1)
    val_idx = min(val_idx, len(days_sorted) - 1)

    if val_idx <= train_idx:
        raise SplitError(
            "Validation boundary collapsed onto the train boundary — "
            "train_frac/val_frac are too fine-grained for this day granularity."
        )

    train_last_day = int(days_sorted[train_idx])
    val_last_day = int(days_sorted[val_idx])

    train_mask = day <= train_last_day
    val_mask = (day > train_last_day) & (day <= val_last_day)
    holdout_mask = day > val_last_day

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    holdout_df = df.loc[holdout_mask].copy()

    def _segment_stats(seg: pd.DataFrame) -> dict:
        n = len(seg)
        n_fraud = int(seg[target_col].sum())
        dt = seg[time_col]
        return {
            "n_rows": n,
            "n_fraud": n_fraud,
            "fraud_rate": n_fraud / n if n else float("nan"),
            "dt_min": int(dt.min()) if n else None,
            "dt_max": int(dt.max()) if n else None,
            "day_min": int(dt.min() // SECONDS_PER_DAY) if n else None,
            "day_max": int(dt.max() // SECONDS_PER_DAY) if n else None,
        }

    report = {
        "train_frac_target": train_frac,
        "val_frac_target": val_frac,
        "train_last_day": train_last_day,
        "val_last_day": val_last_day,
        "total_days": int(days_sorted.max() - days_sorted.min() + 1),
        "train": _segment_stats(train_df),
        "val": _segment_stats(val_df),
        "holdout": _segment_stats(holdout_df),
    }

    return train_df, val_df, holdout_df, report
