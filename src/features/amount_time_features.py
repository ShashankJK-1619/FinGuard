"""Milestone 2: row-level, point-in-time-safe features from TransactionAmt
and TransactionDT.

Every function here is a pure function of a single row's own columns. None
of them look at any other row, so none of them can leak future information
by construction — the only "point-in-time safety" question is whether the
*input* column itself (TransactionAmt, TransactionDT) is known at
transaction time, which it is (both are transaction-time attributes, not
outcomes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
DAYS_PER_WEEK = 7


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds TransactionAmt_log1p = log1p(TransactionAmt).

    log1p rather than log: TransactionAmt can be a small positive value
    close to 0 (observed min in Milestone 1: 0.251), never exactly 0 or
    negative (verified by Milestone 1's non-negativity test), so log1p is
    a safety margin against a future 0-amount row, not a response to one
    observed here.
    """
    out = df.copy()
    out["TransactionAmt_log1p"] = np.log1p(out["TransactionAmt"])
    return out


def add_time_features(df: pd.DataFrame, dt_col: str = "TransactionDT") -> pd.DataFrame:
    """Adds relative day/week/hour/day-of-week features from TransactionDT.

    TransactionDT is documented as a relative offset in seconds from an
    undisclosed reference point (confirmed structurally in Milestone 1,
    docs/time_structure.md). All features below are deterministic bucketing
    of that offset:

    - day_index, week_index: ordinal elapsed-time buckets. Valid as relative
      ordering/velocity inputs regardless of calendar alignment.
    - hour_of_day (0-23), hour_sin/hour_cos: TransactionDT % 86400 is a
      *consistent* cyclical position for every row, because every row shares
      the same unknown reference point. It will still capture real
      time-of-day patterns (e.g. nighttime fraud spikes) even though we
      cannot label which bucket is "3am" — that mapping is unknown and not
      claimed here.
    - day_of_week_index (0-6): day_index % 7, same phase-unknown-but-
      consistent reasoning as hour_of_day. NOT claimed to align with real
      Mon-Sun weekdays.
    """
    out = df.copy()
    dt = out[dt_col]

    out["day_index"] = dt // SECONDS_PER_DAY
    out["week_index"] = out["day_index"] // DAYS_PER_WEEK
    out["hour_of_day"] = (dt % SECONDS_PER_DAY) // SECONDS_PER_HOUR
    out["hour_sin"] = np.sin(2 * np.pi * out["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_of_day"] / 24)
    out["day_of_week_index"] = out["day_index"] % DAYS_PER_WEEK

    return out
