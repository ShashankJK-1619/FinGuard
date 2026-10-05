"""Shared DuckDB registration helper for the SQL-first analytics layer
(kpi.py, funnel.py, retry_recovery.py, segmentation.py, guardrails.py).

Every analytics module in src/payments takes a DuckDB connection with
`payment_sessions`/`payment_attempts` already registered as views over the
in-memory pandas DataFrames (via `duckdb.DuckDBPyConnection.register`, no
copy) and expresses its logic as SQL — the same "query in place, don't
hand-roll pandas aggregation" preference `src.data.loaders` already
established for the raw CSVs, now applied to the synthetic tables too.
"""
from __future__ import annotations

import duckdb
import pandas as pd


def connect_with_payments_views(sessions_df: pd.DataFrame, attempts_df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.register("payment_sessions", sessions_df)
    con.register("payment_attempts", attempts_df)
    return con
