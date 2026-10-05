"""Shared DuckDB registration helper for the SQL-first analytics layer
(kpi.py, funnel.py, retry_recovery.py, segmentation.py, guardrails.py, and
Milestone 2's experiment_metrics.py / experiment_guardrails.py /
experiment_segments.py / experiment_diagnostics.py).

Every analytics module in src/payments takes a DuckDB connection with the
relevant views already registered over the in-memory pandas DataFrames
(via `duckdb.DuckDBPyConnection.register`, no copy) and expresses its
logic as SQL — the same "query in place, don't hand-roll pandas
groupby/aggregation" preference `src.data.loaders` already established
for the raw CSVs, now applied to the synthetic tables too.
"""
from __future__ import annotations

import duckdb
import pandas as pd


def connect_with_payments_views(sessions_df: pd.DataFrame, attempts_df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.register("payment_sessions", sessions_df)
    con.register("payment_attempts", attempts_df)
    return con


def register_experiment_sessions_view(con: duckdb.DuckDBPyConnection, experiment_sessions_df: pd.DataFrame) -> None:
    """Registers the Milestone 2 experiment table (Milestone 1
    `payment_sessions` + `assignment`/`retry_eligible`/
    `treatment_effect_applied`/`experiment_final_outcome`, see
    src.payments.schema.EXPERIMENT_SESSION_COLUMNS) as its own view,
    `payment_experiment_sessions`, on an already-connected connection
    (typically one returned by `connect_with_payments_views`, so
    `payment_attempts` is also available for the joins
    experiment_metrics.py needs)."""
    con.register("payment_experiment_sessions", experiment_sessions_df)
