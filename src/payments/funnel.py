"""Module A, Milestone 1: funnel analysis.

See docs/payments_data_model.md Section 6. Stage `n` ("attempt `n`
reached") = sessions with `attempt_count >= n`. `sessions_reaching_stage`
is monotonically non-increasing in `stage` by construction. Caveat
repeated here from the design doc: this funnel's specific drop-off shape
is an artifact of the synthetic generator's configured sampling
distributions (`configs/payments_synthetic_assumptions.json`), not an
empirical finding about real checkout abandonment.
"""
from __future__ import annotations

import duckdb
import pandas as pd

_FUNNEL_QUERY = """
WITH max_attempts AS (
    SELECT MAX(attempt_count) AS m FROM payment_sessions
),
stages AS (
    SELECT UNNEST(generate_series(1, (SELECT m FROM max_attempts))) AS stage
),
reaching AS (
    SELECT stages.stage, COUNT(*) AS sessions_reaching_stage
    FROM stages
    JOIN payment_sessions ON payment_sessions.attempt_count >= stages.stage
    GROUP BY stages.stage
),
attempt_outcomes AS (
    SELECT attempt_number AS stage,
           SUM(CASE WHEN outcome = 'declined' THEN 1 ELSE 0 END) AS declined_at_stage,
           SUM(CASE WHEN outcome = 'authorized' THEN 1 ELSE 0 END) AS authorized_at_stage
    FROM payment_attempts
    GROUP BY attempt_number
)
SELECT
    stages.stage,
    COALESCE(reaching.sessions_reaching_stage, 0) AS sessions_reaching_stage,
    COALESCE(attempt_outcomes.declined_at_stage, 0) AS declined_at_stage,
    COALESCE(attempt_outcomes.authorized_at_stage, 0) AS authorized_at_stage
FROM stages
LEFT JOIN reaching ON reaching.stage = stages.stage
LEFT JOIN attempt_outcomes ON attempt_outcomes.stage = stages.stage
ORDER BY stages.stage
"""


def compute_funnel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(_FUNNEL_QUERY).df()
    total_sessions = con.execute("SELECT COUNT(*) AS n FROM payment_sessions").df()["n"].iloc[0]
    df["conversion_from_initiated"] = df["sessions_reaching_stage"] / total_sessions if total_sessions else float("nan")
    return df
