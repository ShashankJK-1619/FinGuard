"""Module A, Milestone 2: SQL-first primary/secondary experiment metrics.

See docs/payments_experiment_metric_dictionary.md for the full metric
dictionary (numerator, denominator, grain, interpretation, direction,
tier). All metrics here are computed per arm (`assignment`) by a single
DuckDB SQL query over `payment_experiment_sessions` joined to
`payment_attempts` for the first-attempt-outcome metric — no hand-rolled
pandas groupby duplicates this logic (same convention as Milestone 1's
kpi.py/funnel.py/etc).

Primary metric: payment_completion_rate (the ONLY metric the primary
statistical test in experiment_stats.py is run on).
Secondary metrics: first_attempt_authorization_rate, retry_rate,
retry_recovery_rate, attempts_per_completed_session — reported
descriptively alongside the primary metric, not used for the ship/no-ship
decision rule.
"""
from __future__ import annotations

import duckdb
import pandas as pd

_PRIMARY_AND_SECONDARY_QUERY = """
WITH first_attempt AS (
    SELECT session_id, outcome AS first_attempt_outcome
    FROM payment_attempts
    WHERE attempt_number = 1
),
joined AS (
    SELECT s.*, fa.first_attempt_outcome
    FROM payment_experiment_sessions s
    LEFT JOIN first_attempt fa ON s.session_id = fa.session_id
)
SELECT
    assignment,
    COUNT(*) AS n_sessions,
    SUM(CASE WHEN experiment_final_outcome = 'authorized' THEN 1 ELSE 0 END) AS n_completed,
    AVG(CASE WHEN experiment_final_outcome = 'authorized' THEN 1.0 ELSE 0.0 END) AS payment_completion_rate,
    AVG(CASE WHEN first_attempt_outcome = 'authorized' THEN 1.0 ELSE 0.0 END) AS first_attempt_authorization_rate,
    AVG(CASE WHEN attempt_count > 1 THEN 1.0 ELSE 0.0 END) AS retry_rate,
    SUM(CASE WHEN attempt_count > 1 AND experiment_final_outcome = 'authorized' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END), 0) AS retry_recovery_rate,
    AVG(CASE WHEN experiment_final_outcome = 'authorized' THEN attempt_count ELSE NULL END)
        AS attempts_per_completed_session
FROM joined
GROUP BY assignment
ORDER BY assignment
"""


def compute_primary_and_secondary_metrics(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per arm ('control', 'treatment'). Requires both
    `payment_experiment_sessions` and `payment_attempts` to be registered
    (src.payments.duckdb_views)."""
    return con.execute(_PRIMARY_AND_SECONDARY_QUERY).df()
