"""Module A, Milestone 1: SQL-first KPI layer.

See docs/payments_data_model.md Section 5 for the KPI dictionary
(definitions + denominators). Every number here is computed by a single
DuckDB SQL query over the `payment_sessions`/`payment_attempts` views
(src.payments.duckdb_views.connect_with_payments_views) — no hand-rolled
pandas groupby/aggregation duplicates this logic.
"""
from __future__ import annotations

import duckdb

_OVERALL_KPI_QUERY = """
WITH session_stats AS (
    SELECT
        COUNT(*) AS total_sessions,
        SUM(CASE WHEN final_outcome = 'authorized' THEN 1 ELSE 0 END) AS authorized_sessions,
        SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END) AS retried_sessions,
        SUM(CASE WHEN attempt_count > 1 AND final_outcome = 'authorized' THEN 1 ELSE 0 END) AS recovered_sessions,
        AVG(attempt_count) AS avg_attempts_per_session
    FROM payment_sessions
),
attempt_stats AS (
    SELECT
        COUNT(*) AS total_attempts,
        SUM(CASE WHEN outcome = 'declined' THEN 1 ELSE 0 END) AS declined_attempts
    FROM payment_attempts
)
SELECT * FROM session_stats, attempt_stats
"""


def compute_overall_kpis(con: duckdb.DuckDBPyConnection) -> dict:
    row = con.execute(_OVERALL_KPI_QUERY).df().iloc[0]
    total_sessions = int(row["total_sessions"])
    retried_sessions = int(row["retried_sessions"])
    authorized_sessions = int(row["authorized_sessions"])
    total_attempts = int(row["total_attempts"])
    declined_attempts = int(row["declined_attempts"])
    recovered_sessions = int(row["recovered_sessions"])

    authorization_rate = authorized_sessions / total_sessions if total_sessions else float("nan")
    return {
        "total_sessions": total_sessions,
        "authorized_sessions": authorized_sessions,
        "retried_sessions": retried_sessions,
        "recovered_sessions": recovered_sessions,
        "total_attempts": total_attempts,
        "declined_attempts": declined_attempts,
        "authorization_rate": authorization_rate,
        "session_failure_rate": 1 - authorization_rate if total_sessions else float("nan"),
        "gross_attempt_decline_rate": declined_attempts / total_attempts if total_attempts else float("nan"),
        "retry_rate": retried_sessions / total_sessions if total_sessions else float("nan"),
        "avg_attempts_per_session": float(row["avg_attempts_per_session"]) if total_sessions else float("nan"),
        "recovery_rate": recovered_sessions / retried_sessions if retried_sessions else float("nan"),
    }
