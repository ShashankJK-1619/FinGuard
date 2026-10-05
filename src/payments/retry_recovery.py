"""Module A, Milestone 1: retry/recovery analysis.

See docs/payments_data_model.md Section 7. Sessions are bucketed by
number of PRIOR declines before resolution (`attempt_count - 1`, capped
at "4+"). Recovery is deterministic given `is_synthetic_abandonment` in
this generator (every real-linked session's last attempt is
`authorized`; every abandonment session never reaches `authorized`) —
so this table's real content is the MIX of real vs. abandonment sessions
at each retry depth, not an independently modeled recovery probability.
Stated explicitly per the design doc, not left implicit.
"""
from __future__ import annotations

import duckdb
import pandas as pd

_RETRY_RECOVERY_QUERY = """
WITH session_last_attempt AS (
    SELECT session_id, MAX(attempt_offset_seconds) AS last_attempt_offset
    FROM payment_attempts
    GROUP BY session_id
),
session_depth AS (
    SELECT
        s.session_id,
        s.final_outcome,
        la.last_attempt_offset - s.session_start_offset_seconds AS time_to_resolution_seconds,
        CASE WHEN s.attempt_count - 1 >= 4 THEN '4+' ELSE CAST(s.attempt_count - 1 AS VARCHAR) END AS prior_declines_bucket
    FROM payment_sessions s
    JOIN session_last_attempt la ON la.session_id = s.session_id
)
SELECT
    prior_declines_bucket,
    COUNT(*) AS session_count,
    SUM(CASE WHEN final_outcome = 'authorized' THEN 1 ELSE 0 END) AS recovered_count,
    AVG(time_to_resolution_seconds) AS mean_time_to_resolution_seconds,
    MEDIAN(time_to_resolution_seconds) AS median_time_to_resolution_seconds
FROM session_depth
GROUP BY prior_declines_bucket
ORDER BY prior_declines_bucket
"""


def compute_retry_recovery(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(_RETRY_RECOVERY_QUERY).df()
    df["recovery_rate"] = df["recovered_count"] / df["session_count"]
    return df
