"""Module A, Milestone 1: segmentation.

See docs/payments_data_model.md Section 8. Slices the KPI dictionary
(Section 5) by real, documented-category fields only — `ProductCD`,
`card6`, `DeviceType`, and a `TransactionAmt` quartile bucket — so
segmentation is always defined on real fields even for rows that are
synthetic sessions (whose categorical/amount values are themselves
cloned from a real donor row, per Section 4).
"""
from __future__ import annotations

import duckdb
import pandas as pd

# Allow-list, not an arbitrary f-string column: these are exactly the real
# categorical columns carried onto payment_sessions (src.payments.schema).
ALLOWED_SEGMENT_COLUMNS = ["ProductCD", "card6", "DeviceType"]

_SEGMENT_QUERY_TEMPLATE = """
SELECT
    {segment_col} AS segment,
    COUNT(*) AS session_count,
    SUM(CASE WHEN final_outcome = 'authorized' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS authorization_rate,
    SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS retry_rate,
    AVG(attempt_count) AS avg_attempts_per_session,
    SUM(CASE WHEN attempt_count > 1 AND final_outcome = 'authorized' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END), 0) AS recovery_rate
FROM payment_sessions
GROUP BY {segment_col}
ORDER BY session_count DESC
"""

_AMOUNT_QUARTILE_QUERY = """
WITH bucketed AS (
    SELECT *, NTILE(4) OVER (ORDER BY TransactionAmt) AS amount_quartile
    FROM payment_sessions
)
SELECT
    amount_quartile AS segment,
    MIN(TransactionAmt) AS min_amount_in_quartile,
    MAX(TransactionAmt) AS max_amount_in_quartile,
    COUNT(*) AS session_count,
    SUM(CASE WHEN final_outcome = 'authorized' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS authorization_rate,
    SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS retry_rate,
    AVG(attempt_count) AS avg_attempts_per_session,
    SUM(CASE WHEN attempt_count > 1 AND final_outcome = 'authorized' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN attempt_count > 1 THEN 1 ELSE 0 END), 0) AS recovery_rate
FROM bucketed
GROUP BY amount_quartile
ORDER BY amount_quartile
"""


class SegmentationError(ValueError):
    pass


def compute_segment_by_column(con: duckdb.DuckDBPyConnection, column: str) -> pd.DataFrame:
    if column not in ALLOWED_SEGMENT_COLUMNS:
        raise SegmentationError(f"column must be one of {ALLOWED_SEGMENT_COLUMNS}; got {column!r}")
    query = _SEGMENT_QUERY_TEMPLATE.format(segment_col=column)
    df = con.execute(query).df()
    df.insert(0, "segment_dimension", column)
    return df


def compute_segment_by_amount_quartile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(_AMOUNT_QUARTILE_QUERY).df()
    df.insert(0, "segment_dimension", "TransactionAmt_quartile")
    return df


def compute_all_segments(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frames = [compute_segment_by_column(con, col) for col in ALLOWED_SEGMENT_COLUMNS]
    frames.append(compute_segment_by_amount_quartile(con))
    return pd.concat(frames, ignore_index=True)
