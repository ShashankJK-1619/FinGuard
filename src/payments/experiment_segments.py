"""Module A, Milestone 2: pre-specified segment heterogeneity.

See docs/moduleA_milestone2_experimentation.md Section I. Exactly four
PRE-SPECIFIED dimensions are examined — chosen before looking at any
result, not discovered by searching for significant slices:

  - DeviceType     (real field)
  - ProductCD      (real field)
  - card6          (real field; "debit"/"credit" is used as a payment-
                    method proxy — there is no separate real payment-
                    method field in the IEEE-CIS schema Module A uses)
  - retry_eligible (derived: attempt_count >= eligible_min_attempt_count,
                    i.e. "went through an initial decline and at least one
                    retry" vs. not)

Each slice gets the SAME two-proportion test as the headline primary
metric (src.payments.experiment_stats.compute_two_proportion_test) — nice
to have, but the multiple-testing / subgroup-selection risk this creates
must be stated, not glossed over: running N independent tests at alpha
inflates the chance of at least one false positive well above alpha
(roughly 1 - (1-alpha)^N for independent tests), and NONE of the
comparisons below have a multiple-comparison correction applied. With only
4 pre-registered dimensions (not dozens of ad hoc slices) the inflation is
modest, but any "significant" subgroup result here should be read as
descriptive and hypothesis-generating, not as a confirmed causal claim —
the experiment was not powered or designed to detect subgroup-specific
effects, and assignment, while unconfounded with these real fields by
construction (session_id-hash based, Sec B), was not stratified by them.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.payments.config import ExperimentAssumptions
from src.payments.experiment_stats import compute_two_proportion_test
from src.payments.schema import ASSIGNMENT_COL, ASSIGNMENT_CONTROL, ASSIGNMENT_TREATMENT

SEGMENT_DIMENSIONS = ["DeviceType", "ProductCD", "card6", "retry_eligible"]

_SEGMENT_COUNTS_QUERY_TEMPLATE = """
SELECT
    {dimension} AS segment_value,
    assignment,
    COUNT(*) AS n,
    SUM(CASE WHEN experiment_final_outcome = 'authorized' THEN 1 ELSE 0 END) AS n_completed
FROM payment_experiment_sessions
GROUP BY {dimension}, assignment
ORDER BY {dimension}, assignment
"""


class ExperimentSegmentError(ValueError):
    pass


def compute_segment_heterogeneity(
    con: duckdb.DuckDBPyConnection, dimension: str, assumptions: ExperimentAssumptions
) -> pd.DataFrame:
    if dimension not in SEGMENT_DIMENSIONS:
        raise ExperimentSegmentError(f"dimension must be one of {SEGMENT_DIMENSIONS}; got {dimension!r}")

    raw = con.execute(_SEGMENT_COUNTS_QUERY_TEMPLATE.format(dimension=dimension)).df()
    rows = []
    for value, group in raw.groupby("segment_value", dropna=False):
        indexed = group.set_index(ASSIGNMENT_COL)
        if ASSIGNMENT_CONTROL not in indexed.index or ASSIGNMENT_TREATMENT not in indexed.index:
            continue
        n_control = int(indexed.loc[ASSIGNMENT_CONTROL, "n"])
        n_completed_control = int(indexed.loc[ASSIGNMENT_CONTROL, "n_completed"])
        n_treatment = int(indexed.loc[ASSIGNMENT_TREATMENT, "n"])
        n_completed_treatment = int(indexed.loc[ASSIGNMENT_TREATMENT, "n_completed"])
        if n_control == 0 or n_treatment == 0:
            continue
        test_result = compute_two_proportion_test(
            n_control, n_completed_control, n_treatment, n_completed_treatment, assumptions.alpha
        )
        rows.append({"segment_dimension": dimension, "segment_value": value, **test_result})
    return pd.DataFrame(rows)


def compute_all_segment_heterogeneity(con: duckdb.DuckDBPyConnection, assumptions: ExperimentAssumptions) -> pd.DataFrame:
    frames = [compute_segment_heterogeneity(con, dim, assumptions) for dim in SEGMENT_DIMENSIONS]
    return pd.concat(frames, ignore_index=True)
