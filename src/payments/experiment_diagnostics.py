"""Module A, Milestone 2: experiment-health diagnostics.

See docs/moduleA_milestone2_experimentation.md Section J. Three checks,
run BEFORE any outcome metric is trusted:

1. `check_no_crossover` — fails loudly if the same randomized unit
   (session_id) appears more than once in the assigned table at all
   (which is the only way it could end up in both arms, since assignment
   is a pure function of session_id). This is a defensive regression
   guard, not merely a restatement of the assignment code's logic — the
   same lesson as Milestone 1's DeviceType bug: verify the invariant
   explicitly, don't just trust that the code is right.
2. `compute_srm_check` — Sample Ratio Mismatch: a one-sample proportion
   test of the observed treatment-arm share against the configured
   target allocation. A significant mismatch would indicate a bug in
   assignment (or, in a real system, a bug in the randomization/logging
   pipeline) and should block trusting any downstream result.
3. `compute_pretreatment_covariate_balance` — descriptive comparison of a
   few real, pre-treatment fields across arms. Because assignment here is
   a pure hash of `session_id` alone, these are expected to be close
   between arms; this is a sanity check, not a formal balance test.
"""
from __future__ import annotations

import duckdb
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

from src.payments.config import ExperimentAssumptions
from src.payments.schema import ASSIGNMENT_COL, ASSIGNMENT_TREATMENT, SESSION_ID_COL


class ExperimentDiagnosticError(ValueError):
    pass


def check_no_crossover(assigned_df: pd.DataFrame) -> dict:
    """Fails loudly (raises) if any session_id appears more than once in
    `assigned_df` — the grain of `payment_sessions`/the experiment table
    is one row per session_id, so a duplicate is the only mechanism by
    which the same randomized unit could end up counted in both arms."""
    duplicate_ids = assigned_df[SESSION_ID_COL][assigned_df[SESSION_ID_COL].duplicated()].unique().tolist()
    if duplicate_ids:
        raise ExperimentDiagnosticError(
            f"{len(duplicate_ids)} session_id(s) appear more than once in the assigned sessions table "
            f"(grain should be exactly one row per session_id): {duplicate_ids[:10]}"
        )
    return {"n_sessions": len(assigned_df), "n_unique_session_ids": int(assigned_df[SESSION_ID_COL].nunique())}


def compute_srm_check(assigned_df: pd.DataFrame, assumptions: ExperimentAssumptions) -> dict:
    n_total = len(assigned_df)
    n_treatment = int((assigned_df[ASSIGNMENT_COL] == ASSIGNMENT_TREATMENT).sum())
    n_control = n_total - n_treatment
    expected_fraction = assumptions.treatment_allocation_fraction

    if n_total == 0:
        zstat, pvalue = float("nan"), float("nan")
    else:
        zstat, pvalue = proportions_ztest(count=n_treatment, nobs=n_total, value=expected_fraction)
    return {
        "n_total": n_total,
        "n_treatment": n_treatment,
        "n_control": n_control,
        "observed_treatment_fraction": n_treatment / n_total if n_total else float("nan"),
        "expected_treatment_fraction": expected_fraction,
        "srm_zstat": float(zstat),
        "srm_p_value": float(pvalue),
        "alpha": assumptions.alpha,
        # True => observed allocation differs from target by more than
        # chance at the configured alpha => investigate before trusting
        # any downstream result.
        "srm_flag": bool(pvalue < assumptions.alpha) if n_total else True,
    }


_COVARIATE_BALANCE_QUERY = """
SELECT
    assignment,
    COUNT(*) AS n,
    AVG(TransactionAmt) AS mean_transaction_amt,
    AVG(CASE WHEN ProductCD = 'W' THEN 1.0 ELSE 0.0 END) AS share_productcd_w,
    AVG(CASE WHEN card6 = 'debit' THEN 1.0 ELSE 0.0 END) AS share_card6_debit,
    AVG(CASE WHEN DeviceType = 'mobile' THEN 1.0 ELSE 0.0 END) AS share_devicetype_mobile
FROM payment_experiment_sessions
GROUP BY assignment
ORDER BY assignment
"""


def compute_pretreatment_covariate_balance(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Descriptive only (see module docstring) — a handful of real,
    pre-treatment fields compared across arms. `payment_experiment_sessions`
    must already be registered (src.payments.duckdb_views)."""
    return con.execute(_COVARIATE_BALANCE_QUERY).df()
