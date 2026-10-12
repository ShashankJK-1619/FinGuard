"""Module C, Milestone 2, Section D: target construction against the real
LendingClub file.

**The central empirical finding this module encodes (see
docs/moduleC_milestone2_real_data_validation.md Section D for full detail):
the real accepted-loans file contains NO genuine default/charge-off
event-date column.** It provides only:
  - `issue_d` -- month-granularity origination date.
  - `loan_status` -- a single current-snapshot status, not a timestamped
    event log.
  - `last_pymnt_d` -- month-granularity date of the most recent payment
    received (populated for the vast majority of loans, including
    charged-off ones).
  - A handful of hardship/debt-settlement workflow dates
    (`hardship_start_date`, `debt_settlement_flag_date`, `settlement_date`,
    ...) that are >98% NULL and only exist for loans that entered those
    specific servicing programs -- not a general-purpose event date for
    the whole defaulted population, and already excluded as leakage
    regardless (`configs/credit_risk_feature_eligibility.json`).

Per the hard project rule ("do not manufacture event dates"), this module
therefore builds **two distinct, separately-named targets**, never
conflated:

1. `terminal_adverse_outcome` (`compute_terminal_adverse_outcome_target`) --
   **PRIMARY modeling target for Milestone 3** (see
   `configs/credit_risk_target_governance.json`, the machine-readable
   governance source of truth, and `docs/decision_log.md`'s "Milestone 2
   target-governance correction" entry). Requires NO event date. A
   sufficiently-matured loan (scheduled maturity date, from `issue_d` +
   `term` months, has already passed as of `data_as_of_date`) is labeled 1
   if its final status_category is `charged_off`, 0 if `fully_paid`, and
   censored otherwise (including the rare case where a loan reaches/passes
   maturity while still flagged current/delinquent/severe-default in the
   snapshot -- its true terminal outcome is genuinely unknown to us, and
   coding it either way would be a fabricated resolution). It is built
   only from explicitly mapped resolved terminal `loan_status` categories.
   **This target estimates eventual adverse credit outcome among
   originated loans over a variable, loan-specific horizon (36 or 60
   months, depending on `term`) -- it is NOT a 12-month-window target and
   must never be described or consumed as a 12-month probability of
   default.**

2. `default_12m_last_pymnt_proxy` (`compute_default_12m_proxy_target`) --
   **SECONDARY, sensitivity-analysis-only target.** Never to be promoted
   to primary-target status (see governance config/decision-log entry
   above). Reuses Milestone 1's validated, unmodified
   `compute_default_12m_labels` (true 12-month-window logic), fed a
   *proxy* event date: `last_pymnt_d` for loans whose status is in
   `default_event_statuses`, or `issue_d` itself for the small subset of
   charged-off loans with a NULL `last_pymnt_d` (meaning no payment was
   ever received -- the loan was in default from day one in a practical
   sense, though its formal charge-off date is later; see docs for the
   exact row count). **`last_pymnt_d` is NOT a default-event timestamp --
   it is a last-activity/last-payment-received date, repurposed here only
   as an explicitly-labeled proxy, never presented as a genuine event
   time.** **Known, documented bias direction**: LendingClub's own
   publicly documented servicing policy charges off a loan only after it
   has been 120+ days delinquent -- i.e., the true formal charge-off date
   is systematically LATER than `last_pymnt_d`. Anchoring the event date
   on `last_pymnt_d` therefore makes a default appear earlier than it
   truly was, which can only ever pull a borderline default INTO the
   12-month window, never push one out -- i.e. this proxy is a
   one-directional, conservative-against-the-lender (optimistic-about-
   apparent-within-window-defaults) approximation. It is not a
   measurement of the true event timing and must never be presented as
   one.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.credit_risk.config import TargetDefinitionAssumptions, load_target_definition_assumptions
from src.credit_risk.raw_loader import LOANS_RAW_VIEW, MONTH_YEAR_STRPTIME_FORMAT
from src.credit_risk.schema import (
    DATA_AS_OF_DATE_COL,
    DEFAULT_12M_PROXY_LABEL_COL,
    DEFAULT_EVENT_DATE_COL,
    DEFAULT_EVENT_DATE_IS_PROXY_COL,
    DEFAULT_LABEL_COL,
    DELINQUENCY_EVENT_DATE_COL,
    LABEL_STATUS_COL,
    LABEL_STATUS_12M_PROXY_COL,
    LABEL_STATUS_CENSORED,
    LABEL_STATUS_DEFAULT,
    LABEL_STATUS_NON_DEFAULT,
    LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL,
    LOAN_ID_COL,
    MATURITY_DATE_COL,
    ORIGINATION_DATE_COL,
    STATUS_CATEGORY_CHARGED_OFF,
    STATUS_CATEGORY_FULLY_PAID,
    SUFFICIENT_FOLLOWUP_TERMINAL_ADVERSE_OUTCOME_COL,
    TERMINAL_ADVERSE_OUTCOME_LABEL_COL,
)
from src.credit_risk.status_mapping import load_status_mapping_table, map_status_series
from src.credit_risk.target_definition import compute_default_12m_labels

# Inferred, documented proxy for the file's extract/snapshot date. No
# explicit "as of" field exists in the raw file; this is the max observed
# last_credit_pull_d across the whole file (2019-04-01), the field
# LendingClub's own servicing system refreshes close to monthly for every
# loan still being tracked. See docs/moduleC_milestone2_real_data_validation.md
# Section B for the empirical derivation -- this is an inferred proxy,
# not an asserted fact from the source.
INFERRED_DATA_AS_OF_DATE = pd.Timestamp("2019-04-01")


class TargetConstructionError(ValueError):
    pass


def _parse_month_year_column(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(df[column], format="%b-%Y", errors="coerce")


def _term_to_months(term_series: pd.Series) -> pd.Series:
    """LendingClub's `term` column is a string like ' 36 months'."""
    extracted = term_series.str.extract(r"(\d+)")[0]
    if extracted.isna().any():
        bad = term_series[extracted.isna()].unique().tolist()
        raise TargetConstructionError(f"Could not parse term month count from value(s): {bad}")
    return extracted.astype(int)


def load_loans_base_table(con: duckdb.DuckDBPyConnection, limit: int | None = None) -> pd.DataFrame:
    """Pulls the minimal real columns needed for target construction out of
    `loans_raw` (see raw_loader.register_loans_view) into pandas, with
    explicit, auditable casts -- no silent type inference."""
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
        SELECT
            id AS {LOAN_ID_COL},
            issue_d,
            term,
            loan_status,
            last_pymnt_d
        FROM {LOANS_RAW_VIEW}
        {limit_clause}
    """
    df = con.execute(query).fetchdf()
    df[LOAN_ID_COL] = df[LOAN_ID_COL].astype("int64")
    df[ORIGINATION_DATE_COL] = _parse_month_year_column(df, "issue_d")
    df["last_pymnt_date"] = _parse_month_year_column(df, "last_pymnt_d")
    df["term_months"] = _term_to_months(df["term"])
    df[MATURITY_DATE_COL] = df[ORIGINATION_DATE_COL] + pd.to_timedelta(
        df["term_months"] * 30.44, unit="D"
    )
    # Use calendar-month-accurate maturity via DateOffset per row (vectorized
    # via apply is acceptable here -- this is a one-time per-loan-id column,
    # not a hot loop).
    df[MATURITY_DATE_COL] = [
        o + pd.DateOffset(months=int(m))
        for o, m in zip(df[ORIGINATION_DATE_COL], df["term_months"])
    ]

    status_table = load_status_mapping_table()
    status_df = map_status_series(df["loan_status"], status_table)
    df = pd.concat([df, status_df], axis=1)
    return df


def compute_terminal_adverse_outcome_target(
    df: pd.DataFrame, data_as_of_date: pd.Timestamp = INFERRED_DATA_AS_OF_DATE
) -> pd.DataFrame:
    """Section D target 1 -- PRIMARY modeling target for Milestone 3 (see
    module docstring and `configs/credit_risk_target_governance.json`).
    No event date required. A loan is resolved once it has passed its own
    scheduled maturity date; label is 1 if its final status_category is
    charged_off, 0 if fully_paid, and censored otherwise (including the
    rare case where a loan reaches/passes maturity while still flagged
    current/delinquent/severe-default in the snapshot -- its true terminal
    outcome is genuinely unknown to us, and coding it either way would be a
    fabricated resolution). This estimates eventual adverse credit outcome
    among originated loans over a variable (36- or 60-month) horizon -- it
    is NOT a 12-month probability-of-default estimate."""
    out = df.copy()
    out[SUFFICIENT_FOLLOWUP_TERMINAL_ADVERSE_OUTCOME_COL] = (
        data_as_of_date >= out[MATURITY_DATE_COL]
    )

    is_charged_off = out["status_category"] == STATUS_CATEGORY_CHARGED_OFF
    is_fully_paid = out["status_category"] == STATUS_CATEGORY_FULLY_PAID

    out[TERMINAL_ADVERSE_OUTCOME_LABEL_COL] = pd.array([pd.NA] * len(out), dtype="Int64")
    resolved_default = out[SUFFICIENT_FOLLOWUP_TERMINAL_ADVERSE_OUTCOME_COL] & is_charged_off
    resolved_non_default = out[SUFFICIENT_FOLLOWUP_TERMINAL_ADVERSE_OUTCOME_COL] & is_fully_paid
    out.loc[resolved_default, TERMINAL_ADVERSE_OUTCOME_LABEL_COL] = 1
    out.loc[resolved_non_default, TERMINAL_ADVERSE_OUTCOME_LABEL_COL] = 0

    out[LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL] = LABEL_STATUS_CENSORED
    out.loc[resolved_default, LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL] = LABEL_STATUS_DEFAULT
    out.loc[
        resolved_non_default, LABEL_STATUS_TERMINAL_ADVERSE_OUTCOME_COL
    ] = LABEL_STATUS_NON_DEFAULT
    return out


def compute_default_12m_proxy_target(
    df: pd.DataFrame,
    assumptions: TargetDefinitionAssumptions | None = None,
    data_as_of_date: pd.Timestamp = INFERRED_DATA_AS_OF_DATE,
) -> pd.DataFrame:
    """Section D target 2 -- SECONDARY, sensitivity-analysis-only target
    (never primary; see module docstring and
    `configs/credit_risk_target_governance.json`). Reuses Milestone 1's
    compute_default_12m_labels unmodified, constructing the required
    `default_event_date` input column as an explicitly-labeled PROXY
    (last_pymnt_d, falling back to origination_date only for the
    zero-payment edge case) -- never a fabricated true event date.
    `last_pymnt_d` is NOT a default-event timestamp. See module docstring
    for the documented one-directional bias."""
    assumptions = assumptions or load_target_definition_assumptions()
    out = df.copy()

    is_default_status = out["loan_status"].isin(assumptions.default_event_statuses)
    is_delinquency_status = out["loan_status"].isin(assumptions.delinquency_event_statuses)

    zero_payment_default = is_default_status & out["last_pymnt_date"].isna()
    proxy_event_date = out["last_pymnt_date"].copy()
    proxy_event_date = proxy_event_date.where(~zero_payment_default, out[ORIGINATION_DATE_COL])
    # Only defaulted loans get a non-null event date -- a Current/Late/
    # Fully-Paid loan has no default event at all, so its proxy date must
    # stay NaT regardless of its (irrelevant) last_pymnt_date value.
    proxy_event_date = proxy_event_date.where(is_default_status, pd.NaT)

    out[DEFAULT_EVENT_DATE_COL] = proxy_event_date
    out[DEFAULT_EVENT_DATE_IS_PROXY_COL] = True
    out[DELINQUENCY_EVENT_DATE_COL] = pd.NaT  # carried through, never used to set the label
    out.loc[is_delinquency_status, DELINQUENCY_EVENT_DATE_COL] = out.loc[
        is_delinquency_status, ORIGINATION_DATE_COL
    ]  # month-level proxy anchor only; delinquency timing is informational, not label-setting
    out[DATA_AS_OF_DATE_COL] = data_as_of_date

    labeled = compute_default_12m_labels(out, assumptions)
    labeled = labeled.rename(
        columns={
            DEFAULT_LABEL_COL: DEFAULT_12M_PROXY_LABEL_COL,
            LABEL_STATUS_COL: LABEL_STATUS_12M_PROXY_COL,
        }
    )
    return labeled
