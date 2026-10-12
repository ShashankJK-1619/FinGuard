"""Module C, Milestone 1: default_12m target-definition logic.

See docs/credit_risk_target_definition.md for the full rationale. This
module is a pure, deterministic function of a standardized per-loan
performance table -- it does not read any real dataset (none has been
downloaded for Module C yet) and is exercised only against toy data in
`tests/credit_risk/test_target_definition.py`.

Expected input columns on `loans_df` (see src/credit_risk/schema.py):
  - `loan_id`
  - `origination_date` (datetime-like): observation start (Sec D).
  - `data_as_of_date` (datetime-like): the latest date through which this
    loan's performance has actually been observed -- i.e. how much
    follow-up exists. This is what lets us distinguish a genuine
    "survived 12 months without defaulting" loan from a loan that simply
    hasn't been followed long enough yet to know.
  - `default_event_date` (datetime-like or NaT): the date the loan FIRST
    reached a configured severe-delinquency/default status, if ever
    observed. NaT if never observed (so far).
  - `delinquency_event_date` (datetime-like or NaT, optional): the date
    the loan first reached a configured (lesser) delinquency status, if
    ever observed. Carried through the output for visibility, but never
    used to set `default_12m` -- delinquency and default are deliberately
    treated as separate severities (Sec D), not conflated into one label.

Output adds:
  - `observation_window_end` = `origination_date` + `outcome_window_months`.
  - `sufficient_followup` (bool): whether `data_as_of_date` reaches or
    passes `observation_window_end`.
  - `default_12m`: 1 if `default_event_date` falls on/before
    `observation_window_end` (inclusive -- see doc); 0 if no such event
    AND `sufficient_followup` is True; otherwise left unset (NA) and
    `label_status` is set to the censored state -- a loan that hasn't
    defaulted YET but also hasn't been followed long enough to rule out
    a default later in the window must never be silently coded as a
    confirmed non-default.
  - `label_status`: one of `LABEL_STATUS_DEFAULT`, `LABEL_STATUS_NON_DEFAULT`,
    `LABEL_STATUS_CENSORED`.
"""
from __future__ import annotations

import pandas as pd

from src.credit_risk.config import TargetDefinitionAssumptions
from src.credit_risk.schema import (
    DATA_AS_OF_DATE_COL,
    DEFAULT_EVENT_DATE_COL,
    DEFAULT_LABEL_COL,
    LABEL_STATUS_COL,
    LABEL_STATUS_CENSORED,
    LABEL_STATUS_DEFAULT,
    LABEL_STATUS_NON_DEFAULT,
    LOAN_ID_COL,
    OBSERVATION_WINDOW_END_COL,
    ORIGINATION_DATE_COL,
    SUFFICIENT_FOLLOWUP_COL,
)

REQUIRED_INPUT_COLUMNS = (LOAN_ID_COL, ORIGINATION_DATE_COL, DATA_AS_OF_DATE_COL, DEFAULT_EVENT_DATE_COL)


class TargetDefinitionError(ValueError):
    pass


def compute_default_12m_labels(
    loans_df: pd.DataFrame, assumptions: TargetDefinitionAssumptions
) -> pd.DataFrame:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in loans_df.columns]
    if missing:
        raise TargetDefinitionError(f"loans_df is missing required column(s): {missing}")

    out = loans_df.copy()
    origination_date = pd.to_datetime(out[ORIGINATION_DATE_COL])
    data_as_of_date = pd.to_datetime(out[DATA_AS_OF_DATE_COL])
    default_event_date = pd.to_datetime(out[DEFAULT_EVENT_DATE_COL])

    if (data_as_of_date < origination_date).any():
        bad_ids = out.loc[data_as_of_date < origination_date, LOAN_ID_COL].tolist()
        raise TargetDefinitionError(
            f"data_as_of_date precedes origination_date for loan_id(s): {bad_ids} -- "
            "a loan cannot have less than zero days of follow-up."
        )

    window = pd.DateOffset(months=assumptions.outcome_window_months)
    out[OBSERVATION_WINDOW_END_COL] = origination_date + window
    out[SUFFICIENT_FOLLOWUP_COL] = data_as_of_date >= out[OBSERVATION_WINDOW_END_COL]

    defaulted_in_window = default_event_date.notna() & (default_event_date <= out[OBSERVATION_WINDOW_END_COL])
    non_default_confirmed = (~defaulted_in_window) & out[SUFFICIENT_FOLLOWUP_COL]

    out[DEFAULT_LABEL_COL] = pd.array([pd.NA] * len(out), dtype="Int64")
    out.loc[defaulted_in_window, DEFAULT_LABEL_COL] = 1
    out.loc[non_default_confirmed, DEFAULT_LABEL_COL] = 0

    out[LABEL_STATUS_COL] = LABEL_STATUS_CENSORED
    out.loc[defaulted_in_window, LABEL_STATUS_COL] = LABEL_STATUS_DEFAULT
    out.loc[non_default_confirmed, LABEL_STATUS_COL] = LABEL_STATUS_NON_DEFAULT

    return out
