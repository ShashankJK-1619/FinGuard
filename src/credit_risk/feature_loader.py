"""Module C, Milestone 3, Section B/C support: loads the real raw columns a
given `FeatureSet` (Model A or Model B, see feature_selection.py) needs,
with explicit, auditable casts -- no silent type inference -- and builds
the small number of derived application-time-safe features declared in
`configs/credit_risk_model_feature_sets.json`'s
`model_a_derived_numeric_fields`.

Mirrors `src/credit_risk/target_construction.py::load_loans_base_table`'s
conventions (reuses its month/year date parser and term-month parser so
the two modules never disagree about what "term_months" or
"origination_date" mean) but additionally selects every raw numeric/
categorical modeling field a `FeatureSet` declares, plus the handful of
raw columns (`emp_length`, `earliest_cr_line`, `fico_range_low`,
`fico_range_high`) needed only to build this milestone's derived
features, never passed through to the model as raw columns themselves.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.credit_risk.feature_selection import FeatureSet
from src.credit_risk.raw_loader import LOANS_RAW_VIEW
from src.credit_risk.schema import LOAN_ID_COL, MATURITY_DATE_COL, ORIGINATION_DATE_COL
from src.credit_risk.status_mapping import load_status_mapping_table, map_status_series
from src.credit_risk.target_construction import _parse_month_year_column, _term_to_months

# Explicit, auditable emp_length string -> years mapping (see
# configs/credit_risk_model_feature_sets.json's model_a_derived_numeric_fields
# note for emp_length_years). Any raw value not in this map (other than the
# real file's NULL, which pandas already represents as None/NaN) raises
# rather than being silently coerced.
EMP_LENGTH_YEARS_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}

# Raw columns each derived feature needs, beyond whatever FeatureSet.raw_*
# already requests. Fixed because the derivation rules themselves are fixed
# (see feature_selection.py's derived_feature_notes).
_DERIVED_FEATURE_RAW_INPUTS = {
    "term_months": ("term",),
    "emp_length_years": ("emp_length",),
    "fico_avg_at_application": ("fico_range_low", "fico_range_high"),
    "credit_history_months_at_application": ("earliest_cr_line",),
}

_ALWAYS_NEEDED_BASE_COLUMNS = ("id", "issue_d", "loan_status")


class FeatureLoaderError(ValueError):
    pass


def _columns_to_select(feature_set: FeatureSet) -> list[str]:
    cols = set(_ALWAYS_NEEDED_BASE_COLUMNS)
    cols.update(feature_set.raw_numeric_fields)
    cols.update(feature_set.categorical_fields)
    for derived in feature_set.derived_numeric_fields:
        inputs = _DERIVED_FEATURE_RAW_INPUTS.get(derived)
        if inputs is None:
            raise FeatureLoaderError(
                f"Derived feature '{derived}' has no registered raw-input mapping in "
                "feature_loader.py's _DERIVED_FEATURE_RAW_INPUTS -- add one before using it."
            )
        cols.update(inputs)
    return sorted(cols)


def _cast_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="raise")


def _derive_emp_length_years(emp_length: pd.Series) -> pd.Series:
    non_null = emp_length.dropna()
    unmapped = sorted(set(non_null.unique()) - set(EMP_LENGTH_YEARS_MAP))
    if unmapped:
        raise FeatureLoaderError(
            f"Unrecognized emp_length value(s) {unmapped} -- not present in "
            "feature_loader.py's EMP_LENGTH_YEARS_MAP. Review and add them explicitly "
            "before mapping this column."
        )
    return emp_length.map(lambda v: EMP_LENGTH_YEARS_MAP[v] if pd.notna(v) else pd.NA).astype("Float64")


def _derive_credit_history_months(earliest_cr_line: pd.Series, origination_date: pd.Series) -> pd.Series:
    cr_line_date = pd.to_datetime(earliest_cr_line, format="%b-%Y", errors="coerce")
    months = (origination_date.dt.year - cr_line_date.dt.year) * 12 + (
        origination_date.dt.month - cr_line_date.dt.month
    )
    return months.astype("Float64")


def load_modeling_table(
    con: duckdb.DuckDBPyConnection, feature_set: FeatureSet, limit: int | None = None
) -> pd.DataFrame:
    """Returns one row per real loan with: `loan_id`, `origination_date`,
    `maturity_date`, `status_category`, `meets_credit_policy` (everything
    `target_construction.compute_terminal_adverse_outcome_target` needs),
    plus every raw numeric/categorical field `feature_set` declares (cast
    explicitly) and this milestone's derived numeric features
    (`term_months`, `emp_length_years`, `fico_avg_at_application`,
    `credit_history_months_at_application`)."""
    select_cols = _columns_to_select(feature_set)
    limit_clause = f"LIMIT {limit}" if limit else ""
    cols_sql = ",\n            ".join(select_cols)
    query = f"""
        SELECT
            {cols_sql}
        FROM {LOANS_RAW_VIEW}
        {limit_clause}
    """
    df = con.execute(query).fetchdf()
    df[LOAN_ID_COL] = df["id"].astype("int64")
    df[ORIGINATION_DATE_COL] = _parse_month_year_column(df, "issue_d")

    if "term" in df.columns:
        df["term_months"] = _term_to_months(df["term"])
        df[MATURITY_DATE_COL] = [
            o + pd.DateOffset(months=int(m)) for o, m in zip(df[ORIGINATION_DATE_COL], df["term_months"])
        ]

    if "emp_length" in df.columns:
        df["emp_length_years"] = _derive_emp_length_years(df["emp_length"])

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        _cast_numeric(df, ["fico_range_low", "fico_range_high"])
        df["fico_avg_at_application"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0

    if "earliest_cr_line" in df.columns:
        df["credit_history_months_at_application"] = _derive_credit_history_months(
            df["earliest_cr_line"], df[ORIGINATION_DATE_COL]
        )

    numeric_cols_to_cast = [c for c in feature_set.raw_numeric_fields if c in df.columns]
    _cast_numeric(df, numeric_cols_to_cast)

    status_table = load_status_mapping_table()
    status_df = map_status_series(df["loan_status"], status_table)
    df = pd.concat([df, status_df], axis=1)
    return df


def feature_matrix_columns(feature_set: FeatureSet) -> tuple[list[str], list[str]]:
    """Returns (numeric_columns, categorical_columns) -- the exact column
    names, post-derivation, that the preprocessing pipeline should consume
    for this feature set."""
    return list(feature_set.all_numeric_fields), list(feature_set.categorical_fields)
