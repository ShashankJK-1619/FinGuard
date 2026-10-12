from src.credit_risk.raw_loader import (
    LOANS_RAW_VIEW,
    excluded_non_loan_row_count,
    get_connection,
    raw_file_present,
    register_loans_view,
)
from tests.credit_risk.conftest import requires_credit_risk_raw_data


def test_raw_file_present_reflects_gitignored_download_state():
    # Does not assert True/False (the file may or may not be downloaded on
    # a given machine) -- just that the check runs without error and
    # returns a bool, consistent with Module B's raw_files_present pattern.
    assert isinstance(raw_file_present(), bool)


@requires_credit_risk_raw_data
def test_register_loans_view_filters_non_loan_footer_rows():
    """The real file contains 33 non-loan footer/section-marker rows
    (e.g. 'Total amount funded in policy code 1: ...') interspersed with
    real loan rows -- found empirically while profiling (Section B).
    register_loans_view must filter them out via the id-is-numeric check,
    not via DuckDB's ignore_errors (which was tried first and found to
    silently corrupt unrelated real rows instead)."""
    con = get_connection()
    register_loans_view(con)
    excluded = excluded_non_loan_row_count(con)
    assert excluded == 33


@requires_credit_risk_raw_data
def test_register_loans_view_has_no_duplicate_ids():
    con = get_connection()
    register_loans_view(con)
    dup_count = con.execute(
        f"SELECT COUNT(*) FROM (SELECT id, COUNT(*) c FROM {LOANS_RAW_VIEW} GROUP BY id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert dup_count == 0


@requires_credit_risk_raw_data
def test_register_loans_view_row_count_matches_profiled_value():
    con = get_connection()
    register_loans_view(con)
    n = con.execute(f"SELECT COUNT(*) FROM {LOANS_RAW_VIEW}").fetchone()[0]
    assert n == 2_260_668
