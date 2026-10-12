"""Module C, Milestone 2: DuckDB-backed access to the real, raw LendingClub
accepted-loans CSV.

Mirrors the Module B convention (`src/data/loaders.py`): query the CSV in
place via DuckDB rather than loading the full 1.6GB file into pandas.

One real-data wrinkle this module exists to handle, found empirically
while profiling the downloaded file (not assumed in advance): the raw CSV
is not purely tabular loan rows. Interspersed at ~20 fixed offsets are 33
non-data footer/section-marker lines left over from how this mirror
appears to have concatenated the original per-vintage LendingClub release
files -- e.g. literal text rows reading
`"Total amount funded in policy code 1: 6417608175"` with every other
field empty, and one literal section header
`"Loans that do not meet the credit policy"`. These are not loans and
would corrupt any aggregate if left in (e.g. `id` is not numeric on these
rows, `loan_status` is NULL). `register_loans_view` filters them out
explicitly, by requiring `id` to be all-digits, rather than silently
dropping rows via `ignore_errors` (which was tried first and found to
silently NULL out unrelated real columns on a different, larger set of
rows whenever DuckDB's sampled type inference guessed wrong for a column
-- see docs/moduleC_milestone2_real_data_validation.md Section B for the
full finding). All 151 columns are read as VARCHAR here deliberately, so
that type-casting decisions are made explicitly, per-column, by calling
code (status mapping, target construction, etc.), not silently guessed by
a CSV sniffer.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from src.credit_risk.raw_source import (
    PROJECT_ROOT,
    RawSourceMetadata,
    load_raw_source_metadata,
)

LOANS_RAW_VIEW = "loans_raw"

# LendingClub's own month-granularity date columns are formatted "Mon-YYYY"
# (e.g. "Dec-2011"). This DuckDB strptime pattern is reused everywhere a
# date column must be parsed, so there is exactly one place that encodes
# this assumption.
MONTH_YEAR_STRPTIME_FORMAT = "%b-%Y"


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def raw_file_present(metadata: RawSourceMetadata | None = None) -> bool:
    metadata = metadata or load_raw_source_metadata()
    return (PROJECT_ROOT / metadata.local_path).exists()


def register_loans_view(
    con: duckdb.DuckDBPyConnection,
    metadata: RawSourceMetadata | None = None,
) -> Path:
    """Registers `loans_raw`: every column as VARCHAR, footer/non-loan rows
    filtered out. Raises FileNotFoundError with guidance if the raw file
    is missing (it is gitignored and must be downloaded per
    docs/moduleC_milestone2_real_data_validation.md Section A)."""
    metadata = metadata or load_raw_source_metadata()
    file_path = PROJECT_ROOT / metadata.local_path
    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected raw LendingClub file at {file_path}, but it does not exist. "
            f"Download from {metadata.source_url} ({metadata.mirror_provider}) "
            "and verify its SHA256 against configs/credit_risk_raw_source_metadata.json "
            "(see src.credit_risk.raw_source.verify_local_file_integrity)."
        )
    con.execute(
        f"""
        CREATE OR REPLACE VIEW {LOANS_RAW_VIEW} AS
        SELECT *
        FROM read_csv_auto('{file_path}', all_varchar=true)
        WHERE regexp_matches(id, '^[0-9]+$')
        """
    )
    return file_path


def excluded_non_loan_row_count(con: duckdb.DuckDBPyConnection, metadata: RawSourceMetadata | None = None) -> int:
    """Counts rows in the raw file that `register_loans_view` filters out
    (footer/section-marker lines, not real loan records). Reads the file a
    second time at full-VARCHAR to count without the WHERE filter."""
    metadata = metadata or load_raw_source_metadata()
    file_path = PROJECT_ROOT / metadata.local_path
    total = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{file_path}', all_varchar=true)"
    ).fetchone()[0]
    kept = con.execute(f"SELECT COUNT(*) FROM {LOANS_RAW_VIEW}").fetchone()[0]
    return total - kept
