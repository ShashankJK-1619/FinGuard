"""DuckDB-backed access to the raw IEEE-CIS CSVs.

We query CSVs in place via DuckDB rather than loading everything into
pandas up front. The files are wide (300+ columns) and loading both
transaction tables into memory at once as pandas DataFrames is wasteful
for inspection tasks that only need aggregates.
"""
from __future__ import annotations

import duckdb

from src.data.config import Paths, load_paths


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def register_raw_views(con: duckdb.DuckDBPyConnection, paths: Paths | None = None) -> Paths:
    """Register each raw CSV as a DuckDB view without materializing it.

    Raises FileNotFoundError with a clear message if a raw file is missing,
    since Milestone 1 assumes the user has downloaded the Kaggle files
    into data/raw/ themselves.
    """
    paths = paths or load_paths()
    for key, filename in paths.raw_files.items():
        file_path = paths.raw_file(key)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Expected raw file '{filename}' at {file_path}, but it does not exist. "
                "Download the IEEE-CIS Fraud Detection data from Kaggle and place the "
                "four CSVs (train_transaction, train_identity, test_transaction, "
                "test_identity) into data/raw/."
            )
        con.execute(f"CREATE OR REPLACE VIEW {key} AS SELECT * FROM read_csv_auto('{file_path}')")
    return paths


def raw_files_present(paths: Paths | None = None) -> dict[str, bool]:
    paths = paths or load_paths()
    return {key: paths.raw_file(key).exists() for key in paths.raw_files}
