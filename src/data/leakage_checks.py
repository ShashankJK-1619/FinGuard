"""Milestone 1, section G (empirical half): automatable leakage checks.

Two checks that are cheap to compute and don't require guessing at business
meaning:
  1. TransactionID overlap between train and test (should be zero).
  2. Correlation of each numeric column with isFraud, flagging anything
     suspiciously high for manual review (not auto-dropped).

Run: python -m src.data.leakage_checks
Output: docs/leakage_checks_empirical.md
"""
from __future__ import annotations

import sys

import duckdb
import pandas as pd

from src.data.config import Paths, load_paths
from src.data.loaders import get_connection, raw_files_present, register_raw_views

SUSPICIOUS_CORR_THRESHOLD = 0.95


def transaction_id_overlap(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute(
        "SELECT COUNT(*) FROM ("
        "SELECT TransactionID FROM train_transaction "
        "INTERSECT "
        "SELECT TransactionID FROM test_transaction"
        ")"
    ).fetchone()[0]


def numeric_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    schema = con.execute(f"DESCRIBE {table}").fetchdf()
    numeric_types = {"BIGINT", "DOUBLE", "INTEGER", "HUGEINT", "SMALLINT", "TINYINT", "FLOAT"}
    return [
        row["column_name"]
        for _, row in schema.iterrows()
        if row["column_type"] in numeric_types and row["column_name"] not in ("TransactionID", "isFraud")
    ]

def correlation_scan(con: duckdb.DuckDBPyConnection, table: str = "train_transaction") -> pd.DataFrame:
    cols = numeric_columns(con, table)
    rows = []
    for col in cols:
        try:
            corr = con.execute(f'SELECT CORR("{col}", isFraud) FROM {table}').fetchone()[0]
        except duckdb.Error:
            corr = None
        rows.append({"column": col, "corr_with_isFraud": corr})
    df = pd.DataFrame(rows)
    df["abs_corr"] = df["corr_with_isFraud"].abs()
    return df.sort_values("abs_corr", ascending=False, na_position="last")


def render_markdown(id_overlap: int, corr_df: pd.DataFrame) -> str:
    lines = ["# Empirical Leakage Checks — Milestone 1 Section G", ""]
    lines.append("## Train/test TransactionID overlap")
    lines.append("")
    lines.append(f"Overlapping TransactionID count: **{id_overlap}**")
    lines.append("")
    if id_overlap == 0:
        lines.append("No row-level contamination between train and test via TransactionID.")
    else:
        lines.append("WARNING: unexpected overlap — investigate before any modeling.")
    lines.append("")

    lines.append(f"## Correlation with `isFraud` (flag threshold: |corr| >= {SUSPICIOUS_CORR_THRESHOLD})")
    lines.append("")
    suspicious = corr_df[corr_df["abs_corr"] >= SUSPICIOUS_CORR_THRESHOLD]
    if suspicious.empty:
        lines.append("No column crosses the suspicious-correlation threshold.")
    else:
        lines.append("The following columns are flagged for manual review. They are "
                      "**not** dropped automatically — high correlation with a rare "
                      "event can be a real, legitimate signal, not necessarily leakage. "
                      "Investigate what the column represents before deciding.")
        lines.append("")
        lines.append(suspicious.to_markdown(index=False))
    lines.append("")
    lines.append("### Top 15 columns by |correlation| with isFraud (for context, not a ranking of importance)")
    lines.append("")
    lines.append(corr_df.head(15).to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    paths: Paths = load_paths()
    present = raw_files_present(paths)
    if not all(present.values()):
        missing = [k for k, v in present.items() if not v]
        print("Missing raw files, cannot run leakage checks:", missing, file=sys.stderr)
        print(f"Place the IEEE-CIS CSVs in {paths.raw_dir}", file=sys.stderr)
        return 1

    con = get_connection()
    register_raw_views(con, paths)

    overlap = transaction_id_overlap(con)
    corr_df = correlation_scan(con)

    report = render_markdown(overlap, corr_df)
    paths.docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.docs_dir / "leakage_checks_empirical.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
