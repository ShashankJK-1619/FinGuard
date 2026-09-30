"""Milestone 1, section D: join validation between transaction and identity.

Answers four specific questions for each (transaction, identity) pair:
  1. % of transaction rows with a matching identity row
  2. does identity have duplicate TransactionID values
  3. does a LEFT JOIN change the transaction row count
  4. is there any accidental many-to-many relationship

Run: python -m src.data.validate_join
Output: docs/join_validation.md
"""
from __future__ import annotations

import sys

import duckdb

from src.data.config import Paths, load_paths
from src.data.loaders import get_connection, raw_files_present, register_raw_views


def validate_pair(con: duckdb.DuckDBPyConnection, txn_table: str, id_table: str) -> dict:
    txn_rows = con.execute(f"SELECT COUNT(*) FROM {txn_table}").fetchone()[0]

    id_dupes = con.execute(
        f"SELECT COUNT(*) FROM (SELECT TransactionID FROM {id_table} "
        f"GROUP BY TransactionID HAVING COUNT(*) > 1)"
    ).fetchone()[0]

    matched = con.execute(
        f"SELECT COUNT(*) FROM {txn_table} t "
        f"INNER JOIN {id_table} i ON t.TransactionID = i.TransactionID"
    ).fetchone()[0]

    left_join_rows = con.execute(
        f"SELECT COUNT(*) FROM {txn_table} t "
        f"LEFT JOIN {id_table} i ON t.TransactionID = i.TransactionID"
    ).fetchone()[0]

    # Many-to-many would show up as left_join_rows > txn_rows (identity fanning out)
    # or as duplicate TransactionID within the identity table itself.
    many_to_many = (left_join_rows != txn_rows) or (id_dupes > 0)

    return {
        "txn_table": txn_table,
        "id_table": id_table,
        "txn_rows": txn_rows,
        "matched_rows": matched,
        "match_pct": round(matched / txn_rows * 100, 4) if txn_rows else None,
        "identity_duplicate_transaction_ids": id_dupes,
        "left_join_row_count": left_join_rows,
        "join_changes_row_count": left_join_rows != txn_rows,
        "many_to_many_detected": many_to_many,
    }


def render_markdown(results: list[dict]) -> str:
    lines = ["# Join Validation — Milestone 1 Section D", ""]
    lines.append(
        "Validates the TransactionID relationship between transaction and identity "
        "tables. No modeling performed."
    )
    lines.append("")
    for r in results:
        lines.append(f"## `{r['txn_table']}` ⨝ `{r['id_table']}`")
        lines.append("")
        lines.append(f"- Transaction rows: {r['txn_rows']:,}")
        lines.append(f"- Matched to identity (INNER JOIN): {r['matched_rows']:,} ({r['match_pct']}%)")
        lines.append(f"- Identity rows with duplicate TransactionID: {r['identity_duplicate_transaction_ids']}")
        lines.append(f"- Row count after LEFT JOIN transaction->identity: {r['left_join_row_count']:,}")
        lines.append(f"- LEFT JOIN changes transaction row count: {r['join_changes_row_count']}")
        lines.append(f"- Many-to-many relationship detected: {r['many_to_many_detected']}")
        lines.append("")
        if not r["many_to_many_detected"] and not r["join_changes_row_count"]:
            lines.append(
                "Interpretation: the relationship is one-to-zero-or-one "
                "(each transaction has at most one identity row). Not every "
                "transaction has identity data — that is expected per the "
                "competition's documented design (identity data is only "
                "available for a subset of transactions), not a data quality bug."
            )
        else:
            lines.append(
                "Interpretation: WARNING — join behavior deviates from the "
                "expected one-to-zero-or-one relationship. Investigate before "
                "using a naive LEFT JOIN in feature engineering."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    paths: Paths = load_paths()
    present = raw_files_present(paths)
    if not all(present.values()):
        missing = [k for k, v in present.items() if not v]
        print("Missing raw files, cannot validate joins:", missing, file=sys.stderr)
        print(f"Place the IEEE-CIS CSVs in {paths.raw_dir}", file=sys.stderr)
        return 1

    con = get_connection()
    register_raw_views(con, paths)

    results = [
        validate_pair(con, "train_transaction", "train_identity"),
        validate_pair(con, "test_transaction", "test_identity"),
    ]

    report = render_markdown(results)
    paths.docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.docs_dir / "join_validation.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
