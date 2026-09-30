"""Milestone 1, section H: TransactionDT structure.

Determines empirically whether TransactionDT behaves like a relative
offset or an absolute timestamp, and whether train/test are chronologically
separated (i.e. whether test starts after train ends). Does not create any
train/test split — inspection only.

Run: python -m src.data.analyze_time
Output: docs/time_structure.md
"""
from __future__ import annotations

import sys

import duckdb

from src.data.config import Paths, load_paths
from src.data.loaders import get_connection, raw_files_present, register_raw_views

# If TransactionDT were a Unix timestamp, values covering "recent" dates would
# be on the order of ~1.5-1.7 billion (seconds since 1970). We use this only as
# a sanity check, not as a claim about the true reference point, which the
# competition does not disclose.
UNIX_TIMESTAMP_LOWER_BOUND = 1_000_000_000


def analyze(con: duckdb.DuckDBPyConnection) -> dict:
    train_stats = con.execute(
        "SELECT MIN(TransactionDT) AS min_dt, MAX(TransactionDT) AS max_dt, COUNT(*) AS n "
        "FROM train_transaction"
    ).fetchone()
    test_stats = con.execute(
        "SELECT MIN(TransactionDT) AS min_dt, MAX(TransactionDT) AS max_dt, COUNT(*) AS n "
        "FROM test_transaction"
    ).fetchone()

    train_min, train_max, train_n = train_stats
    test_min, test_max, test_n = test_stats

    looks_like_unix_timestamp = train_min >= UNIX_TIMESTAMP_LOWER_BOUND
    train_range_days = (train_max - train_min) / 86400
    test_starts_after_train_ends = test_min >= train_max

    return {
        "train_min": train_min,
        "train_max": train_max,
        "train_n": train_n,
        "test_min": test_min,
        "test_max": test_max,
        "test_n": test_n,
        "looks_like_unix_timestamp": looks_like_unix_timestamp,
        "train_range_days": round(train_range_days, 1),
        "test_starts_after_train_ends": test_starts_after_train_ends,
    }


def render_markdown(stats: dict) -> str:
    lines = ["# TransactionDT Structure — Milestone 1 Section H", ""]
    lines.append(
        "No model or split created here. This documents what TransactionDT "
        "structurally is, to inform a time-aware split design in a later milestone."
    )
    lines.append("")
    lines.append(f"- train_transaction TransactionDT range: {stats['train_min']:,} to {stats['train_max']:,} "
                  f"({stats['train_range_days']} days if seconds) across {stats['train_n']:,} rows")
    lines.append(f"- test_transaction TransactionDT range: {stats['test_min']:,} to {stats['test_max']:,} "
                  f"across {stats['test_n']:,} rows")
    lines.append("")
    lines.append(
        f"- Values consistent with an absolute Unix timestamp (>= {UNIX_TIMESTAMP_LOWER_BOUND:,}): "
        f"{stats['looks_like_unix_timestamp']}"
    )
    lines.append(f"- test_transaction's minimum TransactionDT >= train_transaction's maximum: "
                  f"{stats['test_starts_after_train_ends']}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if not stats["looks_like_unix_timestamp"]:
        lines.append(
            "TransactionDT values are far too small to be Unix timestamps (which would "
            "be in the billions for any recent date). This is consistent with the "
            "competition's documentation that TransactionDT is a **relative time delta "
            "in seconds from an undisclosed reference point**, not a wall-clock timestamp. "
            "We cannot recover the actual calendar date of any transaction from this field alone."
        )
    else:
        lines.append(
            "WARNING: TransactionDT values are large enough to be consistent with an "
            "absolute Unix timestamp. This contradicts the documented design and should "
            "be re-checked before assuming it is a relative delta."
        )
    lines.append("")
    if stats["test_starts_after_train_ends"]:
        lines.append(
            "test_transaction's TransactionDT values start after train_transaction's "
            "maximum value — the provided train/test split is itself time-based, not "
            "random. This is a strong signal that fraud patterns are expected to drift "
            "over time, and that evaluating on a random split of train data would "
            "overstate how the model performs on genuinely future, unseen traffic."
        )
    else:
        lines.append(
            "test_transaction's TransactionDT range overlaps with train_transaction's. "
            "This needs closer inspection before assuming a strict future/past split exists."
        )
    lines.append("")
    lines.append(
        "## Why random splitting is risky for fraud modeling\n\n"
        "Fraud behavior is non-stationary: new fraud rings, new attack patterns, and "
        "merchant/product mix all shift over time. A random split lets the model "
        "\"see\" transactions from the same time window it will be tested on, including "
        "potentially correlated fraud rings active only in that window. That inflates "
        "offline metrics relative to real-world performance, where the model only ever "
        "sees the future. A time-based split (train on earlier TransactionDT, validate/test "
        "on strictly later TransactionDT) is the defensible design — to be implemented in "
        "a later milestone, not here."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    paths: Paths = load_paths()
    present = raw_files_present(paths)
    if not all(present.values()):
        missing = [k for k, v in present.items() if not v]
        print("Missing raw files, cannot analyze time structure:", missing, file=sys.stderr)
        print(f"Place the IEEE-CIS CSVs in {paths.raw_dir}", file=sys.stderr)
        return 1

    con = get_connection()
    register_raw_views(con, paths)
    stats = analyze(con)

    report = render_markdown(stats)
    paths.docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.docs_dir / "time_structure.md"
    out_path.write_text(report)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
