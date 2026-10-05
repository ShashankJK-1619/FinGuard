"""Module A, Milestone 1: deterministic synthetic payment-session/attempt
generator.

See docs/payments_data_model.md for the full methodology and rationale.
Summary: IEEE-CIS has no funnel/session/retry concept, only realized
transactions. This module builds two synthetic tables on top of the real
`transactions` table:

- `payment_sessions` (one row per session; every real transaction is the
  authorized terminal attempt of exactly one session, plus a configurable
  fraction of entirely-synthetic "abandonment" sessions with no real
  transaction link at all).
- `payment_attempts` (one or more rows per session; retries before the
  session's terminal outcome).

The whole generator is a pure function of
(sorted real transaction rows, `SyntheticAssumptions`, a seed): identical
inputs produce byte-identical output regardless of input row order (the
real-transaction rows are sorted internally first) — verified by
`tests/payments/test_generate_sessions.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.config import Paths
from src.data.loaders import get_connection, register_raw_views
from src.features.categorical_features import add_email_domain_match
from src.payments.config import SyntheticAssumptions, load_synthetic_assumptions
from src.payments.schema import (
    AMOUNT_COL,
    ATTEMPT_COLUMNS,
    ATTEMPT_OUTCOME_AUTHORIZED,
    ATTEMPT_OUTCOME_DECLINED,
    FINAL_OUTCOME_ABANDONED,
    FINAL_OUTCOME_AUTHORIZED,
    ID_COL,
    REAL_TRANSACTION_COLUMNS,
    SESSION_COLUMNS,
    SESSION_ID_COL,
    TARGET_COL,
    TIME_COL,
)

# The donor-cloned profile fields copied onto abandonment-only sessions
# (see docs/payments_data_model.md Sec 4). Not referenced elsewhere in this
# module (there is no loop over this list in the generator below) — kept
# only as documentation of which fields are donor-cloned. Previously
# omitted "DeviceType", which had silently drifted out of sync with the
# actual cloned fields (see the session_rows.append(...) blocks below) when
# DeviceType was added to SESSION_COLUMNS; fixed alongside the DeviceType
# all-NaN bug to avoid leaving a second, inconsistent source of truth.
PROFILE_COLUMNS = ["ProductCD", "card4", "card6", "DeviceType", AMOUNT_COL, "email_domain_match"]


def load_real_transactions(paths: Paths | None = None) -> pd.DataFrame:
    """Reads the minimal real-field set (docs/payments_data_model.md Sec 2a)
    directly from the raw IEEE-CIS CSVs via DuckDB — independent of Module
    B's processed baseline_features_v1.parquet. train_transaction LEFT JOIN
    train_identity on TransactionID (DeviceType is only present in the
    identity table and only for the subset of transactions with an
    identity record)."""
    con = get_connection()
    paths = register_raw_views(con, paths)
    query = f"""
        SELECT
            t.{ID_COL} AS {ID_COL},
            t.{TIME_COL} AS {TIME_COL},
            t.{AMOUNT_COL} AS {AMOUNT_COL},
            t.{TARGET_COL} AS {TARGET_COL},
            t.ProductCD AS ProductCD,
            t.card4 AS card4,
            t.card6 AS card6,
            t.P_emaildomain AS P_emaildomain,
            t.R_emaildomain AS R_emaildomain,
            i.DeviceType AS DeviceType
        FROM train_transaction t
        LEFT JOIN train_identity i ON t.{ID_COL} = i.{ID_COL}
    """
    df = con.execute(query).df()
    df = add_email_domain_match(df)
    return df[REAL_TRANSACTION_COLUMNS + ["email_domain_match"]]


def _sample_attempt_offsets(
    rng: np.random.Generator,
    anchor_time: float,
    attempt_count: int,
    assumptions: SyntheticAssumptions,
) -> tuple[np.ndarray, float]:
    """Builds `attempt_count` strictly-non-decreasing offsets ending EXACTLY
    at `anchor_time` (the one real fact we have — either the real
    TransactionDT of a real-linked session's authorized attempt, or the
    bootstrap-sampled real TransactionDT used as an abandonment session's
    give-up time), working backward via sampled retry gaps. Returns
    (offsets, session_start_offset). Offsets are clamped at >= 0: for
    sessions anchored very close to time zero, large sampled gaps could
    otherwise push an early attempt's offset negative, which has no
    physical meaning (elapsed time can't be negative) — clamping trades a
    compressed/duplicated early offset for a nonsensical negative one, and
    is covered by an explicit boundary test.
    """
    offsets = np.empty(attempt_count, dtype=float)
    offsets[-1] = anchor_time
    if attempt_count > 1:
        gaps = rng.uniform(
            assumptions.retry_gap_seconds_min, assumptions.retry_gap_seconds_max, size=attempt_count - 1
        )
        for i in range(attempt_count - 2, -1, -1):
            offsets[i] = max(0.0, offsets[i + 1] - gaps[i])
    lead_time = rng.uniform(assumptions.session_lead_time_seconds_min, assumptions.session_lead_time_seconds_max)
    session_start_offset = max(0.0, offsets[0] - lead_time)
    return offsets, session_start_offset


def generate_payment_sessions_and_attempts(
    real_df: pd.DataFrame,
    assumptions: SyntheticAssumptions | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure function: (real transactions, config, seed) -> (sessions, attempts).

    `real_df` must contain REAL_TRANSACTION_COLUMNS + 'email_domain_match'
    (the shape `load_real_transactions` returns). Sorted internally by
    (TransactionDT, TransactionID) first, so the caller's row order never
    affects the result (verified by an order-invariance test).
    """
    assumptions = assumptions or load_synthetic_assumptions()
    rng = np.random.default_rng(seed if seed is not None else assumptions.random_seed)

    real_df = real_df.sort_values([TIME_COL, ID_COL], kind="stable").reset_index(drop=True)
    n_real = len(real_df)
    n_abandon = round(assumptions.abandonment_rate * n_real)

    decline_probs = np.asarray(assumptions.decline_before_success_probs)
    abandon_probs = np.asarray(assumptions.abandonment_attempt_count_probs)
    reason_codes = assumptions.decline_reason_codes
    reason_probs = assumptions.decline_reason_probs

    # --- Real-linked sessions: one per real transaction, in sorted order ---
    real_prior_declines = rng.choice(len(decline_probs), size=n_real, p=decline_probs)
    real_attempt_counts = real_prior_declines + 1

    # --- Abandonment-only sessions: donor-cloned profile (Sec 4), no real FK ---
    donor_idx = rng.integers(0, n_real, size=n_abandon) if n_abandon > 0 else np.empty(0, dtype=int)
    abandon_attempt_counts = (
        rng.choice(len(abandon_probs), size=n_abandon, p=abandon_probs) + 1 if n_abandon > 0 else np.empty(0, dtype=int)
    )

    session_rows: list[dict] = []
    attempt_rows: list[dict] = []
    session_idx = 0

    for i in range(n_real):
        row = real_df.iloc[i]
        attempt_count = int(real_attempt_counts[i])
        anchor_time = float(row[TIME_COL])
        session_id = f"S{session_idx:08d}"
        session_idx += 1

        offsets, session_start = _sample_attempt_offsets(rng, anchor_time, attempt_count, assumptions)
        for attempt_number in range(1, attempt_count + 1):
            is_last = attempt_number == attempt_count
            outcome = ATTEMPT_OUTCOME_AUTHORIZED if is_last else ATTEMPT_OUTCOME_DECLINED
            reason = None if is_last else rng.choice(reason_codes, p=reason_probs)
            attempt_rows.append(
                {
                    "attempt_id": f"{session_id}-A{attempt_number}",
                    SESSION_ID_COL: session_id,
                    "attempt_number": attempt_number,
                    "attempt_offset_seconds": float(offsets[attempt_number - 1]),
                    "outcome": outcome,
                    "decline_reason_code": reason,
                }
            )
        session_rows.append(
            {
                SESSION_ID_COL: session_id,
                "transaction_id": row[ID_COL],
                "is_synthetic_abandonment": False,
                "final_outcome": FINAL_OUTCOME_AUTHORIZED,
                "attempt_count": attempt_count,
                "session_start_offset_seconds": session_start,
                "ProductCD": row["ProductCD"],
                "card4": row["card4"],
                "card6": row["card6"],
                "DeviceType": row["DeviceType"],
                AMOUNT_COL: row[AMOUNT_COL],
                "email_domain_match": row["email_domain_match"],
                TARGET_COL: row[TARGET_COL],
            }
        )

    for j in range(n_abandon):
        donor = real_df.iloc[int(donor_idx[j])]
        attempt_count = int(abandon_attempt_counts[j])
        anchor_time = float(donor[TIME_COL])
        session_id = f"S{session_idx:08d}"
        session_idx += 1

        offsets, session_start = _sample_attempt_offsets(rng, anchor_time, attempt_count, assumptions)
        for attempt_number in range(1, attempt_count + 1):
            reason = rng.choice(reason_codes, p=reason_probs)
            attempt_rows.append(
                {
                    "attempt_id": f"{session_id}-A{attempt_number}",
                    SESSION_ID_COL: session_id,
                    "attempt_number": attempt_number,
                    "attempt_offset_seconds": float(offsets[attempt_number - 1]),
                    "outcome": ATTEMPT_OUTCOME_DECLINED,
                    "decline_reason_code": reason,
                }
            )
        session_rows.append(
            {
                SESSION_ID_COL: session_id,
                "transaction_id": None,
                "is_synthetic_abandonment": True,
                "final_outcome": FINAL_OUTCOME_ABANDONED,
                "attempt_count": attempt_count,
                "session_start_offset_seconds": session_start,
                "ProductCD": donor["ProductCD"],
                "card4": donor["card4"],
                "card6": donor["card6"],
                "DeviceType": donor["DeviceType"],
                AMOUNT_COL: donor[AMOUNT_COL],
                "email_domain_match": donor["email_domain_match"],
                TARGET_COL: None,
            }
        )

    sessions_df = pd.DataFrame(session_rows, columns=SESSION_COLUMNS)
    attempts_df = pd.DataFrame(attempt_rows, columns=ATTEMPT_COLUMNS)
    return sessions_df, attempts_df


def build_payments_dataset(
    assumptions: SyntheticAssumptions | None = None, seed: int | None = None, paths: Paths | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orchestration: load real transactions from raw CSVs, generate the
    synthetic layer, write both tables to data/processed/ as Parquet
    (DuckDB/pandas native writer, no new dependency)."""
    from src.data.config import load_paths

    paths = paths or load_paths()
    assumptions = assumptions or load_synthetic_assumptions()
    real_df = load_real_transactions(paths)
    sessions_df, attempts_df = generate_payment_sessions_and_attempts(real_df, assumptions, seed)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    # DuckDB's native Parquet writer (COPY ... TO ... (FORMAT PARQUET)) is
    # used instead of pandas.to_parquet, the same choice
    # src.features.build_baseline_dataset makes — it avoids adding a new
    # dependency (pyarrow/fastparquet) purely to write Parquet when DuckDB
    # (already a project dependency) can do it natively.
    out_con = get_connection()
    out_con.register("sessions_df", sessions_df)
    out_con.register("attempts_df", attempts_df)
    sessions_path = paths.processed_dir / "payment_sessions_v1.parquet"
    attempts_path = paths.processed_dir / "payment_attempts_v1.parquet"
    out_con.execute(f"COPY sessions_df TO '{sessions_path}' (FORMAT PARQUET)")
    out_con.execute(f"COPY attempts_df TO '{attempts_path}' (FORMAT PARQUET)")
    return sessions_df, attempts_df


if __name__ == "__main__":
    sessions_df, attempts_df = build_payments_dataset()
    print(f"payment_sessions: {len(sessions_df):,} rows")
    print(f"payment_attempts: {len(attempts_df):,} rows")
