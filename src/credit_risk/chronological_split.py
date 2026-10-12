"""Module C, Milestone 2, Section G (revised by the target-governance
correction): chronological train/validation/temporal-benchmark split by
real loan origination date.

Design (full rationale in docs/credit_risk_split_design.md): a strict,
non-overlapping partition of `origination_date` into three ordered
windows -- train (oldest), validation (middle), temporal benchmark
(newest fully-resolved) -- plus an explicit fourth bucket,
`excluded_too_recent`, for originations too close to the data snapshot to
reliably evaluate the target's own outcome window. No row is ever
silently dropped: every input row lands in exactly one of the four named
buckets, and the function's return value accounts for all of them.

There are now TWO split designs, loadable via this same module (see
`configs/credit_risk_target_governance.json` for which target each
belongs to):
  - `credit_risk_split_design_primary_terminal.json` (the default --
    `DEFAULT_SPLIT_DESIGN_FILE`) for the PRIMARY `terminal_adverse_outcome`
    target, with `outcome_window_months=60` (the real max LendingClub
    term) so that every benchmark-bucket row has had enough time to reach
    its own maturity, not just a fixed 12-month window.
  - `credit_risk_split_design_secondary_proxy.json`
    (`SECONDARY_PROXY_SPLIT_DESIGN_FILE`) for the SECONDARY
    `default_12m_last_pymnt_proxy` sensitivity-analysis target, unchanged
    from the original Milestone 2 draft, with `outcome_window_months=12`.
The two designs deliberately use different boundaries -- the terminal
target's variable, loan-specific maturity window makes the proxy target's
boundaries invalid for it (see docs/credit_risk_split_design.md for the
full rationale and empirical censoring tables for both).

Within each of the three *evaluation-eligible* buckets (train/validation/
benchmark), rows whose configured `label_status_column` is censored are
further excluded from that bucket's labeled count and reported
separately -- per the explicit project rule, a censored/unresolved
observation must never appear in a labeled train or evaluation set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.credit_risk.schema import (
    LABEL_STATUS_CENSORED,
    LABEL_STATUS_DEFAULT,
    ORIGINATION_DATE_COL,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Default = the PRIMARY terminal-adverse-outcome split (see module
# docstring and configs/credit_risk_target_governance.json).
DEFAULT_SPLIT_DESIGN_FILE = (
    PROJECT_ROOT / "configs" / "credit_risk_split_design_primary_terminal.json"
)
# The SECONDARY, sensitivity-analysis-only proxy-target split.
SECONDARY_PROXY_SPLIT_DESIGN_FILE = (
    PROJECT_ROOT / "configs" / "credit_risk_split_design_secondary_proxy.json"
)

TRAIN = "train"
VALIDATION = "validation"
BENCHMARK = "benchmark"
EXCLUDED_TOO_RECENT = "excluded_too_recent"


class SplitDesignError(ValueError):
    pass


@dataclass(frozen=True)
class SplitDesign:
    data_as_of_date: pd.Timestamp
    outcome_window_months: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    benchmark_start: pd.Timestamp
    benchmark_end: pd.Timestamp
    excluded_too_recent_start: pd.Timestamp
    target_column: str
    label_status_column: str

    def __post_init__(self) -> None:
        boundaries = [
            ("train_start", self.train_start),
            ("train_end", self.train_end),
            ("validation_start", self.validation_start),
            ("validation_end", self.validation_end),
            ("benchmark_start", self.benchmark_start),
            ("benchmark_end", self.benchmark_end),
            ("excluded_too_recent_start", self.excluded_too_recent_start),
        ]
        for i in range(len(boundaries) - 1):
            name_a, val_a = boundaries[i]
            name_b, val_b = boundaries[i + 1]
            if val_a > val_b:
                raise SplitDesignError(
                    f"Split boundaries must be strictly chronological: {name_a}={val_a} "
                    f"comes after {name_b}={val_b}"
                )
        window_end_of_benchmark = self.benchmark_end + pd.DateOffset(months=self.outcome_window_months)
        if window_end_of_benchmark > self.data_as_of_date:
            raise SplitDesignError(
                f"benchmark_end={self.benchmark_end} plus the {self.outcome_window_months}-month "
                f"observation window ({window_end_of_benchmark}) exceeds data_as_of_date="
                f"{self.data_as_of_date} -- the benchmark window would not be fully resolved."
            )


def load_split_design(config_file: Path = DEFAULT_SPLIT_DESIGN_FILE) -> SplitDesign:
    with open(config_file) as f:
        cfg = json.load(f)
    return SplitDesign(
        data_as_of_date=pd.Timestamp(cfg["data_as_of_date"]),
        outcome_window_months=cfg["outcome_window_months"],
        train_start=pd.Timestamp(cfg["train_start"]),
        train_end=pd.Timestamp(cfg["train_end"]),
        validation_start=pd.Timestamp(cfg["validation_start"]),
        validation_end=pd.Timestamp(cfg["validation_end"]),
        benchmark_start=pd.Timestamp(cfg["benchmark_start"]),
        benchmark_end=pd.Timestamp(cfg["benchmark_end"]),
        excluded_too_recent_start=pd.Timestamp(cfg["excluded_too_recent_start"]),
        target_column=cfg["target_column"],
        label_status_column=cfg["label_status_column"],
    )


def assign_split_bucket(origination_date: pd.Series, design: SplitDesign) -> pd.Series:
    bucket = pd.Series([None] * len(origination_date), index=origination_date.index, dtype=object)
    bucket[(origination_date >= design.train_start) & (origination_date <= design.train_end)] = TRAIN
    bucket[
        (origination_date >= design.validation_start) & (origination_date <= design.validation_end)
    ] = VALIDATION
    bucket[
        (origination_date >= design.benchmark_start) & (origination_date <= design.benchmark_end)
    ] = BENCHMARK
    bucket[origination_date >= design.excluded_too_recent_start] = EXCLUDED_TOO_RECENT
    if bucket.isna().any():
        n_unassigned = int(bucket.isna().sum())
        raise SplitDesignError(
            f"{n_unassigned} row(s) fall outside every configured split window "
            f"(before train_start={design.train_start}) -- extend the design or "
            "investigate these rows rather than silently dropping them."
        )
    return bucket


def build_chronological_split(
    df: pd.DataFrame, design: SplitDesign | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_with_split_bucket, summary_df). `summary_df` has one row
    per bucket with: total rows, rows excluded as censored, labeled rows
    kept, default prevalence among labeled rows, and origination date
    bounds actually observed -- this is the Section G report table."""
    design = design or load_split_design()
    out = df.copy()
    out["split_bucket"] = assign_split_bucket(out[ORIGINATION_DATE_COL], design)

    rows = []
    for bucket in (TRAIN, VALIDATION, BENCHMARK, EXCLUDED_TOO_RECENT):
        sub = out[out["split_bucket"] == bucket]
        total = len(sub)
        censored = int((sub[design.label_status_column] == LABEL_STATUS_CENSORED).sum())
        labeled = sub[sub[design.label_status_column] != LABEL_STATUS_CENSORED]
        n_labeled = len(labeled)
        n_default = int((labeled[design.label_status_column] == LABEL_STATUS_DEFAULT).sum())
        prevalence = (n_default / n_labeled * 100) if n_labeled else float("nan")
        rows.append(
            {
                "split_bucket": bucket,
                "total_rows": total,
                "censored_excluded_rows": censored,
                "labeled_rows": n_labeled,
                "default_count": n_default,
                "default_prevalence_pct": round(prevalence, 3) if n_labeled else None,
                "origination_min": sub[ORIGINATION_DATE_COL].min() if total else None,
                "origination_max": sub[ORIGINATION_DATE_COL].max() if total else None,
                "all_labeled_rows_have_sufficient_followup": bool(censored == 0)
                if bucket != EXCLUDED_TOO_RECENT
                else None,
            }
        )
    summary = pd.DataFrame(rows)

    # Hard guarantee check for the three evaluation-eligible buckets: no
    # censored row may ever be included in a *labeled* split output. This
    # does not apply to excluded_too_recent, which is reported but not a
    # labeled split.
    for bucket in (TRAIN, VALIDATION, BENCHMARK):
        bucket_rows = out[out["split_bucket"] == bucket]
        labeled_subset = bucket_rows[bucket_rows[design.label_status_column] != LABEL_STATUS_CENSORED]
        if (labeled_subset[design.label_status_column] == LABEL_STATUS_CENSORED).any():
            raise SplitDesignError(f"Censored rows leaked into labeled bucket '{bucket}'")

    return out, summary
