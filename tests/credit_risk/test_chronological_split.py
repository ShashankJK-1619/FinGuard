import pandas as pd
import pytest

from src.credit_risk.chronological_split import (
    BENCHMARK,
    EXCLUDED_TOO_RECENT,
    SECONDARY_PROXY_SPLIT_DESIGN_FILE,
    TRAIN,
    VALIDATION,
    SplitDesign,
    SplitDesignError,
    assign_split_bucket,
    build_chronological_split,
    load_split_design,
)
from src.credit_risk.schema import (
    LABEL_STATUS_12M_PROXY_COL,
    LABEL_STATUS_CENSORED,
    LABEL_STATUS_DEFAULT,
    LABEL_STATUS_NON_DEFAULT,
    ORIGINATION_DATE_COL,
)

VALID_DESIGN_KWARGS = dict(
    data_as_of_date=pd.Timestamp("2019-04-01"),
    outcome_window_months=12,
    train_start=pd.Timestamp("2007-06-01"),
    train_end=pd.Timestamp("2014-12-31"),
    validation_start=pd.Timestamp("2015-01-01"),
    validation_end=pd.Timestamp("2016-12-31"),
    benchmark_start=pd.Timestamp("2017-01-01"),
    benchmark_end=pd.Timestamp("2017-12-31"),
    excluded_too_recent_start=pd.Timestamp("2018-01-01"),
    target_column="default_12m_last_pymnt_proxy",
    label_status_column="label_status_12m_proxy",
)


def test_load_real_split_design_is_valid():
    """Default = the PRIMARY terminal_adverse_outcome split (governance
    correction) -- see configs/credit_risk_split_design_primary_terminal.json."""
    design = load_split_design()
    assert design.train_end < design.validation_start
    assert design.validation_end < design.benchmark_start
    assert design.benchmark_end < design.excluded_too_recent_start
    assert design.target_column == "terminal_adverse_outcome"
    assert design.outcome_window_months == 60


def test_load_secondary_proxy_split_design_is_valid_and_distinct_from_primary():
    """The SECONDARY, sensitivity-analysis-only proxy-target split is
    preserved unchanged from the original Milestone 2 draft, loadable via
    its own explicit config path -- distinct boundaries from the primary
    split, since the proxy target's fixed 12-month window does not need
    (and should not reuse) the terminal target's 60-month resolution
    requirement."""
    primary = load_split_design()
    secondary = load_split_design(SECONDARY_PROXY_SPLIT_DESIGN_FILE)
    assert secondary.train_end < secondary.validation_start
    assert secondary.validation_end < secondary.benchmark_start
    assert secondary.benchmark_end < secondary.excluded_too_recent_start
    assert secondary.target_column == "default_12m_last_pymnt_proxy"
    assert secondary.outcome_window_months == 12
    assert secondary.target_column != primary.target_column
    assert secondary.train_end != primary.train_end


def test_rejects_out_of_order_boundaries():
    with pytest.raises(SplitDesignError):
        SplitDesign(**{**VALID_DESIGN_KWARGS, "train_end": pd.Timestamp("2020-01-01")})


def test_rejects_benchmark_window_not_yet_resolved():
    # benchmark_end so close to data_as_of_date that +12mo exceeds it.
    with pytest.raises(SplitDesignError):
        SplitDesign(**{**VALID_DESIGN_KWARGS, "benchmark_end": pd.Timestamp("2018-12-31")})


def test_assign_split_bucket_is_chronologically_ordered():
    design = SplitDesign(**VALID_DESIGN_KWARGS)
    dates = pd.Series(
        [
            pd.Timestamp("2010-06-01"),  # train
            pd.Timestamp("2015-06-01"),  # validation
            pd.Timestamp("2017-06-01"),  # benchmark
            pd.Timestamp("2018-06-01"),  # excluded_too_recent
        ]
    )
    buckets = assign_split_bucket(dates, design)
    assert list(buckets) == [TRAIN, VALIDATION, BENCHMARK, EXCLUDED_TOO_RECENT]


def test_assign_split_bucket_raises_on_dates_before_train_start():
    design = SplitDesign(**VALID_DESIGN_KWARGS)
    dates = pd.Series([pd.Timestamp("2000-01-01")])
    with pytest.raises(SplitDesignError):
        assign_split_bucket(dates, design)


def _toy_loans_df():
    return pd.DataFrame(
        {
            "loan_id": [1, 2, 3, 4, 5],
            ORIGINATION_DATE_COL: [
                pd.Timestamp("2010-06-01"),  # train, default
                pd.Timestamp("2015-06-01"),  # validation, non_default
                pd.Timestamp("2017-06-01"),  # benchmark, non_default
                pd.Timestamp("2018-06-01"),  # excluded_too_recent, censored
                pd.Timestamp("2018-01-01"),  # excluded_too_recent, resolved non_default
            ],
            LABEL_STATUS_12M_PROXY_COL: [
                LABEL_STATUS_DEFAULT,
                LABEL_STATUS_NON_DEFAULT,
                LABEL_STATUS_NON_DEFAULT,
                LABEL_STATUS_CENSORED,
                LABEL_STATUS_NON_DEFAULT,
            ],
        }
    )


def test_build_chronological_split_no_censored_rows_in_labeled_buckets():
    design = SplitDesign(**VALID_DESIGN_KWARGS)
    df = _toy_loans_df()
    out, summary = build_chronological_split(df, design)

    for bucket in (TRAIN, VALIDATION, BENCHMARK):
        bucket_rows = out[out["split_bucket"] == bucket]
        assert (bucket_rows[LABEL_STATUS_12M_PROXY_COL] != LABEL_STATUS_CENSORED).all()

    summary_by_bucket = summary.set_index("split_bucket")
    assert summary_by_bucket.loc[TRAIN, "censored_excluded_rows"] == 0
    assert summary_by_bucket.loc[TRAIN, "all_labeled_rows_have_sufficient_followup"] is True
    assert summary_by_bucket.loc[EXCLUDED_TOO_RECENT, "censored_excluded_rows"] == 1
    assert summary_by_bucket.loc[EXCLUDED_TOO_RECENT, "total_rows"] == 2


def test_build_chronological_split_every_row_accounted_for():
    design = SplitDesign(**VALID_DESIGN_KWARGS)
    df = _toy_loans_df()
    out, summary = build_chronological_split(df, design)
    assert summary["total_rows"].sum() == len(df)
    assert out["split_bucket"].notna().all()


def test_build_chronological_split_prevalence_is_computed_on_labeled_rows_only():
    design = SplitDesign(**VALID_DESIGN_KWARGS)
    df = _toy_loans_df()
    _, summary = build_chronological_split(df, design)
    summary_by_bucket = summary.set_index("split_bucket")
    # train has 1 labeled row, 1 default -> 100% prevalence
    assert summary_by_bucket.loc[TRAIN, "default_prevalence_pct"] == 100.0
    # excluded_too_recent: 1 labeled (non_default) out of 2 total (1 censored) -> 0% prevalence, 1 labeled row
    assert summary_by_bucket.loc[EXCLUDED_TOO_RECENT, "labeled_rows"] == 1
    assert summary_by_bucket.loc[EXCLUDED_TOO_RECENT, "default_prevalence_pct"] == 0.0
