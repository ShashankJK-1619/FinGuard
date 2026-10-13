"""Tests for Module C, Milestone 3, Section I:
src/credit_risk/vintage_analysis.py -- descriptive-only stability analysis
across origination years, tagging evaluation-eligibility per year."""
import numpy as np
import pandas as pd
import pytest

from src.credit_risk.vintage_analysis import compute_vintage_stability


def test_mismatched_lengths_raise():
    with pytest.raises(Exception):
        compute_vintage_stability(
            pd.Series([pd.Timestamp("2012-01-01")]),
            pd.Series(["train"]),
            np.array([0, 1]),
            np.array([0.1, 0.2]),
        )


def test_groups_by_origination_year_and_tags_split_buckets_present():
    origination_date = pd.Series(
        [pd.Timestamp("2012-01-01"), pd.Timestamp("2012-06-01"), pd.Timestamp("2013-01-01")]
    )
    split_bucket = pd.Series(["train", "train", "validation"])
    y_true = np.array([0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2])
    out = compute_vintage_stability(origination_date, split_bucket, y_true, y_prob)
    assert sorted(out["origination_year"].tolist()) == [2012, 2013]
    row_2012 = out[out["origination_year"] == 2012].iloc[0]
    assert row_2012["split_buckets_present"] == ["train"]
    assert row_2012["n"] == 2


def test_excluded_too_recent_only_year_is_not_evaluation_eligible():
    origination_date = pd.Series([pd.Timestamp("2015-01-01"), pd.Timestamp("2015-06-01")])
    split_bucket = pd.Series(["excluded_too_recent", "excluded_too_recent"])
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    out = compute_vintage_stability(origination_date, split_bucket, y_true, y_prob)
    row = out[out["origination_year"] == 2015].iloc[0]
    assert bool(row["evaluation_eligible"]) is False


def test_year_with_any_evaluation_eligible_bucket_present_is_eligible():
    # A year whose rows are a MIX of excluded_too_recent and e.g. benchmark
    # (shouldn't really happen given chronological assignment, but the
    # eligibility rule itself is "not purely excluded_too_recent").
    origination_date = pd.Series([pd.Timestamp("2013-01-01"), pd.Timestamp("2013-06-01")])
    split_bucket = pd.Series(["benchmark", "excluded_too_recent"])
    y_true = np.array([0, 1])
    y_prob = np.array([0.1, 0.9])
    out = compute_vintage_stability(origination_date, split_bucket, y_true, y_prob)
    row = out[out["origination_year"] == 2013].iloc[0]
    assert bool(row["evaluation_eligible"]) is True


def test_single_class_year_reports_none_roc_auc_with_explanatory_note():
    origination_date = pd.Series([pd.Timestamp("2014-01-01"), pd.Timestamp("2014-06-01")])
    split_bucket = pd.Series(["excluded_too_recent", "excluded_too_recent"])
    y_true = np.array([0, 0])  # only one class present
    y_prob = np.array([0.1, 0.2])
    out = compute_vintage_stability(origination_date, split_bucket, y_true, y_prob)
    row = out[out["origination_year"] == 2014].iloc[0]
    assert row["roc_auc"] is None
    assert "undefined" in row["note"]
    assert row["brier_score"] is not None


def test_rows_sorted_by_origination_year_ascending():
    origination_date = pd.Series(
        [pd.Timestamp("2014-01-01"), pd.Timestamp("2012-01-01"), pd.Timestamp("2013-01-01")]
    )
    split_bucket = pd.Series(["excluded_too_recent", "train", "benchmark"])
    y_true = np.array([0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2])
    out = compute_vintage_stability(origination_date, split_bucket, y_true, y_prob)
    assert out["origination_year"].tolist() == [2012, 2013, 2014]
