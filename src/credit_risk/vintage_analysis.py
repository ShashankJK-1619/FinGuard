"""Module C, Milestone 3, Section I: vintage/stability analysis across
origination years. Descriptive only -- no retraining, no threshold
re-tuning, no feature changes per year.

Scope note on the `excluded_too_recent` bucket: `docs/
credit_risk_split_design.md` explicitly flags that bucket's already-
resolved rows (loans originated 2014+ that nonetheless matured and
resolved before `data_as_of_date`) as "a useful empirical cross-check,"
while being clear it must "never [be] one of the three evaluation-
eligible splits" for training, threshold selection, or benchmark metrics.
This module honors that distinction: it applies the already-fit (train-
only) model to every labeled row across ALL origination years --
including `excluded_too_recent`'s labeled subset -- purely to describe
how stable performance looks by year, never to fit, tune, or report a
headline benchmark number. Every output row is tagged with its
`split_bucket` so a reader can see exactly which rows are evaluation-
eligible (train/validation/benchmark) vs. descriptive-only
(excluded_too_recent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


class VintageAnalysisError(ValueError):
    pass


def compute_vintage_stability(
    origination_date: pd.Series,
    split_bucket: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    if not (len(origination_date) == len(split_bucket) == len(y_true) == len(y_prob)):
        raise VintageAnalysisError(
            "origination_date, split_bucket, y_true, and y_prob must all be the same length"
        )
    df = pd.DataFrame(
        {
            "origination_year": pd.to_datetime(origination_date).dt.year,
            "split_bucket": np.asarray(split_bucket),
            "y_true": np.asarray(y_true).astype(int),
            "y_prob": np.asarray(y_prob).astype(float),
        }
    )

    rows = []
    for year, sub in df.groupby("origination_year"):
        n = len(sub)
        prevalence = float(sub["y_true"].mean() * 100)
        buckets_present = sorted(sub["split_bucket"].unique().tolist())
        is_evaluation_eligible = buckets_present != ["excluded_too_recent"]
        row = {
            "origination_year": int(year),
            "split_buckets_present": buckets_present,
            "evaluation_eligible": is_evaluation_eligible,
            "n": n,
            "default_prevalence_pct": round(prevalence, 3),
        }
        if sub["y_true"].nunique() < 2:
            row["roc_auc"] = None
            row["brier_score"] = round(float(brier_score_loss(sub["y_true"], sub["y_prob"])), 5)
            row["note"] = "ROC-AUC undefined -- only one class present in this year's labeled rows"
        else:
            row["roc_auc"] = round(float(roc_auc_score(sub["y_true"], sub["y_prob"])), 4)
            row["brier_score"] = round(float(brier_score_loss(sub["y_true"], sub["y_prob"])), 5)
            row["note"] = ""
        rows.append(row)

    return pd.DataFrame(rows).sort_values("origination_year").reset_index(drop=True)
