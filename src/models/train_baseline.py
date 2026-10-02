"""Milestone 3: chronological split, interpretable logistic-regression
baseline, evaluation, and threshold analysis.

Reads Milestone 2's data/processed/baseline_features_v1.parquet only —
train_transaction/train_identity derived, test_transaction/test_identity
are never loaded anywhere in this project. The "holdout" split below is a
time-later slice of this same labeled training data, not the Kaggle test
set, and it is touched exactly once, at the end, for final evaluation —
never for threshold selection (see `main`).

Run: .venv/bin/python -m src.models.train_baseline
Writes: data/interim/milestone3_metrics.json (reproducibility artifact;
the prose report is docs/milestone3_baseline_model.md).
"""
from __future__ import annotations

import json
import sys

import duckdb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.config import Paths, load_paths
from src.models.preprocessing import build_preprocessor
from src.models.schema import MASKED_CODE_NUMERIC_FEATURES, TARGET_COL
from src.models.split import chronological_split

DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_baseline_table(paths: Paths | None = None) -> pd.DataFrame:
    paths = paths or load_paths()
    path = paths.processed_dir / "baseline_features_v1.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `.venv/bin/python -m "
            "src.features.build_baseline_dataset` first (Milestone 2)."
        )
    con = duckdb.connect(database=":memory:")
    return con.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()


def build_model_pipeline(numeric_features: list[str] | None = None) -> Pipeline:
    """class_weight="balanced" rather than resampling (SMOTE etc.): it
    rebalances the loss function to account for the ~3.5% fraud prevalence
    without synthesizing data or adding a resampling step beyond Milestone
    3's scope, and it leaves the fitted model a plain logistic regression
    whose coefficients are still directly interpretable.

    `numeric_features` defaults to schema.NUMERIC_FEATURES (every Milestone
    3 call site is unaffected); Milestone 4's time-feature ablation passes
    schema.NUMERIC_FEATURES_NO_TIME_DRIFT instead, see
    src/models/train_calibrated.py.
    """
    from src.models.schema import NUMERIC_FEATURES

    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Same metric set as `evaluate`, but takes raw arrays rather than a
    fitted pipeline + DataFrame. Added in Milestone 4 so calibrated
    probability arrays (which aren't produced by calling .predict_proba on
    a pipeline — see src/models/calibration.py) can be scored with the
    exact same metric logic as the Milestone 3 pipeline-based `evaluate`,
    instead of a second, divergent implementation.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

    return {
        "n_rows": len(y_true),
        "fraud_prevalence": float(np.mean(y_true)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "calibration_curve": {
            "mean_predicted_prob": [float(x) for x in prob_pred],
            "observed_fraud_rate": [float(x) for x in prob_true],
        },
    }


def evaluate(pipeline: Pipeline, df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> dict:
    y_true = df[TARGET_COL].to_numpy()
    y_prob = pipeline.predict_proba(df)[:, 1]
    return evaluate_probs(y_true, y_prob, threshold)


def threshold_analysis(pipeline: Pipeline, df: pd.DataFrame, thresholds: list[float] = THRESHOLD_GRID) -> list[dict]:
    """Precision/recall/alert-volume tradeoff across thresholds, computed
    on the VALIDATION split only. No threshold here is labeled "best" —
    that requires a business cost function (relative cost of a missed
    fraud vs. a false decline/manual-review alert) that has not been
    defined in this milestone.
    """
    y_true = df[TARGET_COL].to_numpy()
    y_prob = pipeline.predict_proba(df)[:, 1]
    n = len(df)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n_alerts = int(tp + fp)
        rows.append(
            {
                "threshold": t,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "alert_volume": n_alerts,
                "alert_rate": n_alerts / n,
            }
        )
    return rows


def extract_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]

    def _source_column(name: str) -> str:
        # ColumnTransformer prefixes e.g. "numeric__card2" or
        # "categorical__ProductCD_W"; recover the underlying baseline-table
        # column for the masked-code filter below.
        stripped = name.split("__", 1)[-1]
        for col in MASKED_CODE_NUMERIC_FEATURES:
            if stripped == col:
                return col
        return stripped

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
        }
    )
    df["source_column"] = df["feature"].map(_source_column)
    df["is_masked_code"] = df["source_column"].isin(MASKED_CODE_NUMERIC_FEATURES)
    return df.sort_values("coefficient", ascending=False).reset_index(drop=True)


def main() -> int:
    paths = load_paths()
    df = load_baseline_table(paths)

    train_df, val_df, holdout_df, split_report = chronological_split(df)
    print("=== Split report ===")
    print(json.dumps(split_report, indent=2))

    pipeline = build_model_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])

    val_metrics = evaluate(pipeline, val_df)
    holdout_metrics = evaluate(pipeline, holdout_df)
    threshold_rows = threshold_analysis(pipeline, val_df)
    coef_table = extract_coefficients(pipeline)

    print("\n=== Validation metrics ===")
    print(json.dumps(val_metrics, indent=2))
    print("\n=== Holdout metrics ===")
    print(json.dumps(holdout_metrics, indent=2))
    print("\n=== Threshold analysis (validation split) ===")
    print(json.dumps(threshold_rows, indent=2))
    print("\n=== Top 10 positive coefficients ===")
    print(coef_table.head(10).to_string(index=False))
    print("\n=== Top 10 negative coefficients ===")
    print(coef_table.tail(10).to_string(index=False))

    out_path = paths.interim_dir / "milestone3_metrics.json"
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "split_report": split_report,
                "validation_metrics": val_metrics,
                "holdout_metrics": holdout_metrics,
                "threshold_analysis": threshold_rows,
                "coefficients": coef_table.to_dict(orient="records"),
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
