"""Milestone 5: transparent nonlinear model comparison.

Adds one strong nonlinear tabular baseline — `HistGradientBoostingClassifier`
— and compares it fairly against Milestone 3's class-weighted logistic
regression, reusing every piece of leakage-safe infrastructure Milestones
3-4 already built: the same transparent feature set (schema.py; still no
masked `C*`/`D*`/`M*`/`V*`/`id_*` columns anywhere), the same chronological
train/val/temporal-benchmark split (`src.models.split.chronological_split`,
untouched defaults), the same cal_fit/selection sub-split and calibration
methodology (`src.models.calibration`, `src.models.train_calibrated`), and
the same illustrative cost-based decisioning framework
(`src.models.cost_model`, `configs/cost_assumptions.json`, unchanged).

## Why HistGradientBoostingClassifier

Preferred per explicit instruction order: HistGradientBoostingClassifier,
then XGBoost/LightGBM only if justified. It ships with the project's
already-pinned `scikit-learn>=1.4,<2` dependency (no new dependency, no new
runtime/build complexity — consistent with the project's "no Spark, no
cloud infra, no deep learning" minimal-dependency stance in README.md), it
natively supports missing values (no imputation needed for numeric columns)
and native categorical splits (no one-hot explosion), and it is a strong,
well-understood gradient-boosted-tree baseline for tabular data generally.
No dependency/runtime tradeoff was found that justified reaching for
XGBoost or LightGBM instead.

## Preprocessing differs from the linear pipeline, on purpose

`src.models.preprocessing.build_preprocessor` (median-impute + scale +
one-hot, built for a linear model) is NOT reused as-is here. Scaling is
irrelevant to a tree model (splits are threshold-based, invariant to
monotonic rescaling), and one-hot-encoding a categorical column throws away
the native categorical-split capability that is one of this model's actual
advantages. Instead `build_tree_preprocessor` below:
  - passes numeric columns straight through unscaled, un-imputed — HGB
    treats missing numeric values as their own learned split direction
    natively, so SimpleImputer's median-fill would only destroy signal
    ("missing" is informative here, same reasoning `preprocessing.py`
    already uses for `add_indicator=True`, just handled natively instead of
    via an extra indicator column);
  - ordinal-encodes categorical columns (impute missing as an explicit
    "missing" level first, matching `preprocessing.py`'s policy) and tells
    HistGradientBoostingClassifier's own `categorical_features` mask which
    output columns are categorical, so it treats the encoded integers as
    discrete categories, not an ordinal magnitude;
  - imputes binary flags the same way `preprocessing.py` does.
This is still the exact same *feature set* (NUMERIC_FEATURES /
CATEGORICAL_FEATURES / BINARY_FEATURES, optionally
NUMERIC_FEATURES_NO_TIME_DRIFT) — Requirement A's "same transparent feature
set" is about which columns are modeled, not about reusing a
linear-model-shaped preprocessing pipeline verbatim for a model family it
wasn't designed for.

## Hyperparameter tuning (train-only, time-aware, modest)

`tune_hyperparameters` never touches `cal_fit`, `selection`, or
`temporal_benchmark` — only `train_df`, internally resplit by
`sklearn.model_selection.TimeSeriesSplit` (forward-chaining, time-ordered
folds) so even the internal tuning score respects "never evaluate on data
from before the thing you're predicting" at a smaller scale. The search
space is a short, explicitly enumerated list of 4 candidate configurations
(`HGB_PARAM_CANDIDATES`) — not a brute-force grid — varying the three knobs
most likely to matter for this dataset size/imbalance: tree complexity
(`max_leaf_nodes`), the learning-rate/iteration-count tradeoff, and L2
regularization strength. Selection metric is mean PR-AUC (average
precision) across folds, matching the project's existing preference for
PR-AUC over accuracy on a ~3.5% prevalence problem. Runtime is measured and
reported alongside the chosen configuration (see `main`).

## Calibration, decisioning, and feature importance

`calibration_comparison`/`select_best_calibration_method`
(`src.models.train_calibrated`) are reused UNCHANGED on this model's raw
probabilities — fit on `cal_fit`, evaluated on `selection` and
`temporal_benchmark`, never the reverse. `src.models.cost_model` is reused
UNCHANGED with the same `configs/cost_assumptions.json` values and the same
`REVIEW_THRESHOLD_GRID`/`DECLINE_THRESHOLD_GRID` — no cost assumption is
touched to flatter either model. Feature importance uses permutation
importance (`sklearn.inspection.permutation_importance`) on the `selection`
split: HistGradientBoostingClassifier exposes no native
`feature_importances_` (unlike e.g. RandomForestClassifier), so permutation
importance is the only importance signal reported here, and it is reported
as exactly that — a measure of how much held-out predictive SCORE drops
when a column is shuffled, not a causal claim about fraud mechanics (same
caveat the project already applies to logistic coefficients).

Run: .venv/bin/python -m src.models.train_nonlinear
Requires: data/processed/baseline_features_v1.parquet (Milestone 2).
Writes: data/interim/milestone5_metrics.json.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.data.config import load_paths
from src.models.calibration import apply_calibrator, fit_calibrator, reliability_table
from src.models.cost_model import (
    COST_ASSUMPTIONS_DISCLAIMER,
    CostAssumptions,
    evaluate_decision_policy,
    evaluate_policy_grid,
    load_cost_assumptions,
    sensitivity_analysis,
)
from src.models.schema import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MASKED_CODE_NUMERIC_FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_FEATURES_NO_TIME_DRIFT,
    TARGET_COL,
    TIME_COL,
)
from src.models.split import chronological_split, chronological_split_two_way
from src.models.train_baseline import evaluate_probs, extract_coefficients, load_baseline_table
from src.models.train_calibrated import (
    CALIBRATION_FIT_FRAC_OF_VALIDATION,
    DECLINE_THRESHOLD_GRID,
    REVIEW_THRESHOLD_GRID,
    build_cost_variations,
    calibration_comparison,
    fit_baseline_pipeline,
    select_best_calibration_method,
)

RANDOM_STATE = 42

# Requirement B: a short, explicitly enumerated search space, not a
# brute-force grid. Each candidate varies tree complexity, the
# learning-rate/iteration tradeoff, and L2 regularization — see module
# docstring for why these three.
HGB_PARAM_CANDIDATES: list[dict] = [
    {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1, "l2_regularization": 0.0},
    {"max_iter": 200, "max_leaf_nodes": 31, "learning_rate": 0.05, "l2_regularization": 0.0},
    {"max_iter": 100, "max_leaf_nodes": 15, "learning_rate": 0.1, "l2_regularization": 1.0},
    {"max_iter": 200, "max_leaf_nodes": 63, "learning_rate": 0.05, "l2_regularization": 1.0},
]
N_TUNING_FOLDS = 3

_METRIC_KEYS = ("roc_auc", "pr_auc", "brier_score")


def categorical_feature_mask(numeric_features: list[str] | None = None) -> list[bool]:
    """Boolean mask over the tree preprocessor's OUTPUT columns (numeric
    block, then categorical block, then binary block, matching
    `build_tree_preprocessor`'s transformer order) identifying which
    columns `HistGradientBoostingClassifier.categorical_features` should
    treat as discrete categories rather than ordinal magnitudes.
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    return (
        [False] * len(numeric_features)
        + [True] * len(CATEGORICAL_FEATURES)
        + [False] * len(BINARY_FEATURES)
    )


def build_tree_preprocessor(numeric_features: list[str] | None = None) -> ColumnTransformer:
    """Tree-native preprocessing. See module docstring for why this is a
    deliberately different shape from `src.models.preprocessing.build_preprocessor`
    (no scaling, no numeric imputation, ordinal- not one-hot-encoded
    categoricals) despite modeling the exact same transparent feature set.
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    binary_pipeline = Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent"))])

    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
        ],
        remainder="drop",
    )


def build_nonlinear_pipeline(numeric_features: list[str] | None = None, hgb_params: dict | None = None) -> Pipeline:
    """`numeric_features` defaults to schema.NUMERIC_FEATURES; pass
    schema.NUMERIC_FEATURES_NO_TIME_DRIFT for the Requirement G ablation
    variant, mirroring `train_calibrated.fit_baseline_pipeline`'s pattern.
    `class_weight="balanced"` mirrors Milestone 3's logistic baseline so
    both models address the ~3.5% fraud prevalence the same way (fair
    comparison — Requirement A), rather than giving one model a resampling
    trick the other doesn't get.
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    hgb_params = hgb_params if hgb_params is not None else HGB_PARAM_CANDIDATES[0]
    clf = HistGradientBoostingClassifier(
        categorical_features=categorical_feature_mask(numeric_features),
        class_weight="balanced",
        random_state=RANDOM_STATE,
        **hgb_params,
    )
    return Pipeline(steps=[("preprocess", build_tree_preprocessor(numeric_features)), ("clf", clf)])


def tune_hyperparameters(
    train_df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    param_candidates: list[dict] = HGB_PARAM_CANDIDATES,
    n_folds: int = N_TUNING_FOLDS,
) -> dict:
    """Modest, reproducible, TRAIN-ONLY hyperparameter search (Requirement
    B). `train_df` is time-sorted and then split with
    `sklearn.model_selection.TimeSeriesSplit` into `n_folds` forward-chaining
    (expanding-window) folds — each fold's validation block is strictly
    later in time than everything used to fit it, and every fold lives
    entirely inside TRAIN: `cal_fit`/`selection`/`temporal_benchmark` are
    never passed to this function. Selection metric is mean PR-AUC
    (average precision) across folds, matching this project's existing
    preference for PR-AUC over accuracy/ROC-AUC alone on an imbalanced
    target. Returns every candidate's scores plus the chosen best, and the
    measured wall-clock runtime, so the final report can state exactly what
    was tried and why the winner won.
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    ordered = train_df.sort_values(TIME_COL).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_folds)

    start = time.time()
    candidate_rows = []
    for params in param_candidates:
        fold_scores = []
        for fold_train_idx, fold_val_idx in tscv.split(ordered):
            fold_train_df = ordered.iloc[fold_train_idx]
            fold_val_df = ordered.iloc[fold_val_idx]
            pipeline = build_nonlinear_pipeline(numeric_features=numeric_features, hgb_params=params)
            pipeline.fit(fold_train_df, fold_train_df[TARGET_COL])
            fold_prob = pipeline.predict_proba(fold_val_df)[:, 1]
            fold_scores.append(float(average_precision_score(fold_val_df[TARGET_COL].to_numpy(), fold_prob)))
        candidate_rows.append(
            {"params": params, "fold_pr_auc": fold_scores, "mean_pr_auc": float(np.mean(fold_scores))}
        )
    runtime_seconds = time.time() - start

    best_row = max(candidate_rows, key=lambda r: r["mean_pr_auc"])
    return {
        "candidates": candidate_rows,
        "n_folds": n_folds,
        "selection_metric": "mean PR-AUC (average_precision_score) across train-only TimeSeriesSplit forward-chaining folds",
        "best_params": best_row["params"],
        "best_mean_pr_auc": best_row["mean_pr_auc"],
        "tuning_runtime_seconds": runtime_seconds,
    }


def fit_nonlinear_pipeline(
    train_df: pd.DataFrame, hgb_params: dict, numeric_features: list[str] | None = None
) -> Pipeline:
    """Fits the final nonlinear pipeline architecture on TRAIN ONLY, using
    hyperparameters already chosen by `tune_hyperparameters` (never
    re-tuned here). Mirrors `train_calibrated.fit_baseline_pipeline`.
    """
    pipeline = build_nonlinear_pipeline(numeric_features=numeric_features, hgb_params=hgb_params)
    pipeline.fit(train_df, train_df[TARGET_COL])
    return pipeline


def time_feature_ablation_nonlinear(
    train_df: pd.DataFrame, val_df: pd.DataFrame, temporal_benchmark_df: pd.DataFrame, hgb_params: dict
) -> dict:
    """Requirement G: re-run the nonlinear comparison with day_index/
    week_index excluded (schema.NUMERIC_FEATURES_NO_TIME_DRIFT), on
    identical train/val/temporal-benchmark data, reusing the SAME tuned
    hyperparameters for both variants (only the feature set changes) —
    mirrors `train_calibrated.time_feature_ablation`'s structure exactly,
    including the "no calibrator is fit anywhere here" property (both
    variants are fit on train only and evaluated on val/temporal_benchmark,
    so the fit-vs-evaluate-on-same-data concern does not apply).
    """
    full_pipeline = fit_nonlinear_pipeline(train_df, hgb_params, numeric_features=NUMERIC_FEATURES)
    no_time_pipeline = fit_nonlinear_pipeline(train_df, hgb_params, numeric_features=NUMERIC_FEATURES_NO_TIME_DRIFT)

    def _summary(pipeline: Pipeline, df: pd.DataFrame) -> dict:
        y_true = df[TARGET_COL].to_numpy()
        # RAW, uncalibrated predict_proba output — this ablation isolates
        # ranking performance (ROC-AUC/PR-AUC, which calibration cannot
        # change) one feature set at a time, mirroring
        # `train_calibrated.time_feature_ablation`'s own uncalibrated
        # convention exactly. The "brier_score" reported here is therefore
        # NOT comparable to the CALIBRATED Brier reported elsewhere (e.g.
        # `nonlinear_full_pipeline`'s "calibrated_metrics") — the explicit
        # "brier_score_is_calibrated: False" flag exists so no caller or
        # report can accidentally mix the two.
        y_prob = pipeline.predict_proba(df)[:, 1]
        metrics = evaluate_probs(y_true, y_prob)
        summary = {k: metrics[k] for k in _METRIC_KEYS}
        summary["brier_score_is_calibrated"] = False
        return summary

    return {
        "with_time_drift_features": {
            "validation": _summary(full_pipeline, val_df),
            "temporal_benchmark": _summary(full_pipeline, temporal_benchmark_df),
        },
        "without_time_drift_features": {
            "validation": _summary(no_time_pipeline, val_df),
            "temporal_benchmark": _summary(no_time_pipeline, temporal_benchmark_df),
        },
    }


def permutation_importance_table(
    pipeline: Pipeline,
    df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Permutation importance (Requirement F) computed on `df` (intended to
    be the `selection` split — out-of-sample for the pipeline, which was
    fit on `train` only) by permuting each RAW input column in turn and
    measuring the drop in PR-AUC (average precision), consistent with this
    project's existing PR-AUC preference. Operates on the whole `Pipeline`
    (preprocessing included), so importances are reported in terms of the
    original, named, transparent feature columns — not one-hot/ordinal-
    encoded sub-columns. `is_masked_code` flags the same masked numeric
    codes (card2/card3/card5/addr2) the logistic coefficient table already
    flags, for direct comparison.

    Not a causal measure: a large importance score means shuffling that
    column hurts this model's held-out ranking score, nothing about why.
    HistGradientBoostingClassifier has no native `feature_importances_`
    (unlike e.g. RandomForestClassifier), so permutation importance is the
    only importance signal this module reports — see module docstring.
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    feature_cols = numeric_features + CATEGORICAL_FEATURES + BINARY_FEATURES
    X = df[feature_cols]
    y = df[TARGET_COL].to_numpy()

    result = permutation_importance(
        pipeline, X, y, scoring="average_precision", n_repeats=n_repeats, random_state=random_state
    )
    table = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    table["is_masked_code"] = table["feature"].isin(MASKED_CODE_NUMERIC_FEATURES)
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compare_feature_importance(coef_table: pd.DataFrame, importance_table: pd.DataFrame, top_n: int = 10) -> dict:
    """Requirement F: compares the top-N most-important raw source columns
    between the logistic coefficient table (ranked by |coefficient|, since
    sign only indicates direction, not magnitude of effect) and the
    nonlinear permutation-importance table (already ranked by importance).
    Reports the overlap, not a claim that either ranking is "the truth" —
    the two models can legitimately disagree because logistic coefficients
    describe a linear, additive effect while permutation importance
    reflects however this specific nonlinear model used a column, possibly
    through interactions.

    Granularity caveat: `extract_coefficients`'s `source_column` is, for a
    one-hot-encoded categorical feature, the individual LEVEL name (e.g.
    "day_of_week_index_B"), not the raw column name ("day_of_week_index") —
    the logistic model has a separate coefficient per level, while
    permutation importance here shuffles the whole raw categorical column
    at once. This is not fixed up into a single shared naming scheme: doing
    so would hide a real difference in what each table reports, rather than
    clarify it. Numeric/binary columns, which aren't one-hot-encoded,
    compare cleanly on name.
    """
    logistic_top = (
        coef_table.assign(abs_coefficient=coef_table["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)["source_column"]
        .drop_duplicates()
        .head(top_n)
        .tolist()
    )
    nonlinear_top = importance_table["feature"].head(top_n).tolist()
    agree = sorted(set(logistic_top) & set(nonlinear_top))
    only_logistic = sorted(set(logistic_top) - set(nonlinear_top))
    only_nonlinear = sorted(set(nonlinear_top) - set(logistic_top))
    return {
        "top_n": top_n,
        "logistic_top_features": logistic_top,
        "nonlinear_top_features": nonlinear_top,
        "agree_in_both_top_n": agree,
        "only_in_logistic_top_n": only_logistic,
        "only_in_nonlinear_top_n": only_nonlinear,
    }


def decision_policy_comparison(
    y_select: np.ndarray,
    calibrated_probs_select: np.ndarray,
    y_temporal_benchmark: np.ndarray,
    calibrated_probs_temporal_benchmark: np.ndarray,
    costs: CostAssumptions,
) -> dict:
    """Requirement E: the SAME illustrative approve/review/decline cost
    framework and threshold grids as `train_calibrated.main`, applied to
    one model's calibrated probabilities. Chooses the lowest-illustrative-
    cost (review, decline) policy on `selection` only, then evaluates that
    SAME policy once on `temporal_benchmark` — never re-optimized there.
    Returns the full selection-split grid plus the single temporal-
    benchmark evaluation, so logistic and nonlinear results can be read
    side by side under identical cost assumptions.
    """
    grid = evaluate_policy_grid(y_select, calibrated_probs_select, REVIEW_THRESHOLD_GRID, DECLINE_THRESHOLD_GRID, costs)
    best_policy = grid.iloc[0]
    temporal_benchmark_eval = evaluate_decision_policy(
        y_temporal_benchmark,
        calibrated_probs_temporal_benchmark,
        float(best_policy["review_threshold"]),
        float(best_policy["decline_threshold"]),
        costs,
    )
    return {
        "threshold_cost_grid_selection": grid.to_dict(orient="records"),
        "best_policy_selection": best_policy.to_dict(),
        "temporal_benchmark_policy_evaluation": temporal_benchmark_eval,
    }


def nonlinear_full_pipeline(
    train_df: pd.DataFrame,
    cal_fit_df: pd.DataFrame,
    select_df: pd.DataFrame,
    temporal_benchmark_df: pd.DataFrame,
    hgb_params: dict,
    costs: CostAssumptions,
    numeric_features: list[str] | None = None,
) -> dict:
    """Runs ONE nonlinear feature-set configuration through the *entire*
    leakage-safe pipeline end to end, so that every reported number for that
    configuration — calibrated or decisioning — comes from the exact same
    code path as every other configuration this function is called with:

      1. fit on `train_df` only (never cal_fit/selection/temporal_benchmark);
      2. raw probabilities on cal_fit/selection/temporal_benchmark;
      3. sigmoid vs. isotonic comparison (`calibration_comparison`): each
         calibrator is `.fit()` on `cal_fit` ONLY, then evaluated on
         `selection`/`temporal_benchmark` — never the reverse;
      4. `select_best_calibration_method` picks sigmoid unless isotonic
         clears the 3%-relative-Brier-improvement bar (same rule used
         everywhere else in this project);
      5. that single chosen calibrator is applied to `selection` and
         `temporal_benchmark` to produce this configuration's CALIBRATED
         probabilities, and `evaluate_probs` reports CALIBRATED ROC-AUC/
         PR-AUC/Brier from those — never the raw, uncalibrated Brier that
         `time_feature_ablation_nonlinear` reports (see that function's
         docstring: the two must never be compared as if they were the same
         quantity);
      6. `decision_policy_comparison` chooses the lowest-illustrative-cost
         `(review_threshold, decline_threshold)` policy on `selection` only
         (using the `costs` object the CALLER passes in, so every
         configuration this function is called with — logistic, nonlinear
         with day_index/week_index, nonlinear without — is scored against
         byte-for-byte the same cost assumptions and threshold grids; no
         configuration is allowed to get a different `costs` instance), then
         evaluates that same policy once, unmodified, on `temporal_benchmark`.

    This exists so the "preferred configuration" reported in
    docs/milestone5_nonlinear_comparison.md is never a different code path
    (or a bare ranking-only ablation) from the configuration whose
    calibrated-and-decisioning numbers are actually shown in that doc's
    headline tables — see the Milestone 5 consistency-pass note in
    `main()` and in the doc itself.
    """
    numeric_features = numeric_features if numeric_features is not None else NUMERIC_FEATURES
    y_cal_fit = cal_fit_df[TARGET_COL].to_numpy()
    y_select = select_df[TARGET_COL].to_numpy()
    y_temporal_benchmark = temporal_benchmark_df[TARGET_COL].to_numpy()

    pipeline = fit_nonlinear_pipeline(train_df, hgb_params, numeric_features=numeric_features)
    raw_cal_fit = pipeline.predict_proba(cal_fit_df)[:, 1]
    raw_select = pipeline.predict_proba(select_df)[:, 1]
    raw_temporal_benchmark = pipeline.predict_proba(temporal_benchmark_df)[:, 1]

    comparison = calibration_comparison(
        y_cal_fit, raw_cal_fit, y_select, raw_select, y_temporal_benchmark, raw_temporal_benchmark
    )
    best_method = select_best_calibration_method(comparison)
    calibrator = fit_calibrator(best_method, raw_cal_fit, y_cal_fit)
    calibrated_select = apply_calibrator(calibrator, raw_select)
    calibrated_temporal_benchmark = apply_calibrator(calibrator, raw_temporal_benchmark)

    calibrated_metrics = {
        "selection": evaluate_probs(y_select, calibrated_select),
        "temporal_benchmark": evaluate_probs(y_temporal_benchmark, calibrated_temporal_benchmark),
    }
    for split_metrics in calibrated_metrics.values():
        split_metrics["brier_score_is_calibrated"] = True

    policy = decision_policy_comparison(
        y_select, calibrated_select, y_temporal_benchmark, calibrated_temporal_benchmark, costs
    )

    return {
        "numeric_features_used": numeric_features,
        "pipeline": pipeline,
        "calibrated_probs": {"selection": calibrated_select, "temporal_benchmark": calibrated_temporal_benchmark},
        "calibration_comparison": comparison,
        "best_calibration_method": best_method,
        "calibrated_metrics": calibrated_metrics,
        "decision_policy": policy,
    }


def main() -> int:
    paths = load_paths()
    df = load_baseline_table(paths)

    train_df, val_df, temporal_benchmark_df, split_report = chronological_split(df)
    split_report = {("temporal_benchmark" if k == "holdout" else k): v for k, v in split_report.items()}
    print("=== Split report (identical boundaries to Milestones 3-4; final segment is temporal_benchmark) ===")
    print(json.dumps(split_report, indent=2))

    cal_fit_df, select_df, val_subsplit_report = chronological_split_two_way(
        val_df, first_frac=CALIBRATION_FIT_FRAC_OF_VALIDATION
    )
    print("\n=== Validation-period sub-split (identical methodology to Milestone 4: cal_fit vs. selection) ===")
    print(json.dumps(val_subsplit_report, indent=2))

    y_cal_fit = cal_fit_df[TARGET_COL].to_numpy()
    y_select = select_df[TARGET_COL].to_numpy()
    y_temporal_benchmark = temporal_benchmark_df[TARGET_COL].to_numpy()

    # Cost assumptions loaded ONCE, here, and passed as the SAME object to
    # the logistic model and BOTH nonlinear configurations below — no
    # configuration gets different cost assumptions or threshold grids
    # (consistency-pass requirement: do not change cost assumptions or
    # thresholds to favor any one configuration).
    costs = load_cost_assumptions()
    print(f"\n=== Cost assumptions (unchanged from Milestone 4) ===\n{COST_ASSUMPTIONS_DISCLAIMER}")
    print(json.dumps(asdict(costs), indent=2))

    # ---- Logistic baseline, recomputed fresh in this same run so every
    # comparison number below comes from the identical train/cal_fit/
    # selection/temporal_benchmark objects (Requirement A: fair comparison).
    logistic_pipeline = fit_baseline_pipeline(train_df)
    logistic_raw_cal_fit = logistic_pipeline.predict_proba(cal_fit_df)[:, 1]
    logistic_raw_select = logistic_pipeline.predict_proba(select_df)[:, 1]
    logistic_raw_temporal_benchmark = logistic_pipeline.predict_proba(temporal_benchmark_df)[:, 1]
    logistic_comparison = calibration_comparison(
        y_cal_fit, logistic_raw_cal_fit, y_select, logistic_raw_select, y_temporal_benchmark, logistic_raw_temporal_benchmark
    )
    logistic_best_method = select_best_calibration_method(logistic_comparison)
    logistic_best_calibrator = fit_calibrator(logistic_best_method, logistic_raw_cal_fit, y_cal_fit)
    logistic_calibrated_select = apply_calibrator(logistic_best_calibrator, logistic_raw_select)
    logistic_calibrated_temporal_benchmark = apply_calibrator(logistic_best_calibrator, logistic_raw_temporal_benchmark)
    logistic_policy = decision_policy_comparison(
        y_select, logistic_calibrated_select, y_temporal_benchmark, logistic_calibrated_temporal_benchmark, costs
    )

    # ---- Nonlinear model: tune (train-only, time-aware) ONCE; the same
    # chosen hyperparameters are then reused, unchanged, for BOTH feature-set
    # configurations below (only the feature set changes between them).
    print("\n=== Hyperparameter tuning (TRAIN-ONLY, time-aware TimeSeriesSplit folds) ===")
    tuning_report = tune_hyperparameters(train_df)
    print(json.dumps({k: v for k, v in tuning_report.items() if k != "candidates"}, indent=2))
    print(f"Candidates considered: {len(tuning_report['candidates'])}")
    for row in tuning_report["candidates"]:
        print(f"  {row['params']} -> mean PR-AUC {row['mean_pr_auc']:.4f}")
    best_params = tuning_report["best_params"]

    # ---- Consistency pass: BOTH nonlinear feature-set configurations go
    # through the identical full pipeline (`nonlinear_full_pipeline`) —
    # train -> cal_fit -> sigmoid/isotonic comparison -> selection ->
    # temporal_benchmark -> business decisioning. "with_time_index" matches
    # the established Milestone 3/4 logistic baseline's own feature set
    # (fair head-to-head against that specific baseline); "preferred"
    # excludes day_index/week_index per Requirement G's default-exclude
    # instruction and is the configuration this milestone actually
    # recommends going forward — see §4/§6 of
    # docs/milestone5_nonlinear_comparison.md.
    nonlinear_with_time_index = nonlinear_full_pipeline(
        train_df, cal_fit_df, select_df, temporal_benchmark_df, best_params, costs, numeric_features=NUMERIC_FEATURES
    )
    nonlinear_preferred = nonlinear_full_pipeline(
        train_df,
        cal_fit_df,
        select_df,
        temporal_benchmark_df,
        best_params,
        costs,
        numeric_features=NUMERIC_FEATURES_NO_TIME_DRIFT,
    )
    nonlinear_pipeline = nonlinear_with_time_index["pipeline"]  # used by §F feature importance below

    def _calibration_summary(comparison: dict) -> dict:
        return {
            method: {
                split_name: {k: comparison[method][split_name][k] for k in _METRIC_KEYS}
                for split_name in ("selection", "temporal_benchmark")
            }
            for method in comparison
        }

    print(
        "\n=== Calibration comparison (nonlinear, WITH day_index/week_index): "
        "ROC-AUC / PR-AUC / Brier (selection & temporal_benchmark) ==="
    )
    print(json.dumps(_calibration_summary(nonlinear_with_time_index["calibration_comparison"]), indent=2))
    print(f"Best calibration method: {nonlinear_with_time_index['best_calibration_method']}")

    print(
        "\n=== Calibration comparison (nonlinear, PREFERRED config, WITHOUT day_index/week_index): "
        "ROC-AUC / PR-AUC / Brier (selection & temporal_benchmark) ==="
    )
    print(json.dumps(_calibration_summary(nonlinear_preferred["calibration_comparison"]), indent=2))
    print(f"Best calibration method: {nonlinear_preferred['best_calibration_method']}")

    # ---- Requirement C: head-to-head CALIBRATED metrics on identical
    # splits, across all three configurations.
    head_to_head = {
        "selection": {
            "logistic": evaluate_probs(y_select, logistic_calibrated_select),
            "nonlinear_with_time_index": nonlinear_with_time_index["calibrated_metrics"]["selection"],
            "nonlinear_preferred_no_time_index": nonlinear_preferred["calibrated_metrics"]["selection"],
        },
        "temporal_benchmark": {
            "logistic": evaluate_probs(y_temporal_benchmark, logistic_calibrated_temporal_benchmark),
            "nonlinear_with_time_index": nonlinear_with_time_index["calibrated_metrics"]["temporal_benchmark"],
            "nonlinear_preferred_no_time_index": nonlinear_preferred["calibrated_metrics"]["temporal_benchmark"],
        },
    }
    print("\n=== Logistic vs. nonlinear(x2): CALIBRATED head-to-head metrics ===")
    print(
        json.dumps(
            {
                split_name: {
                    model_name: {
                        k: head_to_head[split_name][model_name][k]
                        for k in (*_METRIC_KEYS, "precision", "recall", "f1")
                    }
                    for model_name in ("logistic", "nonlinear_with_time_index", "nonlinear_preferred_no_time_index")
                }
                for split_name in head_to_head
            },
            indent=2,
        )
    )

    # ---- Requirement G: time-drift-feature ablation for the nonlinear
    # model. This is a RANKING-ONLY, UNCALIBRATED, structural ablation
    # (mirrors `train_calibrated.time_feature_ablation`'s own uncalibrated
    # convention exactly) — its "brier_score" is explicitly flagged
    # `brier_score_is_calibrated: False` and must NOT be read side by side
    # with the CALIBRATED Brier scores above/below without noting that
    # label. It exists to isolate the feature-set effect in isolation from
    # calibration choice.
    ablation = time_feature_ablation_nonlinear(train_df, val_df, temporal_benchmark_df, best_params)
    print("\n=== Nonlinear time-feature ablation (day_index/week_index) — UNCALIBRATED, ranking-only ===")
    print(json.dumps(ablation, indent=2))

    # The actual preferred-configuration recommendation is based on the
    # CALIBRATED temporal_benchmark ROC-AUC from the full pipeline above
    # (nonlinear_with_time_index vs. nonlinear_preferred), NOT the raw
    # ablation alone — this is the exact configuration whose calibrated and
    # decisioning metrics are reported in this run.
    tb_auc_with_calibrated = nonlinear_with_time_index["calibrated_metrics"]["temporal_benchmark"]["roc_auc"]
    tb_auc_without_calibrated = nonlinear_preferred["calibrated_metrics"]["temporal_benchmark"]["roc_auc"]
    consistency_check = {
        "selection": {
            "with_time_index": {k: nonlinear_with_time_index["calibrated_metrics"]["selection"][k] for k in _METRIC_KEYS},
            "without_time_index_preferred": {k: nonlinear_preferred["calibrated_metrics"]["selection"][k] for k in _METRIC_KEYS},
        },
        "temporal_benchmark": {
            "with_time_index": {k: nonlinear_with_time_index["calibrated_metrics"]["temporal_benchmark"][k] for k in _METRIC_KEYS},
            "without_time_index_preferred": {
                k: nonlinear_preferred["calibrated_metrics"]["temporal_benchmark"][k] for k in _METRIC_KEYS
            },
        },
    }
    time_drift_recommendation = (
        "exclude day_index/week_index by default (matches Milestone 4's logistic finding; "
        "based on CALIBRATED temporal_benchmark ROC-AUC from the full pipeline, not the raw ablation alone)"
        if tb_auc_without_calibrated >= tb_auc_with_calibrated
        else "include day_index/week_index (CALIBRATED temporal_benchmark ROC-AUC is higher WITH them in this run)"
    )
    print(
        f"\nCALIBRATED temporal_benchmark ROC-AUC: with={tb_auc_with_calibrated:.4f} "
        f"without(preferred)={tb_auc_without_calibrated:.4f} -> {time_drift_recommendation}"
    )

    # ---- Requirement E: business-decision comparison for all three
    # configurations, identical cost assumptions/threshold grids throughout
    # (the SAME `costs` object loaded once above).
    print(
        "\n=== Decision-policy comparison (SELECTION-chosen policy, each configuration's own calibrated probabilities) ==="
    )
    print(
        json.dumps(
            {
                "logistic_best_policy_selection": logistic_policy["best_policy_selection"],
                "nonlinear_with_time_index_best_policy_selection": nonlinear_with_time_index["decision_policy"][
                    "best_policy_selection"
                ],
                "nonlinear_preferred_best_policy_selection": nonlinear_preferred["decision_policy"][
                    "best_policy_selection"
                ],
                "logistic_temporal_benchmark_evaluation": logistic_policy["temporal_benchmark_policy_evaluation"],
                "nonlinear_with_time_index_temporal_benchmark_evaluation": nonlinear_with_time_index[
                    "decision_policy"
                ]["temporal_benchmark_policy_evaluation"],
                "nonlinear_preferred_temporal_benchmark_evaluation": nonlinear_preferred["decision_policy"][
                    "temporal_benchmark_policy_evaluation"
                ],
            },
            indent=2,
        )
    )

    # Sensitivity analysis is run on the PREFERRED configuration (the one
    # this milestone actually recommends), using the same cost-variation
    # construction and threshold grids Milestone 4 used.
    cost_variations = build_cost_variations(costs)
    nonlinear_sensitivity = sensitivity_analysis(
        y_select,
        nonlinear_preferred["calibrated_probs"]["selection"],
        REVIEW_THRESHOLD_GRID,
        DECLINE_THRESHOLD_GRID,
        costs,
        cost_variations,
    )
    print("\n=== Sensitivity analysis (nonlinear, PREFERRED no-time-index config, SELECTION SPLIT ONLY) ===")
    print(nonlinear_sensitivity.to_string(index=False))

    # ---- Requirement F: feature importance. Computed on the WITH-time-
    # index (full feature set) pipeline, matching the established
    # Milestone 3/4 logistic baseline's own feature set — NOT the preferred
    # no-time-drift configuration. This is a deliberate, noted scope choice
    # (not an oversight): day_index/week_index both scored exactly 0.0
    # permutation importance regardless (see the ablation/consistency-check
    # above), so this choice does not change which features matter.
    coef_table = extract_coefficients(logistic_pipeline)
    importance_table = permutation_importance_table(nonlinear_pipeline, select_df)
    importance_comparison = compare_feature_importance(coef_table, importance_table)
    print("\n=== Feature importance: permutation importance (nonlinear, with-time-index pipeline, selection split) ===")
    print(importance_table.to_string(index=False))
    print("\n=== Feature importance comparison (logistic |coefficient| vs. nonlinear permutation importance) ===")
    print(json.dumps(importance_comparison, indent=2))

    out = {
        "split_report": split_report,
        "validation_period_subsplit_report": val_subsplit_report,
        "hyperparameter_tuning": tuning_report,
        "cost_assumptions": {"disclaimer": COST_ASSUMPTIONS_DISCLAIMER, **asdict(costs)},
        "logistic_calibration_comparison": logistic_comparison,
        "logistic_best_calibration_method": logistic_best_method,
        "logistic_decision_policy": logistic_policy,
        "nonlinear_with_time_index_calibration_comparison": nonlinear_with_time_index["calibration_comparison"],
        "nonlinear_with_time_index_best_calibration_method": nonlinear_with_time_index["best_calibration_method"],
        "nonlinear_with_time_index_decision_policy": nonlinear_with_time_index["decision_policy"],
        "nonlinear_preferred_no_time_index_calibration_comparison": nonlinear_preferred["calibration_comparison"],
        "nonlinear_preferred_no_time_index_best_calibration_method": nonlinear_preferred["best_calibration_method"],
        "nonlinear_preferred_no_time_index_decision_policy": nonlinear_preferred["decision_policy"],
        "head_to_head_calibrated_metrics": head_to_head,
        "time_feature_ablation_nonlinear_uncalibrated_ranking_only": ablation,
        "preferred_configuration_consistency_check_calibrated": consistency_check,
        "time_drift_feature_recommendation": time_drift_recommendation,
        "nonlinear_sensitivity_analysis_selection_preferred_config": nonlinear_sensitivity.to_dict(orient="records"),
        "logistic_coefficients": coef_table.to_dict(orient="records"),
        "nonlinear_permutation_importance_selection": importance_table.to_dict(orient="records"),
        "nonlinear_permutation_importance_pipeline_note": (
            "Computed on the WITH-time-index (full feature set) nonlinear pipeline, matching "
            "the established Milestone 3/4 logistic baseline's own feature set, NOT the "
            "no-time-drift preferred configuration. day_index/week_index both scored exactly "
            "0.0 importance regardless, so this does not change which features matter; see "
            "time_feature_ablation_nonlinear_uncalibrated_ranking_only and "
            "preferred_configuration_consistency_check_calibrated for the preferred "
            "configuration's own metrics."
        ),
        "feature_importance_comparison": importance_comparison,
        "preferred_nonlinear_configuration": (
            "no_time_drift (day_index/week_index excluded) -- see "
            "nonlinear_preferred_no_time_index_* keys and "
            "preferred_configuration_consistency_check_calibrated above for this EXACT "
            "configuration's own calibrated ROC-AUC/PR-AUC/Brier and decisioning metrics. "
            "The head-to-head comparison against the logistic baseline in head_to_head_calibrated_metrics "
            "and nonlinear_preferred_no_time_index_decision_policy use this same configuration."
        ),
        "methodology_note": (
            "Milestone 5 reuses Milestone 3/4's chronological split, cal_fit/selection "
            "sub-split, calibration methodology, and cost-decisioning framework unchanged. "
            "The nonlinear model's hyperparameters were tuned on TRAIN ONLY via time-aware "
            "forward-chaining folds, never on selection or temporal_benchmark, and the SAME "
            "chosen hyperparameters are reused unchanged for both the with-time-index and "
            "preferred no-time-index feature-set configurations. Calibration methods are fit "
            "only on cal_fit and evaluated only on selection/temporal_benchmark. Decision-policy "
            "thresholds are chosen on selection only (same cost assumptions/threshold grids for "
            "every configuration) and evaluated once, unmodified, on temporal_benchmark. "
            "temporal_benchmark is NOT a pristine, never-touched final test set (see Milestone 4 "
            "methodology note) — it has now also been used for this milestone's comparisons. "
            "Consistency-pass correction: the preferred-configuration recommendation "
            "(preferred_nonlinear_configuration) is based on this exact no-time-index "
            "configuration's own calibrated and decisioning metrics computed via the same "
            "`nonlinear_full_pipeline` code path as every other configuration — not on the "
            "separate, uncalibrated, ranking-only ablation alone."
        ),
    }
    out_path = paths.interim_dir / "milestone5_metrics.json"
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
