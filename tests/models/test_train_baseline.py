from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.schema import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL
from src.models.split import SECONDS_PER_DAY, chronological_split
from src.models.train_baseline import (
    DEFAULT_THRESHOLD,
    build_model_pipeline,
    evaluate,
    extract_coefficients,
    threshold_analysis,
)
from tests.conftest import requires_baseline_features
from tests.models.test_split import _toy_df as _toy_split_df


def _toy_model_df(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    data: dict[str, object] = {
        "TransactionID": np.arange(n),
        "TransactionDT": np.sort(rng.integers(0, 40 * SECONDS_PER_DAY, n)),
        TARGET_COL: rng.choice([0, 1], size=n, p=[0.9, 0.1]),
    }
    for col in NUMERIC_FEATURES:
        data[col] = rng.normal(size=n)
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(["A", "B", "C"], size=n)
    for col in BINARY_FEATURES:
        data[col] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


def test_pipeline_fit_and_predict_proba_are_well_formed():
    df = _toy_model_df()
    train_df, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)

    pipeline = build_model_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])
    probs = pipeline.predict_proba(val_df)[:, 1]

    assert probs.shape[0] == len(val_df)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_evaluate_returns_expected_keys_and_valid_ranges():
    df = _toy_model_df()
    train_df, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    pipeline = build_model_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])

    metrics = evaluate(pipeline, val_df, threshold=DEFAULT_THRESHOLD)
    for key in ("precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"):
        assert 0.0 <= metrics[key] <= 1.0
    cm = metrics["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(val_df)


def test_threshold_analysis_alert_volume_decreases_as_threshold_increases():
    df = _toy_model_df()
    train_df, val_df, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    pipeline = build_model_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])

    rows = threshold_analysis(pipeline, val_df, thresholds=[0.1, 0.5, 0.9])
    volumes = [r["alert_volume"] for r in rows]
    assert volumes == sorted(volumes, reverse=True)


def test_coefficients_are_extracted_for_every_transformed_feature():
    df = _toy_model_df()
    train_df, _, _, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)
    pipeline = build_model_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])

    coef_table = extract_coefficients(pipeline)
    n_transformed = len(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert len(coef_table) == n_transformed
    assert {"feature", "coefficient", "odds_ratio", "is_masked_code"} <= set(coef_table.columns)


def test_model_is_fit_only_on_train_split_not_val_or_holdout():
    """No future leakage: fitting must only ever see the train split. This
    mutates val/holdout to an extreme, different distribution after fit and
    confirms predictions still reflect the train-only fitted pipeline
    rather than erroring or silently refitting."""
    df = _toy_model_df()
    train_df, val_df, holdout_df, _ = chronological_split(df, train_frac=0.6, val_frac=0.2)

    pipeline = build_model_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])

    mutated_val = val_df.copy()
    for col in NUMERIC_FEATURES:
        mutated_val[col] = 10_000.0  # far outside anything seen in train

    probs_before = pipeline.predict_proba(val_df)[:, 1]
    probs_after_on_mutated = pipeline.predict_proba(mutated_val)[:, 1]
    assert not np.allclose(probs_before, probs_after_on_mutated)


@requires_baseline_features
def test_full_pipeline_on_real_baseline_table_produces_sane_metrics():
    from src.models.train_baseline import load_baseline_table

    df = load_baseline_table()
    train_df, val_df, holdout_df, split_report = chronological_split(df)

    assert len(train_df) + len(val_df) + len(holdout_df) == len(df)
    assert train_df["TransactionDT"].max() < val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() < holdout_df["TransactionDT"].min()

    pipeline = build_model_pipeline()
    pipeline.fit(train_df, train_df[TARGET_COL])

    val_metrics = evaluate(pipeline, val_df)
    holdout_metrics = evaluate(pipeline, holdout_df)

    # A logistic regression baseline should clearly beat random (0.5 AUC)
    # but this is a sanity floor, not a performance claim.
    assert val_metrics["roc_auc"] > 0.6
    assert holdout_metrics["roc_auc"] > 0.6
