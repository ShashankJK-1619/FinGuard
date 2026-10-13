"""Module C, Milestone 3, Section E: evaluation metrics shared by Model A
and Model B, on validation and benchmark.

Threshold discipline: the project spec explicitly warns against treating
0.5 as an unexamined "operational" default. This module instead requires
callers to pass a NAMED threshold (`threshold_name`), and this milestone's
orchestration script (scripts/run_milestone3_credit_baseline.py) sets it
to the TRAIN split's own observed default prevalence
(`train_prevalence_matched`) -- decided once, from train alone, before any
validation/benchmark metric is computed, so it is never tuned to
optimize an evaluation-split metric (that would be a second, more subtle
form of leakage: threshold-snooping instead of feature/label snooping).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class ClassificationMetrics:
    split_name: str
    model_name: str
    threshold_name: str
    threshold_value: float
    n: int
    prevalence_pct: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float

    def as_dict(self) -> dict:
        return {
            "split_name": self.split_name,
            "model_name": self.model_name,
            "threshold_name": self.threshold_name,
            "threshold_value": round(self.threshold_value, 6),
            "n": self.n,
            "prevalence_pct": round(self.prevalence_pct, 3),
            "confusion_matrix": {
                "true_negatives": self.true_negatives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "true_positives": self.true_positives,
            },
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "brier_score": round(self.brier_score, 5),
        }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_value: float,
    threshold_name: str,
    split_name: str,
    model_name: str,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.shape[0] != y_prob.shape[0]:
        raise EvaluationError("y_true and y_prob must have the same length")
    if len(np.unique(y_true)) < 2:
        raise EvaluationError(
            f"split '{split_name}' has only one class present in y_true -- ROC-AUC/PR-AUC "
            "are undefined. Check the upstream split/label construction."
        )
    y_pred = (y_prob >= threshold_value).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        split_name=split_name,
        model_name=model_name,
        threshold_name=threshold_name,
        threshold_value=float(threshold_value),
        n=int(len(y_true)),
        prevalence_pct=float(y_true.mean() * 100),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
    )
