"""Module C, Milestone 3, Section H: lender-signal ablation.

Quantifies the lift from Model A -> Model B (adding grade/sub_grade/
int_rate) on ROC-AUC, PR-AUC, Brier, and calibration. Framed, per the
milestone spec, strictly as "how much signal is embedded in the lender's
own prior underwriting decision," never as a recommendation to deploy
Model B -- `AblationResult`'s `framing_note` restates this on every
result so it cannot be silently dropped by a downstream consumer.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.credit_risk.evaluation import ClassificationMetrics

FRAMING_NOTE = (
    "This lift quantifies signal already embedded in LendingClub's own prior underwriting "
    "decision (grade/sub_grade/int_rate), not a recommendation to use these fields in a "
    "deployed PD model -- see configs/credit_risk_feature_eligibility.json's unresolved "
    "grade/sub_grade/int_rate circularity entries."
)


@dataclass(frozen=True)
class AblationResult:
    split_name: str
    model_a_roc_auc: float
    model_b_roc_auc: float
    roc_auc_lift: float
    model_a_pr_auc: float
    model_b_pr_auc: float
    pr_auc_lift: float
    model_a_brier: float
    model_b_brier: float
    brier_improvement: float  # positive means Model B's Brier is lower (better)
    framing_note: str = FRAMING_NOTE

    def as_dict(self) -> dict:
        return {
            "split_name": self.split_name,
            "roc_auc": {
                "model_a": round(self.model_a_roc_auc, 4),
                "model_b": round(self.model_b_roc_auc, 4),
                "lift": round(self.roc_auc_lift, 4),
            },
            "pr_auc": {
                "model_a": round(self.model_a_pr_auc, 4),
                "model_b": round(self.model_b_pr_auc, 4),
                "lift": round(self.pr_auc_lift, 4),
            },
            "brier": {
                "model_a": round(self.model_a_brier, 5),
                "model_b": round(self.model_b_brier, 5),
                "improvement": round(self.brier_improvement, 5),
            },
            "framing_note": self.framing_note,
        }


def compute_ablation(
    split_name: str, model_a_metrics: ClassificationMetrics, model_b_metrics: ClassificationMetrics
) -> AblationResult:
    if model_a_metrics.split_name != model_b_metrics.split_name:
        raise ValueError(
            f"Model A metrics are for split '{model_a_metrics.split_name}' but Model B metrics "
            f"are for split '{model_b_metrics.split_name}' -- an ablation must compare both "
            "models on the SAME split."
        )
    return AblationResult(
        split_name=split_name,
        model_a_roc_auc=model_a_metrics.roc_auc,
        model_b_roc_auc=model_b_metrics.roc_auc,
        roc_auc_lift=model_b_metrics.roc_auc - model_a_metrics.roc_auc,
        model_a_pr_auc=model_a_metrics.pr_auc,
        model_b_pr_auc=model_b_metrics.pr_auc,
        pr_auc_lift=model_b_metrics.pr_auc - model_a_metrics.pr_auc,
        model_a_brier=model_a_metrics.brier_score,
        model_b_brier=model_b_metrics.brier_score,
        brier_improvement=model_a_metrics.brier_score - model_b_metrics.brier_score,
    )
