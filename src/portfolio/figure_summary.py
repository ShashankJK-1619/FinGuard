"""Visual 7 (optional) -- FinGuard at a glance.

A compact, factual stat panel summarizing scale and headline findings
across all three modules. Every number here is pulled from the same
centralized `metrics_config` used by Visuals 2-6 -- nothing new is
computed or claimed here. Kept intentionally small (6 stats, no new
chart types) to avoid clutter; this is meant as a possible top-of-README
summary, not a replacement for the detailed visuals.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.portfolio.metrics_config import (
    CREDIT_MODEL_A_SLOPE_BENCHMARK,
    CREDIT_MODEL_A_SLOPE_VALIDATION,
    FRAUD_LOGISTIC_PR_AUC,
    FRAUD_N_TRANSACTIONS,
    FRAUD_NONLINEAR_PR_AUC,
    FRAUD_PREVALENCE_PCT,
    PAYMENTS_N_SESSIONS,
    PAYMENTS_PRIMARY_METRIC_TEST,
    TEST_SUITE_FAILED,
    TEST_SUITE_PASSED,
    TEST_SUITE_SKIPPED,
)
from src.portfolio.style import (
    BORDER,
    CAUTION,
    GRID,
    LAVENDER,
    POSITIVE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VIOLET,
    add_source_note,
    apply_style,
    save_figure,
)


def build() -> None:
    apply_style()

    pay = PAYMENTS_PRIMARY_METRIC_TEST.value
    uplift_pp = pay["absolute_uplift"] * 100
    pr_lift_pct = (FRAUD_NONLINEAR_PR_AUC.value / FRAUD_LOGISTIC_PR_AUC.value - 1) * 100

    stats = [
        (
            f"{FRAUD_N_TRANSACTIONS.value:,}",
            f"labeled transactions evaluated\n(~{FRAUD_PREVALENCE_PCT.value:.1f}% fraud prevalence)",
            VIOLET,
        ),
        (
            f"{PAYMENTS_N_SESSIONS.value:,}",
            "synthetic payment sessions\n(A/B experiment layer)",
            LAVENDER,
        ),
        (
            f"+{uplift_pp:.2f}pp",
            "payments experiment uplift\nstatistically significant -- NO-SHIP\n(below +1.0pp predeclared threshold)",
            CAUTION,
        ),
        (
            f"+{pr_lift_pct:.0f}%",
            "fraud PR-AUC, preferred nonlinear\nmodel vs. logistic baseline\n(temporal benchmark)",
            POSITIVE,
        ),
        (
            f"{CREDIT_MODEL_A_SLOPE_VALIDATION.value:.2f} -> {CREDIT_MODEL_A_SLOPE_BENCHMARK.value:.2f}",
            "credit calibration slope, Model A\nvalidation vs. temporal benchmark\n(degrades out-of-time)",
            CAUTION,
        ),
        (
            f"{TEST_SUITE_PASSED.value} passed",
            f"· {TEST_SUITE_SKIPPED.value} skipped · {TEST_SUITE_FAILED.value} failed\nautomated test suite",
            VIOLET,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))

    for ax, (value, label, color) in zip(axes.flat, stats):
        ax.axis("off")
        card = FancyBboxPatch(
            (0.04, 0.08),
            0.92,
            0.84,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            transform=ax.transAxes,
            linewidth=1.2,
            edgecolor=BORDER,
            facecolor=GRID,
            zorder=0,
        )
        ax.add_patch(card)
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=22, fontweight="bold", color=color, transform=ax.transAxes, zorder=1)
        ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=10, color=TEXT_PRIMARY, transform=ax.transAxes, zorder=1)

    fig.suptitle("FinGuard — At a Glance", fontsize=17, fontweight="bold", y=1.02)
    fig.text(
        0.5,
        0.93,
        "Payments analytics, fraud detection, and credit risk -- summary across all three modules",
        ha="center",
        fontsize=11,
        color=TEXT_SECONDARY,
        style="italic",
    )

    add_source_note(
        fig,
        "Source: see metrics_config.py and the individual module visuals (payments_experiment_result.png, "
        "fraud_model_comparison.png, credit_calibration_drift.png) for exact per-figure provenance.",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out = save_figure(fig, "finguard_summary.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
