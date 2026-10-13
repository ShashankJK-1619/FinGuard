"""Visual 1 -- FinGuard architecture diagram.

Pure layout diagram: no metrics, no computed values. Depicts the shared
pipeline shape documented in README.md's "Architecture" section and
restated per-module in docs/decision_log.md: data -> validation ->
feature/KPI layer -> per-module modeling/experimentation -> decisioning
& governance.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.portfolio.style import (
    BACKGROUND,
    LAVENDER,
    LAVENDER_LIGHT,
    PERIWINKLE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    VIOLET,
    apply_style,
    save_figure,
)


def _box(ax, xy, w, h, text, facecolor, textcolor=TEXT_PRIMARY, fontsize=11, fontweight="bold"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=TEXT_SECONDARY,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        fontweight=fontweight,
        linespacing=1.4,
    )
    return box


def _arrow(ax, start, end, color=TEXT_SECONDARY):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.4,
        color=color,
    )
    ax.add_patch(arrow)


def build() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    # Shared top-of-pipeline stages.
    _box(ax, (2.0, 9.1), 6.5, 0.9, "External / Public Data\n(IEEE-CIS · LendingClub · synthetic session/experiment layer)", LAVENDER, fontsize=10.5)
    _arrow(ax, (5.25, 9.1), (5.25, 8.55))
    _box(ax, (2.75, 7.75), 5.0, 0.8, "Validation & Governance\n(schema checks, leakage review, censoring/target rules)", LAVENDER_LIGHT, fontsize=10)
    _arrow(ax, (5.25, 7.75), (5.25, 7.2))
    _box(ax, (2.75, 6.4), 5.0, 0.8, "Feature / KPI Layer\n(point-in-time-safe features, SQL KPI layer)", LAVENDER_LIGHT, fontsize=10)

    # Branches down into the three modules.
    branch_y_top = 6.4
    branch_y_bottom = 5.9
    col_centers = [1.75, 5.25, 8.75]
    for cx in col_centers:
        _arrow(ax, (5.25, branch_y_top), (cx, branch_y_bottom))

    module_defs = [
        (
            col_centers[0],
            "Module A\nPayments / Product Analytics",
            ["KPI layer", "Funnel analysis", "Experimentation (A/B)"],
            PERIWINKLE,
        ),
        (
            col_centers[1],
            "Module B\nFraud / Risk Modeling",
            ["Point-in-time features", "Logistic baseline", "Nonlinear model", "Calibration / decisioning"],
            LAVENDER,
        ),
        (
            col_centers[2],
            "Module C\nCredit Risk / Expected Loss",
            ["Target / censoring design", "Interpretable risk models", "Expected loss", "Stability / governance"],
            LAVENDER_LIGHT,
        ),
    ]

    module_box_w = 3.1
    module_header_h = 0.75
    module_body_top = 5.75
    for cx, header, bullets, color in module_defs:
        x0 = cx - module_box_w / 2
        _box(ax, (x0, module_body_top - module_header_h), module_box_w, module_header_h, header, color, fontsize=10.5)
        body_h = 0.42 * len(bullets) + 0.25
        body_y0 = module_body_top - module_header_h - 0.15 - body_h
        bullet_text = "\n".join(f"• {b}" for b in bullets)
        _box(
            ax,
            (x0, body_y0),
            module_box_w,
            body_h,
            bullet_text,
            BACKGROUND,
            textcolor=TEXT_PRIMARY,
            fontsize=9.3,
            fontweight="normal",
        )
        box = ax.patches[-1]
        box.set_edgecolor(color)
        box.set_linewidth(1.6)
        _arrow(ax, (cx, body_y0), (cx, 2.05), color=TEXT_SECONDARY)

    # Convergence into decisioning & governance.
    _box(
        ax,
        (2.0, 1.15),
        6.5,
        0.9,
        "Decisioning & Governance\n(approve/review/decline · ship/no-ship · expected loss · limitations)",
        VIOLET,
        textcolor="white",
        fontsize=10.5,
    )

    ax.set_title("FinGuard Architecture", fontsize=16, pad=14, color=TEXT_PRIMARY)
    fig.text(
        0.5,
        0.955,
        "Shared pipeline shape across all three modules",
        ha="center",
        fontsize=11,
        color=TEXT_SECONDARY,
    )

    out = save_figure(fig, "finguard_architecture.png")
    print(f"Wrote {out}")


if __name__ == "__main__":
    build()
