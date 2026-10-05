"""Module A, Milestone 2: power and minimum-detectable-effect (MDE)
calculations for the primary (binary) metric.

See docs/moduleA_milestone2_experimentation.md Section G. This is
EXPERIMENT-PLANNING arithmetic — it answers "given a baseline rate, alpha,
target power, and sample size, what effect size could this design reliably
detect?" It is NOT evidence that the synthetic treatment effect modeled in
experiment_effects.py is real; it exists so the milestone demonstrates the
planning step a real experimentation team would do before (and
independently of) looking at any result.

Uses `statsmodels.stats.power.NormalIndPower`, which works in Cohen's h
effect-size units for two independent proportions. Cohen's h is defined as
h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2)) (the arcsine-variance-
stabilizing transform makes power roughly independent of the specific p1,
unlike the raw absolute difference). `solve_power` is used to get the
MINIMUM detectable |h| for a given (n_per_arm, alpha, power); converting
that back to an absolute proportion uplift (the MDE a stakeholder actually
cares about) requires inverting the transform, done explicitly below —
statsmodels has no built-in inverse helper.
"""
from __future__ import annotations

import math

import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

_power_analysis = NormalIndPower()


def _cohens_h_to_absolute_mde(baseline_rate: float, effect_size_h: float) -> float:
    """Given baseline_rate p1 and a (signed) Cohen's h, returns p2 such
    that proportion_effectsize(p1, p2) == effect_size_h, then the absolute
    difference |p2 - p1|. `effect_size_h` from `solve_power` is the
    magnitude of the minimum detectable effect, so the sign choice here
    only affects which direction p2 is probed in — the reported MDE is
    always the absolute magnitude, consistent with a two-sided test
    detecting an uplift OR a drop of that size."""
    phi1 = 2 * math.asin(math.sqrt(baseline_rate))
    phi2 = phi1 - effect_size_h
    # Clamp to a valid arcsine-transform domain in case of a degenerate
    # (very large effect_size_h relative to baseline) edge case.
    phi2 = max(0.0, min(math.pi, phi2))
    p2 = math.sin(phi2 / 2) ** 2
    return abs(p2 - baseline_rate)


def compute_mde(baseline_rate: float, n_per_arm: int, alpha: float, power: float) -> float:
    """Minimum absolute uplift in the primary metric detectable with
    `n_per_arm` sessions per arm, at the given alpha/power, assuming a
    1:1 allocation ratio."""
    effect_size_h = _power_analysis.solve_power(nobs1=n_per_arm, alpha=alpha, power=power, ratio=1.0)
    return _cohens_h_to_absolute_mde(baseline_rate, effect_size_h)


def compute_mde_table(baseline_rate: float, alpha: float, power: float, sample_sizes_per_arm: tuple[int, ...]) -> pd.DataFrame:
    """Shows how MDE shrinks as n_per_arm grows (Sec G) — a reference
    table, not a claim about this run's actual sample size."""
    rows = []
    for n in sample_sizes_per_arm:
        mde_abs = compute_mde(baseline_rate, n, alpha, power)
        rows.append(
            {
                "n_per_arm": n,
                "alpha": alpha,
                "target_power": power,
                "baseline_rate": baseline_rate,
                "mde_absolute": mde_abs,
                "mde_relative": mde_abs / baseline_rate if baseline_rate else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_achieved_power(n_per_arm: int, baseline_rate: float, observed_treatment_rate: float, alpha: float) -> float:
    """Post-hoc power at the OBSERVED effect size and actual n_per_arm.
    Reported as a diagnostic only — post-hoc ("observed") power is a
    direct, monotonic function of the p-value and adds little new
    information once the p-value is already known; it must not be read as
    independent confirmation that the synthetic effect is real or that the
    design was well-powered for whatever the true effect happens to be."""
    effect_size_h = proportion_effectsize(observed_treatment_rate, baseline_rate)
    return float(
        _power_analysis.solve_power(effect_size=abs(effect_size_h), nobs1=n_per_arm, alpha=alpha, power=None, ratio=1.0)
    )
