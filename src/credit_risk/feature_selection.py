"""Module C, Milestone 3, Section B: predeclared Model A / Model B feature
lists.

Design discipline: the actual feature-selection *judgment* lives in
`configs/credit_risk_model_feature_sets.json`, authored and reviewed by a
human, frozen before any model is fit (see that file's `_disclaimer`).
This module's job is to (a) load that config, (b) run a loud, fail-fast
cross-check against `configs/credit_risk_feature_eligibility.json` -- every
raw field Model A proposes to use must be marked
`available_at_application == "yes"` with no recorded `leakage_concern`
there, i.e. `FeatureEligibilityEntry.is_application_time_safe` -- and (c)
expose the two feature sets as simple, explicit dataclasses the rest of
Milestone 3's pipeline (preprocessing, modeling, interpretation) can
consume without re-deriving the list themselves.

Model B is defined structurally as Model A plus exactly
`{grade, sub_grade, int_rate}` -- never an independently-edited list that
could silently drift from Model A plus those three fields.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.credit_risk.feature_eligibility import (
    DEFAULT_FEATURE_ELIGIBILITY_FILE,
    load_feature_eligibility_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_FEATURE_SETS_FILE = PROJECT_ROOT / "configs" / "credit_risk_model_feature_sets.json"

# The three lender-assigned-risk-grade fields that distinguish Model B from
# Model A. Declared once, here, so no other module has to repeat the set.
GRADE_CIRCULARITY_FIELDS = ("grade", "sub_grade", "int_rate")


class FeatureSelectionError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureSet:
    name: str
    raw_numeric_fields: tuple[str, ...]
    derived_numeric_fields: tuple[str, ...]
    categorical_fields: tuple[str, ...]
    derived_feature_notes: dict[str, str] = field(default_factory=dict)

    @property
    def all_numeric_fields(self) -> tuple[str, ...]:
        return self.raw_numeric_fields + self.derived_numeric_fields

    @property
    def all_fields(self) -> tuple[str, ...]:
        return self.all_numeric_fields + self.categorical_fields


def _check_eligibility(raw_fields: tuple[str, ...], eligibility_by_field: dict) -> None:
    for f in raw_fields:
        entry = eligibility_by_field.get(f)
        if entry is None:
            raise FeatureSelectionError(
                f"Field '{f}' is referenced by a Model A/B feature list but has no entry "
                "at all in configs/credit_risk_feature_eligibility.json -- every modeling "
                "field must be covered by that eligibility review first."
            )
        if not entry.is_application_time_safe:
            raise FeatureSelectionError(
                f"Field '{f}' is used as a Model A/B feature but "
                "credit_risk_feature_eligibility.json does not mark it "
                "is_application_time_safe (available_at_application='yes' AND no "
                f"leakage_concern). Got available_at_application={entry.available_at_application!r}, "
                f"leakage_concern={entry.leakage_concern!r}. Fix the feature-set config or the "
                "eligibility table -- never silently use a field the eligibility review flagged."
            )


def load_feature_sets(
    config_file: Path = DEFAULT_MODEL_FEATURE_SETS_FILE,
    eligibility_file: Path = DEFAULT_FEATURE_ELIGIBILITY_FILE,
) -> tuple[FeatureSet, FeatureSet]:
    """Loads and validates Model A and Model B. Raises FeatureSelectionError
    if: a Model A raw field is not application-time-safe per the eligibility
    table; Model B's additional fields are anything other than exactly
    `GRADE_CIRCULARITY_FIELDS`; or any field appears in both Model A and the
    Model B additions (which would make the two feature sets ambiguous)."""
    with open(config_file) as f:
        cfg = json.load(f)

    eligibility_entries = load_feature_eligibility_table(eligibility_file)
    eligibility_by_field = {e.field: e for e in eligibility_entries}

    model_a_raw_numeric = tuple(cfg["model_a_raw_numeric_fields"])
    model_a_derived_numeric = tuple(cfg["model_a_derived_numeric_fields"].keys())
    model_a_categorical = tuple(cfg["model_a_categorical_fields"])
    derived_notes = dict(cfg["model_a_derived_numeric_fields"])

    # Model A's raw (non-derived) fields must each be independently
    # eligibility-cleared. Derived features are checked by the module that
    # builds them (feature_loader.py) against their own raw inputs.
    _check_eligibility(model_a_raw_numeric, eligibility_by_field)
    _check_eligibility(model_a_categorical, eligibility_by_field)

    model_b_extra_numeric = tuple(cfg["model_b_additional_raw_numeric_fields"])
    model_b_extra_categorical = tuple(cfg["model_b_additional_categorical_fields"])
    model_b_extra_all = model_b_extra_numeric + model_b_extra_categorical
    if set(model_b_extra_all) != set(GRADE_CIRCULARITY_FIELDS):
        raise FeatureSelectionError(
            f"Model B's additional fields must be exactly {GRADE_CIRCULARITY_FIELDS}; "
            f"got {model_b_extra_all}. Model B is defined structurally as Model A plus "
            "the lender-assigned-risk-grade circularity group, never an independently "
            "edited list."
        )

    model_a_all = set(model_a_raw_numeric) | set(model_a_derived_numeric) | set(model_a_categorical)
    overlap = model_a_all & set(model_b_extra_all)
    if overlap:
        raise FeatureSelectionError(
            f"Field(s) {overlap} appear in both Model A and Model B's additional fields -- "
            "Model B must add grade/sub_grade/int_rate on top of a disjoint Model A."
        )

    model_a = FeatureSet(
        name="model_a_application_time_safe",
        raw_numeric_fields=model_a_raw_numeric,
        derived_numeric_fields=model_a_derived_numeric,
        categorical_fields=model_a_categorical,
        derived_feature_notes=derived_notes,
    )
    model_b = FeatureSet(
        name="model_b_with_lender_grade_signal",
        raw_numeric_fields=model_a_raw_numeric + model_b_extra_numeric,
        derived_numeric_fields=model_a_derived_numeric,
        categorical_fields=model_a_categorical + model_b_extra_categorical,
        derived_feature_notes=derived_notes,
    )
    return model_a, model_b
