# Module C, Milestone 3, Section B — Model A / Model B Feature Sets

## Why this doc exists

`configs/credit_risk_model_feature_sets.json` is the machine-readable
source of truth (loaded and cross-checked at runtime by
`src/credit_risk/feature_selection.py::load_feature_sets`). This doc is
its human-readable companion: it explains *why* the two feature sets
are shaped the way they are, written before any model was fit (per this
project's "predeclare, never adjust after seeing evaluation-split
performance" discipline).

## The two specifications

| | Model A | Model B |
|---|---|---|
| Name | `model_a_application_time_safe` | `model_b_with_lender_grade_signal` |
| Purpose | A defensible application-time-safe baseline PD model | Model A **plus** LendingClub's own `grade`/`sub_grade`/`int_rate` |
| Used to recommend deployment? | Candidate baseline | **No** — exists only to quantify lift from the lender's prior underwriting signal (Section H ablation) |

Model B is defined **structurally** as Model A plus exactly
`{grade, sub_grade, int_rate}` — `feature_selection.py`'s
`load_feature_sets` raises `FeatureSelectionError` if Model B's
additional-fields config ever drifts from that exact set, or if those
fields overlap with Model A's own list. There is no independently
edited "Model B field list" that could silently diverge.

## Model A: 19 raw numeric + 4 derived numeric + 7 categorical fields

**Raw numeric** (`model_a_raw_numeric_fields`): `loan_amnt`,
`installment`, `annual_inc`, `dti`, `delinq_2yrs`, `inq_last_6mths`,
`mths_since_last_delinq`, `mths_since_last_record`, `open_acc`,
`pub_rec`, `revol_bal`, `revol_util`, `total_acc`,
`collections_12_mths_ex_med`, `acc_now_delinq`, `pub_rec_bankruptcies`,
`tax_liens`, `chargeoff_within_12_mths`, `delinq_amnt`.

**Derived numeric** (`model_a_derived_numeric_fields`, built by
`feature_loader.py`):
- `term_months` — parsed from raw `term` (reuses
  `target_construction.py::_term_to_months`, so the feature and the
  maturity-date derivation never disagree).
- `emp_length_years` — explicit, auditable string→int map
  (`feature_loader.EMP_LENGTH_YEARS_MAP`); any unrecognized raw value
  raises rather than being silently coerced.
- `fico_avg_at_application` — mean of `fico_range_low`/`fico_range_high`.
- `credit_history_months_at_application` — whole months between
  `earliest_cr_line` and `issue_d` (origination date).

**Categorical** (`model_a_categorical_fields`): `home_ownership`,
`verification_status`, `purpose`, `addr_state`, `initial_list_status`,
`application_type`, `disbursement_method`.

Every raw field above is independently cross-checked against
`configs/credit_risk_feature_eligibility.json`'s
`is_application_time_safe` property (`available_at_application=="yes"`
AND no recorded `leakage_concern`) at load time — a field cannot enter
Model A by editing this JSON file alone if the eligibility table
disagrees.

## Model B: Model A + the lender-grade circularity group

`model_b_additional_raw_numeric_fields = ["int_rate"]`,
`model_b_additional_categorical_fields = ["grade", "sub_grade"]`.

These three fields are technically available at-or-before origination
(not a timing leak) but are flagged `"unknown"` in the eligibility
table specifically because they are **LendingClub's own prior risk
assessment and pricing decision** — including them risks measuring
"how well does the lender's own grade predict the lender's own
outcome" rather than independent borrower signal. Model B exists
*solely* to quantify that circularity's effect size (Section H); its
`AblationResult.framing_note` (`src/credit_risk/ablation.py`) restates
this on every result so the distinction cannot be silently dropped
downstream.

## What was excluded from the already-eligibility-cleared pool, and why

`exclusions_from_eligible_pool` in the config documents every field
that passed `is_application_time_safe` but was still kept out of both
Model A and Model B, with one rationale each. In summary, by category:

- **Pure identifiers / anchors, never features**: `id`, `url`,
  `member_id`, `issue_d` (the origination-date anchor itself —
  consumed to derive `term_months`/`credit_history_months_at_
  application`/`maturity_date`, never fed to the model directly).
- **Near-constant**: `policy_code` (≈1 for virtually every row).
- **Free text, out of scope for a baseline logistic regression**:
  `emp_title`, `desc`, `title`.
- **High-cardinality geography, deliberately not used**: `zip_code`
  (~900 distinct values). `addr_state` (51 values) is retained as the
  baseline's geographic control; adding `zip_code` too would make
  Section G's coefficient table intractable and add many near-collinear
  geographic dummies. (See also `subgroup_diagnostics.py`'s *separate*
  decision to exclude `addr_state`/`zip_code` from the Section J
  governance diagnostic as a documented protected-class proxy — a
  stricter standard than "is it a modeling feature," applied only to
  that one diagnostic.)
- **Empirically near-duplicate / collinear fields, verified not
  assumed**: `funded_amnt` (correlation 0.9998 with `loan_amnt` on the
  full real file) and `funded_amnt_inv` (correlation 0.9990) are
  dropped in favor of `loan_amnt`. `fico_range_low`/`fico_range_high`
  (correlation 0.99999998, a constant 4-5-point gap on every row) are
  superseded by the single derived `fico_avg_at_application`.
- **Joint-application sparsity (~95% NULL, populated only when
  `application_type=="Joint App"`)**: `annual_inc_joint`, `dti_joint`,
  `verification_status_joint`, `revol_bal_joint`, and all
  `sec_app_*` co-applicant bureau fields. A naive impute-and-include
  treatment would blur the co-applicant signal into `application_type`
  rather than meaningfully modeling joint applications — flagged as a
  candidate for a dedicated joint-application sub-model in a future
  milestone, not included here.
- **100%-missing-in-train, vintage-introduced-later fields (48 fields,
  e.g. `open_acc_6m`, `mo_sin_old_rev_tl_op`, `bc_util`,
  `total_bal_ex_mort`, ...)**: verified by direct query against
  `loans_raw` restricted to the primary split's train bucket
  (2007-06 to 2011-12), not assumed from column names. An all-NaN-
  in-train column, under the train-only-fit imputer required by
  Section C, contributes no trainable signal and cannot be included
  without leaking information from later vintages into what "train"
  means.

## Uniform missingness policy (no per-field special-casing)

Two numeric fields carry *meaningful* missingness
(`mths_since_last_delinq`, 63% missing in train; `mths_since_last_record`,
91% missing in train — NULL means "no delinquency/public record ever
observed," not random missingness). Rather than hand-coding that
semantics for just these two fields, `preprocessing.py` applies
`SimpleImputer(strategy="median", add_indicator=True)` **uniformly to
every numeric feature** — any column with missingness in train gets its
own `missingindicator_<field>` column automatically, keeping the
"was this missing" signal without fabricating a magnitude for it, and
without a special case that would need separate justification per
field.
