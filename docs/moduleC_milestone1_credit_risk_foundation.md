# Module C, Milestone 1 — Credit Risk Foundation: Data Model, Dataset Choice, Target Definition

## 0. Scope boundary

This milestone establishes the **design foundation** for a credit-risk
module: the lending lifecycle and data model, a dataset recommendation,
the `default_12m` target definition with censoring, and an
application-time feature-eligibility policy with an automated leakage
safety net. **No credit-risk model is trained. No real dataset has been
downloaded.** Every table, config, and function here is validated only
against small toy fixtures in `tests/credit_risk/`. This complements,
and does not touch, Module A (payments/experimentation) or Module B
(fraud detection).

This document is the consolidated summary; the full reasoning lives in
three companion docs:

- `docs/credit_risk_data_model.md` — lending lifecycle, 5-table data
  model, point-in-time feature policy (Sections 0, A, B, E).
- `docs/credit_risk_dataset_comparison.md` — dataset evaluation and
  recommendation (Section C).
- `docs/credit_risk_target_definition.md` — `default_12m` definition,
  censoring, delinquency/default separation, PD/LGD/EAD feasibility,
  chronological-split feasibility (Sections D, F, G).

## 1. What was built

| Artifact | Purpose |
|---|---|
| `src/credit_risk/schema.py` | Logical table/column name constants for the 5-table model: `borrowers`, `loan_applications`, `loans`, `scheduled_payments`, `actual_payments`. |
| `src/credit_risk/config.py` + `configs/credit_risk_target_definition.json` | `TargetDefinitionAssumptions` — outcome window, default/delinquency status vocabularies, follow-up tolerance, loudly validated. |
| `src/credit_risk/target_definition.py` | `compute_default_12m_labels` — the deterministic censoring-aware target-definition function (Section D). |
| `src/credit_risk/feature_eligibility.py` + `configs/credit_risk_feature_eligibility.json` | `FeatureEligibilityEntry`/`load_feature_eligibility_table`/`filter_application_time_safe` — point-in-time feature policy as machine-readable, validated config, with a leakage-keyword safety net. |
| `tests/credit_risk/test_schema.py`, `test_config.py`, `test_target_definition.py`, `test_feature_eligibility.py` | 27 tests, all against toy/synthetic fixtures. |

## 2. Dataset recommendation (full detail in Section C doc)

**LendingClub accepted-loans data is recommended** as the primary
dataset for Module C, chosen over Freddie Mac's Single-Family Loan-Level
Dataset (SFLLD, strongest alternative — structurally cleaner
point-in-time separation, but mortgage-specific, narrower feature set,
registration-gated) and Home Credit Default Risk (third comparison
point — pre-baked target label with no raw timeline to derive
censoring logic from, and a documented history of subtle feature
leakage in public analyses). LendingClub's `loan_status` vocabulary maps
directly onto this module's default/delinquency split, and its
documented column set is well suited to exercising the point-in-time
feature policy.

**Provenance, stated precisely**: LendingClub does not currently offer
unrestricted public download of this historical file directly from its
own platform — it is accessed today via third-party re-hosted public
mirrors (commonly a Kaggle dataset), not a live, openly-available
LendingClub endpoint. This project will use one specific, reproducible
public mirror of the 2007–2018 accepted-loans file; the exact source
URL, license terms, and a file checksum will be recorded in
`docs/credit_risk_dataset_comparison.md` and `docs/decision_log.md` when
that mirror is actually downloaded in Module C Milestone 2 — no mirror
has been selected or downloaded yet, and no claim is made here that the
original LendingClub release is itself currently open-access.

## 3. The `default_12m` target definition (full detail in Section D doc)

```
observation_window_end = origination_date + 12 months
default_12m = 1   if a default event occurred on/before observation_window_end (inclusive)
            = 0   if no default event AND data_as_of_date >= observation_window_end
            = NA  otherwise, label_status = "censored_insufficient_followup"
```

Three design choices are deliberate, tested, and documented, not
incidental:

1. **Inclusive window boundary** — a default landing exactly on day 365
   counts as "within the window."
2. **Censoring, never silent zero-filling** — a loan not yet observed
   long enough to resolve its outcome is explicitly `NA`/`censored`,
   never coded as a confirmed non-default.
3. **Delinquency and default tracked as separate severities** — a
   lesser delinquency event never, by itself, flips `default_12m`.

## 4. Point-in-time feature policy (full detail in data-model doc, Section E)

The rule: a field may be used as an underwriting feature only if it was
genuinely available on or before the credit decision/origination date.
This is enforced as a reviewed, machine-readable table
(`configs/credit_risk_feature_eligibility.json`, ~29 provisional
entries) plus an automated safety net
(`FeatureEligibilityEntry.__post_init__`) that raises
`FeatureEligibilityError` if any field name matching a known
post-origination leakage pattern (`recover`, `total_pymnt`,
`last_pymnt`, `out_prncp`, `charged_off`, `settlement`, `hardship`,
`last_fico`, …) is ever marked `available_at_application = "yes"`.

**`grade`/`sub_grade`/`int_rate` are explicitly flagged `"unknown"`, not
`"yes"`** — these are the lender's own prior risk assessment/pricing,
technically fixed at-or-before origination (so not a timing leak in the
literal sense) but a circularity risk if used as if they were
independent borrower facts, since a model trained on them risks
reproducing LendingClub's own existing risk model rather than learning
borrower-level default risk from first principles. This caution is left
**unresolved** on purpose — whether to exclude these fields entirely,
include them as a deliberate "replicate the lender's pricing policy"
baseline, or something in between, is deferred to Milestone 2.

## 5. PD / LGD / EAD and chronological-split feasibility (full detail in Section D doc, Sections F–G)

- **PD**: feasible and the primary fit — `default_12m` is a direct,
  derivable target from LendingClub's `loan_status` plus an
  extract/`data_as_of_date`. **Scope caveat**: because the dataset
  contains only originated (accepted) loans, the resulting PD model
  estimates default risk *among loans LendingClub already chose to
  fund* — it is not a complete approve/decline underwriting model for
  the full applicant population. See Section 7 (Limitations) for the
  full selection-bias discussion.
- **LGD**: feasible only as a retrospective, post-default-conditional
  illustration (via `recoveries`/`total_rec_prncp`-style fields) —
  never as an application-time underwriting feature, since these fields
  only exist after default has already occurred.
- **EAD**: technically computable but low-value for this specific
  fixed-term, fully amortizing installment product, where exposure is
  close to a deterministic function of origination terms and elapsed
  scheduled amortization rather than a genuinely uncertain quantity.
- **Chronological split**: feasible and recommended over a random split,
  for the same point-in-time reasons already applied in Module B
  (`docs/time_structure.md`) — with one dataset-specific wrinkle flagged
  for the implementing milestone: cutoffs must account for the 12-month
  resolution lag `sufficient_followup` imposes, or a naive "most recent
  12 months" test slice will be overwhelmingly censored.

## 6. Tests

27 new tests added in `tests/credit_risk/` (schema grain/key checks,
config validation, target-definition censoring/boundary/separation
logic, feature-eligibility validation and leakage-keyword safety net).
Full project suite: **208 passed, 1 skipped** (the pre-existing,
unrelated skip from Module B's missing-raw-file guard; unchanged by
this milestone). All 27 Module C tests run against toy fixtures only —
no real credit-risk data was read.

## 7. Limitations

- No real dataset has been downloaded or joined against this schema —
  every table/column, status-vocabulary value, and feature-eligibility
  classification is a provisional design based on secondary-source
  documentation of LendingClub's publicly known column vocabulary, not
  the actual raw file's own data dictionary. Both configs' own
  `_disclaimer` fields say this explicitly.
- `scheduled_payments` is described as fully derivable from origination
  terms via a standard amortization formula, but no amortization
  generator has been implemented yet.
- The `grade`/`sub_grade`/`int_rate` circularity question is flagged,
  not resolved (Section 4 above).
- No split, no feature pipeline, and no model exist yet — this
  milestone is foundation/design only.
- **Selection bias / target-population limitation (not resolved,
  inherent to the recommended dataset)**: the LendingClub accepted-loans
  file contains only loans that were approved and originated — declined
  applications never appear in it. A `default_12m` PD model trained on
  this file therefore estimates default risk **conditional on having
  already been approved and originated**, among the population of loans
  LendingClub's own historical underwriting process chose to fund, not
  among the full population of applicants. **It must not be presented
  as a complete approve/decline underwriting model for all applicants.**
  Addressing this would require reject inference against LendingClub's
  separately published rejected-applications file (a much thinner table
  with no origination/repayment columns, since none exist for a
  declined application) — and even then, declined applicants' true
  repayment outcomes are permanently unobserved, since they were never
  funded; any reject-inference technique substitutes a modeling
  assumption about that unobserved counterfactual for an observed fact.
  Full discussion in `docs/credit_risk_dataset_comparison.md`,
  "Selection bias and target-population limitation." This is not a
  Milestone 2 action item by default — it is recorded as a standing
  limitation of this dataset choice, to be revisited explicitly if and
  when reject inference is ever judged in scope.
- **Dataset provenance has not yet been pinned down to a specific
  file.** LendingClub does not currently provide unrestricted public
  download of this historical file from its own platform; it will be
  obtained via one specific, reproducible public mirror whose source
  URL, license, and checksum are recorded only once actually downloaded
  (Milestone 2) — not asserted here. See
  `docs/credit_risk_dataset_comparison.md` for the current wording.

## 8. Recommended next milestone

**Module C Milestone 2**: download the LendingClub accepted-loans file,
validate the proposed 5-table schema and `default_12m`/feature-
eligibility configs against the real columns and `loan_status` values
(resolving both configs' "provisional, pending verification" caveats),
and build the chronological train/validation/test split — accounting
explicitly for the 12-month censoring-resolution lag noted in Section 5
above.
</content>
