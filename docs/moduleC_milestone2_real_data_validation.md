# Module C, Milestone 2 — Real LendingClub Data Validation, Target Construction & Chronological Split Design

## 0. Scope boundary

This milestone downloads and validates the real LendingClub accepted-loans
file against every design decision Milestone 1 made on secondary-source
documentation alone, builds two honestly-labeled targets (no fabricated
event dates), runs a censoring analysis, re-audits leakage against the
real 151-column schema, and designs a chronological split from the real
origination-date distribution. **No credit-risk model is trained.**
This document is the consolidated summary; companion docs with fuller
detail are `docs/credit_risk_status_mapping.md` (Section C) and
`docs/credit_risk_split_design.md` (Section G).

## A. Dataset acquisition & provenance

Full record: `configs/credit_risk_raw_source_metadata.json`,
`src/credit_risk/raw_source.py`.

- **Source**: DePaul University course-data mirror
  (`bigblue.depaul.edu/jlee141/econdata/LendingClub_LoanData/`), file
  `LendingClub_2007_to_2018Q4.csv`, 1,675,133,810 bytes.
- **SHA256**: `3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a`
  (`verify_local_file_integrity` recomputes and compares this on demand).
- **License**: this mirror shows **no explicit license or usage-terms
  statement** — only a "source:" citation to a Dropbox folder (an origin
  citation, not a license grant). A separate, well-known Kaggle listing
  of apparently the same underlying data (`wordsforthewise/lending-club`)
  displays a "CC0: Public Domain" tag; that tag belongs to Kaggle's own
  page and is **explicitly not inherited** by this DePaul mirror, since
  it cannot be verified to originate from LendingClub itself. No
  authentication or click-through terms were required to download.
- **Reproducibility**: `curl -L -C - --retry 5 --retry-delay 3
  --max-time 1800`, no account/API key.

## B. Real schema profile

Profiled with DuckDB (`all_varchar=true` + `SUMMARIZE`), per Module B's
convention.

- **151 columns** (verified exact match against `configs/
  credit_risk_feature_eligibility.json`, which was rewritten from
  Milestone 1's ~29-entry guess to all 151 real field names).
- **2,260,668 clean loan rows** after excluding 33 non-loan footer/
  section-marker rows (see below). Raw file has 2,260,701 data rows
  total (confirmed well-formed via Python's own quote-aware `csv`
  module: every raw row has exactly 151 fields — the footer rows are not
  malformed CSV, they are genuine extra text rows LendingClub's export
  process appended, e.g. `"Total amount funded in policy code 1:
  6417608175"`).
- **Origination date range** (`issue_d`): 2007-06-01 to 2018-12-01
  (month granularity).
- **No duplicate `id` values** among the 2,260,668 clean rows.

**Footer-row contamination, found and fixed**: the file has 33 rows
where `id` is non-numeric text rather than a loan ID, interspersed at
roughly 20 points in the file. The first fix attempted was DuckDB's
`ignore_errors=true` with its default sampled type inference — this was
**rejected** after testing, because it silently nulled out real columns
on rows unrelated to the footer rows whenever its type-sniffing sample
missed a rarer value elsewhere in the file. The adopted fix,
`src/credit_risk/raw_loader.py::register_loans_view`, reads with
`all_varchar=true` (no casting at the CSV-reader level at all) and
applies an explicit `WHERE regexp_matches(id, '^[0-9]+$')` filter —
isolates exactly the 33 known-bad rows, with no risk of collateral
damage to real rows elsewhere in the 1.6GB file.

## C. Real loan-status vocabulary & mapping

Full detail: `docs/credit_risk_status_mapping.md`.

The real file has **9 distinct `loan_status` values** (not a guess —
enumerated via `SELECT DISTINCT loan_status`), each mapped explicitly in
`configs/credit_risk_status_mapping.json` / `src/credit_risk/
status_mapping.py`:

| loan_status | category | meets_credit_policy | count |
|---|---|---|---|
| Fully Paid | fully_paid | true | 1,076,751 |
| Current | current_performing | true | 878,317 |
| Charged Off | charged_off | true | 268,559 |
| Late (31-120 days) | severe_delinquency_or_default | true | 21,467 |
| In Grace Period | delinquent | true | 8,436 |
| Late (16-30 days) | delinquent | true | 4,349 |
| Does not meet the credit policy. Status:Fully Paid | fully_paid | **false** | 1,988 |
| Does not meet the credit policy. Status:Charged Off | charged_off | **false** | 761 |
| Default | severe_delinquency_or_default | true | 40 |

Two design points carried through from the Milestone 1 plan, now
verified against real data:

1. **"Does not meet the credit policy." statuses map to the same
   outcome category as their unprefixed counterpart.** Underwriting-
   policy vintage (`meets_credit_policy`) is tracked as a separate field
   and never conflated with repayment-outcome severity.
2. **`map_loan_status`/`map_status_series` raise `StatusMappingError`
   on any unmapped value** rather than silently bucketing it — a
   previously-unseen status string must be reviewed and added by a
   human.

`configs/credit_risk_target_definition.json` was updated after this
verification: `default_event_statuses` now reads `["Charged Off",
"Default", "Does not meet the credit policy. Status:Charged Off"]` —
the credit-policy-flagged charge-offs are real charge-offs and must
trigger the label the same way. `"Late (31-120 days)"` is deliberately
**excluded** from `default_event_statuses` despite sharing the
`severe_delinquency_or_default` *category* with `Default` — it is a
real but not-yet-terminal severity state (see its `target_trigger_note`
in `configs/credit_risk_status_mapping.json`), and category
(description) is intentionally decoupled from target-trigger membership
(label-setting rule) throughout this module.

## D. Target construction — the central empirical finding

Full detail and bias reasoning: `src/credit_risk/target_construction.py`
module docstring.

**The real file contains no genuine default/charge-off event-date
column.** It provides only a month-granularity origination date
(`issue_d`), a single current-snapshot `loan_status` (not a timestamped
event log), a month-granularity `last_pymnt_d`, and a handful of
hardship/debt-settlement workflow dates that are >98% NULL, exist only
for loans that entered those specific servicing programs, and are
already excluded as leakage regardless.

Per the explicit project rule against fabricating event dates, this
milestone builds **two distinct, separately-named, never-conflated
targets**. **Governance correction (pre-commit review, recorded in
`docs/decision_log.md`'s "Milestone 2 target-governance correction"
entry, machine-readable in `configs/credit_risk_target_governance.json`):**
the initial draft of this milestone treated target 2 below as the
presumptive primary PD target; that was wrong, and the designation is
corrected here before commit.

**1. `terminal_adverse_outcome`**
(`compute_terminal_adverse_outcome_target`) — **PRIMARY modeling target
for Milestone 3.** Requires no event date at all. Built only from
explicitly mapped resolved terminal `loan_status` categories
(`charged_off`, `fully_paid`). A loan is resolved once it has passed its
own scheduled maturity date (`issue_d` + `term` months); labeled 1 if
`charged_off`, 0 if `fully_paid` as of that point, censored otherwise.
This is a variable-horizon (36- or 60-month) "did this loan ever end in
default by its own maturity" target. **It estimates eventual adverse
credit outcome among originated loans and does NOT represent a
12-month probability of default** — it must never be described or
consumed as a fixed-horizon PD estimate.

**2. `default_12m_last_pymnt_proxy`**
(`compute_default_12m_proxy_target`) — **SECONDARY, sensitivity-
analysis-only target.** Never to be promoted back to primary status.
Reuses Milestone 1's validated, unmodified `compute_default_12m_labels`
12-month-window logic, fed a **proxy** event date: `last_pymnt_d` for
defaulted loans, falling back to `issue_d` only for the small subset of
charged-off loans with no payment ever recorded (`last_pymnt_d` NULL).
**`last_pymnt_d` is NOT a default-event timestamp** — it is the date of
the loan's most recent received payment, repurposed here only as an
explicitly-labeled proxy.

**Documented, one-directional bias (preserved from the original draft,
unchanged by the governance correction)**: LendingClub's publicly
documented servicing policy charges off a loan only after 120+ days
delinquent — the true formal charge-off date is systematically
**later** than `last_pymnt_d`. Anchoring the proxy event date on
`last_pymnt_d` therefore makes a default appear earlier than it truly
was, which can only ever **pull a borderline default into** the
12-month window, never push one out. This proxy is a
conservative-against-the-lender, one-directional approximation —
explicitly not a measurement of true event timing, and the two targets'
columns are never merged or reused interchangeably anywhere in this
codebase.

**`INFERRED_DATA_AS_OF_DATE = 2019-04-01`**: no explicit "as of"/extract
field exists in the raw file. This is the inferred proxy for the file's
snapshot date, derived from the maximum observed `last_credit_pull_d`
across the whole file — the field LendingClub's own servicing system
refreshes close to monthly for every loan still being tracked. Stated
as an inferred proxy, not an asserted fact from the source.

**Empirical edge case found and validated, not treated as a bug**: a
handful of loans (10, inspected directly in the 2013 vintage) are still
`Current`/`Late`/`In Grace Period` in the snapshot even though their
scheduled maturity date has already passed — almost certainly real
loan modifications/restructuring. Per the terminal-target design, these
stay **censored**, never coded either way; this is exactly the behavior
`test_terminal_target_past_maturity_but_still_open_is_censored_not_fabricated`
asserts.

## E. Censoring analysis

Both targets are fully resolved (0% censored) for originations through
2013; censoring grows as expected for the 12-month-window secondary
proxy target only in the final (2018) vintage, and much earlier for the
longer-horizon primary terminal target (which needs each loan's own
36/60-month maturity to pass, not just +12 months). This difference in
resolution speed is exactly why Section G defines two separate
chronological splits rather than reusing one for both targets.

**Primary target (`terminal_adverse_outcome`), censored % by origination year:**

| year | total | censored | default | pct censored |
|---|---|---|---|---|
| 2007-2013 | — | 0 | — | 0.0% |
| 2014 | 235,629 | 55,213 | 27,220 | 23.4% |
| 2015 | 421,095 | 138,069 | 42,132 | 32.8% |
| 2016 | 434,407 | 325,172 | 18,876 | 74.9% |
| 2017 | 443,579 | 443,579 | 0 | 100.0% |
| 2018 | 495,242 | 495,242 | 0 | 100.0% |

Overall: 1,457,285 censored / 679,056 non-default / 124,327 default
(of 2,260,668). This target's long, term-dependent horizon means a
large share of the file's more recent vintages are structurally
unresolvable by 2019-04-01 — expected, not a defect, and why Section G's
**primary** split uses much earlier boundaries (train through 2011,
benchmark = 2013) than the secondary split below.

**Secondary target (`default_12m_last_pymnt_proxy`), censored % by origination year:**

| year | total | censored | default | pct censored |
|---|---|---|---|---|
| 2007-2017 | — | 0 | — | 0.0% |
| 2018 | 495,242 | 341,325 | 8,852 | 68.9% |

Overall: 1,802,809 non-default / 341,325 censored / 116,534 default. The
12-month window resolves every origination year through 2017 cleanly —
this is the empirical basis for Section G's **secondary**-split
boundaries.

## F. Leakage audit of the real 151-column schema

Full table: `configs/credit_risk_feature_eligibility.json`
(`src/credit_risk/feature_eligibility.py`), rewritten from Milestone 1's
~29-entry secondary-source guess to all 151 real column names, verified
by exact-set match against the live file's header.

| available_at_application | count |
|---|---|
| yes (application-time safe) | 108 |
| no (post-origination leakage, or excluded identifier) | 40 |
| unknown (flagged, unresolved) | 3 |

- **40 excluded fields** include all `hardship_*`/`settlement_*`/
  `debt_settlement_*` workflow fields, payment/recovery fields
  (`total_pymnt`, `recoveries`, `last_pymnt_d`, `out_prncp`, …),
  `last_fico_range_*` (a post-origination credit-bureau refresh, not the
  application-time FICO snapshot), and `member_id` (see below).
- **Name-pattern-matching alone is not sufficient and was not used as
  the sole test**: `chargeoff_within_12_mths` *sounds* like an outcome
  field but actually describes charge-offs on the borrower's **other**
  bureau-reported accounts as observed at the application-time credit
  pull — correctly classified `available_at_application = "yes"`
  (`test_credit_bureau_pull_fields_are_safe_despite_outcome_sounding_names`).
- **Governance correction: `member_id` moved from `"unknown"` to `"no"`
  (excluded identifier), not left unresolved.** It is a pure identifier
  column with no defensible predictive business role, and is always
  100% NULL in this public release (LendingClub strips it for privacy) —
  "unknown" is reserved for genuinely ambiguous-availability fields, and
  `member_id` is not one; it is simply unusable and excluded. See
  `docs/decision_log.md`'s "Milestone 2 target-governance correction"
  entry.
- **`grade`/`sub_grade`/`int_rate` remain flagged `"unknown"`,
  deliberately unresolved** — carried forward from Milestone 1's
  circularity concern (these are LendingClub's own prior risk
  assessment/pricing, not independent borrower facts) and still not
  decided one way or the other. Explicitly confirmed, not revisited, by
  the governance correction. This is an explicit Milestone 3+ decision,
  not resolved by this milestone.
- A bug was found and fixed during this audit: informational notes for
  several "yes"-classified bureau-pull fields had been placed in the
  `leakage_concern` field, which made `filter_application_time_safe()`
  undercount safe features (37 instead of 108), since any non-empty
  `leakage_concern` — even a clarifying note, not an actual concern —
  disqualifies an entry from "safe." Fixed by moving all such notes into
  `proposed_use` instead; the automated leakage-keyword safety net in
  `FeatureEligibilityEntry.__post_init__` (Milestone 1) was not broken
  by this and did not need to change.

## G. Chronological split design

Full tables and rationale: `docs/credit_risk_split_design.md`.
**Governance correction: there are now TWO split configs, not one**
(previously a single `configs/credit_risk_split_design.json`) — see
`docs/decision_log.md`'s "Milestone 2 target-governance correction"
entry and `configs/credit_risk_target_governance.json`.

**Primary split** — `configs/credit_risk_split_design_primary_terminal.json`,
for the primary `terminal_adverse_outcome` target. Because terminal
resolution depends on each loan's own 36- or 60-month maturity, not a
fixed window, **the secondary split's boundaries could not be reused**:
under `terminal_adverse_outcome`, the secondary split's benchmark year
(2017) is 100% censored and its validation years (2015-2016) are
33-75% censored (Section E). `SplitDesign.__post_init__`'s existing
`benchmark_end + outcome_window_months <= data_as_of_date` validation
rule generalizes cleanly by setting `outcome_window_months=60` (the
real max LendingClub term — confirmed `term` values `[36, 60]` only),
requiring no new validation code:

| split_bucket | total_rows | censored_excluded | labeled_rows | default_prevalence_pct | origination range |
|---|---|---|---|---|---|
| train | 42,535 | 0 | 42,535 | 15.119% | 2007-06 – 2011-12 |
| validation | 53,367 | 0 | 53,367 | 16.197% | 2012-01 – 2012-12 |
| benchmark | 134,814 | 10 | 134,804 | 15.596% | 2013-01 – 2013-12 |
| excluded_too_recent | 2,029,952 | 1,457,275 | 572,677 | 15.406% (of labeled) | 2014-01 – 2018-12 |

**Secondary split** — `configs/credit_risk_split_design_secondary_proxy.json`
(previously `credit_risk_split_design.json`, unchanged boundaries), for
the secondary `default_12m_last_pymnt_proxy` sensitivity-analysis
target. Boundaries are a direct, non-arbitrary consequence of Section
E's real censoring-by-year table for this target: 2007-2017 is 0%
censored, only the 2018 vintage is censored (68.9%) and is excluded
**in full**, not partially — a loan-level filter within 2018 would keep
only the fastest-to-resolve early-2018 subset, a subtle survivorship
distortion, which the project rule against touching later periods to
optimize performance explicitly rules out.

| split_bucket | total_rows | censored_excluded | labeled_rows | default_prevalence_pct | origination range |
|---|---|---|---|---|---|
| train | 466,345 | 0 | 466,345 | 5.178% | 2007-06 – 2014-12 |
| validation | 855,502 | 0 | 855,502 | 6.508% | 2015-01 – 2016-12 |
| benchmark | 443,579 | 0 | 443,579 | 6.281% | 2017-01 – 2017-12 |
| excluded_too_recent | 495,242 | 341,325 | 153,917 | 5.751% (of labeled) | 2018-01 – 2018-12 |

For each split, every row lands in exactly one of its four buckets
(`build_chronological_split`'s return value accounts for all 2,260,668
rows in both); `excluded_too_recent` is reported for transparency but
is not a labeled split. A hard post-hoc check in
`build_chronological_split` raises if any censored row is ever found
inside a labeled train/validation/benchmark bucket — verified
empirically above (0 censored rows in the labeled subset of each
bucket, both splits). `load_split_design()`'s no-argument default now
returns the **primary** split; the secondary split is loaded explicitly
via `SECONDARY_PROXY_SPLIT_DESIGN_FILE`.

## H. Selection-bias reminder (reiterated from Milestone 1, unresolved)

The LendingClub accepted-loans file contains only originated (accepted)
loans — declined applications never appear in it. Every number in this
document, and any PD model eventually trained on this split, describes
default risk **conditional on having already been approved and
funded**, not across the full applicant population. This is unchanged
from Milestone 1's finding, is not addressed by anything built in this
milestone, and must not be presented as a complete approve/decline
underwriting model. See `docs/credit_risk_dataset_comparison.md` and
`docs/moduleC_milestone1_credit_risk_foundation.md` §7 for the full
discussion.

## I. Engineering & tests

New modules: `src/credit_risk/raw_source.py`, `raw_loader.py`,
`status_mapping.py`, `target_construction.py`, `chronological_split.py`,
`target_governance.py` (the governance-correction addition) (plus
`schema.py` additions for the new target/status-category constants).
New tests across `tests/credit_risk/test_raw_source.py`,
`test_status_mapping.py`, `test_target_construction.py`,
`test_chronological_split.py`, `test_raw_loader.py`, `test_real_
feature_eligibility.py`, `test_target_governance.py` (new), plus
`tests/credit_risk/conftest.py`'s `requires_credit_risk_raw_data` skip
marker (mirrors `tests/conftest.py`'s `requires_raw_data`).

Testing discipline: pure transformation logic (`target_construction`,
`chronological_split`, `target_governance`) is tested against small,
hand-built toy fixtures/configs — never the real file — consistent with
Milestone 1's toy-fixture-only convention; `test_raw_loader.py`'s
real-data assertions (33 excluded rows, no duplicate IDs, 2,260,668 row
count) are gated behind `requires_credit_risk_raw_data` so the suite
still passes cleanly on a machine without the (gitignored) raw file
downloaded.

Full project test suite after this milestone's governance correction:
see the commit/review record for the exact pass/skip counts the test
run reported — the suite grew from the original 42 new Milestone 2
tests to include renamed terminal-target tests, an added secondary-
split-design test, and the new `test_target_governance.py` file.

## J. Limitations carried forward / newly recorded

- **Selection bias** (Section H) — unresolved, inherent to the dataset.
- **`grade`/`sub_grade`/`int_rate` circularity** — still flagged
  `"unknown"`, still not decided. Explicitly re-confirmed (not
  revisited) by the governance correction.
- **`default_12m_last_pymnt_proxy` is a documented, one-directional
  proxy, not a true event-date target, and is SECONDARY (sensitivity-
  analysis-only)** — must always be reported alongside its bias
  direction (Section D), never presented as equivalent to a dataset
  with genuine charge-off timestamps, and never promoted back to
  primary-target status.
- **`terminal_adverse_outcome` (PRIMARY target) has a variable,
  term-dependent horizon** (36 or 60 months depending on the loan) — it
  is not comparable loan-to-loan on a fixed time axis the way the
  12-month proxy target is, and estimates eventual adverse credit
  outcome among originated loans, NOT a 12-month probability of
  default. Its dedicated primary split (Section G) is both
  earlier-boundaried and substantially smaller in labeled row count
  (230,706 vs. 1,765,426 for the secondary split) as a direct,
  foreseeable consequence of this longer horizon — not a defect.
- **No amortization-schedule generator** and **no model** exist yet —
  unchanged scope boundary from Milestone 1, still explicit Milestone 3+
  territory.

## K. Recommended next milestone

**Module C Milestone 3**: build the baseline PD-adjacent model on the
frozen `train`/`validation`/`benchmark` **primary** split defined here
(`configs/credit_risk_split_design_primary_terminal.json`), using the
**primary** `terminal_adverse_outcome` target (per
`configs/credit_risk_target_governance.json`) and the 108
application-time-safe features from Section F — with an explicit,
pre-registered decision on how (or whether) to include the
`grade`/`sub_grade`/`int_rate` circularity group before any fitting
begins. Any sensitivity analysis against the secondary
`default_12m_last_pymnt_proxy` target and its own split
(`configs/credit_risk_split_design_secondary_proxy.json`) should be
clearly labeled secondary throughout, never presented as the headline
result.
