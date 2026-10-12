# Module C, Milestone 1 — Credit Risk Data Model & Lending Lifecycle

## 0. Scope boundary

This milestone designs the **data model and point-in-time discipline** for
a credit-risk/lending module, complementing Module A (Payments/Product
Analytics) and Module B (Fraud Detection). **No credit-risk model is
trained in this milestone.** No dataset has been downloaded yet either —
Section C of `docs/credit_risk_dataset_comparison.md` evaluates candidates
and makes a recommendation, but that recommendation is a plan for the next
milestone, not something exercised against real rows here. Everything in
this document is schema/workflow design, validated only against toy data
in `tests/credit_risk/`.

## A. The lending lifecycle

```
applicant → application → underwriting decision → origination → scheduled repayment → delinquency → default
```

| Stage | What actually happens | Key date |
|---|---|---|
| **Applicant** | A person (or business) who may submit zero, one, or many applications over time. The applicant's existence predates and outlives any single application. | — |
| **Application** | A specific, dated request for a specific loan amount/purpose. This is the grain at which **underwriting information is frozen** — everything the lender used to decide is, by definition, known no later than this point. | `application_date` |
| **Underwriting decision** | The lender evaluates the application (income, credit history, requested terms, policy rules) and reaches a decision. This is a **process**, not a single field — its output is categorical (approve/decline) plus, if approved, the specific terms offered (amount, rate, term), which may differ from what was requested. | `underwriting_decision_date` |
| **Approval / decline** | The binary (or categorical, if counter-offers exist) outcome of underwriting. A decline ends that application's lifecycle — there is no loan, no repayment, no default to observe. Only approved-and-accepted applications proceed to origination. | (same as decision date) |
| **Origination** | The loan is actually funded/disbursed. This is the **observation start** for everything downstream (Section D) — it is the first moment the loan exists as a loan, not merely a request. Loan terms (amount, rate, term, installment schedule) are fixed at this point. | `origination_date` |
| **Scheduled repayment** | The amortization schedule the loan was originated with — a fixed, known-in-advance sequence of due dates and due amounts. This is itself a point-in-time-safe artifact: it is fully determined by origination-time terms (principal, rate, term), so it can be *derived*, not merely observed. | per-installment `scheduled_due_date` |
| **Delinquency** | The borrower misses a scheduled payment, or falls behind by some number of days (e.g. 30/60/90+ days past due). Delinquency is explicitly **not** terminal — a delinquent loan can cure (catch up) and return to current status. Treating delinquency as equivalent to default would conflate a recoverable state with an (effectively) irrecoverable one. | `delinquency_event_date` (first occurrence) |
| **Default** | The loan reaches a configured **severe**, effectively-terminal delinquency/charge-off state (Section D defines this precisely). Default is the outcome this module's target definition is built around. | `default_event_date` (first occurrence) |

**Why these distinctions matter for modeling**: "underwriting decision"
happens *before* a loan exists — so only application-time information can
ever be a legitimate underwriting feature (Section E). "Delinquency" and
"default" are not the same event and must not be collapsed into one label
(Section D) — a model trained to predict "any missed payment" answers a
materially different, and much noisier, question than one trained to
predict "severe, effectively unrecoverable default."

## B. Core data model

Five tables, read top-to-bottom as a clean 1:many lifecycle chain. No
table here has been populated from a real dataset yet — this is the
*target* schema Module C Milestone 2 will map a real dataset's actual
columns onto, not a description of an existing file. Logical column
names are defined in `src/credit_risk/schema.py`.

### B1. `borrowers`

| | |
|---|---|
| **Grain** | One row per borrower. |
| **Primary key** | `borrower_id` |
| **Relevant dates** | `first_observed_date` — when this borrower first appears in the data (not a biographical fact, just a data-provenance marker). |
| **Application-time availability** | By construction, a borrower-level snapshot used for underwriting must reflect only information known as of the *application* it's being used for, not the borrower's full later history. Many real datasets (including the one recommended in Section C) only provide a single borrower snapshot per application, which is actually the **safer** default (no risk of smuggling in later information) at the cost of not supporting "this borrower's 3rd loan" relationship features without extra care. |
| **Columns** | `borrower_id`, `first_observed_date`, `stated_annual_income`, `employment_length_years`, `home_ownership`. |

### B2. `loan_applications`

| | |
|---|---|
| **Grain** | One row per application. A borrower may have **zero, one, or many** applications (many:1 to `borrowers`). |
| **Primary key** | `application_id` |
| **Foreign key** | `borrower_id` → `borrowers` |
| **Relevant dates** | `application_date` (request submitted), `underwriting_decision_date` (decision reached — always >= `application_date`). |
| **Application-time availability** | Every column on this table except `decision`/`decline_reason`/`underwriting_decision_date` is, by definition, available at application time — that is the table's entire reason for existing separately from `loans`. `decision` and `decline_reason` are the underwriting **output**, not an underwriting **input** — never usable as a feature for the decision that produced them. |
| **Columns** | `application_id`, `borrower_id`, `application_date`, `requested_amount`, `stated_purpose`, `underwriting_decision_date`, `decision` (approved/declined), `decline_reason` (populated only when declined). |

### B3. `loans`

| | |
|---|---|
| **Grain** | One row per originated loan. At most one row per `application_id` — a declined application never reaches this table; an approved-and-funded application produces exactly one loan. |
| **Primary key** | `loan_id` |
| **Foreign keys** | `application_id` → `loan_applications`, `borrower_id` → `borrowers` (denormalized for convenience). |
| **Relevant dates** | `origination_date` — the observation start for everything in Section D. |
| **Application-time availability** | This is the first table where some columns are, by construction, **decision outputs**, not inputs: `funded_amount`, `interest_rate`, `installment_amount`, and `risk_grade` are set *as a result of* underwriting, at or immediately before `origination_date`. They describe the loan the lender chose to offer, not a fact about the borrower that existed before the decision — using them as if they were independent "application-time" predictor features (rather than the terms being modeled) is a subtle form of circularity, flagged explicitly in Section E and in `configs/credit_risk_feature_eligibility.json` for the `grade`/`sub_grade`/`int_rate`-style fields of the recommended candidate dataset. |
| **Columns** | `loan_id`, `application_id`, `borrower_id`, `origination_date`, `funded_amount`, `term_months`, `interest_rate`, `installment_amount`, `risk_grade`. |

### B4. `scheduled_payments`

| | |
|---|---|
| **Grain** | One row per `(loan_id, scheduled_payment_number)` — i.e. one row per installment the amortization schedule calls for. 1:many from `loans`. |
| **Primary key** | `(loan_id, scheduled_payment_number)` |
| **Relevant dates** | `scheduled_due_date` per installment. |
| **Application-time availability** | **Fully available at origination time** — not because it was observed, but because it is *derivable*: given `funded_amount`, `interest_rate`, `term_months`, and `origination_date` (all fixed at origination), a standard amortization formula determines every future due date and due amount with certainty. This table therefore carries **zero** leakage risk by construction — it is a deterministic function of already-origination-time facts, never of anything that happens afterward. |
| **Columns** | `loan_id`, `scheduled_payment_number`, `scheduled_due_date`, `scheduled_amount`, `scheduled_principal`, `scheduled_interest`. |

### B5. `actual_payments`

| | |
|---|---|
| **Grain** | One row per `(loan_id, payment_date)` actually received. 1:many from `loans`; generally sparser or denser than `scheduled_payments` for any given loan (missed payments, partial payments, or extra payments all break the 1:1 correspondence with the schedule). |
| **Primary key** | `(loan_id, payment_date)` |
| **Relevant dates** | `payment_date` — strictly a **post-origination, outcome-adjacent** fact; this entire table only exists because time has passed since origination. |
| **Application-time availability** | **Never available at application or origination time, by definition** — a payment cannot be "actually made" before the loan exists. This table (and anything derived from it: running totals, last-payment date, days-past-due) is exactly the category of information Section E's point-in-time rule excludes from underwriting features. It is, however, exactly the right source for (a) determining `delinquency_event_date`/`default_event_date` for the target definition (Section D), and (b) any later LGD/EAD-style loss analysis (Section F), where post-origination information is the whole point. |
| **Columns** | `loan_id`, `payment_date`, `amount_paid`, `principal_paid`, `interest_paid`, `days_past_due_at_payment`. |

### B6. Entity-relationship summary

```
borrowers (1) ──< (many) loan_applications
                         │
                         │ (declined applications stop here — no loan)
                         ▼ (approved & funded, at most 1 per application)
                      loans (1) ──< (many) scheduled_payments
                            │
                            └──< (many) actual_payments
```

## Point-in-time feature policy (Section E)

**The rule**: *a field may be used as an underwriting feature only if it
was genuinely available on or before the credit decision/origination
date.* This is the lending-domain restatement of the same discipline
Module A and Module B already apply to their own data (Module A's
synthetic-vs-real field inventory, Module B's point-in-time-safe feature
functions).

**Likely leakage categories** (all post-origination, all excluded from
underwriting features by default):

| Category | Example fields | Why it leaks |
|---|---|---|
| Future payment behavior | running `total_pymnt`, `last_pymnt_d`, `last_pymnt_amnt` | Only exists once the loan has been serviced for a while; directly encodes how the loan is performing. |
| Later delinquency counts | any "times 30+ days late" counter computed over the loan's life | Accumulates *because* of the outcome being predicted. |
| Recoveries | `recoveries`, `collection_recovery_fee` | Only exists for loans that have already defaulted — this is a post-default fact, not a pre-decision one. |
| Collections outcomes | `debt_settlement_flag`, `hardship_flag` | Both are servicing actions *triggered by* observed repayment distress. |
| Loan status measured after origination | `loan_status` itself, `out_prncp`, `next_pymnt_d` | `loan_status` is literally the outcome source; the others are time-varying post-origination balances/schedule-state. |
| Charge-offs | any "charged off" flag/date | The terminal realization of default — using it as a feature for predicting default is circular by definition. |
| Post-origination balances | `out_prncp`, any running balance field | Time-varying, exists only because the loan has aged. |

**Initial feature-eligibility table**: maintained as machine-readable
config, not just prose, in
`configs/credit_risk_feature_eligibility.json` (loaded/validated by
`src/credit_risk/feature_eligibility.py`), with one entry per field:
`field`, `category`, `available_at_application` (`yes`/`no`/`unknown`),
`proposed_use`, `leakage_concern`. It currently covers ~29 fields from the
recommended candidate dataset's (LendingClub accepted-loans, see
`docs/credit_risk_dataset_comparison.md`) publicly documented column
vocabulary — **provisional**, pending verification against the actual
raw file once it is downloaded in a later milestone. Representative
entries:

| field | available_at_application | proposed_use | leakage_concern |
|---|---|---|---|
| `annual_inc` | yes | underwriting feature | none |
| `dti` | yes | underwriting feature | none |
| `fico_range_low`/`fico_range_high` | yes | underwriting feature | none |
| `purpose`, `home_ownership`, `emp_length` | yes | underwriting feature | none |
| `grade`, `sub_grade`, `int_rate` | **unknown** | treat with caution — not included by default | circularity: these are the lender's own prior risk assessment/pricing, not independent borrower facts |
| `loan_status` | no | target source only, never a feature | this is the outcome itself |
| `total_pymnt`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `out_prncp` | no | exclude | post-origination payment/balance history |
| `recoveries`, `collection_recovery_fee`, `debt_settlement_flag`, `hardship_flag` | no | exclude (LGD/EAD illustration only, later) | post-default collections/servicing events |
| `last_fico_range_high`, `last_credit_pull_d` | no | exclude | refreshed credit pull taken *during* servicing, not at application |

`src/credit_risk/feature_eligibility.py::filter_application_time_safe`
returns only entries marked `yes` with no recorded leakage concern —
`grade`/`sub_grade`/`int_rate` are deliberately excluded from that default
"safe" set (they are `unknown`, not `yes`), and the loader raises loudly
(`FeatureEligibilityError`) if any field name matching a known
leakage-keyword pattern (`recover`, `total_pymnt`, `last_pymnt`,
`out_prncp`, `charged_off`, `settlement`, `hardship`, `last_fico`, …) is
ever mis-marked `available_at_application = "yes"` — this is a safety net
against a future editing mistake in the config, not a substitute for the
human judgment that produced the table.

## Limitations

- No real dataset has been downloaded or joined against this schema yet
  — every table/column above is a target design, validated only on toy
  data (`tests/credit_risk/test_schema.py`).
- `scheduled_payments` is described as fully derivable from origination
  terms, but no amortization-schedule generator has been implemented yet
  — that is Module C Milestone 2 scope, not this milestone's.
- The feature-eligibility table's judgments are based on secondary-source
  documentation of the recommended candidate dataset's column vocabulary,
  not the actual raw file's own data dictionary — explicitly flagged for
  re-verification before any real computation.
- The `grade`/`sub_grade`/`int_rate` circularity question is flagged, not
  resolved — whether to exclude these fields entirely, include them as a
  deliberate "what if we just replicate the lender's own pricing policy"
  baseline, or something in between, is left as an explicit Milestone 2
  decision.
