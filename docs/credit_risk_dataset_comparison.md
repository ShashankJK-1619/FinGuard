# Module C, Milestone 1 — Credit-Risk Dataset Comparison (Section C)

## Scope boundary

This document evaluates candidate public datasets against the data model
in `docs/credit_risk_data_model.md` and **makes a recommendation only** —
per explicit scope, no dataset is downloaded or read in this milestone.
The recommendation is what Module C Milestone 2 will act on.

## Candidates considered

Three publicly available, non-employer-sourced datasets were considered,
chosen to span the realistic design space for a retail-lending default
study: a peer-to-peer consumer-loan dataset with loan-level life-cycle
detail (LendingClub), a prime-mortgage performance dataset with
investor-grade loan-level detail (Freddie Mac Single-Family Loan-Level
Dataset, "SFLLD"), and an alternative-data consumer-credit Kaggle
competition dataset (Home Credit Default Risk).

### 1. LendingClub accepted loans (2007–2018Q4)

- **What it is**: Loan-level accepted-application data originally
  published by LendingClub for its peer-to-peer consumer lending
  platform, covering loans originated 2007–2018. **Provenance caveat,
  stated explicitly**: LendingClub does not currently offer unrestricted
  public download of this historical file directly from its own
  platform the way it may have at original release — in practice the
  file is obtained today via third-party re-hosted public mirrors (most
  commonly a Kaggle dataset), not a live, openly-available LendingClub
  endpoint. This document should not be read as claiming the *original*
  LendingClub download is itself currently open-access without
  restriction; it is not. **This project will use one specific,
  reproducible public mirror of the 2007–2018 accepted-loans file.**
  When that mirror is actually downloaded (Module C Milestone 2), the
  exact source URL, the license/terms as stated by that specific mirror,
  and a file checksum (e.g. SHA-256) of the downloaded file will be
  recorded in this document and in `docs/decision_log.md`, so the
  provenance is reproducible and auditable rather than asserted from
  memory. No mirror has been selected or downloaded yet.
- **Grain**: One row per originated loan, with a `loan_status` field and
  several cumulative payment/balance columns that capture the loan's
  performance as of the file's extract date.
- **Target definition fit**: `loan_status` includes an explicit
  vocabulary (`Fully Paid`, `Current`, `In Grace Period`,
  `Late (16-30 days)`, `Late (31-120 days)`, `Charged Off`, `Default`)
  that maps directly onto this module's default/delinquency severity
  split (Section D) with no invented categories.
- **Application-time feature availability**: Strong — includes
  self-reported borrower fields (`annual_inc`, `emp_length`,
  `home_ownership`, `dti`, `purpose`), credit-bureau-derived fields
  (`fico_range_low/high`, `revol_util`, `open_acc`, `inq_last_6mths`),
  and loan-request terms (`term`, `installment`). It also includes
  `grade`/`sub_grade`/`int_rate`, the lender's own prior risk
  assessment/pricing — flagged as a circularity concern, not treated as
  a free application-time feature (Section E).
- **Leakage surface**: Well understood and documented ahead of time —
  cumulative payment fields (`total_pymnt`, `last_pymnt_d`,
  `last_pymnt_amnt`, `next_pymnt_d`, `out_prncp`), collections/recovery
  fields (`recoveries`, `collection_recovery_fee`), and
  servicing-event flags (`hardship_flag`, `debt_settlement_flag`) are
  all clearly post-origination by name and documented meaning.
- **Size / class balance**: ~2.9M rows across the full 2007–2018Q4
  release; charge-off rate is a double-digit percentage of closed loans
  (public summaries put realized default-and-charge-off status in the
  high single digits to low double digits of originated balance
  depending on vintage/grade mix) — enough minority-class volume for a
  PD model, unlike extremely rare-event fraud data.
- **Documentation quality**: A long-published, stable data dictionary
  (`LCDataDictionary`) with per-column descriptions; widely used in
  public credit-risk tutorials and papers, so independent column-meaning
  cross-checks are easy to find.
- **Reproducibility / license**: Obtained via a third-party public
  mirror (most commonly a Kaggle dataset), not a live open download
  from LendingClub's own platform — LendingClub does not currently
  provide unrestricted public download of this historical file itself.
  Such mirrors commonly self-tag the file CC0/public domain, but that
  tag is the re-hosting party's own claim and has **not yet been
  independently verified** against LendingClub's original distribution
  terms. No competition-account gate and no employer-NDA concern either
  way, but "low access friction" should not be overstated as "openly
  available with no restrictions" before a specific mirror, its stated
  license, and a checksum of the downloaded file are actually recorded
  (Milestone 2 action item, not done yet).
- **Portfolio credibility**: Directly maps to the lifecycle this module
  documents (application → underwriting → origination → repayment →
  delinquency/default) because it *is* that lifecycle, not mortgage
  servicing or an anonymized feature-engineering competition — the
  clearest "real consumer credit risk" story of the three.

### 2. Freddie Mac Single-Family Loan-Level Dataset (SFLLD)

- **What it is**: Loan-level origination and monthly performance data
  for fixed-rate, fully amortizing prime single-family mortgages
  acquired by Freddie Mac, released directly by Freddie Mac for
  research use.
- **Grain**: Origination record + a separate monthly performance panel
  per loan (delinquency status code evolves month over month) — a
  richer time-series performance table than LendingClub's single
  extract-date snapshot, which is attractive for this module's
  `data_as_of_date`/censoring design (Section D) since true monthly
  granularity is native to the data rather than something to
  approximate.
- **Target definition fit**: Monthly delinquency status codes (current,
  30/60/90+ days delinquent, foreclosure, REO) support a default
  definition at least as precise as LendingClub's, arguably more so
  given the monthly cadence.
- **Application-time feature availability**: Narrower than LendingClub
  — origination fields are mortgage-specific (credit score, LTV/CLTV,
  DTI, loan purpose, occupancy status, channel, property type) and
  reflect *prime* mortgage underwriting, not general consumer credit.
  Fewer fields overall than LendingClub's accepted-loan file.
- **Leakage surface**: Clean in principle — origination fields and
  monthly performance fields are in physically separate files/tables, so
  the join itself enforces a point-in-time boundary rather than relying
  on column-name judgment calls the way a single wide LendingClub row
  does. This is a genuine structural advantage.
- **Size / class balance**: Tens of millions of loan-month records
  across tens of millions of loans; however, because Freddie Mac only
  acquires *prime* conforming mortgages, the realized default rate is
  much lower than LendingClub's (prime mortgage credit box, not
  subprime/near-prime consumer credit) — a harder minority-class
  problem for a teaching-scale model, and a narrower risk spectrum to
  learn from.
- **Documentation quality**: A formal, Freddie-Mac-published user guide
  and data dictionary — high quality, arguably more rigorous than
  LendingClub's, since it is produced by a GSE for external researchers.
- **Reproducibility / license**: Requires a free Freddie Mac account
  registration and acceptance of its data-use agreement before
  download; not a one-click anonymous pull like LendingClub's public
  mirror. Still fundamentally reproducible by anyone willing to
  register, and not employer-sourced.
- **Portfolio credibility**: Very strong — real GSE mortgage data is
  arguably *more* prestigious than a P2P lender's self-published file,
  and the clean origination/performance table separation is a better
  illustration of point-in-time discipline "by construction" rather
  than by careful column exclusion. The tradeoffs are the registration
  friction, narrower feature set, and lower-rate/less-diverse target
  compared to LendingClub.

### 3. Home Credit Default Risk (Kaggle competition)

- **What it is**: A Kaggle competition dataset (hosted by Home Credit
  Group) combining an application table with several auxiliary tables
  (previous applications, bureau records, credit-card/POS/installment
  balance histories) for an alternative-lending/underserved-borrower
  population.
- **Grain**: Multi-table, application-centric — richer relationally than
  either LendingClub or SFLLD, closer in spirit to this module's
  `borrowers`/`loan_applications`/`loans` separation.
- **Target definition fit**: Binary `TARGET` (payment difficulty within
  a fixed observation window) provided directly by the competition —
  convenient, but it is a **pre-computed label**, not a status history
  a learner derives themselves the way LendingClub's `loan_status`
  requires (Section D's censoring logic has no real analogue to exercise
  against this file, since there is no raw delinquency-event timeline to
  derive the window from).
- **Application-time feature availability**: Good on the main
  application table, but several auxiliary tables (`bureau_balance`,
  `POS_CASH_balance`, `credit_card_balance`, `installments_payments`)
  require careful, competition-specific judgment about what was known
  as of application — this is exactly the point-in-time ambiguity this
  module is trying to avoid building a target/feature pipeline around
  for a first milestone.
- **Leakage surface**: Historically a notorious subject of
  Kaggle-community leakage write-ups (e.g. some engineered features in
  public kernels correlating suspiciously well with `TARGET`) — a
  known hazard, not a hypothetical one, for exactly the kind of subtle
  circularity this module's leakage policy (Section E) is designed to
  catch.
- **Size / class balance**: ~307K application rows, `TARGET` roughly
  8% positive — reasonable size and balance, smaller than LendingClub.
- **Documentation quality**: Competition-style column descriptions
  (`HomeCredit_columns_description.csv`) — serviceable, but written for
  a Kaggle leaderboard audience, not a loan-servicing data dictionary;
  less authoritative than either LendingClub's or Freddie Mac's.
- **Reproducibility / license**: Kaggle competition rules apply (free
  account required, competition rules acceptance); not as frictionless
  as a self-tagged-CC0 public mirror, comparable in friction to Freddie
  Mac's registration gate.
- **Portfolio credibility**: Good as a "handle messy multi-table
  alternative data" exercise, but weaker than LendingClub or SFLLD as a
  demonstration of this module's specific lifecycle/point-in-time
  design, precisely because its target is pre-baked and its leakage
  surface is less legible from documentation alone.

## Comparison summary

| Criterion | LendingClub | Freddie Mac SFLLD | Home Credit |
|---|---|---|---|
| Realism (real lender, real outcomes) | Yes | Yes | Yes |
| Documentation quality | Strong, stable public dictionary | Strong, formal GSE user guide | Serviceable, competition-style |
| Size | ~2.9M loans | 10s of millions of loan-months | ~307K applications |
| Target definition fits this module's design | Direct (`loan_status` vocabulary maps onto Sec D) | Direct, arguably cleaner (monthly codes) | Pre-baked binary, no raw timeline to derive from |
| Application-time feature availability | Rich, consumer-credit-relevant | Narrower, mortgage-specific | Rich but spread across ambiguous-timing auxiliary tables |
| Leakage avoidance (documentability) | Well-documented, known leakage fields | Structurally clean (separate tables) | Known history of subtle leakage in public work |
| Reproducibility (access friction) | Lower than alternatives — public mirror, no account/competition gate, but mirror source/license/checksum still to be recorded at download time, not a live open LendingClub endpoint | Free registration + data-use agreement | Free Kaggle account + competition rules |
| Portfolio credibility for this module's lifecycle story | Best fit (P2P consumer lending, matches Sec A/B directly) | Strong, but a different (mortgage) lifecycle | Weaker fit for this module's specific point-in-time story |

## Recommendation

**LendingClub accepted-loans data is recommended as the primary dataset**
for Module C. It is the best fit for this module's specific purpose: its
`loan_status` vocabulary maps directly onto the default/delinquency
severity split already designed in Section D without reinterpretation,
its documented column set gives application-time and post-origination
fields that are distinguishable primarily *by name and documented
meaning* (which is exactly the discipline Section E's feature-eligibility
table and leakage-keyword safety net are built to exercise), it has
lower access friction than either alternative, and its consumer-lending
grain matches the applicant → application → origination → repayment →
default lifecycle in Section A more directly than a mortgage or
competition-feature-store dataset would.

**Freddie Mac SFLLD is the strongest alternative**, and is noted
explicitly for a reason beyond "it didn't win": its origination/monthly-
performance table split enforces point-in-time separation structurally,
which is arguably a *more* rigorous teaching example than LendingClub's
single wide row. If a later milestone wants to stress-test this
module's point-in-time discipline against a dataset where the leakage
boundary is enforced by file structure rather than column-name judgment,
SFLLD is the natural second dataset to bring in — not as a replacement,
but as a complementary validation exercise.

**Home Credit Default Risk is retained as a third comparison point**
but is not recommended, primarily because its pre-computed `TARGET`
label gives this module's own target-definition logic
(`compute_default_12m_labels`, Section D) nothing to derive, and its
documented history of subtle feature leakage in public analyses is a
hazard this milestone's design is specifically trying to avoid
inheriting on day one.

## Selection bias and target-population limitation (applies to the recommended dataset)

The LendingClub accepted-loans file contains only loans that were
**approved and originated** — declined applications never appear in it
(Section A's lifecycle explicitly ends "no loan, no repayment, no
default to observe" for a decline). This has a direct, unavoidable
consequence for what any PD model later built on this file can honestly
claim, and it is recorded here so it is not discovered only after a
model exists:

- A `default_12m` model trained on this file estimates default risk
  **conditional on having already been approved and originated** — i.e.
  among the population of applicants LendingClub's own historical
  underwriting process chose to fund, not among the full population of
  people who applied.
- Such a model **must not be presented as a complete approve/decline
  underwriting model for all applicants.** The accepted population is
  not a random sample of applicants — it is already shaped by whatever
  underwriting criteria (observed and unobserved) LendingClub applied
  historically. A model that only ever sees accepted loans has no basis
  for scoring the population of applicants who look sufficiently
  different from anyone LendingClub approved.
- Addressing this rigorously would require **reject inference** using
  LendingClub's separately published rejected-applications file (a much
  thinner table — typically just requested amount, requested
  term/purpose, application date, and a risk score at the time of the
  rejection decision; no origination or repayment columns at all,
  because none exist for a declined application). Obtaining that file is
  not part of this recommendation.
- Even with the reject file in hand, **declined applicants' true
  repayment outcomes are permanently unobserved** — they were never
  funded, so there is no ground truth for "would this person have
  defaulted." Any reject-inference technique (e.g. parceling,
  reweighting, Heckman-style selection correction) substitutes a
  modeling assumption about that unobserved counterfactual outcome for
  an observed fact; it narrows the bias, it does not eliminate it.
- This limitation is inherent to any accepted-loans-only credit dataset
  — it would apply equally to a PD model built on the Freddie Mac SFLLD
  origination file, for example. It is documented here specifically
  because LendingClub's accepted-loans file is the dataset this
  document recommends.

## What this recommendation does and does not authorize

- It authorizes Module C Milestone 2 to download the LendingClub
  accepted-loans file, via a specific, reproducible public mirror whose
  source URL, license, and checksum will be recorded at download time,
  and begin mapping its real columns onto the schema in
  `docs/credit_risk_data_model.md`.
- It does **not** authorize treating any column's provisional
  classification in `configs/credit_risk_feature_eligibility.json` or
  `configs/credit_risk_target_definition.json` as final — both configs'
  own disclaimers already state they must be re-verified against the
  actual raw file's data dictionary once downloaded.
- It does **not** authorize presenting any resulting PD model as a
  complete approve/decline underwriting model for the full applicant
  population — see "Selection bias and target-population limitation"
  above. The model's scope is default risk among originated loans.
- It does **not** itself constitute downloading, reading, or training
  on any real data — no dataset has been fetched in this milestone.
</content>
