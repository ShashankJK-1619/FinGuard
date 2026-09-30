# Leakage Review — Milestone 1 Section G

No columns are dropped in this milestone. This document identifies risk
categories and open questions; decisions about exclusion belong to feature
engineering (a later milestone), made with evidence, not suspicion alone.

## 1. Direct target reveal

- `isFraud` is the target itself — trivially excluded as a feature, not a
  leakage finding.
- No other column is documented as a restatement or encoding of `isFraud`.
  An empirical check (correlation with target, run by
  `src/data/leakage_checks.py` once data is present) flags any column with
  suspiciously high correlation for manual review — see
  [leakage_checks_empirical.md](leakage_checks_empirical.md).

## 2. Future information / post-outcome fields

- **`D1`–`D15`** ("timedelta" features, per documentation) are the highest
  concern in this category. We do not know Vesta's computation methodology:
  if any `Dn` was computed using information only available after the
  transaction was scored or resolved (e.g., time until a later chargeback),
  it would leak the outcome. **Status: unresolved, flagged for investigation
  before use as a model feature.** We will not use any `D` column in
  Milestone 2 without first checking its distribution conditional on
  `TransactionDT` position (early vs. late in the dataset) and its behavior
  in train vs. test.
- **`C1`–`C14`** ("counting" features) are also a concern: if a count (e.g.,
  "how many addresses associated with this card") was computed using the
  *entire* dataset rather than only transactions *prior to* the current one,
  it implicitly encodes future information. **Status: unresolved, same
  treatment as `D` columns.**
- **`V1`–`V339`** (Vesta's own engineered features, described only as
  "ranking, counting, and other entity relations") carry the same
  point-in-time-computation risk as `C` and `D`, compounded by having zero
  per-column documentation. **Status: unresolved.**

None of these are dropped now. They are documented as "unknown computation
methodology, verify before use" rather than excluded or assumed safe.

## 3. Train/test contamination

- Row-level contamination (the same `TransactionID` appearing in both train
  and test) is checked empirically — see
  [leakage_checks_empirical.md](leakage_checks_empirical.md). By competition
  design this should be zero, but we verify rather than assume.
- A subtler contamination risk applies to **feature engineering we will do
  ourselves** later: any aggregate we compute (e.g., "count of transactions
  per card1") must be computed using only training-time information, and
  must not be fit using statistics derived from the test set. This is a
  methodology risk for future milestones, not a property of the raw data,
  and is recorded here so it isn't forgotten.

## 4. Time encoding requiring careful handling

- `TransactionDT` is a relative offset, not a wall-clock timestamp (verified
  in [time_structure.md](time_structure.md)). It must drive any train/val/test
  split design in a time-aware way; using it as a raw numeric model feature
  without care risks the model learning "later offset = more/less fraud" in a
  way that doesn't generalize, rather than learning transferable behavioral
  signal.
- Any `D` column that represents "time since X" inherits the same
  reference-point ambiguity as `TransactionDT`.

## 5. Fields judged lower-risk (documented, pre-transaction attributes)

`ProductCD`, `card1`–`card6`, `addr1`–`addr2`, `dist1`–`dist2`,
`P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`, and the `id_*`
identity fields are documented as attributes that would plausibly be known
*at* transaction time (card/account/address/device/network attributes).
They are lower-risk by nature but are **not independently verified** here —
"lower risk" is a structural judgment based on what the field plausibly
represents, not proof of safety.

## Summary table

| Field group | Concern | Status |
|---|---|---|
| `isFraud` | Is the target | Excluded as feature (trivial) |
| `D1`–`D15` | Possible future-information leakage via unknown computation window | Unresolved — investigate before use |
| `C1`–`C14` | Possible future-information leakage via global (not point-in-time) counts | Unresolved — investigate before use |
| `V1`–`V339` | Same risk as C/D, undocumented | Unresolved — investigate before use |
| `TransactionDT` | Must drive time-aware split, not be treated as an ordinary numeric feature | Documented, handling deferred to modeling milestone |
| Train/test ID overlap | Would indicate contamination | Checked empirically, see leakage_checks_empirical.md |
| Card/address/device/identity fields | Plausibly pre-transaction attributes | Lower risk, not independently verified |
