# Feature Dictionary — Milestone 2 Baseline Dataset

Grain: **one row per `TransactionID`**, built from `train_transaction` LEFT
JOIN `train_identity` only. `test_transaction`/`test_identity` are never
loaded in this milestone — there is no split yet, and test is never used
as labeled evaluation data. Output: `data/processed/baseline_features_v1.parquet`
(590,540 rows, 38 columns).

For every feature: source column(s), grain, point-in-time safety,
business interpretation, and known limitation.

## Identifiers / target (not features)

| Column | Source | Notes |
|---|---|---|
| `TransactionID` | `train_transaction.TransactionID` | Primary key. Not a model input. |
| `isFraud` | `train_transaction.isFraud` | Label. Not a model input. |

## Raw passthrough features (documented category, undisclosed per-column meaning)

These are kept as-is because the competition documents their general
category but not their exact per-column business meaning. There is
nothing defensible to *compute* from them beyond using the value — any
attempt to interpret, say, "card2 is probably the issuing bank code" would
be fabricating meaning the source does not provide.

| Column | Source | Point-in-time safety | Business interpretation | Known limitation |
|---|---|---|---|---|
| `TransactionDT` | transaction | Safe — attribute of the transaction itself | Relative time offset in seconds from an undisclosed reference point (documented) | Not a wall-clock timestamp; only used directly by models that can handle a monotonic-but-uncalibrated time axis |
| `TransactionAmt` | transaction | Safe | Payment amount in USD (documented) | Heavily right-skewed (Milestone 1: mean $135 vs. median $70, max $31,937) — see `TransactionAmt_log1p` below |
| `card1`–`card6` | transaction | Safe | Payment card attributes; documented category (type/category/issuing bank/country) | Exact per-column mapping undisclosed; `card1` is near-continuous-ID-like (13,553 distinct values) and unsuitable for one-hot encoding — used instead as a frequency/recency key (below) |
| `addr1`, `addr2` | transaction | Safe | Address fields; documented category | Which address (billing/shipping) and granularity undisclosed; `addr1` ~11% missing |
| `dist1`, `dist2` | transaction | Safe | Distance fields; documented category | Units and reference points undisclosed; `dist2` is 93.6% missing (Milestone 1) — kept for transparency, very likely unusable at modeling time |
| `P_emaildomain`, `R_emaildomain` | transaction | Safe | Purchaser / recipient email domain (documented) | `R_emaildomain` is frequently null when purchaser == recipient by convention, not because it's unmeasured — do not treat null the same as "unknown" without checking `email_domain_match` below |
| `ProductCD` | transaction | Safe | Product/service code (documented); 5 distinct values (`C`, `H`, `R`, `S`, `W`) in train | Specific code-to-product mapping not disclosed |
| `DeviceType` | identity | Safe | Device category, e.g. mobile/desktop (documented) | Only populated when `has_identity_record == 1` |
| `DeviceInfo` | identity | Safe | Free-text-like device descriptor (documented category, high cardinality) | Not parsed into OS/model substrings — no documented grammar for this field, so any parsing would be an invented assumption. Used as a frequency-encoding key instead (below) |

## Engineered features

### Amount

| Feature | Source | Grain | Point-in-time safety | Business interpretation | Known limitation |
|---|---|---|---|---|---|
| `TransactionAmt_log1p` | `TransactionAmt` | row | Safe — pure function of the row's own amount | `log1p(amount)`, compresses the long right tail so linear/logistic models don't get dominated by a handful of large transactions | `log1p` used instead of `log` as a safety margin against a future amount of exactly 0, even though none exist in train today |

### Time (from `TransactionDT`)

| Feature | Source | Grain | Point-in-time safety | Business interpretation | Known limitation |
|---|---|---|---|---|---|
| `day_index` | `TransactionDT` | row | Safe | `TransactionDT // 86400` — ordinal elapsed-day bucket since the start of data collection | Not a calendar date; only an ordinal counter |
| `week_index` | `day_index` | row | Safe | `day_index // 7` — ordinal elapsed-week bucket | Same caveat as `day_index` |
| `hour_of_day` | `TransactionDT` | row | Safe | `(TransactionDT % 86400) // 3600` — a 0–23 cyclical position. Because every row shares the same unknown reference point, this bucket is internally consistent across all rows and will still capture real time-of-day patterns (e.g. nighttime fraud spikes) | We do **not** know which bucket corresponds to real clock hours (e.g. bucket 3 might be 3am or 3pm) — phase is unknown, only the cyclical consistency is claimed |
| `hour_sin`, `hour_cos` | `hour_of_day` | row | Safe | Sine/cosine transform of `hour_of_day` so hour 23 and hour 0 are numerically adjacent (no artificial cliff at midnight) for models sensitive to that discontinuity | Same phase-unknown caveat as `hour_of_day` |
| `day_of_week_index` | `day_index` | row | Safe | `day_index % 7` — a 0–6 cyclical weekly position, same phase-unknown-but-consistent reasoning as `hour_of_day` | **Not** verified to align with real Mon–Sun weekdays; do not label buckets as specific weekdays |

### Identity availability

| Feature | Source | Grain | Point-in-time safety | Business interpretation | Known limitation |
|---|---|---|---|---|---|
| `has_identity_record` | `TransactionID` presence in `train_identity` | row | Safe, under the assumption that identity-row presence/absence is known at scoring time (Milestone 1: clean one-to-zero-or-one join, not resolved after the fact) | Whether Vesta's identity/device/network signal was captured for this transaction at all | Assumption stated above is not independently re-verified in this milestone beyond the Milestone 1 join check |
| `DeviceType_missing`, `DeviceInfo_missing`, `P_emaildomain_missing`, `R_emaildomain_missing` | respective raw column | row | Safe | Whether that specific signal was captured, independent of `has_identity_record` (a transaction can have an identity record but still be missing a specific sub-field) | Deliberately limited to these four fields. Per-column missingness indicators for the 38 masked `id_01`–`id_38` fields were **not** built — see `docs/milestone2_feature_design.md` for why |

### Email

| Feature | Source | Grain | Point-in-time safety | Business interpretation | Known limitation |
|---|---|---|---|---|---|
| `email_domain_match` | `P_emaildomain`, `R_emaildomain` | row | Safe | 4-state category: `MATCH`, `MISMATCH`, `ONE_MISSING`, `BOTH_MISSING` — whether purchaser/recipient email domains agree | Kept as 4 explicit states rather than a single 0/1 flag specifically so "both missing" is never silently conflated with "match" |

### Frequency encodings (causal / expanding-window — training history only)

All four below are computed by sorting `train_transaction` by
(`TransactionDT`, `TransactionID`) ascending and taking, for each row, the
count of **strictly earlier** rows sharing the same key value
(`groupby(key).cumcount()` on the time-sorted frame). First occurrence of
any key value gets `0`, not `NaN` — "zero prior sightings" is exact and
well-defined. See `src/features/velocity_features.py` for the
implementation and `tests/features/test_velocity_features.py` for the
adversarial leakage tests (mutating a later row must not change an
earlier row's value — verified).

| Feature | Key | Grain | Point-in-time safety | Business interpretation | Known limitation |
|---|---|---|---|---|---|
| `card1_prior_count` | `card1` | row | Safe by construction (expanding window, verified by test) | How many prior transactions (within train, up to this point in time) used this same `card1` value — a classic account-velocity signal | Early in the dataset's time range, every key's count is near 0 purely because little history has accumulated yet — this is a real cold-start property of expanding features, not a bug |
| `addr1_prior_count` | `addr1` | row | Safe (same construction) | Prior-transaction count for this address value | Same cold-start property; ~11% of rows have null `addr1`, handled as a shared "missing" group, not dropped |
| `P_emaildomain_prior_count` | `P_emaildomain` | row | Safe (same construction) | Prior-transaction count for this purchaser email domain | Common domains (e.g. `gmail.com`) will have large counts dominated by popularity, not fraud-specific behavior — this is a coarse signal, not a purpose-built fraud score |
| `DeviceInfo_prior_count` | `DeviceInfo` | row | Safe (same construction) | Prior-transaction count for this device descriptor string | Only populated meaningfully when `has_identity_record == 1`; null `DeviceInfo` values (no identity row, or identity present but device missing) share one "missing" group |

`ProductCD` was considered and deliberately **excluded** as a frequency
key: with only 5 distinct values, its expanding count mostly tracks
elapsed dataset size/time, not a meaningful behavioral signal.

### Recency / velocity

| Feature | Key | Grain | Point-in-time safety | Business interpretation | Known limitation |
|---|---|---|---|---|---|
| `seconds_since_last_card1` | `card1` | row | Safe (expanding `.diff()` after time-sort, verified by adversarial test) | Seconds since the last prior transaction with the same `card1` value — a velocity/burstiness signal (rapid reuse of the same card is a classic fraud pattern) | `NaN` for a key's first occurrence — this means "no prior signal available," not "zero time elapsed." Must not be imputed as 0 at modeling time without documenting that choice explicitly |
| `seconds_since_last_addr1` | `addr1` | row | Safe (same construction) | Seconds since the last prior transaction from the same address value | Same first-occurrence `NaN` caveat; also inherits `addr1`'s ~11% missingness (shared "missing" group) |

Recency keys were deliberately limited to `card1` and `addr1` (not one per
frequency key) to keep the baseline transparent rather than exhaustive —
see `docs/milestone2_feature_design.md`.

## Excluded from this baseline (and why)

| Group | Why excluded |
|---|---|
| `C1`–`C14`, `D1`–`D15`, `V1`–`V339` | Explicitly masked; Milestone 1's leakage review left their computation window (point-in-time vs. global aggregation) **unresolved**. Using them now would risk building on an unverified leakage assumption. |
| `M1`–`M9` | Not requested in the Milestone 2 feature list. Category is documented ("match" flags) but exact per-column match logic is undisclosed — same caveat pattern as the masked groups. Deferred rather than added unprompted. |
| Raw `id_01`–`id_38` values | Not requested as raw features in the Milestone 2 feature list (only `has_identity_record` and "missingness indicators for important identity/device fields" were requested). Their business meaning is masked; including 38 raw masked columns would reintroduce the same undocumented-meaning problem the baseline is trying to avoid. |
