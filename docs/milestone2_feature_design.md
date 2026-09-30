# Milestone 2 — Feature Design & Decision Record

Module B (Fraud Detection), Milestone 2: Point-in-Time Feature Engineering &
Transparent Baseline Dataset. No model has been trained. No train/val/test
split has been created. This document explains the reasoning behind every
inclusion/exclusion decision, so it can be defended in an interview, not
just reproduced.

## 1. What problem this milestone solves

A fraud model is scored at the instant a transaction happens. At that
instant it can only see transactions that already occurred. If an offline
training feature is built by aggregating across the *entire* dataset
(including rows that happen later in time), the model is trained on
information it will never have in production. This is the single most
common cause of a fraud model that looks excellent offline and
underperforms in production. Milestone 2 exists to build a feature set
that cannot make that mistake, and to prove it with tests rather than
assert it by description.

## 2. The core technical pattern: causal (expanding-window) aggregation

The fix for the above is: sort by time, then compute aggregates using only
strictly earlier rows. In pandas this is:

- **Frequency**: `df.sort_values('TransactionDT').groupby(key).cumcount()`
  — gives each row the count of *prior* occurrences of its group, because
  `cumcount()` only ever counts rows that appear earlier in the (now
  time-ordered) DataFrame.
- **Recency**: `df.sort_values('TransactionDT').groupby(key)['TransactionDT'].diff()`
  after a `.shift(1)` — the gap to the previous row in the same group,
  which is always a backward-looking comparison once sorted.

This is implemented generically in `src/features/velocity_features.py`
(`add_expanding_frequency`, `add_time_since_last`) rather than inlined per
feature, because this exact pattern recurs for every velocity feature a
fraud team builds — it's worth getting right once, and testing it hard
once.

### Why this is tested adversarially, not just for arithmetic

`tests/features/test_velocity_features.py` includes a test that takes a
correctly-computed feature set, **mutates a later transaction's key value**
(or timestamp), recomputes, and asserts that every strictly-earlier
transaction's feature value is unchanged. This is a stronger guarantee
than checking the arithmetic on a fixed example: it directly tests the
causal-safety *property* the feature is supposed to have, the same way
you'd want a production feature store's backfill logic tested. A second
test asserts the result is invariant to the input row order (shuffling
rows before calling the function must not change any row's result),
which catches a different class of bug — accidentally depending on
input ordering rather than recomputing a true time-sort internally.

## 3. Grain and scope

- Grain: one row per `TransactionID`.
- Built from `train_transaction` LEFT JOIN `train_identity` only.
  `test_transaction`/`test_identity` are never loaded in this milestone.
  This isn't a stylistic choice — it makes "don't use Kaggle test as
  labeled evaluation data" structurally true rather than a rule someone
  has to remember to follow later.
- No train/validation split is created here. Milestone 1 already
  established (docs/time_structure.md) that any split must be time-based,
  not random — that split is deferred to the modeling milestone, where it
  will actually be used for something. Creating it now, unused, would be
  scope creep.

## 4. Schema normalization (must happen before any cross-table logic)

Milestone 1 found that `train_identity` uses `id_01`..`id_38` (underscore)
while `test_identity` uses `id-01`..`id-38` (hyphen) — same 38 columns,
different delimiter, undocumented by the competition host.
`src/features/schema.py::normalize_identity_columns` fixes this by rename.
It is applied to `train_identity` even though Milestone 2 only touches
train, because the function is written to be reusable at scoring time in a
later milestone without a silent bug (test-time `id_*` columns
all appearing null because the rename never happened). This is a case
where "build it correctly once for reuse" is cheap and the failure mode
it prevents (silently broken test-time features) is expensive and hard to
notice.

## 5. What was included, and the reasoning for each group

| Group | Included? | Reasoning |
|---|---|---|
| `TransactionAmt`, `log1p(TransactionAmt)` | Yes | Documented, row-level, trivially point-in-time safe |
| Time features from `TransactionDT` | Yes | Row-level, safe by construction; phase-unknown-but-consistent cyclical proxies are explicitly documented as such, not claimed as calendar truth |
| `has_identity_record` | Yes | Row-level flag; directly requested |
| Missingness indicators | Yes, but scoped to `DeviceType`, `DeviceInfo`, `P_emaildomain`, `R_emaildomain` only | These are the clearest documented, non-masked "identity/device" fields. Extending this to all 38 masked `id_*` columns was considered and rejected — see §6 |
| Email domain match | Yes, as a 4-state category | A plain 0/1 "match" flag would silently conflate "both missing" with "match" — both are real, distinct, recurring cases in this data (Milestone 1 found `R_emaildomain` is often null) |
| Card/account-like raw features | Yes (`card1`–`card6`, raw) | Documented category; no defensible transformation beyond using the value. `card1` additionally used as a frequency/recency key because of its high cardinality (13,553 distinct values — unsuitable for one-hot, well-suited to "how often have I seen this" framing) |
| Address/location-like raw features | Yes (`addr1`, `addr2`, `dist1`, `dist2`, raw) | Same reasoning as card features. `dist2` is kept despite 93.6% missingness for transparency — it is not silently dropped, but flagged as likely unusable |
| Device type/info | Yes (`DeviceType` raw categorical, `DeviceInfo` raw + frequency-encoded) | `DeviceType` is low-cardinality and documented. `DeviceInfo` is high-cardinality free text with no documented grammar — parsing it into OS/model substrings would be inventing structure the source doesn't provide, so it is used only via its raw value and its frequency count |
| Product/channel-type | Yes (`ProductCD`, raw) | Documented, low-cardinality (5 values) |
| Frequency encodings | Yes, for `card1`, `addr1`, `P_emaildomain`, `DeviceInfo` | Chosen because each has a plausible "familiarity/velocity" interpretation. `ProductCD` was considered and excluded — with only 5 values, its expanding count would mostly track elapsed dataset size, not fraud-relevant behavior |
| Recency/velocity | Yes, for `card1`, `addr1` only | Deliberately narrower than the frequency-key list, to keep the baseline transparent and auditable rather than exhaustive. Extending to every frequency key was considered and rejected as unnecessary for a *baseline* |

## 6. What was deliberately excluded, and why

- **`C1`–`C14`, `D1`–`D15`, `V1`–`V339`**: masked, and Milestone 1's
  leakage review left their computation window (point-in-time vs. global
  aggregation) **unresolved**. Using them now would mean building the
  "transparent baseline" on top of an unverified leakage assumption —
  directly contradicts the milestone's purpose. This is a hard exclusion
  from the Milestone 2 instructions, not just a judgment call.
- **`M1`–`M9`**: not requested anywhere in the Milestone 2 feature list.
  The category ("match" flags) is documented, but the exact per-column
  match logic is not — the same undocumented-meaning pattern as the
  masked groups, just not named explicitly in the exclusion list. Treated
  as "not yet in scope" rather than added because it seemed technically
  permissible. Consistent with "don't add features beyond what the task
  requires."
- **Raw `id_01`–`id_38` values**: the Milestone 2 instructions explicitly
  scoped identity-table usage to `has_identity_record` and "missingness
  indicators for important identity/device fields" — not raw values. Using
  the raw masked values directly would reintroduce exactly the
  high-cardinality-but-undocumented-meaning problem that motivates
  excluding `C*`/`D*`/`V*` in the first place, just via the identity table
  instead of the transaction table. Per-column missingness indicators for
  all 38 of these were also rejected for the same reason — missingness
  *pattern* across 38 masked columns is still 38 columns of masked-feature
  granularity smuggled in through a side door, not truly different from
  using the values themselves for feature-count purposes.

## 7. Assumptions carried into this milestone

- `has_identity_record` assumes identity-row presence/absence is knowable
  at scoring time. This inherits Milestone 1's finding (clean
  one-to-zero-or-one join) but is not independently re-verified here.
- Time features assume `TransactionDT`'s hidden reference point is
  constant across all rows in `train_transaction` (a single dataset-wide
  reference, not a per-row one) — this is what makes `hour_of_day` and
  `day_of_week_index` internally consistent despite the unknown phase. This
  is consistent with, but not separately re-proven beyond, Milestone 1's
  `time_structure.md`.
- Same-second `TransactionDT` ties are broken by `TransactionID` for a
  deterministic sort order. This is a resolution-limit workaround, not a
  claim that `TransactionID` reflects true sub-second ordering.

## 8. Runtime / memory

- Raw column selection is pushed into DuckDB (`SELECT` only the ~19
  needed columns from `train_transaction`'s 394 and `train_identity`'s 41)
  before materializing into pandas — avoids loading the full wide tables.
- Feature construction on the resulting ~590K-row, ~19-column pandas frame
  (schema normalization, row-level transforms, 4 expanding-frequency
  groupbys, 2 expanding-recency groupbys) completes in a few seconds.
- Output is written via DuckDB's native Parquet writer
  (`COPY ... TO ... (FORMAT PARQUET)`), avoiding a new `pyarrow`/
  `fastparquet` dependency — consistent with the project's minimal-
  dependency constraint. The same native support means it can be read
  back with `duckdb.sql("SELECT * FROM read_parquet(...)")` without those
  packages either.
- Result: 590,540 rows × 38 columns, ~35 MB on disk (vs. ~650 MB for the
  raw `train_transaction.csv` alone) — a reduction driven mostly by column
  selection (19 of 394 source columns) and Parquet's columnar compression,
  not by any lossy transformation.
