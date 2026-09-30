# Data Dictionary — IEEE-CIS Fraud Detection

Source: [IEEE-CIS Fraud Detection, Kaggle competition](https://www.kaggle.com/competitions/ieee-fraud-detection),
hosted by the IEEE Computational Intelligence Society with data provided by
Vesta Corporation. This is **public competition data**. It does not come
from, and is not represented as coming from, any company the author has
worked for.

This dictionary distinguishes two kinds of claims:
- **Documented** — stated in the competition's own data description.
- **Masked / unknown** — the competition explicitly withholds the business
  meaning, or the column is not documented at all. For these, this project
  makes **no claim** about what the column represents. "Unknown" is a final
  answer here, not a placeholder to be guessed at later.

Grain: one row per transaction (`TransactionID`). Identity data joins in via
`TransactionID` and is only present for a subset of transactions (see
[join_validation.md](join_validation.md)).

## Identifiers

| Column | Table | Notes |
|---|---|---|
| `TransactionID` | transaction, identity | Primary key of transaction table; join key to identity table. |

## Target

| Column | Table | Notes |
|---|---|---|
| `isFraud` | train_transaction only | Binary label. 1 = reported fraud (per competition's chargeback/report process), 0 = not reported as fraud. Not present in test_transaction (it is the competition's held-out prediction target). This project treats the label's provenance as given — it does not re-derive or audit how Vesta determined `isFraud`. |

## Transaction Features (documented)

| Column | Notes |
|---|---|
| `TransactionAmt` | Transaction payment amount, in USD (documented). |
| `ProductCD` | Product code — which product/service the transaction is for (documented category; specific code-to-product mapping not disclosed). |

## Time-Related Features

| Column | Notes |
|---|---|
| `TransactionDT` | Documented as a **timedelta from a reference datetime**, i.e. a relative offset in seconds, not a wall-clock timestamp. Verified structurally in [time_structure.md](time_structure.md). The reference point itself is not disclosed. |
| `D1`–`D15` | Documented category: "timedelta, such as days between previous transaction, etc." The general category (time deltas) is documented; the exact event each specific `Dn` measures is **not** documented — treated as unknown per-column. |

## Card / Account-Like Features

| Column | Notes |
|---|---|
| `card1`–`card6` | Documented category: payment card information (e.g., card type, category, issuing bank, country). Exact mapping per column not disclosed — treated as unknown per-column beyond "card-related." |

## Address / Location-Like Features

| Column | Notes |
|---|---|
| `addr1`, `addr2` | Documented category: address fields. Which address (billing/shipping) and at what granularity is not disclosed. |
| `dist1`, `dist2` | Documented category: distance (e.g., between addresses). Units and exact reference points not disclosed. |

## Email / Domain-Like Features

| Column | Notes |
|---|---|
| `P_emaildomain` | Purchaser email domain (documented). |
| `R_emaildomain` | Recipient email domain (documented). |

## Device / Identity Features (from identity table)

| Column | Notes |
|---|---|
| `DeviceType` | Documented: device type category (e.g., mobile/desktop). |
| `DeviceInfo` | Documented: free-text-like device descriptor. High cardinality expected — quantified in [data_profile.md](data_profile.md). |
| `id_01`–`id_38` | Documented only at a high level as identity/network/digital-fingerprint information collected by Vesta's fraud protection system (e.g., network, device characteristics). Individual `id_NN` business meaning is **not documented** — treated as unknown per-column. |

## Engineered / Anonymous Numerical Features

| Column | Notes |
|---|---|
| `C1`–`C14` | Documented category: "counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked" (per competition documentation). Category is documented; exact per-column meaning is explicitly masked by the host. |
| `M1`–`M9` | Documented category: "match, such as names on card and address, etc." Exact per-column match logic not disclosed. |
| `V1`–`V339` | Documented only as "Vesta engineered rich features, including ranking, counting, and other entity relations." No further per-column documentation exists publicly. |

## Unknown / Masked Features

Every column in the `C*`, `D*`, `M*`, `V*`, `id_*`, `card*`, `addr*`, `dist*`
groups above has a **documented category** but an **undocumented specific
meaning**. This project will refer to them by category (e.g., "a `V`
feature," "a counting feature") and will not assign specific business
interpretations to individual columns such as `V217` or `id_14`. Any future
milestone that infers behavior for a specific masked column (e.g., via
correlation with `isFraud`) must label that inference explicitly as an
empirical finding, not a documented fact.

## Known limitation of this dictionary

This dictionary was built from the competition's own public data description
page, cross-checked against a live web search on 2026-09-30, and from
structural inspection outputs in this repo. It was **not** built by loading
every individual column and inspecting values by hand — that level of
detail is deferred to feature engineering (Module B, later milestone) and
will be done per-feature, on demand, only when a feature is actually being
considered for use.
