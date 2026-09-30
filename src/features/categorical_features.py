"""Milestone 2: row-level categorical features (email domain match) and the
explicit list of raw documented-category columns kept in the baseline.

Card/account-like, address/location-like, device, and product/channel
columns are kept as *raw* passthrough values rather than re-engineered,
because the competition documents their general category but not their
per-column business meaning (see docs/data_dictionary.md) — there is
nothing defensible to compute from them beyond using the value itself.
Listing them explicitly (rather than leaving raw-column selection implicit
in the orchestrator) keeps the baseline auditable: anyone can read this
file and see exactly which raw columns made it into the modeling table and
why they're grouped the way they are.
"""
from __future__ import annotations

import pandas as pd

# Documented category: payment card attributes; exact per-column mapping
# (e.g. which is "issuing bank") is not disclosed.
CARD_COLUMNS = ["card1", "card2", "card3", "card4", "card5", "card6"]

# Documented category: address/distance attributes; which address
# (billing/shipping), granularity, and distance units are not disclosed.
ADDRESS_COLUMNS = ["addr1", "addr2", "dist1", "dist2"]

# Documented: purchaser/recipient email domain.
EMAIL_COLUMNS = ["P_emaildomain", "R_emaildomain"]

# Documented: device type/descriptor, from the identity table.
DEVICE_COLUMNS = ["DeviceType", "DeviceInfo"]

# Documented: product/service code for the transaction.
PRODUCT_COLUMNS = ["ProductCD"]

RAW_CATEGORY_COLUMNS = (
    CARD_COLUMNS + ADDRESS_COLUMNS + EMAIL_COLUMNS + DEVICE_COLUMNS + PRODUCT_COLUMNS
)


def add_email_domain_match(df: pd.DataFrame) -> pd.DataFrame:
    """Adds email_domain_match, a 4-state categorical comparing P_ and
    R_emaildomain: MATCH, MISMATCH, ONE_MISSING, BOTH_MISSING.

    Point-in-time safe: both source columns are transaction-time attributes
    of the row itself. Kept as an explicit 4-state category rather than a
    single 0/1 flag so "both missing" is never silently conflated with
    "match" or "mismatch" — those are different situations for a fraud
    model (e.g. R_emaildomain is often null when purchaser == recipient by
    design, which is itself informative and distinct from a genuine
    mismatch signal).
    """
    out = df.copy()
    p = out["P_emaildomain"]
    r = out["R_emaildomain"]
    p_missing = p.isna()
    r_missing = r.isna()

    result = pd.Series("MISMATCH", index=out.index, dtype="object")
    result[p_missing & r_missing] = "BOTH_MISSING"
    result[p_missing & ~r_missing] = "ONE_MISSING"
    result[~p_missing & r_missing] = "ONE_MISSING"
    match_mask = ~p_missing & ~r_missing & (p == r)
    result[match_mask] = "MATCH"

    out["email_domain_match"] = result
    return out
