import numpy as np
import pandas as pd

from src.features.amount_time_features import add_amount_features, add_time_features


def test_log1p_amount_matches_numpy_reference():
    df = pd.DataFrame({"TransactionAmt": [0.251, 1.0, 100.0, 31937.391]})
    out = add_amount_features(df)
    expected = np.log1p(df["TransactionAmt"])
    assert np.allclose(out["TransactionAmt_log1p"], expected)


def test_log1p_is_monotonic_in_amount():
    df = pd.DataFrame({"TransactionAmt": [1.0, 2.0, 3.0, 1000.0]})
    out = add_amount_features(df)
    assert out["TransactionAmt_log1p"].is_monotonic_increasing


def test_day_and_week_index_boundaries():
    # exactly 1 day, 1 second before 2 days, exactly 7 days
    df = pd.DataFrame({"TransactionDT": [86_400, 172_799, 7 * 86_400]})
    out = add_time_features(df)
    assert out["day_index"].tolist() == [1, 1, 7]
    assert out["week_index"].tolist() == [0, 0, 1]


def test_hour_of_day_is_bounded_and_cyclical():
    df = pd.DataFrame({"TransactionDT": [0, 3_600 * 23, 3_600 * 23 + 3_599, 86_400]})
    out = add_time_features(df)
    assert out["hour_of_day"].between(0, 23).all()
    # TransactionDT=0 and TransactionDT=86400 are exactly one full day apart
    # and must land on the same hour bucket.
    assert out.loc[0, "hour_of_day"] == out.loc[3, "hour_of_day"]


def test_hour_sin_cos_are_bounded():
    df = pd.DataFrame({"TransactionDT": np.arange(0, 86_400, 3_600)})
    out = add_time_features(df)
    assert out["hour_sin"].between(-1, 1).all()
    assert out["hour_cos"].between(-1, 1).all()


def test_day_of_week_index_is_bounded():
    df = pd.DataFrame({"TransactionDT": [0, 86_400 * 6, 86_400 * 7, 86_400 * 13]})
    out = add_time_features(df)
    assert out["day_of_week_index"].between(0, 6).all()
    # exactly 7 days later must land on the same weekly-cycle bucket
    assert out.loc[0, "day_of_week_index"] == out.loc[2, "day_of_week_index"]
