import numpy as np
import pandas as pd

from src.features.identity_features import add_has_identity_record, add_missingness_indicators


def test_has_identity_record_flags_matching_ids_only():
    df = pd.DataFrame({"TransactionID": [1, 2, 3]})
    identity_ids = pd.Series([1, 3])
    out = add_has_identity_record(df, identity_ids)
    assert out["has_identity_record"].tolist() == [1, 0, 1]


def test_missingness_indicator_matches_isna_exactly():
    df = pd.DataFrame(
        {
            "DeviceType": ["mobile", None, "desktop", np.nan],
            "P_emaildomain": ["a.com", "b.com", None, None],
        }
    )
    out = add_missingness_indicators(df, columns=["DeviceType", "P_emaildomain"])
    assert out["DeviceType_missing"].tolist() == [0, 1, 0, 1]
    assert out["P_emaildomain_missing"].tolist() == [0, 0, 1, 1]


def test_missingness_indicators_skip_absent_columns_without_error():
    df = pd.DataFrame({"DeviceType": ["mobile", None]})
    out = add_missingness_indicators(df, columns=["DeviceType", "DeviceInfo_not_present"])
    assert "DeviceType_missing" in out.columns
    assert "DeviceInfo_not_present_missing" not in out.columns
