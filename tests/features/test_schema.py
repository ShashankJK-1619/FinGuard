import pandas as pd
import pytest

from src.features.schema import normalize_identity_columns


def test_hyphenated_id_columns_are_renamed_to_underscore():
    df = pd.DataFrame({"TransactionID": [1], "id-01": [5], "id-38": [7], "DeviceType": ["mobile"]})
    out = normalize_identity_columns(df)
    assert "id_01" in out.columns
    assert "id_38" in out.columns
    assert "id-01" not in out.columns
    assert "DeviceType" in out.columns


def test_already_underscored_columns_are_left_alone():
    df = pd.DataFrame({"TransactionID": [1], "id_01": [5]})
    out = normalize_identity_columns(df)
    assert list(out.columns) == ["TransactionID", "id_01"]


def test_collision_raises():
    df = pd.DataFrame({"TransactionID": [1], "id-01": [5], "id_01": [9]})
    with pytest.raises(ValueError):
        normalize_identity_columns(df)


def test_normalizing_train_and_test_style_frames_yields_same_columns():
    train_style = pd.DataFrame({"TransactionID": [1], "id_01": [5], "id_02": [6]})
    test_style = pd.DataFrame({"TransactionID": [1], "id-01": [5], "id-02": [6]})
    assert set(normalize_identity_columns(train_style).columns) == set(
        normalize_identity_columns(test_style).columns
    )
