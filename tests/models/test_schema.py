from src.models.schema import (
    ALL_MODEL_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    ID_COL,
    MASKED_CODE_NUMERIC_FEATURES,
    MASKED_COLUMN_PATTERN,
    NUMERIC_FEATURES,
    TARGET_COL,
    TIME_COL,
)


def test_no_masked_column_is_a_model_feature():
    masked = [c for c in ALL_MODEL_FEATURES if MASKED_COLUMN_PATTERN.match(c)]
    assert masked == []


def test_id_target_and_time_columns_are_never_model_features():
    assert ID_COL not in ALL_MODEL_FEATURES
    assert TARGET_COL not in ALL_MODEL_FEATURES
    assert TIME_COL not in ALL_MODEL_FEATURES


def test_feature_groups_are_disjoint():
    numeric, categorical, binary = set(NUMERIC_FEATURES), set(CATEGORICAL_FEATURES), set(BINARY_FEATURES)
    assert numeric & categorical == set()
    assert numeric & binary == set()
    assert categorical & binary == set()


def test_masked_code_numeric_features_are_a_subset_of_numeric_features():
    assert set(MASKED_CODE_NUMERIC_FEATURES) <= set(NUMERIC_FEATURES)


def test_excluded_columns_do_not_overlap_model_features():
    assert set(EXCLUDED_COLUMNS) & set(ALL_MODEL_FEATURES) == set()
