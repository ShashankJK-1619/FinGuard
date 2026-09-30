import pandas as pd

from src.features.categorical_features import add_email_domain_match


def test_email_domain_match_four_states():
    df = pd.DataFrame(
        {
            "P_emaildomain": ["gmail.com", "gmail.com", "gmail.com", None],
            "R_emaildomain": ["gmail.com", "yahoo.com", None, None],
        }
    )
    out = add_email_domain_match(df)
    assert out["email_domain_match"].tolist() == ["MATCH", "MISMATCH", "ONE_MISSING", "BOTH_MISSING"]


def test_email_domain_match_one_missing_is_symmetric():
    df = pd.DataFrame(
        {
            "P_emaildomain": [None, "gmail.com"],
            "R_emaildomain": ["gmail.com", None],
        }
    )
    out = add_email_domain_match(df)
    assert out["email_domain_match"].tolist() == ["ONE_MISSING", "ONE_MISSING"]
