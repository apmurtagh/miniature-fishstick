import os
import pytest

from fraud_thesis.data_cache import load_joined_train_parquet


def test_load_joined_train_parquet_smoke():
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("CI does not have local artifacts cache")
    # This will raise with an actionable message if missing.
    df = load_joined_train_parquet()
    assert df.shape[0] > 100_000
    assert "isFraud" in df.columns
