import os

import pytest

from fraud_thesis.data_cache import load_joined_train_parquet


def test_load_joined_train_parquet_smoke():
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("CI does not have local artifacts cache")

    try:
        df = load_joined_train_parquet()
    except RuntimeError as e:
        msg = str(e)
        if "Cached joined train parquet not found" in msg:
            pytest.skip(
                "Cached joined train parquet not found. Create it by running:\n"
                "  python scripts/materialize_joined_train.py"
            )
        raise

    assert df.shape[0] > 100_000
    assert "isFraud" in df.columns