import os
import pytest

from fraud_thesis.data_paths import get_ieee_cis_paths

def test_get_ieee_cis_paths_smoke():
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("CI does not have IEEE_CIS_DIR / dataset")
    # If you haven't set IEEE_CIS_DIR locally, also skip.
    if not os.getenv("IEEE_CIS_DIR"):
        pytest.skip("IEEE_CIS_DIR not set locally")
    paths = get_ieee_cis_paths()
    assert paths.train_transaction.exists()
