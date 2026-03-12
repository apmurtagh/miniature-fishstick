import os
from pathlib import Path

import pytest

from fraud_thesis.data_paths import get_ieee_cis_paths


def test_get_ieee_cis_paths_smoke():
    # CI does not have IEEE_CIS_DIR / dataset.
    if os.getenv("GITHUB_ACTIONS") == "true":
        pytest.skip("CI does not have IEEE_CIS_DIR / dataset")

    raw = os.getenv("IEEE_CIS_DIR")
    if not raw:
        pytest.skip("IEEE_CIS_DIR not set locally")

    root = Path(raw).expanduser()
    if not root.exists():
        pytest.skip(f"IEEE_CIS_DIR points to a missing path: {root}")

    paths = get_ieee_cis_paths()
    assert paths.root.exists()
    # Keep lightweight assertions only (avoid requiring full dataset files here)
    assert paths.train_identity_csv.exists()
    assert paths.train_transaction_csv.exists()