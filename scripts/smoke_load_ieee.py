from __future__ import annotations

import pandas as pd

from fraud_thesis.data_paths import get_ieee_cis_paths


def main() -> None:
    p = get_ieee_cis_paths()
    df = pd.read_csv(p.train_transaction, nrows=5)
    print("OK:", p.root)
    print(df.shape)
    print(list(df.columns)[:10])


if __name__ == "__main__":
    main()
