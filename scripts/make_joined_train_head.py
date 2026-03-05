from __future__ import annotations

from pathlib import Path

import pandas as pd

from fraud_thesis.data_paths import get_ieee_cis_paths


def main() -> None:
    p = get_ieee_cis_paths()

    tx = pd.read_csv(p.train_transaction)
    idn = pd.read_csv(p.train_identity)

    df = tx.merge(idn, on="TransactionID", how="left")
    
    
    print("fraud_rate:", float(df["isFraud"].mean()))

    if "TransactionDT" in df.columns:
        dt = df["TransactionDT"]
        print("TransactionDT_min:", int(dt.min()))
        print("TransactionDT_max:", int(dt.max()))

    id_coverage = float(df["id_01"].notna().mean()) if "id_01" in df.columns else float(df["DeviceType"].notna().mean())
    print("identity_coverage_proxy:", id_coverage)
    
    
    print("train_transaction:", tx.shape)
    print("train_identity:", idn.shape)
    print("joined:", df.shape)

    # sanity: label present
    if "isFraud" not in df.columns:
        raise RuntimeError("Expected isFraud in joined train data")

    # quick missingness glimpse (top 15)
    miss = df.isna().mean().sort_values(ascending=False).head(15)
    print("\nTop missingness:\n", miss)

    out_dir = Path("artifacts") / "data_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_joined_head.csv"
    df.head(200).to_csv(out_path, index=False)
    print("\nWrote:", out_path)


if __name__ == "__main__":
    main()
