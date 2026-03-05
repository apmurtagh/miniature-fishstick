from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fraud_thesis.data_paths import get_ieee_cis_paths


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    p = get_ieee_cis_paths()

    out_dir = Path("artifacts") / "data_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / "joined_train.parquet"
    out_schema = out_dir / "joined_train_schema_summary.json"

    print("Loading CSVs...")
    tx = pd.read_csv(p.train_transaction)
    idn = pd.read_csv(p.train_identity)

    print("Joining on TransactionID (left join)...")
    df = tx.merge(idn, on="TransactionID", how="left")

    if "isFraud" not in df.columns:
        raise RuntimeError("Expected isFraud in joined train data")

    print("Joined shape:", df.shape)

    print(f"Writing parquet -> {out_parquet}")
    df.to_parquet(out_parquet, index=False)  # uses pyarrow engine by default if installed

    # Lightweight schema summary (keep it fast: don't compute full missingness for every column)
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    missing_top = (
        df.isna().mean()
        .sort_values(ascending=False)
        .head(25)
        .to_dict()
    )

    summary = {
        "created_utc": utc_now_iso(),
        "source": {
            "train_transaction_csv": str(p.train_transaction),
            "train_identity_csv": str(p.train_identity),
        },
        "output": {
            "joined_train_parquet": str(out_parquet),
        },
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "fraud_rate": float(df["isFraud"].mean()),
        "transactiondt_min": int(df["TransactionDT"].min()) if "TransactionDT" in df.columns else None,
        "transactiondt_max": int(df["TransactionDT"].max()) if "TransactionDT" in df.columns else None,
        "dtypes": dtypes,
        "missingness_top25": missing_top,
    }

    out_schema.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote schema summary -> {out_schema}")


if __name__ == "__main__":
    main()
