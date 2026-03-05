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

    tx = pd.read_csv(p.train_transaction, usecols=["TransactionID", "TransactionDT", "isFraud"])
    # For v1 temporal-only we don't need identity; keep it fast and simple.

    if tx["TransactionDT"].isna().any():
        raise RuntimeError("Unexpected NA in TransactionDT")

    # Choose time cutpoints by quantile.
    q_train_end = 0.70
    q_val_end = 0.85

    t_train_end = float(tx["TransactionDT"].quantile(q_train_end))
    t_val_end = float(tx["TransactionDT"].quantile(q_val_end))

    def assign_split(t: float) -> str:
        if t <= t_train_end:
            return "train"
        if t <= t_val_end:
            return "val"
        return "test"

    tx["split"] = tx["TransactionDT"].map(assign_split)

    out_dir = Path("artifacts") / "splits" / "v1_temporal_q70_q85"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        ids = tx.loc[tx["split"] == split, ["TransactionID"]]
        ids.to_csv(out_dir / f"{split}_transaction_ids.csv", index=False)

    # Basic split stats for sanity
    stats = {}
    for split in ("train", "val", "test"):
        part = tx[tx["split"] == split]
        stats[split] = {
            "rows": int(part.shape[0]),
            "fraud_rate": float(part["isFraud"].mean()),
            "TransactionDT_min": int(part["TransactionDT"].min()),
            "TransactionDT_max": int(part["TransactionDT"].max()),
        }

    manifest = {
        "split_id": "v1_temporal_q70_q85",
        "created_utc": utc_now_iso(),
        "source": {
            "train_transaction_csv": str(p.train_transaction),
        },
        "rule": {
            "type": "temporal_quantiles",
            "q_train_end": q_train_end,
            "q_val_end": q_val_end,
            "t_train_end": t_train_end,
            "t_val_end": t_val_end,
        },
        "stats": stats,
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Wrote split to:", out_dir)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
