from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    preds_path = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample" / "test_predictions.csv"
    if not preds_path.exists():
        raise FileNotFoundError(
            f"Missing predictions file: {preds_path}\n"
            "Create it by running:\n"
            "  python scripts/train_lgbm_numeric_v1.py\n"
        )

    preds = pd.read_csv(preds_path)
    for c in ("TransactionID", "y_true", "y_pred"):
        if c not in preds.columns:
            raise RuntimeError(f"Missing column {c} in {preds_path}")
    preds["TransactionID"] = preds["TransactionID"].astype(int)

    joined_path = Path("artifacts") / "data_cache" / "joined_train.parquet"
    if not joined_path.exists():
        raise FileNotFoundError(
            f"Missing joined parquet: {joined_path}\n"
            "Create it by running:\n"
            "  python scripts/materialize_joined_train.py\n"
        )

    cols = ["TransactionID", "id_01", "DeviceType"]
    df = pd.read_parquet(joined_path, columns=cols)
    df["TransactionID"] = df["TransactionID"].astype(int)

    merged = preds.merge(df, on="TransactionID", how="left")

    # Proxy (3): thick if either id_01 OR DeviceType present
    thick = merged["id_01"].notna() | merged["DeviceType"].notna()
    thin = ~thick

    def metrics(mask: pd.Series) -> dict[str, float]:
        y = merged.loc[mask, "y_true"].astype(int)
        p = merged.loc[mask, "y_pred"].astype(float)
        if y.nunique() < 2:
            return {"roc_auc": float("nan"), "pr_auc": float("nan")}
        return {
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
        }

    out = {
        "created_utc": utc_now_iso(),
        "preds_path": str(preds_path),
        "cohort_definition": "thick if (id_01 notna) OR (DeviceType notna); else thin",
        "counts": {"total": int(merged.shape[0]), "thin": int(thin.sum()), "thick": int(thick.sum())},
        "metrics": {"overall": metrics(pd.Series([True] * merged.shape[0])), "thin": metrics(thin), "thick": metrics(thick)},
    }

    out_path = preds_path.parent / "thin_thick_metrics.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(out, indent=2))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
