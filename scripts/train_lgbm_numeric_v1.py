from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_thesis.data_cache import load_joined_train_parquet


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_ids(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=["TransactionID"])
    return df["TransactionID"].astype(np.int64).to_numpy()


def pick_numeric_features(df: pd.DataFrame, *, max_features: int) -> list[str]:
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    for c in ("TransactionID", "isFraud"):
        if c in numeric_cols:
            numeric_cols.remove(c)

    # Always include TransactionDT if present
    if "TransactionDT" in numeric_cols:
        numeric_cols.remove("TransactionDT")
        return ["TransactionDT"] + numeric_cols[: max_features - 1]
    return numeric_cols[:max_features]


def make_xy(df: pd.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = df[feat_cols].copy()

    # Downcast floats to reduce RAM
    for c in X.columns:
        if pd.api.types.is_float_dtype(X[c].dtype):
            X[c] = X[c].astype(np.float32)

    y = df["isFraud"].astype(np.int8).to_numpy()
    return X, y


def subsample_mask(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= n:
        return mask
    choose = rng.choice(idx, size=n, replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out[choose] = True
    return out


def main() -> None:
    # Inputs
    split_dir = Path("artifacts") / "splits" / "v1_temporal_q70_q85"
    train_ids = read_ids(split_dir / "train_transaction_ids.csv")
    val_ids = read_ids(split_dir / "val_transaction_ids.csv")
    test_ids = read_ids(split_dir / "test_transaction_ids.csv")

    df = load_joined_train_parquet()

    for c in ("TransactionID", "isFraud"):
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    # Create split masks
    txid = df["TransactionID"].astype(np.int64).to_numpy()
    m_train = np.isin(txid, train_ids)
    m_val = np.isin(txid, val_ids)
    m_test = np.isin(txid, test_ids)

    rng = np.random.default_rng(42)

    # Memory-safe baseline sizes (you used these successfully)
    n_train = 50_000
    n_val = 20_000
    n_test = 20_000

    m_train_s = subsample_mask(m_train, n_train, rng)
    m_val_s = subsample_mask(m_val, n_val, rng)
    m_test_s = subsample_mask(m_test, n_test, rng)

    df_train = df.loc[m_train_s]
    df_val = df.loc[m_val_s]
    df_test = df.loc[m_test_s]

    # Features (numeric-only)
    feat_cols = pick_numeric_features(df, max_features=40)
    X_train, y_train = make_xy(df_train, feat_cols)
    X_val, y_val = make_xy(df_val, feat_cols)
    X_test, y_test = make_xy(df_test, feat_cols)

    # Median impute using TRAIN only
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_val = X_val.fillna(med)
    X_test = X_test.fillna(med)

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_bin": 15,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
        "n_jobs": 1,
    }

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    p_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    p_test = booster.predict(X_test, num_iteration=booster.best_iteration)

    metrics = {
        "created_utc": utc_now_iso(),
        "model": "lgbm_numeric_v1_subsample",
        "split_id": "v1_temporal_q70_q85",
        "best_iteration": int(booster.best_iteration or 0),
        "n_rows": {"train": int(X_train.shape[0]), "val": int(X_val.shape[0]), "test": int(X_test.shape[0])},
        "n_features": int(X_train.shape[1]),
        "features": feat_cols,
        "val": {"roc_auc": float(roc_auc_score(y_val, p_val)), "pr_auc": float(average_precision_score(y_val, p_val))},
        "test": {"roc_auc": float(roc_auc_score(y_test, p_test)), "pr_auc": float(average_precision_score(y_test, p_test))},
    }

    out_dir = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # NEW: persist model + feature order for reproducible SHAP
    model_path = out_dir / "model.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)

    (out_dir / "feature_names.json").write_text(json.dumps(feat_cols, indent=2) + "\n", encoding="utf-8")

    # NEW: small, deterministic SHAP background sample (imputed, correct column order)
    bg_n = min(1024, int(X_train.shape[0]))
    X_bg = X_train.sample(n=bg_n, random_state=42).copy()
    X_bg.to_parquet(out_dir / "shap_background.parquet", index=False)

    # Write predictions for downstream cohort analysis
    out_preds = out_dir / "test_predictions.csv"
    pd.DataFrame(
        {
            "TransactionID": df_test["TransactionID"].astype(int).to_numpy(),
            "y_true": y_test,
            "y_pred": p_test,
        }
    ).to_csv(out_preds, index=False)

    print(json.dumps(metrics, indent=2))
    print("Wrote:", out_dir / "metrics.json")
    print("Wrote:", model_path)
    print("Wrote:", out_dir / "feature_names.json")
    print("Wrote:", out_dir / "shap_background.parquet")
    print("Wrote:", out_preds)


if __name__ == "__main__":
    main()
