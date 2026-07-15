import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
CACHE = Path("artifacts/data_cache/joined_train.parquet")
SPLIT_DIR = Path("artifacts/splits/v1_temporal_q70_q85")

N_TRAIN = 50000
N_VAL = 20000
N_TEST = 20000
SEED = 42


def read_ids(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=["TransactionID"])
    return df["TransactionID"].astype(np.int64).to_numpy()


def subsample_mask(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= n:
        return mask

    choose = rng.choice(idx, size=n, replace=False)
    out = np.zeros_like(mask, dtype=bool)
    out[choose] = True
    return out


def make_xy(df: pd.DataFrame, feat_cols: list[str]):
    X = df[feat_cols].copy()

    for c in X.columns:
        if pd.api.types.is_float_dtype(X[c].dtype):
            X[c] = X[c].astype(np.float32)

    y = df["isFraud"].astype(np.int8).to_numpy()
    return X, y


def main():
    if not CACHE.exists():
        raise FileNotFoundError(CACHE)

    feat_cols = json.loads((ART / "feature_names.json").read_text(encoding="utf-8"))

    required_cols = ["TransactionID", "isFraud"] + feat_cols
    required_cols = list(dict.fromkeys(required_cols))

    print("[INFO] Loading only required columns from joined_train.parquet", flush=True)
    print("[INFO] Column count:", len(required_cols), flush=True)

    df = pd.read_parquet(CACHE, columns=required_cols)

    train_ids = read_ids(SPLIT_DIR / "train_transaction_ids.csv")
    val_ids = read_ids(SPLIT_DIR / "val_transaction_ids.csv")
    test_ids = read_ids(SPLIT_DIR / "test_transaction_ids.csv")

    txid = df["TransactionID"].astype(np.int64).to_numpy()

    m_train = np.isin(txid, train_ids)
    m_val = np.isin(txid, val_ids)
    m_test = np.isin(txid, test_ids)

    rng = np.random.default_rng(SEED)

    # Preserve original training-script RNG call order.
    m_train_s = subsample_mask(m_train, N_TRAIN, rng)
    m_val_s = subsample_mask(m_val, N_VAL, rng)
    m_test_s = subsample_mask(m_test, N_TEST, rng)

    df_train = df.loc[m_train_s].copy()
    df_test = df.loc[m_test_s].copy()

    X_train, y_train = make_xy(df_train, feat_cols)
    X_test, y_test = make_xy(df_test, feat_cols)

    # Train-only median imputation, matching original baseline.
    med = X_train.median(numeric_only=True)
    X_test = X_test.fillna(med)

    ART.mkdir(parents=True, exist_ok=True)

    x_path = ART / "X_test.parquet"
    y_path = ART / "y_test_model_ready.csv"
    med_path = ART / "feature_medians_train_only.csv"
    manifest_path = ART / "model_ready_x_test_manifest.json"

    X_test.to_parquet(x_path, index=False)

    pd.DataFrame(
        {
            "TransactionID": df_test["TransactionID"].astype(int).to_numpy(),
            "y_true": y_test,
        }
    ).to_csv(y_path, index=False)

    med.to_frame("median").reset_index(names="feature").to_csv(med_path, index=False)

    manifest = {
        "status": "created",
        "source": str(CACHE),
        "seed": SEED,
        "n_rows": {
            "train": int(m_train_s.sum()),
            "val": int(m_val_s.sum()),
            "test": int(m_test_s.sum()),
        },
        "n_features": len(feat_cols),
        "outputs": {
            "X_test": str(x_path),
            "y_test": str(y_path),
            "feature_medians": str(med_path),
        },
    }

    pred_path = ART / "test_predictions.csv"
    if pred_path.exists():
        preds = pd.read_csv(pred_path)

        current_ids = df_test["TransactionID"].astype(int).to_numpy()
        pred_ids = preds["TransactionID"].astype(int).to_numpy()

        same_len = len(current_ids) == len(pred_ids)
        exact_order = same_len and np.array_equal(current_ids, pred_ids)
        matched_positions = int(
            np.sum(
                current_ids[: min(len(current_ids), len(pred_ids))]
                == pred_ids[: min(len(current_ids), len(pred_ids))]
            )
        )

        manifest["prediction_id_match"] = {
            "same_length": bool(same_len),
            "exact_order_match": bool(exact_order),
            "matched_positions": matched_positions,
        }

        model_path = ART / "model.txt"
        if model_path.exists() and exact_order:
            booster = lgb.Booster(model_file=str(model_path))
            p = booster.predict(X_test)
            y_pred_ref = preds["y_pred"].to_numpy()

            max_abs_diff = float(np.max(np.abs(p - y_pred_ref)))
            mean_abs_diff = float(np.mean(np.abs(p - y_pred_ref)))

            manifest["prediction_reproduction"] = {
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "close_at_1e_10": bool(max_abs_diff < 1e-10),
                "close_at_1e_6": bool(max_abs_diff < 1e-6),
            }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("[OK] Materialised model-ready X_test", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
