from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from eo.evidence_object import Driver, EvidenceObject


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_MODEL_PATH = DEFAULT_RUN_DIR / "model.txt"
DEFAULT_FEATURES_PATH = DEFAULT_RUN_DIR / "feature_names.json"
DEFAULT_BG_PATH = DEFAULT_RUN_DIR / "shap_background.parquet"
DEFAULT_JOINED_PARQUET = Path("artifacts") / "data_cache" / "joined_train.parquet"
DEFAULT_EOS_IN = DEFAULT_RUN_DIR / "eos_test.jsonl"
DEFAULT_EOS_OUT = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--feature-names", default=str(DEFAULT_FEATURES_PATH))
    ap.add_argument("--background-parquet", default=str(DEFAULT_BG_PATH))
    ap.add_argument("--joined-parquet", default=str(DEFAULT_JOINED_PARQUET))
    ap.add_argument("--eos-in", default=str(DEFAULT_EOS_IN))
    ap.add_argument("--eos-out", default=str(DEFAULT_EOS_OUT))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="Optional cap for debugging (0 = no cap)")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    feat_path = Path(args.feature_names)
    bg_path = Path(args.background_parquet)
    joined_path = Path(args.joined_parquet)
    eos_in = Path(args.eos_in)
    eos_out = Path(args.eos_out)

    feat_cols = json.loads(feat_path.read_text(encoding="utf-8"))
    if not isinstance(feat_cols, list) or not feat_cols:
        raise ValueError(f"feature_names.json must be a non-empty list; got {type(feat_cols)}")

    booster = lgb.Booster(model_file=str(model_path))

    # Load background for SHAP. Ensure column order matches.
    X_bg = pd.read_parquet(bg_path)
    X_bg = X_bg[feat_cols]

    explainer = shap.TreeExplainer(booster, data=X_bg, feature_perturbation="tree_path_dependent")

    # Collect EO event_ids (TransactionIDs)
    eos: list[EvidenceObject] = []
    for i, obj in enumerate(iter_jsonl(eos_in), start=1):
        eos.append(EvidenceObject.model_validate(obj))
        if args.limit and i >= args.limit:
            break

    txids = [int(e.event_id) for e in eos]

    # Load feature rows for these txids from joined_train.parquet
    df = pd.read_parquet(joined_path, columns=["TransactionID", "isFraud", *feat_cols])
    df = df[df["TransactionID"].isin(txids)].copy()

    # Preserve EO order
    df = df.set_index("TransactionID").loc[txids].reset_index()

    # Reapply the *same imputation strategy* as training: median impute.
    # We use medians computed from background sample (approx) to avoid loading full train.
    med = X_bg.median(numeric_only=True)
    X = df[feat_cols].copy()
    X = X.fillna(med)

    # SHAP values for binary objective: returns (n, features)
    shap_vals = explainer.shap_values(X)
    # shap can return list for multiclass; for binary it may return ndarray or list len=2
    if isinstance(shap_vals, list):
        # common pattern: [neg_class, pos_class]; use pos_class
        if len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        else:
            shap_vals = shap_vals[0]
    shap_vals = np.asarray(shap_vals)

    out_lines = 0
    with eos_out.open("w", encoding="utf-8") as f:
        for i, eo in enumerate(eos):
            vals = shap_vals[i]
            abs_vals = np.abs(vals)

            # top-k indices by |shap|
            k = min(int(args.top_k), abs_vals.size)
            top_idx = np.argpartition(-abs_vals, kth=k - 1)[:k]
            # stable sort by magnitude descending
            top_idx = top_idx[np.argsort(-abs_vals[top_idx])]

            drivers: list[Driver] = []
            for j in top_idx:
                name = feat_cols[int(j)]
                v = float(vals[int(j)])
                drivers.append(
                    Driver(
                        name=name,
                        direction="+" if v >= 0 else "-",
                        magnitude=float(abs(v)),
                        evidence_span={"feature": name, "value": str(X.iloc[i, int(j)])},
                    )
                )

            eo2 = eo.model_copy(update={"top_drivers": drivers})
            f.write(eo2.model_dump_json())
            f.write("\n")
            out_lines += 1

    print("Wrote:", eos_out)
    print("Rows:", out_lines)


if __name__ == "__main__":
    main()
    
