from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eo.evidence_object import (
    DriftStatus,
    EvidenceObject,
    action_from_band,
    band_from_score,
    evidence_strength_from_thin_flag,
)

DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_PREDS_PATH = DEFAULT_RUN_DIR / "test_predictions.csv"
DEFAULT_JOINED_PARQUET = Path("artifacts") / "data_cache" / "joined_train.parquet"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "eos_test.jsonl"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def read_predictions(path: Path) -> pd.DataFrame:
    """
    Read predictions with delimiter auto-detection.
    """
    return pd.read_csv(path, sep=None, engine="python")


def compute_thin_flag(df: pd.DataFrame) -> pd.Series:
    # Proxy (3): thick if (id_01 notna) OR (DeviceType notna); else thin
    thick = df["id_01"].notna() | df["DeviceType"].notna()
    return ~thick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-path", default=str(DEFAULT_PREDS_PATH))
    ap.add_argument("--joined-parquet", default=str(DEFAULT_JOINED_PARQUET))
    ap.add_argument("--out-jsonl", default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--model", default="lgbm_numeric_v1_subsample")
    ap.add_argument("--split-id", default="v1_temporal_q70_q85")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap (0 = no cap)")
    args = ap.parse_args()

    preds_path = Path(args.preds_path)
    joined_path = Path(args.joined_parquet)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preds = read_predictions(preds_path)
    required = {"TransactionID", "y_pred"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {preds_path}: {sorted(missing)}. "
            f"Found columns={list(preds.columns)}"
        )

    joined = pd.read_parquet(joined_path, columns=["TransactionID", "id_01", "DeviceType"])
    df = preds.merge(joined, on="TransactionID", how="left", validate="many_to_one")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    thin_flag = compute_thin_flag(df)

    sha = git_sha()
    created = utc_now()
    drift_status = DriftStatus.OK  # placeholder until drift detector is added

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            r = row._asdict()
            score = float(r["y_pred"])
            is_thin = bool(thin_flag.iloc[n_written])

            evidence_strength = evidence_strength_from_thin_flag(is_thin)
            band = band_from_score(score)
            action = action_from_band(
                band,
                evidence_strength=evidence_strength,
                drift_status=drift_status,
            )

            eo = EvidenceObject(
                created_utc=created,
                git_sha=sha,
                model=args.model,
                split_id=args.split_id,
                event_id=str(r["TransactionID"]),
                score=score,
                risk_band=band,
                recommended_action_class=action,
                top_drivers=[],  # placeholder; add SHAP top-K later
                thin_file_flag=is_thin,
                evidence_strength=evidence_strength,
                data_coverage_score=0.34 if is_thin else 0.85,  # placeholder heuristic
                monitoring={"drift_status": drift_status, "data_quality_flags": []},
                meta={
                    "preds_path": str(preds_path),
                    "has_y_true": "y_true" in df.columns,
                },
            )

            f.write(eo.model_dump_json())
            f.write("\n")
            n_written += 1

    print("Wrote:", out_path)
    print("Rows:", n_written)
    print("Git SHA:", sha or "(unknown)")


if __name__ == "__main__":
    main()
