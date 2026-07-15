#!/usr/bin/env python3
"""
Evaluate proposal-aligned result gaps for dissertation Chapter 6.

Outputs:
- proposal_gap_summary.json
- calibration_bins_test.csv
- stratified_narrative_metrics.csv
- llm_audit_summary.csv
- drift_proxy_by_time_bin.csv

Assumes current artefact layout:
artifacts/baselines/lgbm_numeric_v1_subsample/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


BASE = Path("artifacts/baselines/lgbm_numeric_v1_subsample")

PRED_PATH = BASE / "test_predictions.csv"
EO_PATH = BASE / "eos_test_with_drivers.jsonl"
TEMPLATE_NARR_PATH = BASE / "narratives_ops_triage_template.jsonl"
LLM_NARR_PATHS = [
    BASE / "narratives_ops_triage_llm_5753rows_backup.jsonl",
    BASE / "narratives_ops_triage_llm.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_get(row: dict[str, Any], *keys: str, default=None):
    cur: Any = row
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def normalise_event_id(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    rows = []
    ece = 0.0
    for b, g in df.groupby("bin", observed=False):
        if len(g) == 0:
            continue
        conf = float(g["p"].mean())
        acc = float(g["y"].mean())
        weight = len(g) / len(df)
        gap = abs(acc - conf)
        ece += weight * gap
        rows.append({
            "bin": str(b),
            "n": int(len(g)),
            "mean_score": conf,
            "observed_fraud_rate": acc,
            "abs_gap": gap,
            "weight": weight,
            "weighted_abs_gap": weight * gap,
        })
    return float(ece), pd.DataFrame(rows)


def extract_prediction_columns(preds: pd.DataFrame):
    """
    Identify target and probability columns safely.

    Important:
    - Never infer TransactionID as a score.
    - Prefer probability-like column names.
    - Only fallback to numeric columns bounded in [0, 1].
    """

    y_candidates = [
        "y_true",
        "isFraud",
        "target",
        "label",
        "actual",
    ]

    p_candidates = [
        "y_pred",
        "y_score",
        "y_prob",
        "probability",
        "pred_proba",
        "prediction",
        "pred",
        "score",
        "fraud_score",
        "prob_fraud",
        "isFraud_pred",
        "isFraud_proba",
        "p_fraud",
    ]

    blocked_cols = {
        "TransactionID",
        "transactionid",
        "event_id",
        "EventID",
        "id",
        "ID",
    }

    y_col = next((c for c in y_candidates if c in preds.columns), None)
    if y_col is None:
        raise ValueError(
            f"Could not find target column. Columns: {list(preds.columns)}"
        )

    p_col = next(
        (
            c for c in p_candidates
            if c in preds.columns
            and c != y_col
            and c not in blocked_cols
            and pd.api.types.is_numeric_dtype(preds[c])
        ),
        None,
    )

    if p_col is None:
        bounded_numeric = []
        for c in preds.columns:
            if c == y_col or c in blocked_cols:
                continue
            if not pd.api.types.is_numeric_dtype(preds[c]):
                continue

            col = preds[c].dropna()
            if col.empty:
                continue

            if col.min() >= 0 and col.max() <= 1 and col.nunique() > 2:
                bounded_numeric.append(c)

        if len(bounded_numeric) == 1:
            p_col = bounded_numeric[0]
        elif len(bounded_numeric) > 1:
            raise ValueError(
                "Multiple bounded numeric candidate score columns found. "
                f"Please choose one explicitly: {bounded_numeric}. "
                f"All columns: {list(preds.columns)}"
            )
        else:
            raise ValueError(
                "Could not safely infer probability column. "
                "Expected a numeric probability column bounded in [0, 1]. "
                f"Columns: {list(preds.columns)}"
            )

    return y_col, p_col


def eo_dataframe() -> pd.DataFrame:
    eos = read_jsonl(EO_PATH)
    rows = []
    for eo in eos:
        top_drivers = eo.get("top_drivers", []) or eo.get("drivers", []) or []
        rows.append({
            "event_id": normalise_event_id(eo.get("event_id")),
            "score": eo.get("score"),
            "risk_band": eo.get("risk_band") or eo.get("calibration_band"),
            "recommended_action": eo.get("recommended_action_class") or eo.get("recommended_action"),
            "thin_file_flag": eo.get("thin_file_flag"),
            "evidence_strength": eo.get("evidence_strength"),
            "drift_status": safe_get(eo, "monitoring", "drift_status", default=eo.get("drift_status")),
            "timestamp": eo.get("timestamp") or eo.get("TransactionDT"),
            "top_driver_names": [d.get("name") for d in top_drivers if isinstance(d, dict)],
            "top_driver_directions": {d.get("name"): d.get("direction") for d in top_drivers if isinstance(d, dict)},
            "top_driver_count": len(top_drivers),
        })
    return pd.DataFrame(rows)


def narrative_dataframe(path: Path, condition: str) -> pd.DataFrame:
    rows = read_jsonl(path)
    out = []
    for r in rows:
        text = (
            r.get("narrative")
            or r.get("text")
            or r.get("summary")
            or safe_get(r, "output", "narrative")
            or safe_get(r, "output", "summary")
            or ""
        )
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)

        out.append({
            "event_id": normalise_event_id(r.get("event_id")),
            "condition": condition,
            "raw_text": str(text),
            "validator_status": r.get("validator_status"),
            "fallback_used": bool(r.get("fallback_used", False)),
            "attempt_count": r.get("attempt_count"),
            "rejection_reason": r.get("rejection_reason") or r.get("fallback_reason"),
        })
    return pd.DataFrame(out)


def compute_narrative_metrics(narr: pd.DataFrame, eos: pd.DataFrame) -> pd.DataFrame:
    df = narr.merge(eos, on="event_id", how="inner")
    metric_rows = []

    for _, row in df.iterrows():
        text = str(row["raw_text"])
        drivers = row["top_driver_names"] if isinstance(row["top_driver_names"], list) else []
        mentioned = [d for d in drivers if d and str(d) in text]
        expected_disclosure = bool(row["thin_file_flag"]) or str(row["evidence_strength"]).upper() == "LOW"
        disclosure_present = any(
            phrase in text.lower()
            for phrase in ["thin file", "thin-file", "limited evidence", "low evidence", "sparse"]
        )
        risk_band = str(row["risk_band"])
        action = str(row["recommended_action"])

        metric_rows.append({
            "event_id": row["event_id"],
            "condition": row["condition"],
            "risk_band": risk_band,
            "recommended_action": action,
            "thin_file_flag": row["thin_file_flag"],
            "evidence_strength": row["evidence_strength"],
            "drift_status": row["drift_status"],
            "top_driver_count": len(drivers),
            "drivers_mentioned_count": len(mentioned),
            "topk_overlap": len(mentioned) / len(drivers) if drivers else np.nan,
            "at_least_one_driver": len(mentioned) > 0 if drivers else False,
            "all_drivers_mentioned": len(mentioned) == len(drivers) if drivers else False,
            "risk_band_mentioned": risk_band.lower() in text.lower(),
            "action_mentioned": action.lower() in text.lower(),
            "disclosure_expected": expected_disclosure,
            "disclosure_present": disclosure_present if expected_disclosure else np.nan,
            "validator_status": row.get("validator_status"),
            "fallback_used": row.get("fallback_used"),
            "attempt_count": row.get("attempt_count"),
            "rejection_reason": row.get("rejection_reason"),
        })

    return pd.DataFrame(metric_rows)


def summarise_stratified(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition", "risk_band", "recommended_action", "thin_file_flag", "evidence_strength"]
    rows = []

    for keys, g in metrics.groupby(group_cols, dropna=False):
        d = dict(zip(group_cols, keys))
        rows.append({
            **d,
            "n": int(len(g)),
            "at_least_one_driver_rate": float(g["at_least_one_driver"].mean()),
            "all_drivers_mentioned_rate": float(g["all_drivers_mentioned"].mean()),
            "avg_topk_overlap": float(g["topk_overlap"].mean()),
            "risk_band_mentioned_rate": float(g["risk_band_mentioned"].mean()),
            "action_mentioned_rate": float(g["action_mentioned"].mean()),
            "disclosure_expected_n": int(g["disclosure_expected"].sum()),
            "disclosure_present_when_expected_rate": (
                float(g.loc[g["disclosure_expected"], "disclosure_present"].mean())
                if g["disclosure_expected"].any()
                else np.nan
            ),
            "fallback_rate": float(g["fallback_used"].fillna(False).mean()),
        })

    return pd.DataFrame(rows).sort_values(["condition", "risk_band", "recommended_action"])


def audit_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in metrics.groupby("condition"):
        rows.append({
            "condition": cond,
            "rows_evaluated": int(len(g)),
            "validator_status_available_n": int(g["validator_status"].notna().sum()),
            "accepted_n": int((g["validator_status"] == "accepted").sum()),
            "fallback_n": int(g["fallback_used"].fillna(False).sum()),
            "fallback_rate": float(g["fallback_used"].fillna(False).mean()),
            "attempt_count_available_n": int(g["attempt_count"].notna().sum()),
            "mean_attempt_count": float(pd.to_numeric(g["attempt_count"], errors="coerce").mean())
                if g["attempt_count"].notna().any() else np.nan,
            "rejection_reason_available_n": int(g["rejection_reason"].notna().sum()),
        })
    return pd.DataFrame(rows)


def drift_proxy(eos: pd.DataFrame) -> pd.DataFrame:
    df = eos.copy()
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return pd.DataFrame([{
            "note": "No timestamp / TransactionDT available in EO file; drift proxy not generated."
        }])

    df["timestamp_numeric"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp_numeric", "score"])
    if df.empty:
        return pd.DataFrame([{
            "note": "Timestamp or score unavailable after coercion; drift proxy not generated."
        }])

    df["time_bin"] = pd.qcut(df["timestamp_numeric"], q=5, duplicates="drop")
    rows = []
    overall_mean = df["score"].mean()
    overall_std = df["score"].std(ddof=0) or 1.0

    for tb, g in df.groupby("time_bin", observed=False):
        rows.append({
            "time_bin": str(tb),
            "n": int(len(g)),
            "mean_score": float(g["score"].mean()),
            "score_std": float(g["score"].std(ddof=0)),
            "mean_score_shift_z": float((g["score"].mean() - overall_mean) / overall_std),
            "high_risk_rate": float((g["risk_band"].astype(str).str.upper() == "HIGH").mean()),
            "thin_file_rate": float(g["thin_file_flag"].fillna(False).astype(bool).mean()),
            "low_evidence_rate": float((g["evidence_strength"].astype(str).str.upper() == "LOW").mean()),
        })

    return pd.DataFrame(rows)


def main():
    out_dir = BASE
    preds = pd.read_csv(PRED_PATH)
    y_col, p_col = extract_prediction_columns(preds)

    y = preds[y_col].astype(int)
    p = preds[p_col].astype(float).clip(0, 1)

    ece, bins = expected_calibration_error(y, p, n_bins=10)
    brier = brier_score_loss(y, p)
    roc_auc = roc_auc_score(y, p)
    pr_auc = average_precision_score(y, p)

    bins.to_csv(out_dir / "calibration_bins_test.csv", index=False)

    eos = eo_dataframe()

    narr_frames = []
    if TEMPLATE_NARR_PATH.exists():
        narr_frames.append(narrative_dataframe(TEMPLATE_NARR_PATH, "template"))

    llm_path = next((p for p in LLM_NARR_PATHS if p.exists()), None)
    if llm_path:
        narr_frames.append(narrative_dataframe(llm_path, "llm"))

    if narr_frames:
        narr = pd.concat(narr_frames, ignore_index=True)
        metrics = compute_narrative_metrics(narr, eos)
        strat = summarise_stratified(metrics)
        audit = audit_summary(metrics)

        metrics.to_csv(out_dir / "proposal_gap_narrative_row_metrics.csv", index=False)
        strat.to_csv(out_dir / "stratified_narrative_metrics.csv", index=False)
        audit.to_csv(out_dir / "llm_audit_summary.csv", index=False)
    else:
        metrics = pd.DataFrame()
        strat = pd.DataFrame()
        audit = pd.DataFrame()

    drift = drift_proxy(eos)
    drift.to_csv(out_dir / "drift_proxy_by_time_bin.csv", index=False)

    summary = {
        "calibration": {
            "target_column": y_col,
            "score_column": p_col,
            "rows": int(len(preds)),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
            "ece_10_bin": float(ece),
        },
        "narrative_metrics": {
            "row_metrics_path": str(out_dir / "proposal_gap_narrative_row_metrics.csv"),
            "stratified_metrics_path": str(out_dir / "stratified_narrative_metrics.csv"),
            "audit_summary_path": str(out_dir / "llm_audit_summary.csv"),
        },
        "drift_proxy": {
            "path": str(out_dir / "drift_proxy_by_time_bin.csv"),
        },
    }

    with (out_dir / "proposal_gap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
