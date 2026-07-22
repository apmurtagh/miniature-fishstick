from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_PRED_PATH = DEFAULT_RUN_DIR / "test_predictions.csv"
DEFAULT_OUT_JSON = DEFAULT_RUN_DIR / "calibration_summary.json"
DEFAULT_OUT_MD = DEFAULT_RUN_DIR / "calibration_summary.md"
DEFAULT_OUT_CSV = DEFAULT_RUN_DIR / "calibration_bins.csv"


LABEL_CANDIDATES = [
    "isFraud", "is_fraud", "y_true", "label", "target", "actual", "truth"
]

PROB_CANDIDATES = [
    "pred_prob", "prob", "probability", "score", "prediction", "pred", "y_score",
    "fraud_probability", "p_fraud", "model_score"
]


def infer_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    # fallback heuristics
    if kind == "label":
        binary_cols = []
        for c in df.columns:
            s = df[c].dropna()
            vals = set(s.unique()[:20])
            if vals and vals.issubset({0, 1, False, True}):
                binary_cols.append(c)
        if len(binary_cols) == 1:
            return binary_cols[0]

    if kind == "prob":
        numeric_cols = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                s = df[c].dropna()
                if len(s) and float(s.min()) >= 0.0 and float(s.max()) <= 1.0:
                    numeric_cols.append(c)
        # avoid obvious label-like cols
        numeric_cols = [c for c in numeric_cols if c.lower() not in [x.lower() for x in LABEL_CANDIDATES]]
        if len(numeric_cols) == 1:
            return numeric_cols[0]
        for c in numeric_cols:
            if any(tok in c.lower() for tok in ["score", "prob", "pred", "fraud"]):
                return c

    return None


def brier_score(y, p):
    return float(((p - y) ** 2).mean())


def calibration_bins(y, p, n_bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    # Fixed-width bins on [0,1]
    df["bin"] = pd.cut(
        df["p"],
        bins=[i / n_bins for i in range(n_bins + 1)],
        include_lowest=True,
        labels=False,
    )

    rows = []
    total = len(df)
    ece = 0.0

    for b in range(n_bins):
        part = df[df["bin"] == b]
        left = b / n_bins
        right = (b + 1) / n_bins
        if len(part) == 0:
            rows.append({
                "bin": b,
                "bin_left": left,
                "bin_right": right,
                "n": 0,
                "avg_pred": None,
                "observed_rate": None,
                "abs_gap": None,
                "weight": 0.0,
                "ece_component": 0.0,
            })
            continue

        avg_pred = float(part["p"].mean())
        observed = float(part["y"].mean())
        gap = abs(avg_pred - observed)
        weight = len(part) / total
        comp = weight * gap
        ece += comp

        rows.append({
            "bin": b,
            "bin_left": left,
            "bin_right": right,
            "n": int(len(part)),
            "avg_pred": avg_pred,
            "observed_rate": observed,
            "abs_gap": gap,
            "weight": weight,
            "ece_component": comp,
        })

    return rows, float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(DEFAULT_PRED_PATH))
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--n-bins", type=int, default=10)
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "summary_version": "calibration_summary_v1",
        "predictions": str(pred_path),
        "status": "not_run",
        "reason": None,
    }

    if not pred_path.exists():
        summary["status"] = "skipped"
        summary["reason"] = f"Predictions file not found: {pred_path}"
    else:
        df = pd.read_csv(pred_path)
        label_col = infer_column(df, LABEL_CANDIDATES, "label")
        prob_col = infer_column(df, PROB_CANDIDATES, "prob")

        summary["columns"] = list(df.columns)
        summary["label_col"] = label_col
        summary["prob_col"] = prob_col

        if label_col is None or prob_col is None:
            summary["status"] = "skipped"
            summary["reason"] = (
                "Could not infer label/probability columns. "
                "Expected a binary label column and a probability/score column in [0,1]."
            )
        else:
            work = df[[label_col, prob_col]].copy()
            work = work.dropna()
            y = work[label_col].astype(float)
            p = work[prob_col].astype(float).clip(0, 1)

            bins, ece = calibration_bins(y, p, n_bins=args.n_bins)
            brier = brier_score(y, p)

            pd.DataFrame(bins).to_csv(out_csv, index=False)

            summary.update({
                "status": "completed",
                "reason": "ok",
                "n_rows": int(len(work)),
                "n_bins": int(args.n_bins),
                "brier_score": brier,
                "ece": ece,
                "positive_rate": float(y.mean()),
                "mean_predicted_probability": float(p.mean()),
                "bins_csv": str(out_csv),
            })

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Calibration Summary\n\n")
        f.write(f"- Predictions file: `{pred_path}`\n")
        f.write(f"- Status: `{summary['status']}`\n")
        f.write(f"- Reason: `{summary.get('reason')}`\n\n")

        if summary["status"] == "completed":
            f.write("| Metric | Value |\n")
            f.write("|---|---:|\n")
            f.write(f"| Rows evaluated | {summary['n_rows']} |\n")
            f.write(f"| Brier score | {summary['brier_score']:.6f} |\n")
            f.write(f"| Expected Calibration Error | {summary['ece']:.6f} |\n")
            f.write(f"| Observed positive rate | {summary['positive_rate']:.6f} |\n")
            f.write(f"| Mean predicted probability | {summary['mean_predicted_probability']:.6f} |\n\n")
            f.write(
                "Interpretation: lower Brier and ECE values indicate better probability calibration. "
                "These metrics contextualise the fraud baseline as a decisioning substrate, but the main thesis contribution remains EO-grounded narrative faithfulness and disclosure evaluation.\n"
            )
        else:
            f.write(
                "Calibration metrics could not be computed from the available prediction artefact. "
                "This should be reported as a limitation/future extension unless label and probability columns are added to the predictions file.\n"
            )

    print(json.dumps(summary, indent=2))
    print("Wrote:", out_json)
    print("Wrote:", out_md)
    if out_csv.exists():
        print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
