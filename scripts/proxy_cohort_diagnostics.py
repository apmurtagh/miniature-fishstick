import json
from pathlib import Path

import numpy as np
import pandas as pd


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
CACHE = Path("artifacts/data_cache/joined_train.parquet")
PREDS = ART / "test_predictions.csv"

OUT = ART / "proxy_cohort_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

OUT_JSON = OUT / "proxy_cohort_diagnostics_summary.json"
OUT_MD = OUT / "proxy_cohort_diagnostics_summary.md"
OUT_CSV = OUT / "proxy_cohort_diagnostics_by_group.csv"
DOC_MD = Path("docs/thesis_final/proxy_cohort_diagnostics.md")

CANDIDATE_FIELDS = [
    "ProductCD",
    "card4",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
]

MIN_GROUP_N = 100
TOP_N_LEVELS = 10


def normalise_group_values(s: pd.Series, top_n: int = TOP_N_LEVELS) -> pd.Series:
    out = s.astype("object").where(~s.isna(), "MISSING").astype(str)
    vc = out.value_counts(dropna=False)
    keep = set(vc.head(top_n).index)
    return out.where(out.isin(keep), "OTHER")


def main():
    if not CACHE.exists():
        raise FileNotFoundError(CACHE)
    if not PREDS.exists():
        raise FileNotFoundError(PREDS)

    preds = pd.read_csv(PREDS)
    if not {"TransactionID", "y_true", "y_pred"}.issubset(preds.columns):
        raise ValueError("Prediction file must contain TransactionID, y_true and y_pred")

    # Read only available lightweight cohort columns.
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(CACHE)
    available_cols = set(pf.schema_arrow.names)

    read_cols = ["TransactionID", "isFraud"] + [c for c in CANDIDATE_FIELDS if c in available_cols]
    raw = pd.read_parquet(CACHE, columns=read_cols)

    df = preds.merge(raw, on="TransactionID", how="left", validate="one_to_one")

    # Prefer y_true from predictions; keep isFraud only as a cross-check.
    df["fraud_label"] = df["y_true"].astype(int)
    df["score"] = df["y_pred"].astype(float)

    high_score_threshold = float(df["score"].quantile(0.95))
    df["high_score_top5pct"] = df["score"] >= high_score_threshold
    df["model_positive_050"] = df["score"] >= 0.5

    group_rows = []
    field_summaries = []

    for field in [c for c in CANDIDATE_FIELDS if c in df.columns]:
        gcol = f"{field}_cohort"
        df[gcol] = normalise_group_values(df[field])

        grouped = (
            df.groupby(gcol, dropna=False)
            .agg(
                rows=("TransactionID", "size"),
                fraud_rate=("fraud_label", "mean"),
                mean_score=("score", "mean"),
                high_score_top5pct_rate=("high_score_top5pct", "mean"),
                model_positive_050_rate=("model_positive_050", "mean"),
            )
            .reset_index()
            .rename(columns={gcol: "cohort"})
        )

        grouped.insert(0, "field", field)
        grouped = grouped.sort_values(["rows", "cohort"], ascending=[False, True])

        eligible = grouped[grouped["rows"] >= MIN_GROUP_N].copy()

        if len(eligible) >= 2:
            summary = {
                "field": field,
                "n_groups_total": int(len(grouped)),
                "n_groups_eligible": int(len(eligible)),
                "min_group_n": MIN_GROUP_N,
                "score_mean_min": float(eligible["mean_score"].min()),
                "score_mean_max": float(eligible["mean_score"].max()),
                "score_mean_gap": float(eligible["mean_score"].max() - eligible["mean_score"].min()),
                "fraud_rate_min": float(eligible["fraud_rate"].min()),
                "fraud_rate_max": float(eligible["fraud_rate"].max()),
                "fraud_rate_gap": float(eligible["fraud_rate"].max() - eligible["fraud_rate"].min()),
                "high_score_top5pct_rate_min": float(eligible["high_score_top5pct_rate"].min()),
                "high_score_top5pct_rate_max": float(eligible["high_score_top5pct_rate"].max()),
                "high_score_top5pct_rate_gap": float(
                    eligible["high_score_top5pct_rate"].max()
                    - eligible["high_score_top5pct_rate"].min()
                ),
            }
        else:
            summary = {
                "field": field,
                "n_groups_total": int(len(grouped)),
                "n_groups_eligible": int(len(eligible)),
                "min_group_n": MIN_GROUP_N,
                "note": "Fewer than two eligible groups; no gap computed.",
            }

        field_summaries.append(summary)
        group_rows.append(grouped)

    all_groups = pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame()
    all_groups.to_csv(OUT_CSV, index=False)

    result = {
        "status": "run",
        "rows": int(len(df)),
        "candidate_fields": CANDIDATE_FIELDS,
        "fields_evaluated": [s["field"] for s in field_summaries],
        "min_group_n": MIN_GROUP_N,
        "top_n_levels": TOP_N_LEVELS,
        "high_score_threshold_top5pct": high_score_threshold,
        "field_summaries": field_summaries,
        "outputs": {
            "csv": str(OUT_CSV),
            "json": str(OUT_JSON),
            "md": str(OUT_MD),
            "docs_md": str(DOC_MD),
        },
        "caveat": (
            "This is an operational proxy-cohort diagnostic. The IEEE-CIS fields are not protected-class labels, "
            "so this is not a fairness audit and does not establish fairness performance."
        ),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("# Proxy Cohort Diagnostics Summary\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This diagnostic inspects model score and outcome differences across available operational proxy cohorts. "
        "It is included to address the proposal's proxy-disparity theme without claiming protected-class fairness evidence.\n\n"
    )

    lines.append("## Scope Boundary\n\n")
    lines.append(
        "IEEE-CIS proxy fields such as product, card, address, email-domain and device attributes are operational fields, not protected-class labels. "
        "The results below are therefore proxy cohort diagnostics, not a fairness audit and not fairness-performance evidence.\n\n"
    )

    lines.append("## Overall Settings\n\n")
    lines.append(f"- Rows evaluated: {len(df)}\n")
    lines.append(f"- Minimum group size for gap summaries: {MIN_GROUP_N}\n")
    lines.append(f"- Top score threshold, 95th percentile: {high_score_threshold:.6f}\n")
    lines.append(f"- Cohort fields evaluated: {', '.join(result['fields_evaluated'])}\n\n")

    lines.append("## Field-Level Gap Summary\n\n")
    lines.append("| Field | Eligible groups | Mean score gap | Fraud-rate gap | High-score top-5% rate gap |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for s in field_summaries:
        if "score_mean_gap" in s:
            lines.append(
                f"| {s['field']} | {s['n_groups_eligible']} | "
                f"{s['score_mean_gap']:.4f} | {s['fraud_rate_gap']:.4f} | "
                f"{s['high_score_top5pct_rate_gap']:.4f} |\n"
            )
        else:
            lines.append(f"| {s['field']} | {s['n_groups_eligible']} |  |  |  |\n")

    lines.append("\n## Interpretation\n\n")
    lines.append(
        "Large gaps across operational cohorts should be treated as prompts for further portfolio-specific investigation. "
        "They do not identify protected-class disparity and do not prove unfairness or fairness. "
        "A production fairness review would require legally and ethically appropriate cohort definitions, governance approval, and calibrated fairness metrics.\n\n"
    )

    lines.append("## Artefacts\n\n")
    lines.append(f"- Group-level CSV: `{OUT_CSV}`\n")
    lines.append(f"- Machine-readable summary: `{OUT_JSON}`\n")

    md = "".join(lines)
    OUT_MD.write_text(md, encoding="utf-8")
    DOC_MD.write_text(md, encoding="utf-8")

    print("[OK] Proxy cohort diagnostics complete")
    print(json.dumps(result, indent=2)[:8000])
    print("")
    print("== Summary preview ==")
    print(md)


if __name__ == "__main__":
    main()
