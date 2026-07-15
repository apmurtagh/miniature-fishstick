import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp, wasserstein_distance
except Exception:
    ks_2samp = None
    wasserstein_distance = None

from sklearn.metrics import roc_auc_score, average_precision_score


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
OUT = ART / "drift_metric_suite"
OUT.mkdir(parents=True, exist_ok=True)

EO_PATH = ART / "eos_test_with_drivers_with_transactiondt.jsonl"
PRED_PATH = ART / "test_predictions.csv"

OUT_JSON = OUT / "drift_metric_suite_results.json"
OUT_CSV = OUT / "drift_metric_suite_by_window.csv"
OUT_MD = OUT / "drift_metric_suite_summary.md"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ece_score(y_true, y_score, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    n = len(y_true)

    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)

        if mask.sum() == 0:
            continue

        total += (mask.sum() / n) * abs(y_true[mask].mean() - y_score[mask].mean())

    return float(total)


def get_event_id(eo):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in eo:
            return eo[k]
    return None


def get_action(eo):
    return str(
        eo.get("recommended_action_class")
        or eo.get("recommended_action")
        or eo.get("action")
        or ""
    )


def get_evidence_strength(eo):
    return str(eo.get("evidence_strength") or eo.get("evidence_band") or "")


def get_top_driver_1(eo):
    drivers = eo.get("top_drivers") or eo.get("drivers") or []
    if not drivers:
        return "NONE"
    d = drivers[0]
    if isinstance(d, dict):
        return str(d.get("name") or d.get("feature") or d.get("driver") or "UNKNOWN")
    return str(d)


def psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)

    if len(expected) == 0 or len(actual) == 0:
        return None

    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))

    if len(cuts) < 3:
        lo = min(expected.min(), actual.min())
        hi = max(expected.max(), actual.max())
        cuts = np.unique(np.linspace(lo, hi, bins + 1))

    if len(cuts) < 3:
        return None

    expected_counts, _ = np.histogram(expected, bins=cuts)
    actual_counts, _ = np.histogram(actual, bins=cuts)

    expected_pct = expected_counts / max(1, expected_counts.sum())
    actual_pct = actual_counts / max(1, actual_counts.sum())

    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def total_variation_distance(a_counts, b_counts):
    keys = sorted(set(a_counts) | set(b_counts))
    a_total = sum(a_counts.values()) or 1
    b_total = sum(b_counts.values()) or 1

    total = 0.0
    for k in keys:
        total += abs(a_counts.get(k, 0) / a_total - b_counts.get(k, 0) / b_total)

    return 0.5 * total


def js_divergence(a_counts, b_counts):
    keys = sorted(set(a_counts) | set(b_counts))
    a_total = sum(a_counts.values()) or 1
    b_total = sum(b_counts.values()) or 1

    p = np.array([a_counts.get(k, 0) / a_total for k in keys], dtype=float)
    q = np.array([b_counts.get(k, 0) / b_total for k in keys], dtype=float)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        if mask.sum() == 0:
            return 0.0
        return float(np.sum(x[mask] * np.log2(x[mask] / np.clip(y[mask], 1e-12, None))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_analysis_frame():
    if not EO_PATH.exists():
        raise FileNotFoundError(f"Missing EO file with TransactionDT: {EO_PATH}")

    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PRED_PATH}")

    eos = read_jsonl(EO_PATH)
    pred = pd.read_csv(PRED_PATH)

    n = min(len(eos), len(pred))
    eos = eos[:n]
    pred = pred.iloc[:n].copy()

    pred["event_id"] = [get_event_id(eo) for eo in eos]
    pred["transaction_dt"] = [
        eo.get("transaction_dt") or eo.get("temporal", {}).get("transaction_dt")
        for eo in eos
    ]
    pred["temporal_window"] = [
        eo.get("temporal_window") or eo.get("temporal", {}).get("window")
        for eo in eos
    ]
    pred["recommended_action"] = [get_action(eo) for eo in eos]
    pred["evidence_strength"] = [get_evidence_strength(eo) for eo in eos]
    pred["top_driver_1"] = [get_top_driver_1(eo) for eo in eos]

    if "y_true" not in pred.columns:
        for c in ["isFraud", "target", "label"]:
            if c in pred.columns:
                pred["y_true"] = pred[c]
                break

    if "y_pred" not in pred.columns:
        for c in ["score", "prediction", "pred", "probability"]:
            if c in pred.columns:
                pred["y_pred"] = pred[c]
                break

    if "y_true" not in pred.columns or "y_pred" not in pred.columns:
        raise RuntimeError(f"Could not identify y_true/y_pred columns. Columns={list(pred.columns)}")

    if pred["transaction_dt"].isna().all():
        raise RuntimeError("No TransactionDT available in enriched EO artefact.")

    if pred["temporal_window"].isna().any():
        q33, q66 = pred["transaction_dt"].quantile([0.3333, 0.6667])
        pred["temporal_window"] = np.where(
            pred["transaction_dt"] <= q33,
            "early",
            np.where(pred["transaction_dt"] <= q66, "middle", "late"),
        )

    return pred


def compute_window_metrics(pred):
    windows = ["early", "middle", "late"]
    rows = []

    for win in windows:
        g = pred[pred["temporal_window"] == win].copy()
        y = g["y_true"].astype(float).to_numpy()
        s = g["y_pred"].astype(float).to_numpy()

        row = {
            "window": win,
            "rows": int(len(g)),
            "fraud_rate": float(np.mean(y)),
            "mean_score": float(np.mean(s)),
            "score_std": float(np.std(s)),
            "brier": float(np.mean((s - y) ** 2)),
            "ece_10": ece_score(y, s),
            "allow_rate": float((g["recommended_action"] == "allow").mean()),
            "step_up_rate": float((g["recommended_action"] == "step-up").mean()),
            "review_rate": float((g["recommended_action"] == "review").mean()),
            "low_evidence_rate": float((g["evidence_strength"].str.upper() == "LOW").mean()),
        }

        if len(np.unique(y)) > 1:
            row["roc_auc"] = float(roc_auc_score(y, s))
            row["pr_auc"] = float(average_precision_score(y, s))
        else:
            row["roc_auc"] = None
            row["pr_auc"] = None

        rows.append(row)

    return rows


def compute_pairwise_metrics(pred):
    base = pred[pred["temporal_window"] == "early"].copy()
    pairwise = {}

    for win in ["middle", "late"]:
        g = pred[pred["temporal_window"] == win].copy()

        base_score = base["y_pred"].astype(float)
        g_score = g["y_pred"].astype(float)

        action_base = Counter(base["recommended_action"])
        action_g = Counter(g["recommended_action"])

        evidence_base = Counter(base["evidence_strength"])
        evidence_g = Counter(g["evidence_strength"])

        driver_base = Counter(base["top_driver_1"])
        driver_g = Counter(g["top_driver_1"])

        pairwise[f"early_vs_{win}"] = {
            "score_psi": psi(base_score, g_score, bins=10),
            "score_ks": None if ks_2samp is None else float(ks_2samp(base_score, g_score).statistic),
            "score_wasserstein": None if wasserstein_distance is None else float(wasserstein_distance(base_score, g_score)),
            "action_total_variation": total_variation_distance(action_base, action_g),
            "evidence_total_variation": total_variation_distance(evidence_base, evidence_g),
            "top_driver_js_divergence": js_divergence(driver_base, driver_g),
        }

    return pairwise


def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.4f}"


def write_outputs(rows, pairwise):
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    result = {
        "source_eo": str(EO_PATH),
        "source_predictions": str(PRED_PATH),
        "rows": rows,
        "pairwise_against_early": pairwise,
        "interpretation": {
            "scope": "Offline production-style drift metric suite over retained TransactionDT windows.",
            "caveat": "This is not live production monitoring and does not model feedback loops, intervention effects, or label maturation."
        }
    }

    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = []
    md.append("# Production-Style Drift Metric Suite\n\n")
    md.append("## Temporal-window performance and calibration\n\n")
    md.append("| Window | Rows | Fraud rate | Mean score | ROC-AUC | PR-AUC | Brier | ECE | Allow | Step-up | Review | LOW evidence |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for r in rows:
        md.append(
            f"| {r['window']} | {r['rows']} | {fmt(r['fraud_rate'])} | {fmt(r['mean_score'])} | "
            f"{fmt(r['roc_auc'])} | {fmt(r['pr_auc'])} | {fmt(r['brier'])} | {fmt(r['ece_10'])} | "
            f"{fmt(r['allow_rate'])} | {fmt(r['step_up_rate'])} | {fmt(r['review_rate'])} | {fmt(r['low_evidence_rate'])} |\n"
        )

    md.append("\n## Pairwise drift metrics versus early window\n\n")
    md.append("| Comparison | Score PSI | Score KS | Score Wasserstein | Action TVD | Evidence TVD | Top-driver JSD |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")

    for k, v in pairwise.items():
        md.append(
            f"| {k} | {fmt(v['score_psi'])} | {fmt(v['score_ks'])} | {fmt(v['score_wasserstein'])} | "
            f"{fmt(v['action_total_variation'])} | {fmt(v['evidence_total_variation'])} | {fmt(v['top_driver_js_divergence'])} |\n"
        )

    md.append(
        "\nInterpretation: This suite strengthens the earlier TransactionDT temporal-slice proxy by adding score-distribution, "
        "action-distribution, evidence-strength and top-driver distribution metrics. It remains an offline production-style "
        "diagnostic rather than live production monitoring.\n"
    )

    OUT_MD.write_text("".join(md), encoding="utf-8")
    return "".join(md)


def main():
    pred = load_analysis_frame()
    rows = compute_window_metrics(pred)
    pairwise = compute_pairwise_metrics(pred)
    summary = write_outputs(rows, pairwise)

    print("[OK] Drift metric suite complete")
    print(f"[OK] CSV: {OUT_CSV}")
    print(f"[OK] JSON: {OUT_JSON}")
    print(f"[OK] MD: {OUT_MD}")
    print("")
    print("== Drift metric suite summary ==")
    print(summary)


if __name__ == "__main__":
    main()
