from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha() -> str:
    """
    Best-effort git SHA for reproducibility.
    Returns empty string if git is unavailable or we're not in a git worktree.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def _git_is_dirty() -> str:
    """
    Returns 'true' if there are uncommitted changes, else 'false'.
    Empty string if git is unavailable.
    """
    try:
        subprocess.check_call(
            ["git", "diff", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["git", "diff", "--quiet", "--staged"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "false"
    except subprocess.CalledProcessError:
        return "true"
    except Exception:
        return ""


def _get(d: dict[str, Any], *path: str, default: Any = "") -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def main() -> None:
    run_dir = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
    metrics_path = run_dir / "metrics.json"
    thin_thick_path = run_dir / "thin_thick_metrics.json"

    metrics = _load_json(metrics_path)
    cohort = _load_json(thin_thick_path)

    # Prefer the explicit top5 file if you generated it; else fall back.
    drv_path_top5 = run_dir / "driver_overlap_metrics_top5.json"
    drv_path_default = run_dir / "driver_overlap_metrics.json"
    driver = _load_json_if_exists(drv_path_top5) or _load_json_if_exists(drv_path_default)

    # Prefer the explicit top5 sign file if you generated it; else fall back.
    sign_path_top5 = run_dir / "driver_sign_metrics_top5.json"
    sign_path_default = run_dir / "driver_sign_metrics.json"
    sign = _load_json_if_exists(sign_path_top5) or _load_json_if_exists(sign_path_default)

    # Prefer the explicit top5 leakage file if you generated it; else fall back.
    leak_path_top5 = run_dir / "driver_leakage_metrics_top5.json"
    leak_path_default = run_dir / "driver_leakage_metrics.json"
    leak = _load_json_if_exists(leak_path_top5) or _load_json_if_exists(leak_path_default)

    sha = _git_sha()
    dirty = _git_is_dirty()

    # Prepare a single-row CSV that’s easy to paste into thesis tables later
    row = {
        "created_utc": utc_now_iso(),
        "git_sha": sha,
        "git_dirty": dirty,
        "run_dir": str(run_dir),
        "model": str(metrics.get("model", "")),
        "split_id": str(metrics.get("split_id", "")),
        "best_iteration": str(metrics.get("best_iteration", "")),
        "n_features": str(metrics.get("n_features", "")),
        "n_train": str((metrics.get("n_rows", {}) or {}).get("train", "")),
        "n_val": str((metrics.get("n_rows", {}) or {}).get("val", "")),
        "n_test": str((metrics.get("n_rows", {}) or {}).get("test", "")),
        "test_roc_auc_overall": str(((cohort.get("metrics", {}) or {}).get("overall", {}) or {}).get("roc_auc", "")),
        "test_pr_auc_overall": str(((cohort.get("metrics", {}) or {}).get("overall", {}) or {}).get("pr_auc", "")),
        "test_roc_auc_thin": str(((cohort.get("metrics", {}) or {}).get("thin", {}) or {}).get("roc_auc", "")),
        "test_pr_auc_thin": str(((cohort.get("metrics", {}) or {}).get("thin", {}) or {}).get("pr_auc", "")),
        "test_roc_auc_thick": str(((cohort.get("metrics", {}) or {}).get("thick", {}) or {}).get("roc_auc", "")),
        "test_pr_auc_thick": str(((cohort.get("metrics", {}) or {}).get("thick", {}) or {}).get("pr_auc", "")),
        "n_thin": str((cohort.get("counts", {}) or {}).get("thin", "")),
        "n_thick": str((cohort.get("counts", {}) or {}).get("thick", "")),
        "cohort_definition": str(cohort.get("cohort_definition", "")),

        # Driver overlap metrics (optional; blank if file missing)
        "driver_overlap_k": str(driver.get("k", "")),
        "driver_overlap_mean_overall": str(_get(driver, "overall", "overlap_at_k", "mean")),
        "driver_overlap_p50_overall": str(_get(driver, "overall", "overlap_at_k", "p50")),
        "driver_overlap_mean_thin": str(_get(driver, "thin", "overlap_at_k", "mean")),
        "driver_overlap_p50_thin": str(_get(driver, "thin", "overlap_at_k", "p50")),
        "driver_overlap_mean_thick": str(_get(driver, "thick", "overlap_at_k", "mean")),
        "driver_overlap_p50_thick": str(_get(driver, "thick", "overlap_at_k", "p50")),
        "driver_mention_any_rate_overall": str(_get(driver, "overall", "mention_any_topk_rate", "rate")),
        "driver_mention_any_rate_thin": str(_get(driver, "thin", "mention_any_topk_rate", "rate")),
        "driver_mention_any_rate_thick": str(_get(driver, "thick", "mention_any_topk_rate", "rate")),
        "driver_metrics_path": str(drv_path_top5 if drv_path_top5.exists() else (drv_path_default if drv_path_default.exists() else "")),

        # Driver sign-faithfulness metrics (optional; blank if file missing)
        "driver_sign_k": str(sign.get("k", "")),
        "driver_sign_acc_mean_overall": str(_get(sign, "overall", "per_row_sign_accuracy", "mean")),
        "driver_sign_acc_p50_overall": str(_get(sign, "overall", "per_row_sign_accuracy", "p50")),
        "driver_sign_any_error_rate_overall": str(_get(sign, "overall", "any_sign_error_rate", "rate")),
        "driver_sign_acc_mean_thin": str(_get(sign, "thin", "per_row_sign_accuracy", "mean")),
        "driver_sign_acc_p50_thin": str(_get(sign, "thin", "per_row_sign_accuracy", "p50")),
        "driver_sign_any_error_rate_thin": str(_get(sign, "thin", "any_sign_error_rate", "rate")),
        "driver_sign_acc_mean_thick": str(_get(sign, "thick", "per_row_sign_accuracy", "mean")),
        "driver_sign_acc_p50_thick": str(_get(sign, "thick", "per_row_sign_accuracy", "p50")),
        "driver_sign_any_error_rate_thick": str(_get(sign, "thick", "any_sign_error_rate", "rate")),
        "driver_sign_metrics_path": str(
            sign_path_top5 if sign_path_top5.exists() else (sign_path_default if sign_path_default.exists() else "")
        ),

        # Driver leakage metrics (optional; blank if file missing)
        "driver_leak_k": str(leak.get("k", "")),
        "driver_leak_any_rate_overall": str(_get(leak, "overall", "leak_any_rate", "rate")),
        "driver_leak_count_mean_overall": str(_get(leak, "overall", "leak_count", "mean")),
        "driver_leak_count_p50_overall": str(_get(leak, "overall", "leak_count", "p50")),
        "driver_leak_count_p95_overall": str(_get(leak, "overall", "leak_count", "p95")),
        "driver_leak_count_max_overall": str(_get(leak, "overall", "leak_count", "max")),
        "driver_leak_any_rate_thin": str(_get(leak, "thin", "leak_any_rate", "rate")),
        "driver_leak_count_mean_thin": str(_get(leak, "thin", "leak_count", "mean")),
        "driver_leak_count_p50_thin": str(_get(leak, "thin", "leak_count", "p50")),
        "driver_leak_count_p95_thin": str(_get(leak, "thin", "leak_count", "p95")),
        "driver_leak_count_max_thin": str(_get(leak, "thin", "leak_count", "max")),
        "driver_leak_any_rate_thick": str(_get(leak, "thick", "leak_any_rate", "rate")),
        "driver_leak_count_mean_thick": str(_get(leak, "thick", "leak_count", "mean")),
        "driver_leak_count_p50_thick": str(_get(leak, "thick", "leak_count", "p50")),
        "driver_leak_count_p95_thick": str(_get(leak, "thick", "leak_count", "p95")),
        "driver_leak_count_max_thick": str(_get(leak, "thick", "leak_count", "max")),
        "driver_leak_metrics_path": str(
            leak_path_top5 if leak_path_top5.exists() else (leak_path_default if leak_path_default.exists() else "")
        ),
    }

    out_csv = run_dir / "report_v2.csv"
    is_new = (not out_csv.exists()) or (out_csv.stat().st_size == 0)

    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)

    # Also print a compact console summary
    print("Wrote:", out_csv)
    print("Git SHA:", sha or "(unknown)", "Dirty:", dirty or "(unknown)")
    print("Model:", row["model"], "Split:", row["split_id"], "Best iter:", row["best_iteration"])
    print("Test overall AUC/PR-AUC:", row["test_roc_auc_overall"], row["test_pr_auc_overall"])
    print("Test thin   AUC/PR-AUC:", row["test_roc_auc_thin"], row["test_pr_auc_thin"])
    print("Test thick  AUC/PR-AUC:", row["test_roc_auc_thick"], row["test_pr_auc_thick"])
    print("Counts thin/thick:", row["n_thin"], row["n_thick"])

    if row["driver_overlap_k"]:
        print(
            "Driver overlap@k (mean/p50 overall):",
            row["driver_overlap_mean_overall"],
            row["driver_overlap_p50_overall"],
            "k=",
            row["driver_overlap_k"],
        )

    if row["driver_sign_k"]:
        print(
            "Driver sign-faithfulness (mean/p50 overall):",
            row["driver_sign_acc_mean_overall"],
            row["driver_sign_acc_p50_overall"],
            "any_error_rate=",
            row["driver_sign_any_error_rate_overall"],
            "k=",
            row["driver_sign_k"],
        )

    if row["driver_leak_k"]:
        print(
            "Driver leakage (any rate overall):",
            row["driver_leak_any_rate_overall"],
            "mean leaked count overall:",
            row["driver_leak_count_mean_overall"],
            "k=",
            row["driver_leak_k"],
        )


if __name__ == "__main__":
    main()
