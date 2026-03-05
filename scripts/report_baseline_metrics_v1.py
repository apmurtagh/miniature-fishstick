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


def main() -> None:
    run_dir = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
    metrics_path = run_dir / "metrics.json"
    thin_thick_path = run_dir / "thin_thick_metrics.json"

    metrics = _load_json(metrics_path)
    cohort = _load_json(thin_thick_path)

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
    }

    out_csv = run_dir / "report.csv"
    is_new = not out_csv.exists()

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


if __name__ == "__main__":
    main()
