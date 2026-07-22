import json
from pathlib import Path

import numpy as np
import pandas as pd


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
OUT = ART / "feature_masking_rescore"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ART / "model.txt"
FEATURES_PATH = ART / "feature_names.json"

OUT_CSV = OUT / "feature_masking_rescore_1000.csv"
OUT_MD = OUT / "feature_masking_rescore_summary.md"
OUT_JSON = OUT / "feature_masking_rescore_summary.json"
LIMITATION_MD = OUT / "feature_masking_rescore_limitation.md"

N = 1000
SEED = 42
MASK_RATES = [0.0, 0.3, 0.6]


FEATURE_MATRIX_CANDIDATES = [
    ART / "X_test.parquet",
    ART / "x_test.parquet",
    ART / "test_features.parquet",
    ART / "X_test.csv",
    ART / "x_test.csv",
    ART / "test_features.csv",
    ART / "features_test.parquet",
    ART / "features_test.csv",
    Path("artifacts/data_cache/X_test.parquet"),
    Path("artifacts/data_cache/test_features.parquet"),
]


def write_limitation(reason):
    data = {
        "status": "not_run",
        "reason": reason,
        "recommended_dissertation_wording": (
            "A full feature-masking re-score was attempted, but the exact model-ready test feature matrix "
            "was not available as a persisted artefact. The dissertation therefore retains the EO-level "
            "masking proxy as bounded thin-file evidence and treats full model/action recalibration under "
            "masked feature re-scoring as future work."
        ),
    }

    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

    LIMITATION_MD.write_text(
        "# Feature-Masking Re-score Limitation\n\n"
        "Status: not run\n\n"
        f"Reason: {reason}\n\n"
        "Dissertation wording: A full feature-masking re-score was attempted, but the exact model-ready "
        "test feature matrix was not available as a persisted artefact. The dissertation therefore retains "
        "the EO-level masking proxy as bounded thin-file evidence and treats full model/action recalibration "
        "under masked feature re-scoring as future work.\n",
        encoding="utf-8",
    )

    print("[WARN] Feature-masking re-score not run")
    print(reason)
    print(f"[OK] Limitation written: {LIMITATION_MD}")


def risk_band(score):
    if score >= 0.80:
        return "HIGH"
    if score >= 0.30:
        return "MED"
    return "LOW"


def action_from_score(score):
    if score >= 0.80:
        return "review"
    if score >= 0.30:
        return "step-up"
    return "allow"


def load_feature_names():
    if not FEATURES_PATH.exists():
        return None, f"Missing feature names file: {FEATURES_PATH}"

    try:
        obj = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"Could not parse feature_names.json: {repr(e)}"

    if isinstance(obj, list):
        names = obj
    elif isinstance(obj, dict):
        names = obj.get("feature_names") or obj.get("features") or obj.get("columns")
        if names is None:
            names = list(obj.values())
    else:
        return None, "Unsupported feature_names.json structure"

    names = [str(x) for x in names]
    return names, None


def load_feature_matrix(feature_names):
    for path in FEATURE_MATRIX_CANDIDATES:
        if not path.exists():
            continue

        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                continue
        except Exception as e:
            print(f"[WARN] Could not load candidate {path}: {repr(e)}")
            continue

        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            print(f"[WARN] Candidate {path} missing model features. First missing: {missing[:8]}")
            continue

        print(f"[OK] Loaded model-ready feature matrix candidate: {path}")
        return df[feature_names].copy(), str(path), None

    return None, None, (
        "No persisted model-ready test feature matrix was found among common candidate paths. "
        "Candidates checked: " + ", ".join(str(p) for p in FEATURE_MATRIX_CANDIDATES)
    )


def coerce_numeric_and_impute(X):
    X2 = X.copy()

    for c in X2.columns:
        X2[c] = pd.to_numeric(X2[c], errors="coerce")

    medians = X2.median(numeric_only=True).fillna(0.0)
    X2 = X2.fillna(medians)

    return X2, medians


def main():
    try:
        import lightgbm as lgb
    except Exception as e:
        write_limitation(f"lightgbm import failed: {repr(e)}")
        return

    if not MODEL_PATH.exists():
        write_limitation(f"Missing LightGBM model file: {MODEL_PATH}")
        return

    feature_names, err = load_feature_names()
    if err:
        write_limitation(err)
        return

    X_raw, matrix_path, err = load_feature_matrix(feature_names)
    if err:
        write_limitation(err)
        return

    X, medians = coerce_numeric_and_impute(X_raw)

    if len(X) < N:
        write_limitation(f"Feature matrix has only {len(X)} rows; expected at least {N}.")
        return

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(np.arange(len(X)), size=N, replace=False)
    X_sample = X.iloc[sample_idx].copy().reset_index(drop=True)

    try:
        model = lgb.Booster(model_file=str(MODEL_PATH))
    except Exception as e:
        write_limitation(f"Could not load LightGBM model: {repr(e)}")
        return

    try:
        base_scores = model.predict(X_sample)
    except Exception as e:
        write_limitation(f"Model prediction failed on loaded feature matrix: {repr(e)}")
        return

    records = []

    for mask_rate in MASK_RATES:
        X_masked = X_sample.copy()

        if mask_rate > 0:
            k = max(1, int(round(len(feature_names) * mask_rate)))
            masked_features = list(rng.choice(feature_names, size=k, replace=False))

            for c in masked_features:
                X_masked[c] = medians.get(c, 0.0)
        else:
            masked_features = []

        try:
            masked_scores = model.predict(X_masked)
        except Exception as e:
            write_limitation(f"Model prediction failed at mask_rate={mask_rate}: {repr(e)}")
            return

        for i, score in enumerate(masked_scores):
            base_score = float(base_scores[i])
            masked_score = float(score)

            base_action = action_from_score(base_score)
            masked_action = action_from_score(masked_score)

            records.append({
                "mask_rate": mask_rate,
                "row_index": int(i),
                "base_score": base_score,
                "masked_score": masked_score,
                "score_delta": masked_score - base_score,
                "base_risk_band": risk_band(base_score),
                "masked_risk_band": risk_band(masked_score),
                "base_action": base_action,
                "masked_action": masked_action,
                "action_changed": base_action != masked_action,
                "masked_feature_count": len(masked_features),
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)

    summary_rows = []
    for rate, g in df.groupby("mask_rate"):
        summary_rows.append({
            "mask_rate": float(rate),
            "rows": int(len(g)),
            "mean_base_score": float(g["base_score"].mean()),
            "mean_masked_score": float(g["masked_score"].mean()),
            "mean_abs_score_delta": float(g["score_delta"].abs().mean()),
            "action_change_rate": float(g["action_changed"].mean()),
            "step_up_or_review_rate": float(g["masked_action"].isin(["step-up", "review"]).mean()),
            "high_risk_rate": float((g["masked_risk_band"] == "HIGH").mean()),
        })

    summary = {
        "status": "run",
        "source_model": str(MODEL_PATH),
        "source_feature_matrix": matrix_path,
        "rows_sampled": N,
        "mask_rates": MASK_RATES,
        "summary_rows": summary_rows,
        "caveat": (
            "This is an inference-only feature-masking re-score using a persisted model-ready feature matrix. "
            "It is a controlled robustness stress test, not a production enrichment-outage simulation."
        ),
        "outputs": {
            "csv": str(OUT_CSV),
            "md": str(OUT_MD),
            "json": str(OUT_JSON),
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Feature-Masking Re-score Summary\n\n")
    md.append("Status: run\n\n")
    md.append(f"Source feature matrix: `{matrix_path}`\n\n")
    md.append("| Mask rate | Rows | Mean base score | Mean masked score | Mean abs score delta | Action change rate | Step-up/review rate | HIGH risk rate |\n")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for r in summary_rows:
        md.append(
            f"| {r['mask_rate']:.1f} | {r['rows']} | {r['mean_base_score']:.4f} | "
            f"{r['mean_masked_score']:.4f} | {r['mean_abs_score_delta']:.4f} | "
            f"{r['action_change_rate']:.4f} | {r['step_up_or_review_rate']:.4f} | "
            f"{r['high_risk_rate']:.4f} |\n"
        )

    md.append(
        "\nInterpretation: This inference-only masking re-score applies 0%, 30% and 60% masking to a persisted "
        "model-ready feature matrix and re-scores the trained LightGBM model. It strengthens thin-file robustness evidence "
        "if the persisted feature matrix is the same matrix used for the reported baseline evaluation. It remains a controlled "
        "stress test rather than a production outage simulation.\n"
    )

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print("[OK] Feature-masking re-score completed")
    print(json.dumps(summary, indent=2))
    print("")
    print("== Summary ==")
    print("".join(md))


if __name__ == "__main__":
    main()
