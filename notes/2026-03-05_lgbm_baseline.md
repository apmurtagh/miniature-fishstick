# 2026-03-05 — LightGBM baseline + thin/thick cohort eval

## Summary

This repo now includes an end-to-end, memory-safe baseline for the IEEE-CIS fraud dataset:

- A numeric-only LightGBM model trained on a subsample of rows/features (to avoid OOM on commodity hardware).
- A thin vs thick cohort evaluator to quantify performance differences when identity information is present.

Artifacts are written under `artifacts/` (gitignored).

## Scripts

### Train baseline

```bash
python scripts/train_lgbm_numeric_v1.py
```

Writes:
- `artifacts/baselines/lgbm_numeric_v1_subsample/metrics.json`
- `artifacts/baselines/lgbm_numeric_v1_subsample/test_predictions.csv`

### Evaluate thin vs thick

```bash
python scripts/eval_thin_thick_v1.py
```

Writes:
- `artifacts/baselines/lgbm_numeric_v1_subsample/thin_thick_metrics.json`

### Write a one-line CSV report (append-only)

```bash
python scripts/report_baseline_metrics_v1.py
```

Writes/appends:
- `artifacts/baselines/lgbm_numeric_v1_subsample/report.csv`

## Cohort definition (proxy 3)

We define:

- **thick** if `(id_01 is not null) OR (DeviceType is not null)`
- **thin** otherwise

This proxy is intentionally simple and intended as a first-pass separation between identity-rich vs identity-poor files.

## Notes / interpretation

- Expect thick-file performance to be substantially higher than thin-file performance.
- For PR-AUC in particular, thin-file PR-AUC can be much lower even when overall AUC remains decent; this can matter a lot operationally because most review capacity is spent on the high-score tail.
- This baseline is not intended to be “best possible”; it is intended to:
  1) validate pipeline + split logic end-to-end
  2) provide a reproducible reference point
  3) surface cohort performance gaps early

## Reproducibility checklist

- Confirm `artifacts/data_cache/joined_train.parquet` was materialized from the intended raw sources.
- Confirm splits in `artifacts/splits/v1_temporal_q70_q85/` match the experiment definition.
- Record the git commit SHA when capturing/reporting numbers (the reporting CSV includes a timestamp; optionally add commit SHA later).