# miniature-fishstick

Thesis repo scaffolding.

## Dev setup

```bash
python -m pip install -r requirements-dev.txt
```

## Run management CLI

### Installed usage (recommended)

```bash
python -m pip install -e .

fraud-thesis show-runs-dir

RUN_DIR="$(fraud-thesis init-run \
  --split-version v1 --seed 1 --model demo --narrative-mode templates --runs-dir /tmp/fraud_runs)"

fraud-thesis finalize-run \
  --run-dir "$RUN_DIR" --runs-dir /tmp/fraud_runs \
  --metrics-json '{"auc":0.91}'
```

### Module usage (no install)

```bash
PYTHONPATH=src python -m fraud_thesis.cli show-runs-dir

RUN_DIR="$(PYTHONPATH=src python -m fraud_thesis.cli init-run \
  --split-version v1 --seed 1 --model demo --narrative-mode templates --runs-dir /tmp/fraud_runs)"

PYTHONPATH=src python -m fraud_thesis.cli finalize-run \
  --run-dir "$RUN_DIR" --runs-dir /tmp/fraud_runs \
  --metrics-json '{"auc":0.91}'
```

### Module usage (installed, no console script)

```bash
python -m fraud_thesis show-runs-dir
```

## Baselines (IEEE-CIS)

### LightGBM numeric baseline (subsample, memory-safe)

This repo includes a numeric-only LightGBM baseline intended to run on modest hardware by subsampling rows and limiting features.

**What it writes (gitignored under `artifacts/`):**
- `artifacts/baselines/lgbm_numeric_v1_subsample/metrics.json`
- `artifacts/baselines/lgbm_numeric_v1_subsample/test_predictions.csv`
- `artifacts/baselines/lgbm_numeric_v1_subsample/thin_thick_metrics.json`
- `artifacts/baselines/lgbm_numeric_v1_subsample/report.csv` (append-only)

**How to run:**
```bash
python scripts/train_lgbm_numeric_v1.py
python scripts/eval_thin_thick_v1.py
python scripts/report_baseline_metrics_v1.py
```

### Thin vs thick cohort definition

The thin/thick evaluator uses proxy (3):

- **thick** if `id_01` is present **OR** `DeviceType` is present
- **thin** otherwise

See `notes/2026-03-05_lgbm_baseline.md` for a short write-up and interpretation guidance.

### Baseline reporting

Baseline metrics are appended to:

- `artifacts/baselines/lgbm_numeric_v1_subsample/report_v3.csv` (current; includes driver overlap metrics)

Note: `report.csv` and `report_v2.csv` are deprecated (older schema) and may contain mixed columns from earlier runs.





