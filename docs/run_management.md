# Run Management Guide

This document describes how to use the `runs` module to track experiments
throughout the MSc fraud-detection thesis project.

---

## Table of contents

1. [Motivation](#motivation)
2. [RUNS_DIR configuration](#runs_dir-configuration)
3. [Creating a run](#creating-a-run)
4. [Writing a manifest](#writing-a-manifest)
5. [Writing metrics to the global index](#writing-metrics-to-the-global-index)
6. [CLI reference](#cli-reference)
7. [Proposed workflow by work package](#proposed-workflow-by-work-package)
8. [Columns in runs_index.csv](#columns-in-runs_indexcsv)

---

## Motivation

The thesis involves many experiments across work packages (baseline model,
evidential ordinals, narratives, evaluation, reporting).  Each experiment
produces large binary artefacts (serialised models, predictions, SHAP values)
that must **not** be committed to Git.  The `runs` module provides:

- A deterministic, sortable `run_id` per experiment.
- A self-contained **run directory** on an external drive containing all
  artefacts.
- A **manifest** (`run_manifest.json`) capturing metadata at creation time
  (Git hash, parameters, dataset versions, seeds).
- A lightweight **global CSV index** (`runs_index.csv`) for quick comparison
  of runs across metrics.

---

## RUNS_DIR configuration

All run artefacts live under a configurable root called **RUNS_DIR**.

| Precedence | Method | Example |
|---|---|---|
| 1 (highest) | CLI flag | `python -m runs --runs-dir /mnt/…/runs create` |
| 2 | Environment variable | `export FRAUD_THESIS_RUNS_DIR=/mnt/…/runs` |
| 3 (default) | Compiled default | `/mnt/Seagate Expansion Drive/DSI/MSc/runs` |

**Recommended:** add the export to your shell profile so it persists:

```bash
echo 'export FRAUD_THESIS_RUNS_DIR="/mnt/Seagate Expansion Drive/DSI/MSc/runs"' \
  >> ~/.zshrc
```

---

## Creating a run

### Python API

```python
from runs import create_run

manifest = create_run(
    params={
        "model":          "xgboost",
        "eo_schema":      "v2",
        "narrative_mode": "full",
        "masking_rate":   0.1,
        "persona":        "analyst",
    },
    seeds={
        "seed_global": 42,
        "seed_data":   7,
    },
    dataset_versions={
        "train": "v1.2",
        "test":  "v1.2",
    },
    # capture_provenance=True  ← default; set False to skip pip freeze
)

run_id  = manifest["run_id"]   # e.g. "20260303_142201_a3f7c1b2"
run_dir = manifest["run_dir"]  # absolute path string
```

`create_run` will:
1. Generate a sortable `run_id` (format: `YYYYMMDD_HHMMSS_<8-hex>`).
2. Create `<RUNS_DIR>/YYYY-MM-DD/<run_id>/` with sub-directories:
   `meta/`, `models/`, `eo/`, `narratives/`, `eval/`, `reports/`.
3. Write `meta/run_manifest.json` (see next section).
4. Capture environment provenance into `meta/` (python version, pip freeze,
   conda history if available).

### CLI

```bash
python -m runs create \
    --params model=xgboost eo_schema=v2 masking_rate=0.1
# Prints: {"run_id": "...", "run_dir": "..."}
```

---

## Writing a manifest

The manifest is written automatically by `create_run`.  Its structure:

```json
{
  "run_id":            "20260303_142201_a3f7c1b2",
  "timestamp":         "2026-03-03T14:22:01.123456+00:00",
  "git": {
    "commit": "abc123def456...",
    "dirty":  false
  },
  "dataset_versions":  {"train": "v1.2", "test": "v1.2"},
  "seeds":             {"seed_global": 42, "seed_data": 7},
  "params": {
    "model":          "xgboost",
    "eo_schema":      "v2",
    "narrative_mode": "full",
    "masking_rate":   "0.1",
    "persona":        "analyst"
  },
  "artifact_paths":    {},
  "run_dir":           "/mnt/.../runs/2026-03-03/20260303_142201_a3f7c1b2"
}
```

You can update `artifact_paths` after writing artefacts by reading and
re-writing the JSON file, or by passing `artifact_paths` to `create_run`.

---

## Writing metrics to the global index

After a run is complete, append a row to `runs_index.csv`:

### Python API

```python
from runs import append_to_index

append_to_index({
    "run_id":       run_id,
    "model":        "xgboost",
    "eo_schema":    "v2",
    "masking_rate": "0.1",
    # Classification
    "auc_roc":    0.9213,
    "ece":        0.0312,
    "brier_score": 0.0764,
    # EO / ranking
    "overlap_at_k":       0.78,
    "direction_accuracy": 0.81,
    "order_agreement":    0.74,
    # Narratives
    "action_consistency": 0.88,
    # Validator rates
    "validator_pass_rate":     0.92,
    "validator_retry_rate":    0.06,
    "validator_fallback_rate": 0.02,
    # Strata
    "drift_stratum":     "high",
    "thin_file_stratum": "low",
    "notes": "first full EO run",
})
```

Unknown keys are silently ignored; missing keys produce empty cells.
Appends are serialised via `fcntl.flock` to avoid corruption from concurrent
processes.

### CLI

```bash
python -m runs finish 20260303_142201_a3f7c1b2 \
    --metrics auc_roc=0.9213 ece=0.0312 brier_score=0.0764
```

---

## CLI reference

```
python -m runs [--runs-dir PATH] <command> [options]

Commands:
  create   Initialise a new run directory and manifest.
             --params KEY=VAL ...    Embed parameters in manifest.
             --no-provenance         Skip pip freeze / conda export.

  finish   Record metrics in the global index for a completed run.
    <run_id>
             --metrics KEY=VAL ...   Metric values to record.

  list     Print a summary of all runs in the index.
```

---

## Proposed workflow by work package

| Work Package | Action | `params` keys to set |
|---|---|---|
| **WP1 – Baseline model** | `create_run` → train model → save to `run_dir/models/` → `append_to_index` | `model`, `seed_global`, `seed_data`, dataset versions |
| **WP2 – Evidential ordinals** | Same, but also write EO outputs to `run_dir/eo/` | `model`, `eo_schema` |
| **WP3 – Narratives** | Write generated narratives to `run_dir/narratives/`; record validator rates | `narrative_mode`, `masking_rate`, `persona` |
| **WP4 – Evaluation** | Write eval outputs to `run_dir/eval/`; append full metric suite to index | all metric columns |
| **WP5 – Reports** | Execute notebook → save to `notebooks/reports/`; link to `run_dir/reports/` | — |

---

## Columns in `runs_index.csv`

| Column | Description |
|---|---|
| `run_id` | Unique run identifier |
| `timestamp` | ISO-8601 UTC timestamp when the run was created |
| `git_commit` | Git commit hash at run creation |
| `git_dirty` | Whether the working tree was dirty |
| `model` | Model type (e.g. `xgboost`, `logistic_regression`) |
| `eo_schema` | Evidential ordinal schema version |
| `narrative_mode` | Narrative generation mode |
| `masking_rate` | Feature masking rate for narrative ablation |
| `persona` | Target reader persona |
| `dataset_version` | Version string for the dataset |
| `split_version` | Version string for the train/test split |
| `seed_global` | Global random seed |
| `seed_data` | Data-split random seed |
| `auc_roc` | Area under the ROC curve |
| `ece` | Expected calibration error |
| `brier_score` | Brier score |
| `overlap_at_k` | Overlap@K between EO ranking and ground truth |
| `direction_accuracy` | Direction accuracy of ordinal predictions |
| `order_agreement` | Spearman / Kendall order agreement |
| `action_consistency` | Consistency of recommended actions across narratives |
| `validator_pass_rate` | Fraction of narratives that passed validation |
| `validator_retry_rate` | Fraction of narratives that required retry |
| `validator_fallback_rate` | Fraction of narratives that fell back to default |
| `drift_stratum` | Data-drift stratum label (e.g. `low`, `medium`, `high`) |
| `thin_file_stratum` | Thin-file stratum label |
| `notes` | Free-text notes |
