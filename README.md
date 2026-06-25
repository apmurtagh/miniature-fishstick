# miniature-fishstick

MSc Data Science thesis repository.

## Contents

1. Thesis Summary
2. High-Level Pipeline
3. Quick Start
4. Key Outputs
5. Reproducibility Notes
6. Runbook
7. Technical Appendix


## 1. Thesis Summary

This project investigates governance-ready fraud decisioning narratives using the IEEE-CIS Fraud Detection dataset.

The repository implements a reproducible pipeline that:

1. trains a baseline fraud detection model
2. generates prediction-level Evidence Objects (EOs)
3. attaches SHAP-based explanatory drivers
4. produces both deterministic template narratives and constrained LLM narratives
5. evaluates narrative faithfulness, disclosure behaviour and driver coverage

The objective is to compare template and LLM-generated explanations while maintaining traceability to structured evidence and supporting reproducible governance workflows.

---

## 2. High-Level Pipeline

```text
IEEE-CIS Data
      │
      ▼
Baseline Fraud Model
      │
      ▼
Predictions
      │
      ▼
Evidence Objects (EOs)
      │
      ▼
SHAP Driver Attachment
      │
      ▼
 ┌──────────────────────┐
 │ Template Narratives  │
 └──────────────────────┘
            │

 ┌──────────────────────┐
 │ LLM Narratives       │
 └──────────────────────┘
            │
            ▼
      Evaluation
```

---

## 3. Quick Start

### Environment

```bash
cd /workspaces/miniature-fishstick

export IEEE_CIS_DIR=/workspaces/miniature-fishstick/data/ieee-cis
export PYTHONPATH=src
export OPENAI_MODEL=gpt-4.1-mini
```

### Install dependencies

```bash
python -m pip install -r requirements-dev.txt
```

### Smoke test dataset access

```bash
PYTHONPATH=src python scripts/smoke_load_ieee.py
```

---

## 4. Key Outputs

The current cleaned pipeline produces the following core artefacts:

### Model outputs

```text
artifacts/baselines/lgbm_numeric_v1_subsample/test_predictions.csv
artifacts/baselines/lgbm_numeric_v1_subsample/metrics.json
```

### Evidence outputs

```text
artifacts/baselines/lgbm_numeric_v1_subsample/eos_test.jsonl
artifacts/baselines/lgbm_numeric_v1_subsample/eos_test_with_drivers.jsonl
```

### Template narrative outputs

```text
artifacts/baselines/lgbm_numeric_v1_subsample/narratives_ops_triage_template.jsonl
artifacts/baselines/lgbm_numeric_v1_subsample/eval_summary_template.json
```

### LLM narrative outputs

```text
artifacts/baselines/lgbm_numeric_v1_subsample/narratives_ops_triage_llm.jsonl
artifacts/baselines/lgbm_numeric_v1_subsample/eval_summary_llm.json
```

---

## 5. Reproducibility Notes

- Raw IEEE-CIS data is not committed to the repository.
- Generated outputs are written under `artifacts/`.
- Template narratives are fully reproducible without external services.
- LLM narratives require a valid OpenAI API key for live execution.
- The LLM orchestration path supports graceful fallback when credentials are unavailable.
- Commands should generally be executed from repository root using:

```bash
PYTHONPATH=src python scripts/<script>.py
```

---

## 6. Runbook

This repository supports a governance-ready fraud decisioning pipeline built on the IEEE-CIS Fraud Detection dataset, with:

- a reproducible baseline fraud model
- Evidence Object (EO) emission
- SHAP-based top-driver attachment
- deterministic template narratives
- optional LLM narratives with validator/fallback behaviour
- lightweight evaluation summaries


# 7. Technical Appendix

This appendix contains implementation-focused details that support reproducibility and experimentation but are not required to understand the core thesis workflow.

---

## Run Management CLI

The repository includes a lightweight run-management CLI for creating and finalising experiment runs.

### Installed usage (recommended)

```bash
python -m pip install -e .

fraud-thesis show-runs-dir

RUN_DIR="$(fraud-thesis init-run \
  --split-version v1 \
  --seed 1 \
  --model demo \
  --narrative-mode templates \
  --runs-dir /tmp/fraud_runs)"

fraud-thesis finalize-run \
  --run-dir "$RUN_DIR" \
  --runs-dir /tmp/fraud_runs \
  --metrics-json '{"auc":0.91}'
```

### Module usage (no install)

```bash
PYTHONPATH=src python -m fraud_thesis.cli show-runs-dir

RUN_DIR="$(PYTHONPATH=src python -m fraud_thesis.cli init-run \
  --split-version v1 \
  --seed 1 \
  --model demo \
  --narrative-mode templates \
  --runs-dir /tmp/fraud_runs)"

PYTHONPATH=src python -m fraud_thesis.cli finalize-run \
  --run-dir "$RUN_DIR" \
  --runs-dir /tmp/fraud_runs \
  --metrics-json '{"auc":0.91}'
```

### Module usage (installed, no console script)

```bash
python -m fraud_thesis show-runs-dir
```

---

## Baseline Implementation

### LightGBM Numeric Baseline

The repository includes a memory-safe numeric-only LightGBM baseline intended to run on modest hardware by:

- subsampling rows
- limiting feature count
- reducing memory pressure during experimentation

Primary outputs:

```text
artifacts/baselines/lgbm_numeric_v1_subsample/metrics.json
artifacts/baselines/lgbm_numeric_v1_subsample/test_predictions.csv
artifacts/baselines/lgbm_numeric_v1_subsample/thin_thick_metrics.json
artifacts/baselines/lgbm_numeric_v1_subsample/report_v3.csv
```

### Baseline Scripts

```bash
PYTHONPATH=src python scripts/train_lgbm_numeric_v1.py
PYTHONPATH=src python scripts/eval_thin_thick_v1.py
PYTHONPATH=src python scripts/report_baseline_metrics_v1.py
```

---

## Thin vs Thick Cohort Definition

The thin/thick evaluator currently uses proxy definition (3):

### Thick

```text
id_01 present
OR
DeviceType present
```

### Thin

```text
otherwise
```

Further discussion is available in:

```text
notes/2026-03-05_lgbm_baseline.md
```

---

## Narrative Generation Components

### Template Narrative Path

```text
EO
  ↓
Template Engine
  ↓
narratives_ops_triage_template.jsonl
  ↓
eval_summary_template.json
```

Key scripts:

```bash
PYTHONPATH=src python scripts/orchestrate_narrative_experiments.py
PYTHONPATH=src python scripts/orchestrate_evaluation.py
```

### LLM Narrative Path

```text
EO
  ↓
LLM Engine
  ↓
Validator
  ↓
narratives_ops_triage_llm.jsonl
  ↓
eval_summary_llm.json
```

Key scripts:

```bash
PYTHONPATH=src python scripts/orchestrate_llm_narratives.py
PYTHONPATH=src python scripts/orchestrate_evaluation.py \
  --narratives-jsonl artifacts/baselines/lgbm_numeric_v1_subsample/narratives_ops_triage_llm.jsonl \
  --out-json artifacts/baselines/lgbm_numeric_v1_subsample/eval_summary_llm.json
```

---

## LLM Fallback Behaviour

The LLM orchestration path supports graceful degradation.

When a valid `OPENAI_API_KEY` is unavailable:

- the orchestration script remains runnable
- deterministic template narratives are emitted instead
- output records explicit fallback metadata
- repository functionality remains demonstrable without credentials

Example metadata:

```json
{
  "validator_status": "fallback",
  "fallback_used": true,
  "validator_reason": "missing_openai_api_key"
}
```

---

## Artifact Locations

Primary thesis artefacts are written under:

```text
artifacts/
```

Key subdirectories:

```text
artifacts/data_cache/
artifacts/splits/
artifacts/baselines/lgbm_numeric_v1_subsample/
```

Raw IEEE-CIS source data is expected under:

```text
data/ieee-cis/
```

and is intentionally excluded from version control.

---

## Development Notes

Recommended execution pattern:

```bash
cd /workspaces/miniature-fishstick

export IEEE_CIS_DIR=/workspaces/miniature-fishstick/data/ieee-cis
export PYTHONPATH=src
export OPENAI_MODEL=gpt-4.1-mini
```

Most repository commands should then be executed as:

```bash
PYTHONPATH=src python scripts/<script>.py
```

This is the execution pattern used throughout repository validation and testing.