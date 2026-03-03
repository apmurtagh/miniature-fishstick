# miniature-fishstick

MSc fraud-detection thesis codebase — run-management framework and experiment pipeline.

---

## Quick start

```bash
# 1. Clone & install (editable)
git clone https://github.com/apmurtagh/miniature-fishstick.git
cd miniature-fishstick
pip install -e ".[dev]"

# 2. (Optional) tell the framework where your external drive is mounted
export FRAUD_THESIS_RUNS_DIR="/mnt/Seagate Expansion Drive/DSI/MSc/runs"

# 3. Create a run, record metrics, inspect the index
python -m runs create --params model=xgboost masking_rate=0.1
# → {"run_id": "20260303_142201_a3f7c1b2", "run_dir": "…/2026-03-03/20260303_…"}

python -m runs finish 20260303_142201_a3f7c1b2 --metrics auc_roc=0.92 ece=0.04
python -m runs list
```

---

## Artifact storage policy

| Location | What lives here |
|---|---|
| **This repo** (`git`) | Source code, schemas, configs, stripped notebooks, reports notebooks |
| **`RUNS_DIR`** (external drive) | All run artifacts — models, predictions, manifests, metrics, full notebooks |

**Rule:** large artifacts (models, datasets, serialised predictions, etc.) are
**never** committed to git.  Only source code and lightweight metadata belong
in the repo.

---

## Configuring RUNS_DIR

The default is:

```
/mnt/Seagate Expansion Drive/DSI/MSc/runs
```

Override in order of precedence:

1. **CLI flag** `--runs-dir /path/to/runs` (highest precedence)
2. **Environment variable** `FRAUD_THESIS_RUNS_DIR=/path/to/runs`
3. Compiled-in default (above)

Add the export to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.) so it
persists across sessions.

---

## Run-management workflow

See [`docs/run_management.md`](docs/run_management.md) for the full guide.
The short version:

```python
from runs import create_run, append_to_index

# 1. Initialise — creates directory + writes manifest + captures provenance
manifest = create_run(
    params={"model": "xgboost", "eo_schema": "v2", "masking_rate": 0.1},
    seeds={"seed_global": 42},
    dataset_versions={"train": "v1.2", "test": "v1.2"},
)
run_id  = manifest["run_id"]
run_dir = manifest["run_dir"]   # Path string

# 2. … run your experiment, write artifacts under run_dir …

# 3. Record metrics in the global index
append_to_index({
    "run_id":    run_id,
    "model":     "xgboost",
    "auc_roc":   0.92,
    "ece":       0.04,
    "brier_score": 0.08,
})
```

---

## Notebook hygiene

- **Work notebooks** (`notebooks/`) — must have outputs stripped before commit.
  Run `nbstripout notebooks/my_notebook.ipynb` or rely on the pre-commit hook.
- **Report notebooks** (`notebooks/reports/`) — executed outputs are preserved
  intentionally; they are excluded from the strip hook.

Install the pre-commit hook (once, per clone):

```bash
pip install pre-commit
pre-commit install
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests use temporary directories and do **not** require the external drive.

---

## Repository layout

```
miniature-fishstick/
├── src/
│   ├── runs/          ← run-management module (this PR)
│   │   ├── config.py      – RUNS_DIR resolution
│   │   ├── run_id.py      – sortable ID generation
│   │   ├── manager.py     – directory creation + manifest
│   │   ├── provenance.py  – pip/conda environment capture
│   │   ├── index.py       – global runs_index.csv
│   │   └── cli.py         – `python -m runs` CLI
│   ├── data/          ← data loading utilities
│   ├── eo/            ← evidential ordinal modules
│   ├── eval/          ← evaluation metrics
│   ├── explain/       ← SHAP / explanation utilities
│   ├── features/      ← feature engineering
│   ├── models/        ← model wrappers
│   ├── narratives/    ← narrative generation
│   └── utils/         ← shared helpers
├── tests/             ← unit tests
├── notebooks/         ← work notebooks (outputs stripped)
│   └── reports/       ← final report notebooks (outputs kept)
├── configs/           ← YAML/JSON experiment configs
├── schemas/           ← JSON schemas for EO / narratives
├── docs/              ← supplementary documentation
├── .pre-commit-config.yaml
├── hooks/pre-commit   ← local notebook-output guard
└── pyproject.toml
```