# miniature-fishstick

Thesis repo scaffolding.

## Dev setup

```bash
python -m pip install -r requirements-dev.txt
```

## Run management CLI (without packaging)

```bash
# Show the resolved default runs dir (env override supported)
PYTHONPATH=src python -m fraud_thesis.cli show-runs-dir

# Create a run and capture the printed run dir
RUN_DIR="$(PYTHONPATH=src python -m fraud_thesis.cli init-run \
  --split-version v1 --seed 1 --model demo --narrative-mode templates --runs-dir /tmp/fraud_runs)"

# Finalize the run (writes metrics + appends runs_index.csv)
PYTHONPATH=src python -m fraud_thesis.cli finalize-run \
  --run-dir "$RUN_DIR" --runs-dir /tmp/fraud_runs \
  --metrics-json '{"auc":0.91}'
```