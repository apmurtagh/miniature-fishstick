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