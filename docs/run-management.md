# Run management (external artifacts, manifests, and CSV index)

## Policy
- Git contains: code, schemas, configs, and small examples.
- Full run artifacts are stored outside the repo in RUNS_DIR.

## RUNS_DIR configuration
Default:
- /mnt/Seagate Expansion Drive/DSI/MSc/runs

Override:
- export FRAUD_THESIS_RUNS_DIR=/some/other/path

## Notebook hygiene
- `notebooks/` must be committed without outputs (pre-commit enforced).
- Executed notebooks are allowed in `notebooks/reports/`.
