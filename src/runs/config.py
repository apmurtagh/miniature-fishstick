"""
Run-management configuration.

RUNS_DIR resolution order (highest precedence first):
  1. Explicit ``runs_dir`` argument passed to a function / CLI flag.
  2. Environment variable ``FRAUD_THESIS_RUNS_DIR``.
  3. Compiled-in default ``/mnt/Seagate Expansion Drive/DSI/MSc/runs``.
"""

import os
from pathlib import Path

_DEFAULT_RUNS_DIR = "/mnt/Seagate Expansion Drive/DSI/MSc/runs"
_ENV_VAR = "FRAUD_THESIS_RUNS_DIR"


def get_runs_dir(override: str | None = None) -> Path:
    """Return the resolved RUNS_DIR as a :class:`pathlib.Path`.

    Parameters
    ----------
    override:
        Optional explicit path (e.g. supplied via a CLI flag).  Takes
        precedence over everything else.
    """
    if override is not None:
        return Path(override)
    env_val = os.environ.get(_ENV_VAR)
    if env_val:
        return Path(env_val)
    return Path(_DEFAULT_RUNS_DIR)
