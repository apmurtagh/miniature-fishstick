"""
Global run index (CSV) management.

The index lives at ``<RUNS_DIR>/runs_index.csv`` and has one row per
completed run.  Appends are made atomic (best-effort) via a file-lock
implemented with :func:`fcntl.flock`.
"""

import csv
import io
import os
from pathlib import Path
from typing import Any

from .config import get_runs_dir

# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------

#: Ordered list of column names.  New optional columns may be appended in
#: future; existing columns are never removed to preserve backward compat.
INDEX_COLUMNS: list[str] = [
    # --- Identity ---
    "run_id",
    "timestamp",
    "git_commit",
    "git_dirty",
    # --- Key parameters ---
    "model",
    "eo_schema",
    "narrative_mode",
    "masking_rate",
    "persona",
    # --- Dataset / split versions ---
    "dataset_version",
    "split_version",
    # --- Seeds ---
    "seed_global",
    "seed_data",
    # --- Classification metrics ---
    "auc_roc",
    "ece",
    "brier_score",
    # --- EO / ranking metrics ---
    "overlap_at_k",
    "direction_accuracy",
    "order_agreement",
    # --- Narrative / action consistency ---
    "action_consistency",
    # --- Validator pass rates ---
    "validator_pass_rate",
    "validator_retry_rate",
    "validator_fallback_rate",
    # --- Drift / thin-file strata ---
    "drift_stratum",
    "thin_file_stratum",
    # --- Free-text notes ---
    "notes",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_flock(fd: int, op: int) -> bool:
    """Attempt fcntl.flock; silently succeed on platforms that lack it."""
    try:
        import fcntl

        fcntl.flock(fd, op)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append_to_index(
    row: dict[str, Any],
    *,
    runs_dir: str | Path | None = None,
) -> Path:
    """Append *row* to the global run index CSV.

    Unknown keys in *row* are silently ignored; missing keys produce empty
    cells.  The header is written on first use.  Appends are serialised via
    ``fcntl.flock`` (POSIX) so concurrent processes do not corrupt the file.

    Parameters
    ----------
    row:
        Mapping of column name → value.  At minimum ``run_id`` should be set.
    runs_dir:
        Override for the RUNS_DIR root.

    Returns
    -------
    Path
        Absolute path to the updated ``runs_index.csv``.
    """
    root = get_runs_dir(override=str(runs_dir) if runs_dir is not None else None)
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "runs_index.csv"

    # Serialise values to strings
    str_row = {k: str(v) if v is not None else "" for k, v in row.items()}

    # Determine if header is needed before opening for append
    write_header = not index_path.exists() or index_path.stat().st_size == 0

    with open(index_path, "a", newline="", encoding="utf-8") as fh:
        import fcntl as _fcntl  # noqa: PLC0415

        try:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
        except Exception:
            pass  # non-POSIX or no flock support – proceed without lock

        # Re-check after acquiring lock
        if write_header:
            fh.seek(0, os.SEEK_END)
            write_header = fh.tell() == 0

        writer = csv.DictWriter(
            fh,
            fieldnames=INDEX_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(str_row)

        try:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
        except Exception:
            pass

    return index_path


def read_index(*, runs_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Read the run index and return all rows as a list of dicts.

    Returns an empty list if the index does not yet exist.
    """
    root = get_runs_dir(override=str(runs_dir) if runs_dir is not None else None)
    index_path = root / "runs_index.csv"
    if not index_path.exists():
        return []
    with open(index_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)
