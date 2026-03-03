"""
Run directory creation and manifest writing.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_runs_dir
from .provenance import capture_env
from .run_id import generate_run_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _git_info() -> dict[str, str]:
    """Return the current git commit hash and dirty flag."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = len(status) > 0
    except Exception:
        commit = "unknown"
        dirty = False
    return {"commit": commit, "dirty": dirty}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_run(
    *,
    runs_dir: str | Path | None = None,
    run_id: str | None = None,
    params: dict[str, Any] | None = None,
    dataset_versions: dict[str, str] | None = None,
    seeds: dict[str, int] | None = None,
    artifact_paths: dict[str, str] | None = None,
    capture_provenance: bool = True,
) -> dict[str, Any]:
    """Create a new run directory and write the run manifest.

    Parameters
    ----------
    runs_dir:
        Override for the RUNS_DIR root.  Resolved via :func:`get_runs_dir`
        when *None*.
    run_id:
        Explicit run ID.  Generated via :func:`generate_run_id` when *None*.
    params:
        Key model/experiment parameters to embed in the manifest.  Recognised
        top-level keys include (but are not limited to): ``model``,
        ``eo_schema``, ``narrative_mode``, ``masking_rate``, ``persona``.
    dataset_versions:
        Mapping of dataset/split name → version string.
    seeds:
        Mapping of seed name → integer value.
    artifact_paths:
        Mapping of artifact label → path *relative to the run directory*.
    capture_provenance:
        When *True* (default), write provenance files into ``meta/``.

    Returns
    -------
    dict
        The manifest dictionary that was written to disk.  Notably includes
        ``run_dir`` (absolute path string) and ``run_id``.
    """
    root = get_runs_dir(override=str(runs_dir) if runs_dir is not None else None)

    now = datetime.now(tz=timezone.utc)
    rid = run_id or generate_run_id(dt=now)
    date_str = now.strftime("%Y-%m-%d")

    run_dir = root / date_str / rid
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Sub-directories aligned to the proposal work packages
    for sub in ("models", "eo", "narratives", "eval", "reports"):
        (run_dir / sub).mkdir(exist_ok=True)

    git = _git_info()

    manifest: dict[str, Any] = {
        "run_id": rid,
        "timestamp": now.isoformat(),
        "git": git,
        "dataset_versions": dataset_versions or {},
        "seeds": seeds or {},
        "params": params or {},
        "artifact_paths": artifact_paths or {},
        "run_dir": str(run_dir),
    }

    manifest_path = meta_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if capture_provenance:
        capture_env(meta_dir)

    return manifest
