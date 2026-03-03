"""
Unit tests for the runs management framework.

All tests use temporary directories so they never touch the external drive or
any real RUNS_DIR.
"""

import csv
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# run_id
# ---------------------------------------------------------------------------


def test_run_id_format():
    from src.runs.run_id import generate_run_id

    rid = generate_run_id()
    pattern = r"^\d{8}_\d{6}_[0-9a-f]{8}$"
    assert re.match(pattern, rid), f"run_id {rid!r} does not match expected pattern"


def test_run_id_sortable():
    from src.runs.run_id import generate_run_id

    dt1 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    dt2 = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    id1 = generate_run_id(dt=dt1)
    id2 = generate_run_id(dt=dt2)
    assert id1 < id2, "Earlier datetime should produce a lexicographically smaller ID"


def test_run_id_unique():
    from src.runs.run_id import generate_run_id

    dt = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ids = {generate_run_id(dt=dt) for _ in range(50)}
    assert len(ids) == 50, "UUIDs should ensure uniqueness even for same timestamp"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_get_runs_dir_override():
    from src.runs.config import get_runs_dir

    result = get_runs_dir(override="/tmp/my_runs")
    assert result == Path("/tmp/my_runs")


def test_get_runs_dir_env(monkeypatch):
    from src.runs.config import get_runs_dir, _ENV_VAR

    monkeypatch.setenv(_ENV_VAR, "/tmp/env_runs")
    result = get_runs_dir()
    assert result == Path("/tmp/env_runs")


def test_get_runs_dir_default(monkeypatch):
    from src.runs.config import get_runs_dir, _ENV_VAR, _DEFAULT_RUNS_DIR

    monkeypatch.delenv(_ENV_VAR, raising=False)
    result = get_runs_dir()
    assert result == Path(_DEFAULT_RUNS_DIR)


# ---------------------------------------------------------------------------
# manager – directory creation
# ---------------------------------------------------------------------------


def test_create_run_creates_directories(tmp_path):
    from src.runs.manager import create_run

    manifest = create_run(
        runs_dir=tmp_path,
        params={"model": "xgboost"},
        capture_provenance=False,
    )
    run_dir = Path(manifest["run_dir"])

    assert run_dir.exists(), "Run directory should be created"
    assert (run_dir / "meta").is_dir()
    assert (run_dir / "models").is_dir()
    assert (run_dir / "eo").is_dir()
    assert (run_dir / "narratives").is_dir()
    assert (run_dir / "eval").is_dir()
    assert (run_dir / "reports").is_dir()


def test_create_run_manifest_content(tmp_path):
    from src.runs.manager import create_run

    params = {
        "model": "logistic_regression",
        "eo_schema": "v2",
        "narrative_mode": "full",
        "masking_rate": "0.1",
        "persona": "analyst",
    }
    seeds = {"seed_global": 42}
    dataset_versions = {"train": "v1.0", "test": "v1.0"}

    manifest = create_run(
        runs_dir=tmp_path,
        params=params,
        seeds=seeds,
        dataset_versions=dataset_versions,
        capture_provenance=False,
    )

    manifest_path = Path(manifest["run_dir"]) / "meta" / "run_manifest.json"
    assert manifest_path.exists()

    with manifest_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["run_id"] == manifest["run_id"]
    assert "timestamp" in data
    assert "git" in data
    assert data["params"] == params
    assert data["seeds"] == seeds
    assert data["dataset_versions"] == dataset_versions


def test_create_run_explicit_run_id(tmp_path):
    from src.runs.manager import create_run

    manifest = create_run(
        runs_dir=tmp_path,
        run_id="20250101_000000_testtest",
        capture_provenance=False,
    )
    assert manifest["run_id"] == "20250101_000000_testtest"


# ---------------------------------------------------------------------------
# manager – provenance
# ---------------------------------------------------------------------------


def test_create_run_provenance(tmp_path):
    from src.runs.manager import create_run

    manifest = create_run(runs_dir=tmp_path, capture_provenance=True)
    meta_dir = Path(manifest["run_dir"]) / "meta"

    assert (meta_dir / "python_version.txt").exists()
    assert (meta_dir / "pip_freeze.txt").exists()


# ---------------------------------------------------------------------------
# index – append_to_index / read_index
# ---------------------------------------------------------------------------


def test_append_to_index_creates_file(tmp_path):
    from src.runs.index import append_to_index

    index_path = append_to_index(
        {"run_id": "20250101_000000_aabbccdd", "model": "xgboost"},
        runs_dir=tmp_path,
    )
    assert index_path.exists()
    assert index_path.name == "runs_index.csv"


def test_append_to_index_header(tmp_path):
    from src.runs.index import append_to_index, INDEX_COLUMNS

    append_to_index({"run_id": "r1"}, runs_dir=tmp_path)
    with open(tmp_path / "runs_index.csv", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == INDEX_COLUMNS


def test_append_to_index_multiple_rows(tmp_path):
    from src.runs.index import append_to_index, read_index

    append_to_index({"run_id": "r1", "model": "xgb", "auc_roc": "0.90"}, runs_dir=tmp_path)
    append_to_index({"run_id": "r2", "model": "lgbm", "auc_roc": "0.92"}, runs_dir=tmp_path)

    rows = read_index(runs_dir=tmp_path)
    assert len(rows) == 2
    assert rows[0]["run_id"] == "r1"
    assert rows[1]["run_id"] == "r2"
    assert rows[1]["auc_roc"] == "0.92"


def test_append_to_index_missing_cols_empty(tmp_path):
    from src.runs.index import append_to_index, read_index

    append_to_index({"run_id": "r1"}, runs_dir=tmp_path)
    rows = read_index(runs_dir=tmp_path)
    assert rows[0]["auc_roc"] == ""


def test_read_index_empty_when_no_file(tmp_path):
    from src.runs.index import read_index

    rows = read_index(runs_dir=tmp_path)
    assert rows == []


def test_append_ignores_unknown_columns(tmp_path):
    from src.runs.index import append_to_index, read_index

    # Should not raise even with an unknown key
    append_to_index(
        {"run_id": "r1", "nonexistent_column": "boom"},
        runs_dir=tmp_path,
    )
    rows = read_index(runs_dir=tmp_path)
    assert len(rows) == 1
    assert "nonexistent_column" not in rows[0]
