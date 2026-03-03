from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from fraud_thesis.run_manager import (
    ENV_VAR_RUNS_DIR,
    append_to_index,
    create_run_dir,
    load_manifest,
    make_run_id,
    write_manifest,
)


def test_make_run_id_format():
    run_id = make_run_id(
        split_version="v1",
        model="lgbm_v1",
        narrative_mode="constrained",
        seed=42,
        eo_schema_version="eo_v1",
        narrative_schema_version="narr_v1",
        masking_rate=0.2,
        extra_tags=("demo",),
        timestamp_utc="2026-03-03T104730Z",
    )
    assert run_id.startswith("2026-03-03T104730Z__")
    assert "split_v1" in run_id
    assert "model_lgbm_v1" in run_id
    assert "narr_constrained" in run_id
    assert "seed42" in run_id
    assert "eo_eo_v1" in run_id
    assert "ns_narr_v1" in run_id
    assert "mask0p2" in run_id
    assert run_id.endswith("__demo")


def test_create_run_dir_creates_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_VAR_RUNS_DIR, str(tmp_path))

    run_id = "2026-03-03T104730Z__split_v1__model_x__narr_y__seed1"
    paths = create_run_dir(run_id)

    assert paths.run_dir.exists()
    assert (paths.run_dir / "meta").is_dir()
    assert (paths.run_dir / "data").is_dir()
    assert (paths.run_dir / "model").is_dir()
    assert (paths.run_dir / "eo").is_dir()
    assert (paths.run_dir / "narratives").is_dir()
    assert (paths.run_dir / "eval").is_dir()
    assert (paths.run_dir / "reports").is_dir()
    assert (paths.run_dir / "logs").is_dir()


def test_manifest_roundtrip(tmp_path: Path):
    run_dir = tmp_path / "r"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"run_id": "x", "params": {"seed": 1}}
    write_manifest(run_dir, manifest)

    loaded = load_manifest(run_dir)
    assert loaded["run_id"] == "x"
    assert loaded["params"]["seed"] == 1


def test_append_to_index(tmp_path: Path):
    row = {
        "run_id": "r1",
        "timestamp_utc": "2026-03-03T104730Z",
        "git_commit": "abc",
        "git_dirty": False,
        "run_dir": "/tmp/r1",
        "split_version": "v1",
        "seed": 1,
        "model": "m",
        "eo_schema_version": "eo_v1",
        "narrative_mode": "templates",
        "narrative_schema_version": "n_v1",
        "masking_rate": 0.2,
        "persona": "p",
        "metrics_path": "eval/metrics.json",
    }

    index_path = append_to_index(runs_dir=tmp_path, row=row)
    assert index_path.exists()

    with index_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["metrics_path"] == "eval/metrics.json"
