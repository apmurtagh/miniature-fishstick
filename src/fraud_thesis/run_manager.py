from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ENV_VAR_RUNS_DIR = "FRAUD_THESIS_RUNS_DIR"
DEFAULT_RUNS_DIR = Path("/mnt/Seagate Expansion Drive/DSI/MSc/runs")

MANIFEST_REL_PATH = Path("meta/run_manifest.json")
GIT_META_REL_PATH = Path("meta/git.json")


def utc_now_compact() -> str:
    # Example: 2026-03-03T104730Z (filesystem safe; sortable)
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def get_runs_dir() -> Path:
    val = os.getenv(ENV_VAR_RUNS_DIR)
    if val:
        return Path(val).expanduser().resolve()
    return DEFAULT_RUNS_DIR


def slugify(s: str) -> str:
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        elif ch in (" ", "\t"):
            out.append("_")
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "x"


def make_run_id(
    *,
    split_version: str,
    model: str,
    narrative_mode: str,
    seed: int,
    eo_schema_version: str | None = None,
    narrative_schema_version: str | None = None,
    masking_rate: float | None = None,
    extra_tags: Iterable[str] = (),
    timestamp_utc: str | None = None,
) -> str:
    ts = timestamp_utc or utc_now_compact()

    parts: list[str] = [
        ts,
        f"split_{slugify(split_version)}",
        f"model_{slugify(model)}",
        f"narr_{slugify(narrative_mode)}",
        f"seed{seed}",
    ]
    if eo_schema_version:
        parts.append(f"eo_{slugify(eo_schema_version)}")
    if narrative_schema_version:
        parts.append(f"ns_{slugify(narrative_schema_version)}")
    if masking_rate is not None:
        parts.append(f"mask{str(masking_rate).replace('.', 'p')}")
    for t in extra_tags:
        parts.append(slugify(t))

    return "__".join(parts)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    meta_dir: Path
    data_dir: Path
    model_dir: Path
    eo_dir: Path
    narratives_dir: Path
    eval_dir: Path
    reports_dir: Path
    logs_dir: Path

    @staticmethod
    def from_run_dir(run_dir: Path) -> "RunPaths":
        return RunPaths(
            run_dir=run_dir,
            meta_dir=run_dir / "meta",
            data_dir=run_dir / "data",
            model_dir=run_dir / "model",
            eo_dir=run_dir / "eo",
            narratives_dir=run_dir / "narratives",
            eval_dir=run_dir / "eval",
            reports_dir=run_dir / "reports",
            logs_dir=run_dir / "logs",
        )


def create_run_dir(
    run_id: str,
    *,
    runs_dir: Path | None = None,
    allow_existing: bool = False,
) -> RunPaths:
    runs_dir = (runs_dir or get_runs_dir()).expanduser().resolve()
    date_prefix = run_id.split("T", 1)[0] if "T" in run_id else run_id[:10]
    run_dir = runs_dir / date_prefix / run_id

    if run_dir.exists() and not allow_existing:
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    paths = RunPaths.from_run_dir(run_dir)
    for d in (
        paths.meta_dir,
        paths.data_dir,
        paths.model_dir,
        paths.eo_dir,
        paths.narratives_dir,
        paths.eval_dir,
        paths.reports_dir,
        paths.logs_dir,
    ):
        d.mkdir(parents=True, exist_ok=allow_existing)

    return paths


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", "not found"


def capture_git_meta(run_dir: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = repo_root or Path.cwd()

    code, head, _ = _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    code2, porcelain, _ = _run_cmd(["git", "status", "--porcelain"], cwd=repo_root)

    meta = {
        "git_available": (code == 0),
        "commit": head if code == 0 else None,
        "dirty": (code2 == 0 and porcelain != ""),
        "status_porcelain": porcelain if code2 == 0 else None,
    }

    out_path = run_dir / GIT_META_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def capture_env_meta(run_dir: Path) -> None:
    meta_dir = run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    (meta_dir / "python_version.txt").write_text(
        f"{platform.python_version()} ({platform.python_implementation()})\n",
        encoding="utf-8",
    )

    code, out, err = _run_cmd(["python", "-m", "pip", "freeze"])
    (meta_dir / "pip_freeze.txt").write_text(
        (out if code == 0 else f"# pip freeze failed\n# {err}\n") + "\n",
        encoding="utf-8",
    )

    code, out, err = _run_cmd(["conda", "env", "export", "--from-history"])
    if code == 0 and out:
        (meta_dir / "conda_env_from_history.yml").write_text(out + "\n", encoding="utf-8")


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    path = run_dir / MANIFEST_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / MANIFEST_REL_PATH
    return json.loads(path.read_text(encoding="utf-8"))


INDEX_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "git_commit",
    "git_dirty",
    "run_dir",
    "split_version",
    "seed",
    "model",
    "eo_schema_version",
    "narrative_mode",
    "narrative_schema_version",
    "masking_rate",
    "persona",
    "metrics_path",
]


def append_to_index(
    *,
    runs_dir: Path,
    row: dict[str, Any],
    index_filename: str = "runs_index.csv",
) -> Path:
    runs_dir = runs_dir.expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    index_path = runs_dir / index_filename
    is_new = not index_path.exists()

    out_row: dict[str, str] = {}
    for c in INDEX_COLUMNS:
        v = row.get(c, "")
        out_row[c] = "" if v is None else str(v)

    with index_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow(out_row)

    return index_path
