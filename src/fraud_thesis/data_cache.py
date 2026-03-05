from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataCachePaths:
    root: Path
    joined_train_parquet: Path
    joined_train_schema_summary: Path


def get_default_data_cache_dir(*, repo_root: Path | None = None) -> Path:
    """
    Default cache location is inside the repo under artifacts/, which is gitignored.
    """
    repo_root = repo_root or Path.cwd()
    return (repo_root / "artifacts" / "data_cache").resolve()


def get_data_cache_paths(*, repo_root: Path | None = None) -> DataCachePaths:
    root = get_default_data_cache_dir(repo_root=repo_root)
    return DataCachePaths(
        root=root,
        joined_train_parquet=root / "joined_train.parquet",
        joined_train_schema_summary=root / "joined_train_schema_summary.json",
    )


def load_joined_train_parquet(*, repo_root: Path | None = None) -> pd.DataFrame:
    """
    Load the cached joined IEEE-CIS train dataset.

    If the cache is missing, raises a RuntimeError with a friendly next-step.
    """
    paths = get_data_cache_paths(repo_root=repo_root)

    if not paths.joined_train_parquet.exists():
        raise RuntimeError(
            "Cached joined train parquet not found.\n"
            f"Expected: {paths.joined_train_parquet}\n\n"
            "Create it by running:\n"
            "  python scripts/materialize_joined_train.py\n"
        )

    return pd.read_parquet(paths.joined_train_parquet)
