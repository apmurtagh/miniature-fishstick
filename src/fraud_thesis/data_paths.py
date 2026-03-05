from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ENV_VAR_IEEE_CIS_DIR = "IEEE_CIS_DIR"


@dataclass(frozen=True)
class IeeeCisPaths:
    root: Path
    train_transaction: Path
    train_identity: Path
    test_transaction: Path
    test_identity: Path
    sample_submission: Path


def get_ieee_cis_paths() -> IeeeCisPaths:
    raw = os.getenv(ENV_VAR_IEEE_CIS_DIR)
    if not raw:
        raise RuntimeError(
            f"{ENV_VAR_IEEE_CIS_DIR} is not set. Example:\n"
            f'  export {ENV_VAR_IEEE_CIS_DIR}="/mnt/Seagate Expansion Drive/DSI/MSc/ieee-fraud-detection"\n'
        )

    root = Path(raw).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"{ENV_VAR_IEEE_CIS_DIR} points to a missing path: {root}")

    paths = IeeeCisPaths(
        root=root,
        train_transaction=root / "train_transaction.csv",
        train_identity=root / "train_identity.csv",
        test_transaction=root / "test_transaction.csv",
        test_identity=root / "test_identity.csv",
        sample_submission=root / "sample_submission.csv",
    )

    missing = [p for p in (
        paths.train_transaction,
        paths.train_identity,
        paths.test_transaction,
        paths.test_identity,
        paths.sample_submission,
    ) if not p.exists()]

    if missing:
        msg = "\n".join(f"- {p}" for p in missing)
        raise RuntimeError(f"IEEE-CIS expected files not found under {root}:\n{msg}")

    return paths
