from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from eo.evidence_object import EvidenceObject
from src.narratives.template_engine import generate_template_narrative


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test.jsonl"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "template_narratives_ops_triage.jsonl"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos-jsonl", default=str(DEFAULT_EOS_PATH))
    ap.add_argument("--out-jsonl", default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--persona", default="ops_triage", choices=["ops_triage"])
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many EO top_drivers to mention (0 = mention all available).",
    )
    args = ap.parse_args()

    eos_path = Path(args.eos_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
