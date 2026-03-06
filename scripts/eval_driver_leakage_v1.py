from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

from eo.evidence_object import EvidenceObject


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"
DEFAULT_NARR_PATH = DEFAULT_RUN_DIR / "template_narratives_ops_triage_with_drivers_top5.jsonl"
DEFAULT_FEATURE_NAMES_PATH = DEFAULT_RUN_DIR / "feature_names.json"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "driver_leakage_metrics_top5.json"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_eos(path: Path) -> Dict[str, EvidenceObject]:
    out: Dict[str, EvidenceObject] = {}
    for obj in iter_jsonl(path):
        eo = EvidenceObject.model_validate(obj)
        out[eo.event_id] = eo
    return out


def load_feature_names(path: Path) -> List[str]:
    """
    feature_names.json is expected to be a JSON list of strings.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature names file: {path}\n"
            "Expected artifacts/baselines/<run>/feature_names.json to exist.\n"
        )
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise RuntimeError(f"Expected {path} to be a JSON list[str], got: {type(obj)}")
    return list(obj)


def _tokenize_words(text: str) -> Set[str]:
    """
    Conservative tokenization used for leakage detection.

    We treat model feature names (like 'V14', 'card1', 'id_01') as "words",
    and look for exact word matches in the narrative. This avoids substring false positives.
    """
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return set(toks)


def compute_metrics(
    eos_by_id: Dict[str, EvidenceObject],
    narratives_path: Path,
    *,
    k: int,
    feature_names: List[str],
    allowed_extra: Set[str] | None = None,
) -> dict:
    """
    Leakage definition (global-vocabulary):
      - Let V be the global model feature vocabulary (feature_names.json).
      - Let A be the allowed set for a specific EO = EO.top_drivers[:k] (plus optional allowlist).
      - Let M be the set of features from V that are mentioned in the narrative text.
      - Leakage set L = M \\ A
      - leak_any = 1 if L non-empty else 0
      - leak_count = |L|

    This catches mention of *any* model feature outside the EO-provided top-k drivers.
    """
    allowed_extra = allowed_extra or set()

    vocab = list(feature_names)

