from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np

from eo.evidence_object import EvidenceObject

print("Leakage eval script started.", flush=True)

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
    """Conservative tokenization used for leakage detection.
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
    """Leakage definition (global-vocabulary):
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

    results = []
    for narr in iter_jsonl(narratives_path):
        event_id = str(narr.get("event_id"))
        eo = eos_by_id.get(event_id)
        if not eo:
            results.append({
                "event_id": event_id,
                "leak_any": None,
                "leak_count": None,
                "leak_set": [],
                "reason": "eo_not_found"
            })
            continue

        # Accept both ['drivers'] from template and ['drivers_used'] from openai
        allowed = set()
        if "drivers" in narr:
            allowed.update(narr["drivers"])
        if "drivers_used" in narr:
            allowed.update(narr["drivers_used"])
        if eo.top_drivers:
            allowed.update([d.get("name") for d in eo.top_drivers[:k] if isinstance(d, dict) and d.get("name")])
        allowed.update(allowed_extra)

        text = narr.get("text", "")
        tokens = _tokenize_words(text)
        mentioned = set([f for f in vocab if f.lower() in tokens])

        leakage_set = [f for f in mentioned if f not in allowed]
        results.append({
            "event_id": event_id,
            "leak_any": int(bool(leakage_set)),
            "leak_count": len(leakage_set),
            "leak_set": leakage_set,
            "allowed": sorted(allowed),
            "mentioned": sorted(mentioned)
        })

    # Aggregate
    num = len(results)
    leak_any = sum(1 for r in results if r["leak_any"] == 1)
    leak_count = sum(r["leak_count"] or 0 for r in results if r["leak_count"] is not None)
    metrics = {
        "num": num,
        "leak_any_count": leak_any,
        "leak_any_rate": leak_any / num if num else None,
        "total_leak_count": leak_count,
        "mean_leak_count": leak_count / num if num else None,
        "results": results,
    }
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos-jsonl", default=str(DEFAULT_EOS_PATH))
    ap.add_argument("--narratives-jsonl", default=str(DEFAULT_NARR_PATH))
    ap.add_argument("--feature-names-json", default=str(DEFAULT_FEATURE_NAMES_PATH))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_PATH))
    args = ap.parse_args()

    eos = load_eos(Path(args.eos_jsonl))
    print(f"Loaded {len(eos)} EOs", flush=True)
    narratives_path = Path(args.narratives_jsonl)

    feature_names = load_feature_names(Path(args.feature_names_json))
    print(f"Loaded {len(feature_names)} feature names", flush=True)

    metrics = compute_metrics(
        eos_by_id=eos,
        narratives_path=narratives_path,
        k=args.k,
        feature_names=feature_names,
        allowed_extra=None
    )

    print(f"Writing metrics to {args.out_json}", flush=True)
    Path(args.out_json).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()