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


def _tokenize_words(text: str) -> Set[str]:
    """
    Conservative tokenization used for leakage detection.

    We treat candidate feature names (which are usually like 'V14', 'card1', 'id_01') as "words",
    and look for exact word matches in the narrative. This avoids substring false positives.
    """
    # Keep letters/digits/underscore, treat everything else as separator.
    # This matches the same boundary assumption as the other eval scripts.
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return set(toks)


def compute_metrics(
    eos_by_id: Dict[str, EvidenceObject],
    narratives_path: Path,
    *,
    k: int,
    allowed_extra: Set[str] | None = None,
) -> dict:
    """
    Leakage definition:
      narrative_mentions_outside_topk = (# feature names mentioned that are in EO.top_drivers but not in top-k) +
                                       (# feature names mentioned that are not in EO.top_drivers at all)
    In practice, since we don't have a global feature list here, we approximate by:
      - take EO.top_drivers names as the "feature vocabulary"
      - consider it leakage if the narrative mentions any EO feature name not in EO.top_drivers[:k]
    This captures "mentioned lower-ranked drivers" leakage, which is a common failure mode for LLM narratives.

    NOTE: If you later want "mentions any model feature not in top-k", you can extend this to load
    feature_names.json from the run_dir and treat that as the vocabulary.
    """
    allowed_extra = allowed_extra or set()

    leak_any: List[int] = []
    leak_count: List[int] = []

    leak_any_thin: List[int] = []
    leak_count_thin: List[int] = []

    leak_any_thick: List[int] = []
    leak_count_thick: List[int] = []

    n_missing_eo = 0

    for obj in iter_jsonl(narratives_path):
        event_id = str(obj["event_id"])
        text = str(obj["text"])

        eo = eos_by_id.get(event_id)
        if eo is None:
            n_missing_eo += 1
            continue

        top = eo.top_drivers
        topk = top[: int(k)]

        top_names = {d.name for d in top}
        topk_names = {d.name for d in topk}

        toks = _tokenize_words(text)

        # Mentioned EO feature names (restricted vocabulary)
        mentioned = {name for name in top_names if name.lower() in toks}

        # Leakage = mentioned names that are not in top-k and not explicitly allowed
        leaked = {m for m in mentioned if (m not in topk_names) and (m not in allowed_extra)}

        leak_any_i = 1 if leaked else 0
        leak_cnt_i = len(leaked)

        leak_any.append(leak_any_i)
        leak_count.append(leak_cnt_i)

        if eo.thin_file_flag:
            leak_any_thin.append(leak_any_i)
            leak_count_thin.append(leak_cnt_i)
        else:
            leak_any_thick.append(leak_any_i)
            leak_count_thick.append(leak_cnt_i)

    def summarize_rate(xs: List[int]) -> dict:
        if not xs:
            return {"n": 0, "rate": None}
        arr = np.asarray(xs, dtype=float)
        return {"n": int(arr.size), "rate": float(arr.mean())}

    def summarize_counts(xs: List[int]) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
        arr = np.asarray(xs, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
        }

    return {
        "schema_version": "driver_leakage_metrics_v0",
        "k": int(k),
        "n_missing_eo": int(n_missing_eo),
        "definition": "Leakage = narrative mentions any EO driver name outside EO.top_drivers[:k] (restricted to EO.top_drivers vocabulary).",
        "overall": {
            "leak_any_rate": summarize_rate(leak_any),
            "leak_count": summarize_counts(leak_count),
        },
        "thin": {
            "leak_any_rate": summarize_rate(leak_any_thin),
            "leak_count": summarize_counts(leak_count_thin),
        },
        "thick": {
            "leak_any_rate": summarize_rate(leak_any_thick),
            "leak_count": summarize_counts(leak_count_thick),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos-jsonl", default=str(DEFAULT_EOS_PATH))
    ap.add_argument("--narratives-jsonl", default=str(DEFAULT_NARR_PATH))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_PATH))
    args = ap.parse_args()

    eos_by_id = load_eos(Path(args.eos_jsonl))
    metrics = compute_metrics(eos_by_id, Path(args.narratives_jsonl), k=int(args.k))

    Path(args.out_json).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Wrote:", args.out_json)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
