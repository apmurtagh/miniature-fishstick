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
DEFAULT_NARR_PATH = DEFAULT_RUN_DIR / "template_narratives_ops_triage_with_drivers.jsonl"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "driver_overlap_metrics.json"


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


def extract_mentions(text: str, *, candidates: List[str]) -> Set[str]:
    """
    Return the subset of `candidates` that are mentioned in `text`.

    Uses token-boundary matching to avoid substring false positives.
    """
    t = text.lower()
    mentioned: Set[str] = set()

    for feat in candidates:
        f = feat.lower()
        pat = rf"(?<![a-z0-9_]){re.escape(f)}(?![a-z0-9_])"
        if re.search(pat, t):
            mentioned.add(feat)

    return mentioned


def overlap_at_k(topk: List[str], mentioned: Set[str]) -> float:
    if not topk:
        return 0.0
    return len(set(topk) & mentioned) / float(len(topk))


def compute_metrics(eos_by_id: Dict[str, EvidenceObject], narratives_path: Path, *, k: int) -> dict:
    overlaps: List[float] = []
    mention_any: List[int] = []

    overlaps_thin: List[float] = []
    mention_any_thin: List[int] = []

    overlaps_thick: List[float] = []
    mention_any_thick: List[int] = []

    n_missing_eo = 0

    for obj in iter_jsonl(narratives_path):
        event_id = str(obj["event_id"])
        text = str(obj["text"])

        eo = eos_by_id.get(event_id)
        if eo is None:
            n_missing_eo += 1
            continue

        top_drivers = eo.top_drivers[: int(k)]
        topk_names = [d.name for d in top_drivers]

        mentioned = extract_mentions(text, candidates=topk_names)

        ov = overlap_at_k(topk_names, mentioned)
        any_hit = 1 if len(mentioned) > 0 else 0

        overlaps.append(ov)
        mention_any.append(any_hit)

        if eo.thin_file_flag:
            overlaps_thin.append(ov)
            mention_any_thin.append(any_hit)
        else:
            overlaps_thick.append(ov)
            mention_any_thick.append(any_hit)

    def summarize(xs: List[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "p50": None}
        arr = np.asarray(xs, dtype=float)
        return {"n": int(arr.size), "mean": float(arr.mean()), "p50": float(np.quantile(arr, 0.50))}

    def summarize_rate(xs: List[int]) -> dict:
        if not xs:
            return {"n": 0, "rate": None}
        arr = np.asarray(xs, dtype=float)
        return {"n": int(arr.size), "rate": float(arr.mean())}

    return {
        "schema_version": "driver_overlap_metrics_v0",
        "k": int(k),
        "n_missing_eo": int(n_missing_eo),
        "overall": {
            "overlap_at_k": summarize(overlaps),
            "mention_any_topk_rate": summarize_rate(mention_any),
        },
        "thin": {
            "overlap_at_k": summarize(overlaps_thin),
            "mention_any_topk_rate": summarize_rate(mention_any_thin),
        },
        "thick": {
            "overlap_at_k": summarize(overlaps_thick),
            "mention_any_topk_rate": summarize_rate(mention_any_thick),
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
