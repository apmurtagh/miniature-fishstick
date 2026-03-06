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

    This is the metric you’ll want for LLM narratives: it catches mention of *any* model feature
    outside the EO-provided top-k drivers.
    """
    allowed_extra = allowed_extra or set()

    vocab = list(feature_names)
    vocab_set = set(vocab)

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

        topk = eo.top_drivers[: int(k)]
        allowed = {d.name for d in topk} | set(allowed_extra)

        toks = _tokenize_words(text)

        # Mentioned model features (global vocabulary)
        mentioned = {name for name in vocab_set if name.lower() in toks}

        # Leakage = mentioned model features that are not in allowed top-k
        leaked = {m for m in mentioned if m not in allowed}

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
        "schema_version": "driver_leakage_metrics_v1_global_vocab",
        "k": int(k),
        "n_missing_eo": int(n_missing_eo),
        "feature_names_path": str(DEFAULT_FEATURE_NAMES_PATH),
        "n_vocab_features": int(len(vocab)),
        "definition": "Leakage = narrative mentions any model feature from feature_names.json that is not in EO.top_drivers[:k] (plus allowlist).",
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
    ap.add_argument("--feature-names-json", default=str(DEFAULT_FEATURE_NAMES_PATH))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_PATH))
    args = ap.parse_args()

    eos_by_id = load_eos(Path(args.eos_jsonl))
    feature_names = load_feature_names(Path(args.feature_names_json))

    metrics = compute_metrics(
        eos_by_id,
        Path(args.narratives_jsonl),
        k=int(args.k),
        feature_names=feature_names,
    )

    Path(args.out_json).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Wrote:", args.out_json)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
