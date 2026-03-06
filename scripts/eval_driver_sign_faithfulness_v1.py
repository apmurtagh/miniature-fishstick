from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np

from eo.evidence_object import EvidenceObject


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"
DEFAULT_NARR_PATH = DEFAULT_RUN_DIR / "template_narratives_ops_triage_with_drivers_top5.jsonl"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "driver_sign_metrics_top5.json"


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


_TOKEN_BOUNDARY = r"(?<![a-z0-9_]){tok}(?![a-z0-9_])"


def extract_list_after_label(text: str, label: str) -> str:
    """
    Extract the substring after `label` up to the next period.
    Example: "Top risk-increasing drivers: A, B, C. Foo bar"
             -> "A, B, C"
    Returns "" if label not found.
    """
    t = text
    i = t.find(label)
    if i < 0:
        return ""
    rest = t[i + len(label) :]
    j = rest.find(".")
    if j < 0:
        return rest.strip()
    return rest[:j].strip()


def extract_mentions_from_segment(segment: str, candidates: List[str]) -> Set[str]:
    """
    Return candidate features mentioned in `segment` using token-boundary regex.
    """
    seg = segment.lower()
    out: Set[str] = set()
    for feat in candidates:
        tok = re.escape(feat.lower())
        pat = _TOKEN_BOUNDARY.format(tok=tok)
        if re.search(pat, seg):
            out.add(feat)
    return out


def compute_sign_metrics(
    eos_by_id: Dict[str, EvidenceObject],
    narratives_path: Path,
    *,
    k: int,
    inc_label: str = "Top risk-increasing drivers:",
    dec_label: str = "Top risk-decreasing drivers:",
) -> dict:
    per_row_accuracy: List[float] = []
    any_error: List[int] = []

    per_row_accuracy_thin: List[float] = []
    any_error_thin: List[int] = []

    per_row_accuracy_thick: List[float] = []
    any_error_thick: List[int] = []

    n_missing_eo = 0
    n_no_sign_sections = 0

    for obj in iter_jsonl(narratives_path):
        event_id = str(obj["event_id"])
        text = str(obj["text"])

        eo = eos_by_id.get(event_id)
        if eo is None:
            n_missing_eo += 1
            continue

        top = eo.top_drivers[: int(k)]
        names = [d.name for d in top]
        expected: Dict[str, str] = {d.name: d.direction for d in top}

        inc_seg = extract_list_after_label(text, inc_label)
        dec_seg = extract_list_after_label(text, dec_label)

        if not inc_seg and not dec_seg:
            n_no_sign_sections += 1

        mentioned_inc = extract_mentions_from_segment(inc_seg, names) if inc_seg else set()
        mentioned_dec = extract_mentions_from_segment(dec_seg, names) if dec_seg else set()

        # If a feature is mentioned in both buckets, that's automatically an error for that feature.
        both = mentioned_inc & mentioned_dec

        mentioned_all = mentioned_inc | mentioned_dec
        if not mentioned_all:
            # No polarity claims were made about the top-k drivers; define accuracy as 1.0 only if
            # there were no sign sections (otherwise this is "missing"), but keep it simple:
            # treat as NaN and exclude from mean.
            acc = float("nan")
            err = 0
        else:
            correct = 0
            total = 0
            err = 0

            for feat in sorted(mentioned_all):
                total += 1
                if feat in both:
                    err = 1
                    continue

                exp = expected.get(feat)
                if exp == "+" and feat in mentioned_inc:
                    correct += 1
                elif exp == "-" and feat in mentioned_dec:
                    correct += 1
                else:
                    err = 1

            acc = correct / float(total) if total else float("nan")

        if not np.isnan(acc):
            per_row_accuracy.append(float(acc))
        any_error.append(int(err))

        if eo.thin_file_flag:
            if not np.isnan(acc):
                per_row_accuracy_thin.append(float(acc))
            any_error_thin.append(int(err))
        else:
            if not np.isnan(acc):
                per_row_accuracy_thick.append(float(acc))
            any_error_thick.append(int(err))

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
        "schema_version": "driver_sign_metrics_v0",
        "k": int(k),
        "n_missing_eo": int(n_missing_eo),
        "n_no_sign_sections": int(n_no_sign_sections),
        "overall": {
            "per_row_sign_accuracy": summarize(per_row_accuracy),
            "any_sign_error_rate": summarize_rate(any_error),
        },
        "thin": {
            "per_row_sign_accuracy": summarize(per_row_accuracy_thin),
            "any_sign_error_rate": summarize_rate(any_error_thin),
        },
        "thick": {
            "per_row_sign_accuracy": summarize(per_row_accuracy_thick),
            "any_sign_error_rate": summarize_rate(any_error_thick),
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
    metrics = compute_sign_metrics(eos_by_id, Path(args.narratives_jsonl), k=int(args.k))

    Path(args.out_json).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Wrote:", args.out_json)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
