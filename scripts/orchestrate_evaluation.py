from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"
DEFAULT_NARRATIVES_PATH = DEFAULT_RUN_DIR / "narratives_ops_triage_template.jsonl"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "eval_summary_template.json"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalise_text(text: str) -> str:
    return text.lower().strip()


def contains_term(text: str, term: str) -> bool:
    """
    Conservative mention check:
    - case-insensitive
    - word-boundary based where possible
    """
    text_norm = normalise_text(text)
    term_norm = term.lower().strip()
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(term_norm)}(?![A-Za-z0-9_])"
    return re.search(pattern, text_norm) is not None


def extract_top_driver_names(eo_obj: dict, k: int) -> List[str]:
    top_drivers = eo_obj.get("top_drivers") or []
    if k > 0:
        top_drivers = top_drivers[:k]
    return [d.get("name") for d in top_drivers if d.get("name")]


def compute_overlap_metrics(
    eo_map: Dict[str, dict],
    narrative_rows: List[dict],
    *,
    top_k: int,
) -> Dict[str, float]:
    total = 0
    any_driver_mentioned = 0
    all_topk_mentioned = 0
    avg_overlap = 0.0
    eos_with_nonempty_topdrivers = 0

    for row in narrative_rows:
        event_id = str(row["event_id"])
        eo = eo_map.get(event_id)
        if eo is None:
            continue

        total += 1
        text = row.get("text", "")
        top_names = extract_top_driver_names(eo, top_k)

        if top_names:
            eos_with_nonempty_topdrivers += 1

        if not top_names:
            continue

        mentioned = [name for name in top_names if contains_term(text, name)]
        overlap = len(mentioned) / len(top_names)

        if mentioned:
            any_driver_mentioned += 1
        if len(mentioned) == len(top_names):
            all_topk_mentioned += 1

        avg_overlap += overlap

    return {
        "rows_compared": total,
        "rows_with_nonempty_top_drivers": eos_with_nonempty_topdrivers,
        "pct_any_topk_driver_mentioned": (any_driver_mentioned / total) if total else 0.0,
        "pct_all_topk_drivers_mentioned": (all_topk_mentioned / total) if total else 0.0,
        "avg_topk_overlap": (avg_overlap / total) if total else 0.0,
    }


def compute_required_phrase_metrics(
    eo_map: Dict[str, dict],
    narrative_rows: List[dict],
) -> Dict[str, float]:
    total = 0
    risk_band_mentioned = 0
    action_field_present = 0
    thinfile_disclosure_expected = 0
    thinfile_disclosure_present = 0

    for row in narrative_rows:
        event_id = str(row["event_id"])
        eo = eo_map.get(event_id)
        if eo is None:
            continue

        total += 1
        text = row.get("text", "")
        text_norm = normalise_text(text)

        risk_band = eo.get("risk_band")
        recommended_action = eo.get("recommended_action_class")

        # EO fields may already be strings depending on serialization
        if isinstance(risk_band, dict):
            risk_band_val = risk_band.get("value")
        else:
            risk_band_val = risk_band

        if isinstance(recommended_action, dict):
            action_val = recommended_action.get("value")
        else:
            action_val = recommended_action

        if risk_band_val and contains_term(text_norm, str(risk_band_val)):
            risk_band_mentioned += 1

        if action_val and row.get("recommended_action_class"):
            action_field_present += 1

        evidence_strength = eo.get("evidence_strength")
        if isinstance(evidence_strength, dict):
            evidence_strength_val = evidence_strength.get("value")
        else:
            evidence_strength_val = evidence_strength

        thin_file_flag = eo.get("thin_file_flag", False)

        if thin_file_flag or str(evidence_strength_val).upper() == "LOW":
            thinfile_disclosure_expected += 1
            if (
                "thin-file" in text_norm
                or "low evidence" in text_norm
                or "evidence is limited" in text_norm
            ):
                thinfile_disclosure_present += 1

    return {
        "rows_compared": total,
        "pct_risk_band_mentioned_in_text": (risk_band_mentioned / total) if total else 0.0,
        "pct_recommended_action_field_present": (action_field_present / total) if total else 0.0,
        "rows_thinfile_disclosure_expected": thinfile_disclosure_expected,
        "pct_thinfile_disclosure_present_when_expected": (
            thinfile_disclosure_present / thinfile_disclosure_expected
            if thinfile_disclosure_expected
            else 0.0
        ),
    }


def build_summary(
    eo_rows: List[dict],
    narrative_rows: List[dict],
    *,
    eos_path: Path,
    narratives_path: Path,
    top_k: int,
) -> dict:
    eo_map = {str(row["event_id"]): row for row in eo_rows}
    narrative_event_ids = {str(row["event_id"]) for row in narrative_rows}
    eo_event_ids = set(eo_map.keys())
    matched_event_ids = eo_event_ids & narrative_event_ids

    overlap_metrics = compute_overlap_metrics(
        eo_map=eo_map,
        narrative_rows=narrative_rows,
        top_k=top_k,
    )

    phrase_metrics = compute_required_phrase_metrics(
        eo_map=eo_map,
        narrative_rows=narrative_rows,
    )

    return {
        "summary_version": "eval_summary_template_v1",
        "inputs": {
            "eos_jsonl": str(eos_path),
            "narratives_jsonl": str(narratives_path),
            "top_k": top_k,
        },
        "row_counts": {
            "eo_rows": len(eo_rows),
            "narrative_rows": len(narrative_rows),
            "matched_event_ids": len(matched_event_ids),
        },
        "overlap_metrics": overlap_metrics,
        "required_phrase_metrics": phrase_metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate enriched template narratives against enriched EO JSONL."
    )
    ap.add_argument(
        "--eos-jsonl",
        default=str(DEFAULT_EOS_PATH),
        help="Path to enriched EO JSONL.",
    )
    ap.add_argument(
        "--narratives-jsonl",
        default=str(DEFAULT_NARRATIVES_PATH),
        help="Path to generated narrative JSONL.",
    )
    ap.add_argument(
        "--out-json",
        default=str(DEFAULT_OUT_PATH),
        help="Path to evaluation summary JSON.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K EO drivers to evaluate for mention overlap.",
    )
    args = ap.parse_args()

    eos_path = Path(args.eos_jsonl)
    narratives_path = Path(args.narratives_jsonl)
    out_path = Path(args.out_json)

    if not eos_path.exists():
        raise FileNotFoundError(
            f"EO file not found: {eos_path}\n"
            f"Run upstream steps first:\n"
            f"  PYTHONPATH=src python scripts/emit_eos_v1.py\n"
            f"  PYTHONPATH=src python scripts/attach_shap_drivers_v1.py"
        )

    if not narratives_path.exists():
        raise FileNotFoundError(
            f"Narratives file not found: {narratives_path}\n"
            f"Run upstream step first:\n"
            f"  PYTHONPATH=src python scripts/orchestrate_narrative_experiments.py"
        )

    eo_rows = list(iter_jsonl(eos_path))
    narrative_rows = list(iter_jsonl(narratives_path))

    summary = build_summary(
        eo_rows=eo_rows,
        narrative_rows=narrative_rows,
        eos_path=eos_path,
        narratives_path=narratives_path,
        top_k=int(args.top_k),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Reading EO from: {eos_path}")
    print(f"Reading narratives from: {narratives_path}")
    print(f"Wrote evaluation summary to: {out_path}")
    print(json.dumps(summary["row_counts"], indent=2))
    print(json.dumps(summary["overlap_metrics"], indent=2))
    print(json.dumps(summary["required_phrase_metrics"], indent=2))


if __name__ == "__main__":
    main()
