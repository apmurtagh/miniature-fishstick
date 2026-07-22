from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"

DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"
DEFAULT_TEMPLATE_PATH = DEFAULT_RUN_DIR / "narratives_ops_triage_template.jsonl"
DEFAULT_LLM_PATH = DEFAULT_RUN_DIR / "narratives_ops_triage_llm_5753rows_backup.jsonl"

DEFAULT_OUT_JSON = DEFAULT_RUN_DIR / "sanity_check_summary.json"
DEFAULT_OUT_MD = DEFAULT_RUN_DIR / "sanity_check_summary.md"
DEFAULT_OUT_CSV = DEFAULT_RUN_DIR / "sanity_check_detail.csv"


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_key(row):
    return str(row.get("event_id"))


def normalise(text):
    return (text or "").lower().strip()


def contains_term(text, term):
    text_norm = normalise(text)
    term_norm = normalise(term)
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(term_norm) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text_norm) is not None


def top_driver_names(eo, top_k):
    drivers = eo.get("top_drivers") or []
    if top_k > 0:
        drivers = drivers[:top_k]
    return [d.get("name") for d in drivers if d.get("name")]


def overlap_against_names(text, names):
    if not names:
        return {
            "count": 0,
            "total": 0,
            "overlap": 0.0,
            "mentioned": [],
        }

    mentioned = [n for n in names if contains_term(text, n)]
    return {
        "count": len(mentioned),
        "total": len(names),
        "overlap": len(mentioned) / len(names),
        "mentioned": mentioned,
    }


def build_randomised_driver_map(eo_map, matched_ids, seed, top_k):
    """
    Randomisation sanity check:
    - Keep narrative text fixed.
    - Replace each event's EO top-driver list with another event's top-driver list.
    - Faithful narratives should show materially lower overlap under this mismatch.
    """
    rng = random.Random(seed)

    ids = list(matched_ids)
    shuffled = ids[:]

    # Ensure we do not accidentally keep the identical assignment where possible.
    for _ in range(20):
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(ids, shuffled)):
            break

    random_driver_map = {}
    for original_id, donor_id in zip(ids, shuffled):
        donor_eo = eo_map[donor_id]
        random_driver_map[original_id] = top_driver_names(donor_eo, top_k)

    return random_driver_map


def evaluate_condition(name, eo_map, narrative_map, top_k, seed):
    matched_ids = sorted(set(eo_map) & set(narrative_map))
    random_driver_map = build_randomised_driver_map(
        eo_map=eo_map,
        matched_ids=matched_ids,
        seed=seed,
        top_k=top_k,
    )

    detail_rows = []

    real_any = 0
    real_all = 0
    real_overlap_sum = 0.0

    random_any = 0
    random_all = 0
    random_overlap_sum = 0.0

    compared = 0

    for event_id in matched_ids:
        eo = eo_map[event_id]
        narrative = narrative_map[event_id]
        text = narrative.get("text", "")

        real_names = top_driver_names(eo, top_k)
        random_names = random_driver_map[event_id]

        real = overlap_against_names(text, real_names)
        rand = overlap_against_names(text, random_names)

        compared += 1

        if real["count"] > 0:
            real_any += 1
        if real["total"] > 0 and real["count"] == real["total"]:
            real_all += 1
        real_overlap_sum += real["overlap"]

        if rand["count"] > 0:
            random_any += 1
        if rand["total"] > 0 and rand["count"] == rand["total"]:
            random_all += 1
        random_overlap_sum += rand["overlap"]

        detail_rows.append({
            "condition": name,
            "event_id": event_id,
            "real_overlap": real["overlap"],
            "randomised_overlap": rand["overlap"],
            "real_drivers": ", ".join(real_names),
            "real_mentioned": ", ".join(real["mentioned"]),
            "randomised_drivers": ", ".join(random_names),
            "randomised_mentioned": ", ".join(rand["mentioned"]),
        })

    if compared == 0:
        return {
            "condition": name,
            "rows_compared": 0,
            "real": {},
            "randomised": {},
            "degradation": {},
        }, detail_rows

    real_avg = real_overlap_sum / compared
    random_avg = random_overlap_sum / compared
    degradation_abs = real_avg - random_avg
    degradation_pct = degradation_abs / real_avg if real_avg else 0.0

    summary = {
        "condition": name,
        "rows_compared": compared,
        "real": {
            "pct_any_topk_driver_mentioned": real_any / compared,
            "pct_all_topk_drivers_mentioned": real_all / compared,
            "avg_topk_overlap": real_avg,
        },
        "randomised": {
            "pct_any_topk_driver_mentioned": random_any / compared,
            "pct_all_topk_drivers_mentioned": random_all / compared,
            "avg_topk_overlap": random_avg,
        },
        "degradation": {
            "avg_topk_overlap_abs_drop": degradation_abs,
            "avg_topk_overlap_pct_drop_vs_real": degradation_pct,
        },
    }

    return summary, detail_rows


def pct(x):
    return f"{100*x:.1f}%"


def main():
    ap = argparse.ArgumentParser(
        description="Run randomisation sanity checks for EO-grounded narratives."
    )
    ap.add_argument("--eos-jsonl", default=str(DEFAULT_EOS_PATH))
    ap.add_argument("--template-jsonl", default=str(DEFAULT_TEMPLATE_PATH))
    ap.add_argument("--llm-jsonl", default=str(DEFAULT_LLM_PATH))
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eos_path = Path(args.eos_jsonl)
    template_path = Path(args.template_jsonl)
    llm_path = Path(args.llm_jsonl)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    print("Reading EO from:", eos_path)
    print("Reading template narratives from:", template_path)
    print("Reading LLM narratives from:", llm_path)

    eos = load_jsonl(eos_path)
    template_rows = load_jsonl(template_path)
    llm_rows = load_jsonl(llm_path)

    eo_map = {row_key(r): r for r in eos}
    template_map = {row_key(r): r for r in template_rows}
    llm_map = {row_key(r): r for r in llm_rows}

    template_summary, template_detail = evaluate_condition(
        name="template",
        eo_map=eo_map,
        narrative_map=template_map,
        top_k=int(args.top_k),
        seed=int(args.seed),
    )

    llm_summary, llm_detail = evaluate_condition(
        name="llm_5753",
        eo_map=eo_map,
        narrative_map=llm_map,
        top_k=int(args.top_k),
        seed=int(args.seed),
    )

    summary = {
        "summary_version": "randomisation_sanity_check_v1",
        "inputs": {
            "eos_jsonl": str(eos_path),
            "template_jsonl": str(template_path),
            "llm_jsonl": str(llm_path),
            "top_k": int(args.top_k),
            "seed": int(args.seed),
        },
        "conditions": [
            template_summary,
            llm_summary,
        ],
        "interpretation": (
            "Faithful narratives should show materially lower driver overlap when EO top-driver lists "
            "are randomised across events. A large drop in randomised overlap provides a sanity-check "
            "that the original overlap metrics are not artefacts of generic wording alone."
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    detail_rows = template_detail + llm_detail
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "condition",
            "event_id",
            "real_overlap",
            "randomised_overlap",
            "real_drivers",
            "real_mentioned",
            "randomised_drivers",
            "randomised_mentioned",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Randomisation Sanity Check Summary\n\n")
        f.write("This sanity check keeps the narrative text fixed but randomises EO top-driver lists across events. ")
        f.write("Faithful narratives should show materially lower overlap when evaluated against mismatched driver evidence.\n\n")

        f.write("## Inputs\n\n")
        f.write(f"- EO file: `{eos_path}`\n")
        f.write(f"- Template narratives: `{template_path}`\n")
        f.write(f"- LLM narratives: `{llm_path}`\n")
        f.write(f"- Top-k: `{args.top_k}`\n")
        f.write(f"- Seed: `{args.seed}`\n\n")

        f.write("## Results\n\n")
        f.write("| Condition | Rows | Real avg overlap | Randomised avg overlap | Absolute drop | Relative drop |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for condition in summary["conditions"]:
            real_avg = condition["real"]["avg_topk_overlap"]
            rand_avg = condition["randomised"]["avg_topk_overlap"]
            abs_drop = condition["degradation"]["avg_topk_overlap_abs_drop"]
            pct_drop = condition["degradation"]["avg_topk_overlap_pct_drop_vs_real"]
            f.write(
                f"| {condition['condition']} | {condition['rows_compared']} | "
                f"{real_avg:.3f} | {rand_avg:.3f} | {abs_drop:.3f} | {pct(pct_drop)} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write(
            "The randomised-driver condition is expected to reduce overlap materially. "
            "A substantial drop indicates that the original narrative-driver overlap is linked to the correct EO evidence rather than generic reuse of common feature names. "
            "This provides a lightweight sanity check supporting the faithfulness interpretation of the main evaluation metrics.\n"
        )

    print("\n=== Randomisation Sanity Check Complete ===\n")
    print(json.dumps(summary, indent=2))
    print("\nWrote JSON:", out_json)
    print("Wrote Markdown:", out_md)
    print("Wrote CSV:", out_csv)


if __name__ == "__main__":
    main()
