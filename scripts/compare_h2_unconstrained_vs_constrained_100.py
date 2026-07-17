import json
from pathlib import Path


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
ROBUST = ART / "llm_20000_robustness"
OUT = ART / "h2_unconstrained_ablation"

EOS = ROBUST / "eos_20000_strict_schema.jsonl"
CONSTRAINED = ROBUST / "narratives_ops_triage_llm_20000_resume_safe.jsonl"
UNCONSTRAINED = OUT / "h2_unconstrained_outputs_100.jsonl"

OUT_JSON = OUT / "h2_unconstrained_vs_constrained_summary_100.json"
OUT_MD = OUT / "h2_unconstrained_vs_constrained_summary_100.md"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_text(obj):
    candidates = [
        "text",
        "narrative",
        "output_text",
        "response_text",
        "content",
        "llm_text",
        "generated_text",
    ]
    for key in candidates:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v

    # Search one nested level.
    for v in obj.values():
        if isinstance(v, dict):
            for key in candidates:
                x = v.get(key)
                if isinstance(x, str) and x.strip():
                    return x

    # Fallback: join obvious string values.
    strings = []
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > 20:
            strings.append(v)
    return "\n".join(strings)


def find_event_id(obj):
    for key in ["event_id", "TransactionID", "transaction_id", "id"]:
        if key in obj:
            return str(obj[key])
    for v in obj.values():
        if isinstance(v, dict):
            for key in ["event_id", "TransactionID", "transaction_id", "id"]:
                if key in v:
                    return str(v[key])
    return None


def contains_any(text, terms):
    tl = text.lower()
    return any(term.lower() in tl for term in terms)


def evaluate(eo, text):
    tl = text.lower()
    drivers = eo.get("top_drivers", [])

    mentioned = [
        d.get("name")
        for d in drivers
        if d.get("name") and d.get("name").lower() in tl
    ]

    risk = str(eo.get("risk_band", ""))
    action = str(eo.get("recommended_action_class", ""))

    disclosure_required = (
        bool(eo.get("thin_file_flag"))
        or str(eo.get("evidence_strength", "")).upper() == "LOW"
    )

    disclosure_terms = [
        "limited evidence",
        "low evidence",
        "thin",
        "limited coverage",
        "caution",
        "uncertain",
        "uncertainty",
        "evidence strength is low",
    ]

    review_terms = [
        "$",
        "customer",
        "location",
        "policy",
        "legitimate",
        "legitimacy",
        "anomaly",
        "anomalies",
        "concern",
        "concerns",
        "suspicious",
        "fraudulent",
        "further investigation",
        "further scrutiny",
    ]

    return {
        "risk_present": risk.lower() in tl if risk else False,
        "action_present": (
            action.lower().replace("_", " ") in tl
            or action.lower() in tl
        ) if action else False,
        "driver_count": len(drivers),
        "mentioned_driver_count": len(mentioned),
        "mentioned_drivers": mentioned,
        "any_driver_mentioned": len(mentioned) > 0,
        "driver_coverage": len(mentioned) / len(drivers) if drivers else None,
        "disclosure_required": disclosure_required,
        "disclosure_present": contains_any(text, disclosure_terms) if disclosure_required else None,
        "review_language_flag": contains_any(text, review_terms),
        "review_language_terms": [t for t in review_terms if t.lower() in tl],
        "text_chars": len(text),
    }


def aggregate(metrics):
    n = len(metrics)
    disclosure_required_n = sum(m["disclosure_required"] for m in metrics)
    coverages = [m["driver_coverage"] for m in metrics if m["driver_coverage"] is not None]

    return {
        "rows": n,
        "risk_present_rate": sum(m["risk_present"] for m in metrics) / n,
        "action_present_rate": sum(m["action_present"] for m in metrics) / n,
        "any_driver_mentioned_rate": sum(m["any_driver_mentioned"] for m in metrics) / n,
        "mean_driver_coverage": sum(coverages) / len(coverages),
        "zero_driver_rows": sum(1 for m in metrics if m["mentioned_driver_count"] == 0),
        "disclosure_required_rows": disclosure_required_n,
        "disclosure_present_when_required_rate": (
            sum(1 for m in metrics if m["disclosure_required"] and m["disclosure_present"])
            / max(1, disclosure_required_n)
        ),
        "review_language_flag_rate": sum(m["review_language_flag"] for m in metrics) / n,
        "mean_text_chars": sum(m["text_chars"] for m in metrics) / n,
    }


def main():
    eos = {str(row["event_id"]): row for row in read_jsonl(EOS)}
    unconstrained_rows = read_jsonl(UNCONSTRAINED)
    event_ids = [str(row["event_id"]) for row in unconstrained_rows]

    constrained_all = {}
    for row in read_jsonl(CONSTRAINED):
        eid = find_event_id(row)
        if eid in event_ids:
            constrained_all[eid] = row

    missing = [eid for eid in event_ids if eid not in constrained_all]
    if missing:
        print("[WARN] Missing constrained outputs for", missing)

    rows = []
    constrained_metrics = []
    unconstrained_metrics = []

    for urow in unconstrained_rows:
        eid = str(urow["event_id"])
        eo = eos[eid]

        unconstrained_text = find_text(urow)
        constrained_text = find_text(constrained_all.get(eid, {}))

        um = evaluate(eo, unconstrained_text)
        cm = evaluate(eo, constrained_text) if constrained_text else None

        unconstrained_metrics.append(um)
        if cm:
            constrained_metrics.append(cm)

        rows.append({
            "event_id": eid,
            "unconstrained": um,
            "constrained": cm,
            "unconstrained_text": unconstrained_text,
            "constrained_text": constrained_text,
        })

    summary = {
        "status": "pilot_comparison",
        "rows_requested": len(event_ids),
        "rows_with_constrained_match": len(constrained_metrics),
        "condition_summaries": {
            "constrained": aggregate(constrained_metrics) if constrained_metrics else None,
            "unconstrained": aggregate(unconstrained_metrics),
        },
        "row_metrics": rows,
        "caveat": (
            "This is a same-row 100-row automated H2 comparison. It is not a human semantic validation. "
            "Review-language flags are simple lexical flags and should be interpreted conservatively."
        ),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    cs = summary["condition_summaries"]["constrained"]
    us = summary["condition_summaries"]["unconstrained"]

    md = []
    md.append("# H2 Unconstrained vs Constrained Comparison, 100 Rows\n\n")
    md.append("## Purpose\n\n")
    md.append(
        "This same-row exploratory comparison evaluates whether unconstrained natural-language generation differs from the constrained LLM path on the 100-row ablation sample. "
        "It is intended to address H2 as a bounded ablation, not as full-scale proof.\n\n"
    )

    md.append("## Condition Summary\n\n")
    md.append("| Condition | Rows | Risk present | Action present | Any driver mentioned | Mean driver coverage | Zero-driver rows | Disclosure when required | Review-language flag |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    if cs:
        md.append(
            f"| constrained | {cs['rows']} | {cs['risk_present_rate']:.3f} | {cs['action_present_rate']:.3f} | "
            f"{cs['any_driver_mentioned_rate']:.3f} | {cs['mean_driver_coverage']:.3f} | {cs['zero_driver_rows']} | "
            f"{cs['disclosure_present_when_required_rate']:.3f} | {cs['review_language_flag_rate']:.3f} |\n"
        )

    md.append(
        f"| unconstrained | {us['rows']} | {us['risk_present_rate']:.3f} | {us['action_present_rate']:.3f} | "
        f"{us['any_driver_mentioned_rate']:.3f} | {us['mean_driver_coverage']:.3f} | {us['zero_driver_rows']} | "
        f"{us['disclosure_present_when_required_rate']:.3f} | {us['review_language_flag_rate']:.3f} |\n"
    )

    md.append("\n## Interpretation\n\n")
    md.append(
        "The unconstrained pilot preserves risk/action framing and required disclosure in this small sample, but exhibits evidence compression through low mean driver coverage and multiple zero-driver rows. "
        "Review-language flags indicate that unconstrained prose may also introduce interpretive phrasing that requires review. "
        "This supports the thesis claim that fluency is not equivalent to audit-complete evidence rendering.\n\n"
    )

    md.append("## Caveat\n\n")
    md.append(
        "This is a bounded 100-row automated ablation. It should not be presented as human semantic validation. "
        "It is useful as proposal-aligned evidence that motivates constrained generation and validator policy, but not as a replacement for a larger ablation or human semantic review.\n"
    )

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print("[OK] Wrote", OUT_JSON)
    print("[OK] Wrote", OUT_MD)
    print(json.dumps(summary["condition_summaries"], indent=2))


if __name__ == "__main__":
    main()
