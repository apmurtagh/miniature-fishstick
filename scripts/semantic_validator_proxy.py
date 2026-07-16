import json
from pathlib import Path

import pandas as pd


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
ROBUST = ART / "llm_20000_robustness"

EOS = ROBUST / "eos_20000_strict_schema.jsonl"
NARR = ROBUST / "narratives_ops_triage_llm_20000_resume_safe.jsonl"

OUT = ART / "semantic_validator_proxy"
OUT.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT / "semantic_validator_proxy_rows.csv"
OUT_JSON = OUT / "semantic_validator_proxy_summary.json"
OUT_MD = OUT / "semantic_validator_proxy_summary.md"

RISK_TERMS = {
    "LOW": ["low"],
    "MED": ["med", "medium"],
    "HIGH": ["high"],
}

ACTION_TERMS = {
    "allow": ["allow"],
    "step-up": ["step-up", "step up", "additional verification", "otp"],
    "review": ["review"],
    "investigate": ["investigate"],
    "enhanced_review": ["enhanced review", "enhanced_review"],
    "hold": ["hold"],
}

DISCLOSURE_TERMS = [
    "limited evidence",
    "low evidence",
    "thin",
    "sparse",
    "limited coverage",
    "coverage is limited",
    "treat with caution",
    "uncertain",
    "uncertainty",
]

POSITIVE_RISK_TERMS = [
    "increase",
    "increases",
    "increasing",
    "elevated",
    "higher",
    "raises",
    "riskier",
    "driven by",
    "drives",
]

NEGATIVE_RISK_TERMS = [
    "reduce",
    "reduces",
    "reducing",
    "lower",
    "lowers",
    "mitigate",
    "mitigates",
    "mitigating",
    "offset",
    "decrease",
    "decreases",
]


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_text(obj):
    parts = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            parts.append(str(x))

    walk(obj)
    return " ".join(parts)


def narrative_text(row):
    for k in ["narrative", "text", "summary", "output", "response", "llm_output", "template_narrative"]:
        if k in row:
            return flatten_text(row[k])
    return flatten_text(row)


def get_event_id(row):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in row:
            return str(row[k])
    return ""


def get_action(eo):
    return str(
        eo.get("recommended_action_class")
        or eo.get("recommended_action")
        or eo.get("action")
        or ""
    )


def get_risk(eo):
    return str(eo.get("risk_band") or eo.get("calibration_band") or eo.get("score_band") or "")


def get_evidence(eo):
    return str(eo.get("evidence_strength") or eo.get("evidence_band") or "")


def get_drivers(eo):
    drivers = eo.get("top_drivers") or eo.get("drivers") or []
    out = []
    for d in drivers:
        if isinstance(d, dict):
            name = d.get("name") or d.get("feature") or d.get("driver")
            direction = d.get("direction") or d.get("sign") or ""
            if name:
                out.append((str(name), str(direction)))
        else:
            out.append((str(d), ""))
    return out


def contains_any(text, terms):
    t = text.lower()
    return any(term.lower() in t for term in terms)


def main():
    if not EOS.exists():
        raise FileNotFoundError(EOS)
    if not NARR.exists():
        raise FileNotFoundError(NARR)

    eos = read_jsonl(EOS)
    narr = read_jsonl(NARR)
    n = min(len(eos), len(narr))

    rows = []

    for i in range(n):
        eo = eos[i]
        nr = narr[i]
        text = narrative_text(nr)
        tl = text.lower()

        risk = get_risk(eo)
        action = get_action(eo)
        evidence = get_evidence(eo)

        risk_present = contains_any(tl, RISK_TERMS.get(risk.upper(), [risk])) if risk else False
        action_present = contains_any(tl, ACTION_TERMS.get(action, [action.replace("_", " ")])) if action else False

        disclosure_required = evidence.upper() == "LOW" or bool(eo.get("thin_file_flag", False))
        disclosure_present = contains_any(tl, DISCLOSURE_TERMS)

        drivers = get_drivers(eo)
        mentioned_drivers = []
        direction_checks = []

        for name, direction in drivers:
            if name.lower() in tl:
                mentioned_drivers.append(name)
                if direction == "+":
                    direction_checks.append(contains_any(tl, POSITIVE_RISK_TERMS))
                elif direction == "-":
                    direction_checks.append(contains_any(tl, NEGATIVE_RISK_TERMS))
                else:
                    direction_checks.append(None)

        direction_applicable = [x for x in direction_checks if x is not None]
        direction_proxy_ok = all(direction_applicable) if direction_applicable else None

        rows.append({
            "row_index": i,
            "event_id": get_event_id(eo),
            "risk_band": risk,
            "risk_present_proxy": risk_present,
            "recommended_action": action,
            "action_present_proxy": action_present,
            "evidence_strength": evidence,
            "disclosure_required": disclosure_required,
            "disclosure_present_proxy": disclosure_present,
            "driver_count": len(drivers),
            "mentioned_driver_count": len(mentioned_drivers),
            "any_driver_mentioned": len(mentioned_drivers) > 0,
            "direction_proxy_applicable": direction_proxy_ok is not None,
            "direction_proxy_ok": direction_proxy_ok,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    summary = {
        "rows_evaluated": int(len(df)),
        "risk_present_rate": float(df["risk_present_proxy"].mean()),
        "action_present_rate": float(df["action_present_proxy"].mean()),
        "any_driver_mentioned_rate": float(df["any_driver_mentioned"].mean()),
        "disclosure_required_rows": int(df["disclosure_required"].sum()),
        "disclosure_present_when_required_rate": (
            float(df.loc[df["disclosure_required"], "disclosure_present_proxy"].mean())
            if df["disclosure_required"].sum() else None
        ),
        "direction_proxy_applicable_rows": int(df["direction_proxy_applicable"].sum()),
        "direction_proxy_ok_rate_when_applicable": (
            float(df.loc[df["direction_proxy_applicable"], "direction_proxy_ok"].mean())
            if df["direction_proxy_applicable"].sum() else None
        ),
        "caveat": (
            "This is a lightweight rule-based proxy. It strengthens automated semantic checks "
            "but does not replace human semantic review or formal natural language inference."
        ),
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = []
    md.append("# Semantic Validator Proxy Summary\n\n")
    md.append("| Metric | Value |\n")
    md.append("|---|---:|\n")
    md.append(f"| Rows evaluated | {summary['rows_evaluated']} |\n")
    md.append(f"| Risk band present proxy rate | {summary['risk_present_rate']:.4f} |\n")
    md.append(f"| Action present proxy rate | {summary['action_present_rate']:.4f} |\n")
    md.append(f"| Any driver mentioned rate | {summary['any_driver_mentioned_rate']:.4f} |\n")
    md.append(f"| Disclosure required rows | {summary['disclosure_required_rows']} |\n")
    if summary["disclosure_present_when_required_rate"] is not None:
        md.append(f"| Disclosure present when required rate | {summary['disclosure_present_when_required_rate']:.4f} |\n")
    md.append(f"| Direction proxy applicable rows | {summary['direction_proxy_applicable_rows']} |\n")
    if summary["direction_proxy_ok_rate_when_applicable"] is not None:
        md.append(f"| Direction proxy OK when applicable rate | {summary['direction_proxy_ok_rate_when_applicable']:.4f} |\n")

    md.append(
        "\nInterpretation: This lightweight rule-based validator adds a semantic proxy layer over the existing "
        "lexical driver/action/risk checks. It remains a proxy and does not replace human semantic review.\n"
    )

    OUT_MD.write_text("".join(md), encoding="utf-8")

    print("[OK] Semantic validator proxy complete")
    print(json.dumps(summary, indent=2))
    print("")
    print("== Summary ==")
    print("".join(md))


if __name__ == "__main__":
    main()
