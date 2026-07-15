import csv
import json
from pathlib import Path
from collections import Counter

ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
PILOT = ART / "audited_llm_pilot"

EOS = PILOT / "audited_llm_pilot_eos_1000_strict_schema.jsonl"
NARR = PILOT / "narratives_ops_triage_llm_audited_pilot_1000.jsonl"

AUDIT_JSONL = PILOT / "llm_generation_attempt_audit_1000.jsonl"
TAXONOMY_CSV = PILOT / "llm_validator_taxonomy_1000.csv"
SUMMARY_JSON = PILOT / "llm_audited_pilot_summary_1000.json"
SUMMARY_MD = PILOT / "llm_audited_pilot_summary_1000.md"


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def event_id(row):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in row and row[k] is not None:
            return row[k]
    return None


def top_driver_names(eo):
    drivers = eo.get("top_drivers") or eo.get("drivers") or []
    out = []
    for d in drivers:
        if isinstance(d, dict):
            name = d.get("name") or d.get("feature") or d.get("driver")
        else:
            name = str(d)
        if name is not None:
            out.append(str(name))
    return out


def risk_band(eo):
    return str(eo.get("risk_band") or eo.get("calibration_band") or eo.get("score_band") or eo.get("band") or "")


def action_class(eo):
    return str(eo.get("recommended_action_class") or eo.get("recommended_action") or eo.get("action") or eo.get("decision") or "")


def evidence_strength(eo):
    return str(eo.get("evidence_strength") or eo.get("evidence_band") or "").upper()


def thin_file(eo):
    v = eo.get("thin_file_flag")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"true", "1", "yes", "y"}
    return False


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


def mentions(text, token):
    return bool(token) and str(token).lower() in text.lower()


def disclosure_detected(text):
    lower = text.lower()
    phrases = ["thin", "limited evidence", "low evidence", "sparse", "limited supporting"]
    return any(p in lower for p in phrases)


def classify_attempt(eo, out, row_index):
    text = narrative_text(out)
    drivers = top_driver_names(eo)
    mentioned = [d for d in drivers if mentions(text, d)]

    expected_disclosure = thin_file(eo) or evidence_strength(eo) == "LOW"
    fallback_used = bool(out.get("fallback_used", False))
    validator_status = str(out.get("validator_status") or "").lower()

    risk_ok = mentions(text, risk_band(eo))
    action_ok = mentions(text, action_class(eo))
    at_least_one_driver_ok = len(mentioned) > 0
    all_drivers_ok = set(drivers).issubset(set(mentioned))
    disclosure_ok = True if not expected_disclosure else disclosure_detected(text)

    failure_reasons = []
    if not at_least_one_driver_ok:
        failure_reasons.append("no_driver_mentioned")
    if drivers and not all_drivers_ok:
        failure_reasons.append("driver_omission")
    if not risk_ok:
        failure_reasons.append("risk_band_missing")
    if not action_ok:
        failure_reasons.append("recommended_action_missing")
    if not disclosure_ok:
        failure_reasons.append("disclosure_missing_when_expected")
    if fallback_used:
        failure_reasons.append("fallback_used")
    if validator_status and validator_status not in {"accepted", "ok", "pass", "passed"}:
        failure_reasons.append(f"validator_status_{validator_status}")

    if fallback_used:
        final_status = "fallback"
    elif failure_reasons:
        final_status = "accepted_with_validation_flags"
    else:
        final_status = "accepted_clean"

    return {
        "row_index": row_index,
        "event_id": event_id(eo),
        "attempt_number": 1,
        "prompt_version": out.get("prompt_version") or out.get("prompt_hash") or "not_recorded",
        "validator_version": out.get("validator_version") or "output_level_audit_v1",
        "model_name": out.get("model_name") or out.get("model") or "not_recorded",
        "validator_status_raw": out.get("validator_status"),
        "validator_reason_raw": out.get("validator_reason"),
        "fallback_used": fallback_used,
        "final_status": final_status,
        "failure_reasons": failure_reasons,
        "schema_valid_proxy": True,
        "closed_world_driver_proxy_at_least_one": at_least_one_driver_ok,
        "closed_world_driver_proxy_all_topk": all_drivers_ok,
        "risk_band_present": risk_ok,
        "recommended_action_present": action_ok,
        "disclosure_expected": expected_disclosure,
        "disclosure_detected": disclosure_detected(text),
        "disclosure_valid": disclosure_ok,
        "topk_driver_count": len(drivers),
        "drivers_mentioned_count": len(mentioned),
        "topk_overlap": len(set(drivers) & set(mentioned)) / max(1, len(set(drivers))),
        "risk_band": risk_band(eo),
        "recommended_action": action_class(eo),
        "evidence_strength": evidence_strength(eo),
        "thin_file_flag": thin_file(eo),
    }


def main():
    eos = read_jsonl(EOS)
    outs = read_jsonl(NARR)
    n = min(len(eos), len(outs))
    if n == 0:
        raise RuntimeError("No rows available for audit")

    audit_rows = [classify_attempt(eo, out, i) for i, (eo, out) in enumerate(zip(eos[:n], outs[:n]))]

    with AUDIT_JSONL.open("w", encoding="utf-8") as f:
        for row in audit_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    fail_counter = Counter()
    status_counter = Counter()

    for row in audit_rows:
        status_counter[row["final_status"]] += 1
        if not row["failure_reasons"]:
            fail_counter["none"] += 1
        else:
            for reason in row["failure_reasons"]:
                fail_counter[reason] += 1

    with TAXONOMY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["failure_reason", "count", "rate"])
        for reason, count in fail_counter.most_common():
            writer.writerow([reason, count, count / n])

    avg_overlap = sum(r["topk_overlap"] for r in audit_rows) / n
    all_topk = sum(r["closed_world_driver_proxy_all_topk"] for r in audit_rows) / n
    at_least_one = sum(r["closed_world_driver_proxy_at_least_one"] for r in audit_rows) / n
    risk_present = sum(r["risk_band_present"] for r in audit_rows) / n
    action_present = sum(r["recommended_action_present"] for r in audit_rows) / n
    disclosure_expected_rows = [r for r in audit_rows if r["disclosure_expected"]]
    disclosure_rate = (
        sum(r["disclosure_valid"] for r in disclosure_expected_rows) / len(disclosure_expected_rows)
        if disclosure_expected_rows else None
    )

    summary = {
        "rows_audited": n,
        "final_status_counts": dict(status_counter),
        "failure_reason_counts": dict(fail_counter),
        "metrics": {
            "avg_topk_overlap": avg_overlap,
            "all_topk_drivers_mentioned_rate": all_topk,
            "at_least_one_driver_mentioned_rate": at_least_one,
            "risk_band_present_rate": risk_present,
            "recommended_action_present_rate": action_present,
            "disclosure_expected_rows": len(disclosure_expected_rows),
            "disclosure_valid_when_expected_rate": disclosure_rate,
        },
        "outputs": {
            "audit_jsonl": str(AUDIT_JSONL),
            "taxonomy_csv": str(TAXONOMY_CSV),
            "summary_json": str(SUMMARY_JSON),
            "summary_md": str(SUMMARY_MD),
        },
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Audited LLM Pilot Summary - 1,000-row stratified sample\n\n")
    md.append(f"Rows audited: {n}\n\n")

    md.append("## Final status counts\n\n")
    md.append("| Status | Count |\n|---|---:|\n")
    for k, v in status_counter.items():
        md.append(f"| {k} | {v} |\n")

    md.append("\n## Failure taxonomy\n\n")
    md.append("| Failure reason | Count | Rate |\n|---|---:|---:|\n")
    for k, v in fail_counter.most_common():
        md.append(f"| {k} | {v} | {v / n:.3f} |\n")

    md.append("\n## Faithfulness and disclosure metrics\n\n")
    for k, v in summary["metrics"].items():
        if isinstance(v, float):
            md.append(f"- {k}: {v:.3f}\n")
        else:
            md.append(f"- {k}: {v}\n")

    md.append(
        "\nInterpretation: This is an attempt-level output audit with one recorded generation attempt per sampled EO. "
        "It captures accepted, fallback and validation-flagged outputs at row level. It does not recover hidden provider-side retries, "
        "but it materially improves the dissertation audit trail compared with accepted-output-only reporting.\n"
    )

    SUMMARY_MD.write_text("".join(md), encoding="utf-8")

    print("[OK] Audit complete")
    print(f"[OK] Audit JSONL: {AUDIT_JSONL}")
    print(f"[OK] Taxonomy CSV: {TAXONOMY_CSV}")
    print(f"[OK] Summary MD: {SUMMARY_MD}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
