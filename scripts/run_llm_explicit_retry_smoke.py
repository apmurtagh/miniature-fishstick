import json
import subprocess
import sys
import time
from pathlib import Path


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
PILOT = ART / "audited_llm_pilot"
OUT = ART / "explicit_retry_smoke"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = PILOT / "audited_llm_pilot_eos_1000_strict_schema.jsonl"
if not SOURCE.exists():
    SOURCE = ART / "eos_test_with_drivers.jsonl"

N = 100
MAX_RETRIES = 2

RUN_EOS = OUT / "explicit_retry_smoke_eos_100.jsonl"
FINAL_NARR = OUT / "explicit_retry_smoke_narratives_100.jsonl"
AUDIT = OUT / "explicit_retry_attempt_audit_100.jsonl"
SUMMARY_MD = OUT / "explicit_retry_smoke_summary.md"
SUMMARY_JSON = OUT / "explicit_retry_smoke_summary.json"


def read_jsonl(path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def event_id(row):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in row:
            return row[k]
    return None


def top_driver_names(eo):
    drivers = eo.get("top_drivers") or eo.get("drivers") or []
    names = []
    for d in drivers:
        if isinstance(d, dict):
            names.append(str(d.get("name") or d.get("feature") or d.get("driver") or "UNKNOWN"))
        else:
            names.append(str(d))
    return names


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


def risk_band(eo):
    return str(eo.get("risk_band") or eo.get("calibration_band") or eo.get("score_band") or "")


def action_class(eo):
    return str(eo.get("recommended_action_class") or eo.get("recommended_action") or eo.get("action") or "")


def validate_output(eo, out):
    text = narrative_text(out)
    drivers = top_driver_names(eo)
    mentioned = [d for d in drivers if mentions(text, d)]

    failures = []

    if not mentioned:
        failures.append("no_driver_mentioned")

    if drivers and not set(drivers).issubset(set(mentioned)):
        failures.append("driver_omission")

    if not mentions(text, risk_band(eo)):
        failures.append("risk_band_missing")

    if not mentions(text, action_class(eo)):
        failures.append("recommended_action_missing")

    if bool(out.get("fallback_used", False)):
        failures.append("fallback_used")

    return {
        "failures": failures,
        "topk_overlap": len(set(drivers) & set(mentioned)) / max(1, len(set(drivers))),
        "drivers_mentioned_count": len(mentioned),
        "topk_driver_count": len(drivers),
    }


def main():
    rows = read_jsonl(SOURCE, limit=N)
    write_jsonl(RUN_EOS, rows)

    if FINAL_NARR.exists():
        FINAL_NARR.unlink()
    if AUDIT.exists():
        AUDIT.unlink()

    accepted = 0
    fallback = 0
    failed_rows = 0
    attempt_rows = []
    final_outputs = []

    for i, eo in enumerate(rows):
        row_ok = False
        last_error = None

        one_eo_path = OUT / f"tmp_eo_{i}.jsonl"
        one_out_path = OUT / f"tmp_out_{i}.jsonl"

        write_jsonl(one_eo_path, [eo])

        for attempt in range(1, MAX_RETRIES + 2):
            start = time.time()

            cmd = [
                sys.executable,
                "scripts/orchestrate_llm_narratives.py",
                "--eos-jsonl",
                str(one_eo_path),
                "--out-jsonl",
                str(one_out_path),
                "--limit",
                "1",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - start

            out_obj = None
            validation = None
            status = "error"

            if proc.returncode == 0 and one_out_path.exists() and one_out_path.stat().st_size > 0:
                try:
                    outs = read_jsonl(one_out_path, limit=1)
                    out_obj = outs[0] if outs else None

                    if out_obj:
                        validation = validate_output(eo, out_obj)
                        status = "accepted_with_validation_flags" if validation["failures"] else "accepted_clean"
                        row_ok = True
                        final_outputs.append(out_obj)

                        if out_obj.get("fallback_used"):
                            fallback += 1
                        else:
                            accepted += 1

                except Exception as e:
                    last_error = repr(e)
                    status = "parse_error"
            else:
                last_error = (proc.stderr or proc.stdout or "").strip()[-1000:]
                status = "subprocess_error"

            audit = {
                "row_index": i,
                "event_id": event_id(eo),
                "attempt_number": attempt,
                "max_client_retries": MAX_RETRIES,
                "sdk_retry_policy": "not_observable_in_existing_orchestrator",
                "provider_hidden_retries_observable": False,
                "provider_request_id": "not_captured",
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "status": status,
                "retry_decision": "stop" if row_ok else ("retry" if attempt <= MAX_RETRIES else "exhausted"),
                "error_tail": last_error,
                "validation": validation,
            }

            attempt_rows.append(audit)

            with AUDIT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(audit, ensure_ascii=False) + "\n")

            if row_ok:
                break

        if not row_ok:
            failed_rows += 1
            fallback_record = {
                "event_id": event_id(eo),
                "narrative": "Generation failed after explicit client-controlled retries.",
                "validator_status": "fallback",
                "fallback_used": True,
                "validator_reason": "explicit_retry_exhausted",
            }
            final_outputs.append(fallback_record)

        if one_eo_path.exists():
            one_eo_path.unlink()
        if one_out_path.exists():
            one_out_path.unlink()

    write_jsonl(FINAL_NARR, final_outputs)

    attempt_count = len(attempt_rows)
    retry_count = sum(1 for a in attempt_rows if a["attempt_number"] > 1)

    status_counts = {}
    for a in attempt_rows:
        status_counts[a["status"]] = status_counts.get(a["status"], 0) + 1

    summary = {
        "rows_requested": N,
        "final_outputs": len(final_outputs),
        "accepted_nonfallback_rows": accepted,
        "fallback_rows": fallback + failed_rows,
        "failed_rows_after_retries": failed_rows,
        "attempt_count": attempt_count,
        "retry_attempt_count": retry_count,
        "status_counts": status_counts,
        "outputs": {
            "input_eos": str(RUN_EOS),
            "final_narratives": str(FINAL_NARR),
            "attempt_audit": str(AUDIT),
            "summary_md": str(SUMMARY_MD),
            "summary_json": str(SUMMARY_JSON),
        },
        "caveat": (
            "This wrapper makes application-level attempts and retries explicit. Hidden provider-side retries "
            "remain unobservable unless exposed by the provider or SDK."
        ),
    }

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = []
    md.append("# Explicit Retry LLM Smoke Test\n\n")
    md.append(f"Rows requested: {N}\n\n")
    md.append("| Metric | Value |\n")
    md.append("|---|---:|\n")
    md.append(f"| Final outputs | {len(final_outputs)} |\n")
    md.append(f"| Accepted non-fallback rows | {accepted} |\n")
    md.append(f"| Fallback/failed rows | {fallback + failed_rows} |\n")
    md.append(f"| Attempt records | {attempt_count} |\n")
    md.append(f"| Retry attempts | {retry_count} |\n")

    md.append("\n## Attempt status counts\n\n")
    md.append("| Status | Count |\n")
    md.append("|---|---:|\n")
    for k, v in status_counts.items():
        md.append(f"| {k} | {v} |\n")

    md.append(
        "\nInterpretation: The smoke test wraps the existing LLM orchestrator in an explicit client-controlled "
        "retry loop and records one audit line per visible application-level attempt. Provider-internal retries "
        "remain unobservable unless exposed by the provider or SDK, but application-level retry behaviour is now auditable.\n"
    )

    SUMMARY_MD.write_text("".join(md), encoding="utf-8")

    print("[OK] Explicit retry smoke test complete")
    print(json.dumps(summary, indent=2))
    print("")
    print("== Summary ==")
    print("".join(md))


if __name__ == "__main__":
    main()
