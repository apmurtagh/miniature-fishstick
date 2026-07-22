import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
ROBUST = ART / "llm_20000_robustness"
ROBUST.mkdir(parents=True, exist_ok=True)

SOURCE = ROBUST / "eos_20000_strict_schema.jsonl"
FINAL_NARR = ROBUST / "narratives_ops_triage_llm_20000_resume_safe.jsonl"
AUDIT = ROBUST / "llm_20000_attempt_audit.jsonl"
PROGRESS = ROBUST / "llm_20000_progress.json"
SUMMARY_MD = ROBUST / "llm_20000_run_summary.md"
SUMMARY_JSON = ROBUST / "llm_20000_run_summary.json"


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def event_id(row):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in row:
            return row[k]
    return None


def count_lines(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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


def fallback_record(eo, reason):
    return {
        "event_id": event_id(eo),
        "narrative": "Generation failed after explicit client-controlled retries.",
        "validator_status": "fallback",
        "fallback_used": True,
        "validator_reason": reason,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()

    eos = read_jsonl(SOURCE)
    total_available = len(eos)
    target = min(args.max_rows, total_available)

    already_done = count_lines(FINAL_NARR)

    if already_done >= target:
        print(f"[OK] Already complete: {already_done}/{target}")
        return

    print(f"[INFO] Source rows available: {total_available}")
    print(f"[INFO] Target rows: {target}")
    print(f"[INFO] Existing outputs: {already_done}")
    print(f"[INFO] Resuming from row index: {already_done}")

    accepted_nonfallback = 0
    fallback_rows = 0
    failed_rows = 0
    attempt_count = 0
    retry_attempt_count = 0

    tmp_dir = ROBUST / "tmp_single_row"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    start_run = time.time()

    for i in range(already_done, target):
        eo = eos[i]
        one_eo = tmp_dir / f"eo_{i}.jsonl"
        one_out = tmp_dir / f"out_{i}.jsonl"

        write_jsonl(one_eo, [eo])

        row_ok = False
        final_out = None

        for attempt in range(1, args.max_retries + 2):
            attempt_count += 1
            if attempt > 1:
                retry_attempt_count += 1

            cmd = [
                sys.executable,
                "scripts/orchestrate_llm_narratives.py",
                "--eos-jsonl",
                str(one_eo),
                "--out-jsonl",
                str(one_out),
                "--limit",
                "1",
            ]

            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0

            status = "subprocess_error"
            error_tail = None
            validation = None
            out_obj = None

            if proc.returncode == 0 and one_out.exists() and one_out.stat().st_size > 0:
                try:
                    outs = read_jsonl(one_out)
                    out_obj = outs[0] if outs else None
                    if out_obj:
                        validation = validate_output(eo, out_obj)
                        status = "accepted_with_validation_flags" if validation["failures"] else "accepted_clean"
                        row_ok = True
                        final_out = out_obj
                except Exception as e:
                    status = "parse_error"
                    error_tail = repr(e)
            else:
                error_tail = (proc.stderr or proc.stdout or "").strip()[-1500:]

            audit_row = {
                "row_index": i,
                "event_id": event_id(eo),
                "attempt_number": attempt,
                "max_client_retries": args.max_retries,
                "returncode": proc.returncode,
                "elapsed_seconds": elapsed,
                "status": status,
                "retry_decision": "stop" if row_ok else ("retry" if attempt <= args.max_retries else "exhausted"),
                "provider_hidden_retries_observable": False,
                "provider_request_id": "not_captured",
                "sdk_retry_policy": "not_observable_in_existing_orchestrator",
                "error_tail": error_tail,
                "validation": validation,
            }
            append_jsonl(AUDIT, audit_row)

            if row_ok:
                break

            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        if not row_ok:
            failed_rows += 1
            final_out = fallback_record(eo, "explicit_retry_exhausted")

        append_jsonl(FINAL_NARR, final_out)

        if final_out.get("fallback_used"):
            fallback_rows += 1
        else:
            accepted_nonfallback += 1

        if one_eo.exists():
            one_eo.unlink()
        if one_out.exists():
            one_out.unlink()

        done = i + 1
        if done % 25 == 0 or done == target:
            progress = {
                "target": target,
                "done": done,
                "remaining": target - done,
                "elapsed_seconds_this_run": time.time() - start_run,
                "accepted_nonfallback_this_run": accepted_nonfallback,
                "fallback_rows_this_run": fallback_rows,
                "failed_rows_this_run": failed_rows,
                "attempt_count_this_run": attempt_count,
                "retry_attempt_count_this_run": retry_attempt_count,
            }
            PROGRESS.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            print(f"[PROGRESS] {done}/{target} rows complete")

    print("[OK] Run finished or target reached")
    print(PROGRESS.read_text(encoding="utf-8") if PROGRESS.exists() else "")


if __name__ == "__main__":
    main()
