from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"
DEFAULT_LLM = DEFAULT_RUN_DIR / "narratives_ops_triage_llm_5753rows_backup.jsonl"
DEFAULT_AUDIT = DEFAULT_RUN_DIR / "narratives_ops_triage_llm_audit.jsonl"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    rows.append({"__parse_error__": True, "raw": line})
    return rows


def row_key(row):
    return str(row.get("event_id"))


def stable_hash(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_fingerprint(row):
    """
    Fingerprint used to detect newly emitted rows after an orchestrator run.
    Uses event_id plus full stable row hash to avoid hiding duplicates/changed rows.
    """
    return f"{row.get('event_id')}::{stable_hash(row)}"


def boolish(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    return str(x).strip().lower() in {"true", "1", "yes", "y"}


def append_audit(audit_path: Path, obj: dict):
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def infer_planned_targets(eos_path: Path, llm_path: Path, limit: int | None):
    """
    Best-effort planned target inference:
    - load all EO event_ids,
    - remove event_ids already present in LLM output,
    - take first N if limit supplied.

    This is deliberately labelled as inferred targets because the existing
    orchestrator may apply its own resume/skip/order logic.
    """
    eos = load_jsonl(eos_path)
    llm = load_jsonl(llm_path)
    existing = {row_key(r) for r in llm if r.get("event_id") is not None}

    targets = []
    for r in eos:
        eid = row_key(r)
        if eid not in existing:
            targets.append(eid)
        if limit is not None and len(targets) >= limit:
            break

    return targets


def main():
    ap = argparse.ArgumentParser(
        description="Lightweight wrapper-level audit instrumentation for LLM narrative generation."
    )
    ap.add_argument("--eos-jsonl", default=str(DEFAULT_EOS))
    ap.add_argument("--llm-jsonl", default=str(DEFAULT_LLM))
    ap.add_argument("--audit-jsonl", default=str(DEFAULT_AUDIT))
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument(
        "--cmd",
        default=None,
        help=(
            "Command to run. If omitted, defaults to "
            "`PYTHONPATH=src python scripts/orchestrate_llm_narratives.py --limit <limit>`."
        ),
    )
    ap.add_argument(
        "--no-run",
        action="store_true",
        help="Only write planned-target audit records; do not invoke the LLM orchestrator.",
    )
    args = ap.parse_args()

    eos_path = Path(args.eos_jsonl)
    llm_path = Path(args.llm_jsonl)
    audit_path = Path(args.audit_jsonl)

    run_id = f"llm_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    limit = args.limit

    cmd = args.cmd
    if cmd is None:
        cmd = f"PYTHONPATH=src python scripts/orchestrate_llm_narratives.py --limit {limit}"

    before_rows = load_jsonl(llm_path)
    before_fps = {row_fingerprint(r) for r in before_rows}

    planned_targets = infer_planned_targets(eos_path, llm_path, limit)

    append_audit(audit_path, {
        "audit_event": "run_start",
        "audit_level": "wrapper",
        "run_id": run_id,
        "created_utc": utc_now(),
        "cmd": cmd,
        "limit": limit,
        "eos_jsonl": str(eos_path),
        "llm_jsonl": str(llm_path),
        "audit_jsonl": str(audit_path),
        "rows_present_before": len(before_rows),
        "planned_target_count_inferred": len(planned_targets),
        "scope_note": (
            "Lightweight wrapper audit. Captures wrapper run lifecycle, inferred targets, and newly emitted rows. "
            "It does not observe internal API retries unless the underlying orchestrator emits them."
        ),
    })

    for order, eid in enumerate(planned_targets, start=1):
        append_audit(audit_path, {
            "audit_event": "planned_target_inferred",
            "audit_level": "wrapper",
            "run_id": run_id,
            "created_utc": utc_now(),
            "event_id": eid,
            "planned_order": order,
            "attempted_by_wrapper": not args.no_run,
            "scope_note": (
                "Target inferred from EO order excluding event_ids already present in LLM JSONL. "
                "Underlying orchestrator may apply its own targeting/resume logic."
            ),
        })

    returncode = None
    stderr_tail = None
    stdout_tail = None
    command_error = None

    if not args.no_run:
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            returncode = proc.returncode
            stdout_tail = (proc.stdout or "")[-4000:]
            stderr_tail = (proc.stderr or "")[-4000:]
        except Exception as e:
            returncode = -999
            command_error = repr(e)

    after_rows = load_jsonl(llm_path)
    after_fps = {row_fingerprint(r) for r in after_rows}
    new_fps = after_fps - before_fps

    emitted_new = 0
    emitted_event_ids = set()

    for r in after_rows:
        fp = row_fingerprint(r)
        if fp not in new_fps:
            continue

        emitted_new += 1
        eid = row_key(r)
        emitted_event_ids.add(eid)

        fallback_used = boolish(r.get("fallback_used"))
        validator_status = r.get("validator_status")
        validator_reason = r.get("validator_reason")

        if fallback_used:
            outcome = "fallback_emitted"
        elif validator_status == "accepted":
            outcome = "accepted_emitted"
        elif validator_status:
            outcome = f"{validator_status}_emitted"
        else:
            outcome = "emitted_unknown_validator_status"

        append_audit(audit_path, {
            "audit_event": "row_emitted_observed",
            "audit_level": "wrapper",
            "run_id": run_id,
            "created_utc": utc_now(),
            "event_id": eid,
            "row_hash": stable_hash(r),
            "outcome": outcome,
            "validator_status": validator_status,
            "validator_reason": validator_reason,
            "fallback_used": fallback_used,
            "text_present": bool(r.get("text")),
            "observed_after_run": True,
            "scope_note": (
                "Observed as a newly emitted row by comparing LLM JSONL before and after wrapper run. "
                "Internal retry count is not observable unless emitted by underlying orchestrator."
            ),
        })

    planned_not_observed = [eid for eid in planned_targets if eid not in emitted_event_ids]

    for eid in planned_not_observed:
        append_audit(audit_path, {
            "audit_event": "planned_target_not_observed",
            "audit_level": "wrapper",
            "run_id": run_id,
            "created_utc": utc_now(),
            "event_id": eid,
            "outcome": "not_observed_as_new_row",
            "scope_note": (
                "Inferred target was not observed as a new LLM JSONL row after this wrapper run. "
                "This may mean it was skipped, already processed by underlying logic, failed without emission, "
                "or the wrapper target inference did not match orchestrator targeting."
            ),
        })

    append_audit(audit_path, {
        "audit_event": "run_end",
        "audit_level": "wrapper",
        "run_id": run_id,
        "created_utc": utc_now(),
        "cmd": cmd,
        "returncode": returncode,
        "command_error": command_error,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "rows_present_after": len(after_rows),
        "new_rows_observed": emitted_new,
        "planned_targets_not_observed": len(planned_not_observed),
        "outcome": "completed" if returncode == 0 else "command_failed_or_not_run",
        "scope_note": (
            "Wrapper-level operational audit completed. "
            "For full internal retry/rejection taxonomy, instrument the underlying LLM orchestrator directly."
        ),
    })

    print(json.dumps({
        "run_id": run_id,
        "cmd": cmd,
        "returncode": returncode,
        "rows_present_before": len(before_rows),
        "rows_present_after": len(after_rows),
        "new_rows_observed": emitted_new,
        "planned_target_count_inferred": len(planned_targets),
        "planned_targets_not_observed": len(planned_not_observed),
        "audit_jsonl": str(audit_path),
        "scope_note": "Wrapper-level audit; internal retries are not observable unless underlying orchestrator emits them."
    }, indent=2))


if __name__ == "__main__":
    main()
