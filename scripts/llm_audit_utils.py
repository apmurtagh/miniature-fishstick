from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def make_audit_record(
    *,
    run_id: str,
    event_id: Any,
    audit_record_type: str,
    attempt_number: int | None = None,
    outcome: str | None = None,
    validator_status: str | None = None,
    validator_reason: str | None = None,
    fallback_used: bool | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    model: str | None = None,
    prompt_version: str | None = None,
    eo_text: str | None = None,
    narrative_text: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = {
        "created_utc": utc_now(),
        "run_id": run_id,
        "event_id": event_id,
        "audit_record_type": audit_record_type,
        "attempt_number": attempt_number,
        "outcome": outcome,
        "validator_status": validator_status,
        "validator_reason": validator_reason,
        "fallback_used": fallback_used,
        "error_type": error_type,
        "error_message": error_message,
        "model": model,
        "prompt_version": prompt_version,
        "eo_hash": sha256_text(eo_text),
        "narrative_hash": sha256_text(narrative_text),
    }
    if extra:
        rec.update(extra)
    return rec
