from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_audit_utils import append_jsonl, make_audit_record


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_LLM_JSONL = DEFAULT_RUN_DIR / "narratives_ops_triage_llm_5753rows_backup.jsonl"
DEFAULT_AUDIT_JSONL = DEFAULT_RUN_DIR / "narratives_ops_triage_llm_audit_backfilled.jsonl"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-jsonl", default=str(DEFAULT_LLM_JSONL))
    ap.add_argument("--audit-jsonl", default=str(DEFAULT_AUDIT_JSONL))
    ap.add_argument("--run-id", default="llm_accepted_sample_backfill_v1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    llm_path = Path(args.llm_jsonl)
    audit_path = Path(args.audit_jsonl)

    if not llm_path.exists():
        raise SystemExit(f"LLM JSONL not found: {llm_path}")

    if audit_path.exists() and args.overwrite:
        audit_path.unlink()

    count = 0
    for row in load_jsonl(llm_path):
        text = row.get("text")
        rec = make_audit_record(
            run_id=args.run_id,
            event_id=row.get("event_id"),
            audit_record_type="backfilled_output_row",
            attempt_number=1,
            outcome="accepted_present_in_output_jsonl",
            validator_status=row.get("validator_status"),
            validator_reason=row.get("validator_reason"),
            fallback_used=row.get("fallback_used"),
            narrative_text=text,
            extra={
                "source_llm_jsonl": str(llm_path),
                "scope_note": (
                    "Backfilled from rows present in LLM output JSONL. "
                    "This is not evidence of all historical attempts/retries/rejections."
                ),
            },
        )
        append_jsonl(audit_path, rec)
        count += 1

    print(f"Wrote {count} backfilled audit rows to {audit_path}")


if __name__ == "__main__":
    main()
