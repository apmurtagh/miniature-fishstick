from __future__ import annotations

from pathlib import Path

from llm_audit_utils import append_jsonl, make_audit_record


OUT = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample" / "narratives_ops_triage_llm_audit_smoke_test.jsonl"

if OUT.exists():
    OUT.unlink()

# Synthetic examples to demonstrate intended prospective operational logging.
records = [
    make_audit_record(
        run_id="audit_smoke_test",
        event_id="synthetic_001",
        audit_record_type="attempt",
        attempt_number=1,
        outcome="rejected",
        validator_status="rejected",
        validator_reason="missing_required_disclosure",
        fallback_used=False,
        model="smoke-test-model",
        prompt_version="ops_triage_v1",
        narrative_text="Synthetic narrative rejected for missing disclosure.",
    ),
    make_audit_record(
        run_id="audit_smoke_test",
        event_id="synthetic_001",
        audit_record_type="attempt",
        attempt_number=2,
        outcome="accepted",
        validator_status="accepted",
        validator_reason="ok",
        fallback_used=False,
        model="smoke-test-model",
        prompt_version="ops_triage_v1",
        narrative_text="Synthetic narrative accepted after retry.",
    ),
    make_audit_record(
        run_id="audit_smoke_test",
        event_id="synthetic_002",
        audit_record_type="attempt",
        attempt_number=1,
        outcome="fallback",
        validator_status="fallback",
        validator_reason="max_retries_exceeded",
        fallback_used=True,
        model="smoke-test-model",
        prompt_version="ops_triage_v1",
        narrative_text="Synthetic deterministic fallback narrative.",
    ),
]

for rec in records:
    append_jsonl(OUT, rec)

print(f"Wrote smoke-test audit log: {OUT}")
