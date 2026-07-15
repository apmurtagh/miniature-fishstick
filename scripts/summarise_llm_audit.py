from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_AUDIT_JSONL = DEFAULT_RUN_DIR / "narratives_ops_triage_llm_audit_backfilled.jsonl"
DEFAULT_OUT_JSON = DEFAULT_RUN_DIR / "llm_audit_summary.json"
DEFAULT_OUT_MD = DEFAULT_RUN_DIR / "llm_audit_summary.md"
DEFAULT_OUT_CSV = DEFAULT_RUN_DIR / "llm_audit_counts.csv"


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(x):
    if x is None:
        return "__missing__"
    if x == "":
        return "__blank__"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-jsonl", default=str(DEFAULT_AUDIT_JSONL))
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    args = ap.parse_args()

    audit_path = Path(args.audit_jsonl)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(audit_path)

    event_ids = [norm(r.get("event_id")) for r in rows]
    unique_event_ids = sorted(set(event_ids))

    counts = {
        "audit_record_type": Counter(norm(r.get("audit_record_type")) for r in rows),
        "outcome": Counter(norm(r.get("outcome")) for r in rows),
        "validator_status": Counter(norm(r.get("validator_status")) for r in rows),
        "validator_reason": Counter(norm(r.get("validator_reason")) for r in rows),
        "fallback_used": Counter(norm(r.get("fallback_used")) for r in rows),
        "error_type": Counter(norm(r.get("error_type")) for r in rows),
    }

    attempts_by_event = defaultdict(int)
    for r in rows:
        if norm(r.get("audit_record_type")) == "attempt":
            attempts_by_event[norm(r.get("event_id"))] += 1

    records_are_backfilled_only = (
        len(rows) > 0
        and set(counts["audit_record_type"].keys()) == {"backfilled_output_row"}
    )

    summary = {
        "summary_version": "llm_audit_summary_v1",
        "audit_jsonl": str(audit_path),
        "rows": len(rows),
        "unique_event_ids": len(unique_event_ids),
        "counts": {k: dict(v) for k, v in counts.items()},
        "attempted_events": len(attempts_by_event),
        "mean_attempts_per_attempted_event": (
            sum(attempts_by_event.values()) / len(attempts_by_event)
            if attempts_by_event else None
        ),
        "records_are_backfilled_only": records_are_backfilled_only,
        "scope_note": (
            "If records_are_backfilled_only=true, this audit file was reconstructed from present output rows "
            "and does not contain full historical attempted/retried/rejected events. Prospective instrumentation "
            "should log audit_record_type='attempt' for every generation attempt."
        ),
    }

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "value", "count"])
        writer.writeheader()
        for category, counter in counts.items():
            for value, count in counter.items():
                writer.writerow({"category": category, "value": value, "count": count})

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# LLM Audit Summary\n\n")
        f.write(f"- Audit JSONL: `{audit_path}`\n")
        f.write(f"- Rows: `{summary['rows']}`\n")
        f.write(f"- Unique event IDs: `{summary['unique_event_ids']}`\n")
        f.write(f"- Records are backfilled only: `{summary['records_are_backfilled_only']}`\n\n")

        f.write("## Core Counts\n\n")
        f.write("| Category | Value | Count |\n")
        f.write("|---|---|---:|\n")
        for category, counter in counts.items():
            for value, count in counter.items():
                f.write(f"| {category} | {value} | {count} |\n")

        f.write("\n## Operational Interpretation\n\n")
        if records_are_backfilled_only:
            f.write(
                "This audit file is a backfilled audit view created from rows already present in the LLM output JSONL. "
                "It supports accepted-sample traceability, but it does not evidence every historical attempt, retry, rejection or fallback. "
                "For production-grade operational audit, the LLM orchestration loop should append an audit record for every generation attempt.\n"
            )
        else:
            f.write(
                "This audit file contains prospective attempt-level records. Counts can be used to report attempted, accepted, rejected, retried and fallback outcomes.\n"
            )

    print(json.dumps(summary, indent=2))
    print("Wrote:", out_json)
    print("Wrote:", out_md)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
