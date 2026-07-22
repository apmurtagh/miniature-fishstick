from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_LLM_PATH = DEFAULT_RUN_DIR / "narratives_ops_triage_llm_5753rows_backup.jsonl"
DEFAULT_OUT_JSON = DEFAULT_RUN_DIR / "validator_metadata_summary.json"
DEFAULT_OUT_MD = DEFAULT_RUN_DIR / "validator_metadata_summary.md"
DEFAULT_OUT_CSV = DEFAULT_RUN_DIR / "validator_metadata_counts.csv"


def load_jsonl(path):
    rows = []
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
    ap.add_argument("--llm-jsonl", default=str(DEFAULT_LLM_PATH))
    ap.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    ap.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    args = ap.parse_args()

    llm_path = Path(args.llm_jsonl)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not llm_path.exists():
        summary = {
            "summary_version": "validator_metadata_summary_v1",
            "status": "skipped",
            "reason": f"LLM JSONL not found: {llm_path}",
        }
        rows = []
    else:
        rows = load_jsonl(llm_path)

        status_counts = Counter(norm(r.get("validator_status")) for r in rows)
        reason_counts = Counter(norm(r.get("validator_reason")) for r in rows)
        fallback_counts = Counter(norm(r.get("fallback_used")) for r in rows)

        missing_text = sum(1 for r in rows if not r.get("text"))
        missing_event_id = sum(1 for r in rows if r.get("event_id") is None)
        accepted = status_counts.get("accepted", 0)

        summary = {
            "summary_version": "validator_metadata_summary_v1",
            "status": "completed",
            "llm_jsonl": str(llm_path),
            "rows_present": len(rows),
            "accepted_rows": accepted,
            "accepted_rate_within_present_rows": accepted / len(rows) if rows else 0.0,
            "validator_status_counts": dict(status_counts),
            "validator_reason_counts": dict(reason_counts),
            "fallback_used_counts": dict(fallback_counts),
            "missing_text_rows": missing_text,
            "missing_event_id_rows": missing_event_id,
            "interpretation_scope": (
                "This is a metadata summary for rows present in the LLM JSONL artefact. "
                "It is not a full attempted/retried/rejected operational taxonomy unless upstream audit logs contain attempted rows."
            ),
        }

    count_rows = []
    if summary.get("status") == "completed":
        for k, v in summary["validator_status_counts"].items():
            count_rows.append({"category": "validator_status", "value": k, "count": v})
        for k, v in summary["validator_reason_counts"].items():
            count_rows.append({"category": "validator_reason", "value": k, "count": v})
        for k, v in summary["fallback_used_counts"].items():
            count_rows.append({"category": "fallback_used", "value": k, "count": v})

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "value", "count"])
        writer.writeheader()
        writer.writerows(count_rows)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Validator Metadata Summary\n\n")
        f.write(f"- LLM JSONL: `{llm_path}`\n")
        f.write(f"- Status: `{summary.get('status')}`\n\n")

        if summary.get("status") == "completed":
            f.write("| Metric | Value |\n")
            f.write("|---|---:|\n")
            f.write(f"| Rows present | {summary['rows_present']} |\n")
            f.write(f"| Accepted rows | {summary['accepted_rows']} |\n")
            f.write(f"| Accepted rate within present rows | {100*summary['accepted_rate_within_present_rows']:.1f}% |\n")
            f.write(f"| Missing text rows | {summary['missing_text_rows']} |\n")
            f.write(f"| Missing event_id rows | {summary['missing_event_id_rows']} |\n\n")

            f.write("## Validator status counts\n\n")
            f.write("| Status | Count |\n")
            f.write("|---|---:|\n")
            for k, v in summary["validator_status_counts"].items():
                f.write(f"| {k} | {v} |\n")

            f.write("\n## Fallback used counts\n\n")
            f.write("| Fallback used | Count |\n")
            f.write("|---|---:|\n")
            for k, v in summary["fallback_used_counts"].items():
                f.write(f"| {k} | {v} |\n")

            f.write("\n## Scope note\n\n")
            f.write(
                "This summary covers metadata for rows present in the LLM JSONL artefact. "
                "It does not prove full attempted/retry/rejection coverage unless a separate audit log records attempted events.\n"
            )
        else:
            f.write(f"Reason: {summary.get('reason')}\n")

    print(json.dumps(summary, indent=2))
    print("Wrote:", out_json)
    print("Wrote:", out_md)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()
