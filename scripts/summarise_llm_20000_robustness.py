import json
from pathlib import Path
from collections import Counter

ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
ROBUST = ART / "llm_20000_robustness"

NARR = ROBUST / "narratives_ops_triage_llm_20000_resume_safe.jsonl"
AUDIT = ROBUST / "llm_20000_attempt_audit.jsonl"
SUMMARY_JSON = ROBUST / "llm_20000_robustness_summary.json"
SUMMARY_MD = ROBUST / "llm_20000_robustness_summary.md"

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

narr = read_jsonl(NARR)
audit = read_jsonl(AUDIT)

status_counts = Counter()
failure_counts = Counter()
overlaps = []

for a in audit:
    status_counts[a.get("status", "UNKNOWN")] += 1
    validation = a.get("validation") or {}
    failures = validation.get("failures") or []
    for f in failures:
        failure_counts[f] += 1
    if validation.get("topk_overlap") is not None:
        overlaps.append(float(validation["topk_overlap"]))

fallback_rows = sum(1 for r in narr if r.get("fallback_used"))
accepted_nonfallback = len(narr) - fallback_rows

summary = {
    "narrative_rows": len(narr),
    "audit_attempt_rows": len(audit),
    "accepted_nonfallback_rows": accepted_nonfallback,
    "fallback_rows": fallback_rows,
    "attempt_status_counts": dict(status_counts),
    "failure_counts": dict(failure_counts),
    "avg_topk_overlap_from_audit": sum(overlaps) / len(overlaps) if overlaps else None,
}

SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

md = []
md.append("# 20,000-row LLM Robustness Summary\n\n")
md.append("| Metric | Value |\n")
md.append("|---|---:|\n")
md.append(f"| Narrative rows | {len(narr)} |\n")
md.append(f"| Audit attempt rows | {len(audit)} |\n")
md.append(f"| Accepted non-fallback rows | {accepted_nonfallback} |\n")
md.append(f"| Fallback rows | {fallback_rows} |\n")
if overlaps:
    md.append(f"| Average top-k overlap from audit | {sum(overlaps) / len(overlaps):.3f} |\n")

md.append("\n## Attempt status counts\n\n")
md.append("| Status | Count |\n")
md.append("|---|---:|\n")
for k, v in status_counts.most_common():
    md.append(f"| {k} | {v} |\n")

md.append("\n## Validation failure counts\n\n")
md.append("| Failure | Count |\n")
md.append("|---|---:|\n")
for k, v in failure_counts.most_common():
    md.append(f"| {k} | {v} |\n")

md.append(
    "\nInterpretation: This resume-safe 20,000-row robustness run scales the audited LLM generation evidence beyond the stratified 1,000-row pilot. "
    "It records visible application-level attempts and output-level validation flags. Provider-internal retries remain unobservable unless exposed by the provider/SDK.\n"
)

SUMMARY_MD.write_text("".join(md), encoding="utf-8")

print("[OK] 20,000 robustness summary written")
print(json.dumps(summary, indent=2))
print("")
print("== Summary ==")
print("".join(md))
