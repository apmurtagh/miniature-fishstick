import csv
import json
import re
from collections import Counter
from pathlib import Path

CSV_IN = Path("docs/thesis_final/manual_driver_omission_spot_check_sample_50.csv")
CSV_OUT = Path("docs/thesis_final/manual_driver_omission_spot_check_sample_50_final.csv")
OUT_MD = Path("docs/thesis_final/manual_driver_omission_spot_check_completed_summary.md")
OUT_JSON = Path("docs/thesis_final/manual_driver_omission_spot_check_completed_summary.json")
OUT_APPENDIX = Path("docs/thesis_final/manual_driver_omission_spot_check_appendix_c.md")
OUT_AUDIT = Path("docs/thesis_final/manual_driver_omission_spot_check_integrity_audit.json")

ALLOWED = {
    "A_TRUE_OMISSION",
    "B_LEXICAL_MISMATCH",
    "C_SALIENCE_COMPRESSION",
    "D_SUMMARY_LENGTH_COMPRESSION",
    "E_AMBIGUOUS_REVIEW",
}

ORDERED = [
    ("A_TRUE_OMISSION", "True omission"),
    ("B_LEXICAL_MISMATCH", "Lexical mismatch"),
    ("C_SALIENCE_COMPRESSION", "Salience compression"),
    ("D_SUMMARY_LENGTH_COMPRESSION", "Summary-length compression"),
    ("E_AMBIGUOUS_REVIEW", "Ambiguous / review required"),
]

def split_drivers(s):
    if not s:
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]

def normalise_for_match(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def driver_mentioned(driver, narrative):
    # Exact-ish matching for feature names such as C5, D10, card2, TransactionDT.
    d = str(driver).strip()
    if not d:
        return False

    narrative_l = str(narrative).lower()

    # Preserve alphanumeric token boundaries.
    pattern = r"(?<![a-z0-9])" + re.escape(d.lower()) + r"(?![a-z0-9])"
    if re.search(pattern, narrative_l):
        return True

    # Fallback for punctuation/spacing variants.
    return normalise_for_match(d) in normalise_for_match(narrative)

def make_note(mentioned, omitted, category):
    mentioned_txt = "/".join(mentioned) if mentioned else "None"
    omitted_txt = "/".join(omitted) if omitted else "none"

    if category == "A_TRUE_OMISSION":
        return f"No clear EO driver rendered; omits {omitted_txt}."
    if category == "B_LEXICAL_MISMATCH":
        return f"Possible paraphrase; exact drivers unclear. Omitted: {omitted_txt}."
    if category == "C_SALIENCE_COMPRESSION":
        return f"Mentions {mentioned_txt}; omits {omitted_txt}."
    if category == "D_SUMMARY_LENGTH_COMPRESSION":
        return f"Brief summary mentions {mentioned_txt}; omits {omitted_txt}."
    if category == "E_AMBIGUOUS_REVIEW":
        return f"Ambiguous coverage; review omitted: {omitted_txt}."
    return f"Mentions {mentioned_txt}; omits {omitted_txt}."

if not CSV_IN.exists():
    raise SystemExit(f"Missing {CSV_IN}")

rows = list(csv.DictReader(CSV_IN.open(encoding="utf-8", newline="")))
if not rows:
    raise SystemExit("No rows found in manual spot-check CSV.")

problems = []
final_rows = []

for i, row in enumerate(rows, start=1):
    event_id = (row.get("event_id") or "").strip()
    category = (row.get("manual_category") or "").strip()

    if not event_id:
        problems.append({"row": i, "issue": "blank_event_id"})
    if not category:
        problems.append({"row": i, "event_id": event_id, "issue": "blank_manual_category"})
    elif category not in ALLOWED:
        problems.append({"row": i, "event_id": event_id, "issue": "invalid_manual_category", "value": category})

    eo_drivers = split_drivers(row.get("eo_top_drivers_if_available", ""))
    narrative = row.get("narrative_excerpt", "")

    mentioned_exact = [d for d in eo_drivers if driver_mentioned(d, narrative)]
    omitted_exact = [d for d in eo_drivers if d not in mentioned_exact]

    if not eo_drivers:
        problems.append({"row": i, "event_id": event_id, "issue": "blank_eo_driver_list"})
    if not narrative:
        problems.append({"row": i, "event_id": event_id, "issue": "blank_narrative_excerpt"})
    if eo_drivers and not omitted_exact:
        problems.append({"row": i, "event_id": event_id, "issue": "no_omitted_drivers_detected"})

    # Do not overwrite the human category. Standardise the note so all omissions are fully reflected.
    final_note = make_note(mentioned_exact, omitted_exact, category or "C_SALIENCE_COMPRESSION")

    out = dict(row)
    out["detected_mentioned_eo_drivers"] = "; ".join(mentioned_exact)
    out["detected_omitted_eo_drivers"] = "; ".join(omitted_exact)
    out["manual_reviewer_note_original"] = row.get("manual_reviewer_note", "")
    out["manual_reviewer_note"] = final_note
    final_rows.append(out)

audit = {
    "status": "complete" if not problems else "review_required",
    "rows": len(rows),
    "problems": problems,
    "problem_count": len(problems),
    "note": "manual_category is preserved; reviewer notes are standardised from EO driver list and narrative exact-match detection."
}

OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

if problems:
    print(json.dumps(audit, indent=2))
    raise SystemExit("Review required before thesis insertion. See integrity audit JSON.")

fieldnames = list(final_rows[0].keys())
with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in final_rows:
        writer.writerow(row)

counts = Counter(row["manual_category"] for row in final_rows)
n = len(final_rows)

summary = {
    "status": "manual_coding_complete",
    "sample_size": n,
    "source_population": "12,006 driver-omission-flagged rows from the 20,000-row LLM robustness audit",
    "counts": {k: counts.get(k, 0) for k, _ in ORDERED},
    "shares": {k: counts.get(k, 0) / n for k, _ in ORDERED},
    "claim_boundary": "Manual error-analysis spot-check only; not a human-subjects study or independent readability, usability, trust-calibration or semantic-validity evaluation."
}
OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

md = []
md.append("# Manual Driver-Omission Spot-Check Completed Summary")
md.append("")
md.append("This manual spot-check contextualises automated driver-omission flags from the 20,000-row LLM robustness audit. It is not a human-subjects study and does not test readability, utility, trust calibration or analyst decision quality.")
md.append("")
md.append("- Source population: `12,006` driver-omission-flagged rows")
md.append(f"- Manual sample size: `{n}`")
md.append("- No additional LLM generation was performed.")
md.append("")
md.append("## Manual classification results")
md.append("")
md.append("| Category | Count | Share |")
md.append("|---|---:|---:|")
for key, label in ORDERED:
    c = counts.get(key, 0)
    md.append(f"| {label} | {c} | {c / n:.1%} |")
md.append("")
md.append("## Interpretation")
md.append("")
md.append("The spot-check provides a bounded qualitative error-analysis layer for the automated driver-omission finding. It helps distinguish substantive omission from validator-recognition limits and summarisation/compression effects. The result should not be interpreted as independent human semantic validation or evidence of user utility.")
md.append("")
md.append("## Claim boundary")
md.append("")
md.append("This spot-check is a manual error-analysis of existing artefacts only. It does not establish improved readability, trust calibration, analyst utility, fairness, production readiness or regulatory audit completeness.")
OUT_MD.write_text("\n".join(md), encoding="utf-8")

appendix = []
appendix.append("# Appendix C. Manual Driver-Omission Spot-Check")
appendix.append("")
appendix.append("## Purpose")
appendix.append("")
appendix.append("This appendix reports a bounded manual error-analysis spot-check of driver-omission-flagged narratives from the 20,000-row LLM robustness audit. The purpose is to contextualise the 12,006 automated driver-omission flags by distinguishing substantive omission from compression and validator-recognition effects. No additional LLM generation was performed.")
appendix.append("")
appendix.append("## Scope boundary")
appendix.append("")
appendix.append("The spot-check is not a human-subjects study and does not test readability, analyst utility, trust calibration, fairness or production readiness. It is a manual review of existing artefacts only and should be interpreted as qualitative error analysis rather than independent semantic validation.")
appendix.append("")
appendix.append("## Coding categories")
appendix.append("")
appendix.append("- **True omission:** one or more EO top drivers are absent from the narrative and not reasonably paraphrased.")
appendix.append("- **Lexical mismatch:** the narrative appears to refer to an EO driver using different wording, grouping or paraphrase, but the validator may not recognise it as coverage.")
appendix.append("- **Salience compression:** the narrative foregrounds dominant drivers and omits weaker EO drivers while preserving risk/action framing.")
appendix.append("- **Summary-length compression:** the narrative is concise and operational, so omission appears related to brevity rather than contradiction.")
appendix.append("- **Ambiguous / review required:** the reviewer cannot confidently distinguish omission from paraphrase or compression.")
appendix.append("")
appendix.append("## Results")
appendix.append("")
appendix.append("| Category | Count | Share |")
appendix.append("|---|---:|---:|")
for key, label in ORDERED:
    c = counts.get(key, 0)
    appendix.append(f"| {label} | {c} | {c / n:.1%} |")
appendix.append("")
appendix.append("## Interpretation")
appendix.append("")
appendix.append("The manual spot-check supports the interpretation that driver-omission flags should not be treated as a single undifferentiated failure mode. In sampled cases, the reviewer classification helps separate true omission from salience compression, summary-length compression, lexical mismatch and ambiguous review cases. This strengthens the thesis interpretation that operations-summary acceptance and validator-defined audit-complete rendering are distinct governance states.")
appendix.append("")
appendix.append("## Claim boundary")
appendix.append("")
appendix.append("The spot-check does not establish that users understand, trust or act on the narratives more effectively. Independent human semantic validation remains future work, as described in Appendix B.")
OUT_APPENDIX.write_text("\n".join(appendix), encoding="utf-8")

print("[OK] wrote", CSV_OUT)
print("[OK] wrote", OUT_MD)
print("[OK] wrote", OUT_JSON)
print("[OK] wrote", OUT_APPENDIX)
print("[OK] wrote", OUT_AUDIT)
print(json.dumps(summary, indent=2))
