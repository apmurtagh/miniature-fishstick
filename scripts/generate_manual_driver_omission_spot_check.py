import csv
import json
import random
import re
from pathlib import Path

ROOT = Path.cwd()
ART = ROOT / "artifacts" / "baselines" / "lgbm_numeric_v1_subsample"
ROBUST = ART / "llm_20000_robustness"
H2_JSON = ART / "h2_unconstrained_ablation" / "h2_unconstrained_vs_constrained_summary_100.json"
OUT_DIR = ROOT / "docs" / "thesis_final"

SEED = 20260723
TARGET_N = 50

OUT_CSV = OUT_DIR / "manual_driver_omission_spot_check_sample_50.csv"
OUT_MD = OUT_DIR / "manual_driver_omission_spot_check_sample_50.md"
OUT_GUIDE = OUT_DIR / "manual_driver_omission_spot_check_coding_guide.md"
OUT_SUMMARY = OUT_DIR / "manual_driver_omission_spot_check_summary.json"
OUT_DIAG = OUT_DIR / "manual_driver_omission_spot_check_diagnostics.json"

def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            pass
    return rows

def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def event_id(row):
    for k in ["event_id", "id", "transaction_id", "row_id"]:
        if isinstance(row, dict) and row.get(k) not in [None, ""]:
            return str(row.get(k))
    return ""

def compact(x, n=700):
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()[:n]

def boolish(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return bool(int(x))
    return str(x).strip().lower() in ["1", "true", "yes", "y", "flag", "flagged", "review", "present"]

def driver_names_from_eo(eo):
    names = []
    if not isinstance(eo, dict):
        return names
    for d in eo.get("top_drivers", []) or []:
        if isinstance(d, dict):
            nm = d.get("name") or d.get("feature") or d.get("driver")
            if nm:
                names.append(str(nm))
    return names

def narrative_text(row):
    if not isinstance(row, dict):
        return ""
    for k in ["text", "narrative", "output_text", "response", "content"]:
        if row.get(k):
            return compact(row.get(k), 900)
    return ""

def write_outputs(rows, source_label, source_file):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    if len(rows) > TARGET_N:
        sample = rng.sample(rows, TARGET_N)
    else:
        sample = rows[:]

    sample = sorted(sample, key=lambda r: str(r.get("event_id", "")))

    fields = [
        "event_id",
        "sample_source",
        "source_file",
        "condition",
        "driver_count",
        "mentioned_driver_count",
        "driver_coverage",
        "review_language_flag",
        "mentioned_drivers",
        "eo_top_drivers_if_available",
        "missing_drivers_if_available",
        "narrative_excerpt",
        "manual_category",
        "manual_reviewer_note"
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sample:
            w.writerow({k: r.get(k, "") for k in fields})

    guide = """# Manual Driver-Omission Spot-Check Coding Guide

## Purpose

This guide supports a bounded manual error-analysis spot-check of automated driver-omission flags. It is not a human-subjects study and does not test readability, utility, trust calibration or analyst decision quality.

## Coding categories

Use exactly one category per sampled row.

- A_TRUE_OMISSION: one or more EO top drivers are absent from the narrative and not reasonably paraphrased.
- B_LEXICAL_MISMATCH: the narrative appears to refer to an EO driver using different wording, grouping or paraphrase, but the validator does not recognise it as coverage.
- C_SALIENCE_COMPRESSION: the narrative foregrounds one or two dominant drivers and omits weaker EO drivers while preserving risk/action framing.
- D_SUMMARY_LENGTH_COMPRESSION: the narrative is concise and operational, so omission appears related to brevity or output economy rather than contradiction.
- E_AMBIGUOUS_REVIEW: the reviewer cannot confidently distinguish omission from paraphrase or compression using the available artefacts.

## Claim boundary

This spot-check contextualises automated omission flags. It does not establish independent semantic validity, human usability, fairness, production readiness or regulatory audit completeness.
"""
    OUT_GUIDE.write_text(guide, encoding="utf-8")

    md = []
    md.append("# Manual Driver-Omission Spot-Check Sample")
    md.append("")
    md.append(f"- Sample source: `{source_label}`")
    md.append(f"- Source file: `{source_file}`")
    md.append(f"- Candidate omission rows found: `{len(rows)}`")
    md.append(f"- Sample target: `{TARGET_N}`")
    md.append(f"- Sample generated: `{len(sample)}`")
    md.append(f"- Random seed: `{SEED}`")
    md.append("")
    md.append("This file provides a coding template. Complete `manual_category` and `manual_reviewer_note` in the CSV using the coding guide.")
    md.append("")
    md.append("## Coding categories")
    md.append("")
    md.append("- `A_TRUE_OMISSION`")
    md.append("- `B_LEXICAL_MISMATCH`")
    md.append("- `C_SALIENCE_COMPRESSION`")
    md.append("- `D_SUMMARY_LENGTH_COMPRESSION`")
    md.append("- `E_AMBIGUOUS_REVIEW`")
    md.append("")
    md.append("## First 20 sampled rows")
    md.append("")
    md.append("| # | event_id | condition | driver_coverage | mentioned / driver count | review flag |")
    md.append("|---:|---|---|---:|---|---|")
    for i, r in enumerate(sample[:20], start=1):
        cnt = f"{r.get('mentioned_driver_count','')} / {r.get('driver_count','')}"
        md.append(f"| {i} | {r.get('event_id','')} | {r.get('condition','')} | {r.get('driver_coverage','')} | {cnt} | {r.get('review_language_flag','')} |")
    md.append("")
    md.append("## Claim boundary")
    md.append("")
    md.append("This is a manual error-analysis spot-check, not a human evaluation. It should not be used to claim improved readability, human trust calibration, analyst utility, fairness, production readiness or independent semantic truth.")
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    summary = {
        "status": "sample_generated",
        "sample_source": source_label,
        "source_file": str(source_file),
        "candidate_omission_rows_found": len(rows),
        "sample_target": TARGET_N,
        "sample_generated": len(sample),
        "seed": SEED,
        "outputs": {
            "csv": str(OUT_CSV),
            "markdown": str(OUT_MD),
            "coding_guide": str(OUT_GUIDE),
            "diagnostics": str(OUT_DIAG)
        },
        "claim_boundary": "Manual error-analysis spot-check only; not a human-subjects study or independent utility/readability validation."
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary

def try_20k():
    eo_file = ROBUST / "eos_20000_strict_schema.jsonl"
    narr_file = ROBUST / "narratives_ops_triage_llm_20000_resume_safe.jsonl"
    audit_file = ROBUST / "llm_20000_attempt_audit.jsonl"

    eos = read_jsonl(eo_file)
    narrs = read_jsonl(narr_file)
    audits = read_jsonl(audit_file)

    eo_by_id = {event_id(e): e for e in eos if event_id(e)}
    text_by_id = {event_id(n): narrative_text(n) for n in narrs if event_id(n)}

    candidates = []
    examples = []

    for a in audits:
        eid = event_id(a)
        if not eid:
            if len(examples) < 3:
                examples.append({"missing_event_id_keys": sorted(a.keys())})
            continue

        # Search for common audit metric keys.
        driver_count = a.get("driver_count") or a.get("top_driver_count") or a.get("eo_driver_count")
        mentioned_count = a.get("mentioned_driver_count") or a.get("covered_driver_count") or a.get("matched_driver_count")
        coverage = a.get("driver_coverage") or a.get("top_driver_coverage") or a.get("coverage") or a.get("top_k_overlap")
        review = a.get("review_language_flag") or a.get("review_flag")

        try:
            driver_count_i = int(float(driver_count)) if driver_count is not None else ""
        except Exception:
            driver_count_i = ""
        try:
            mentioned_count_i = int(float(mentioned_count)) if mentioned_count is not None else ""
        except Exception:
            mentioned_count_i = ""
        try:
            coverage_f = float(coverage) if coverage is not None else ""
        except Exception:
            coverage_f = ""

        inferred = False
        if driver_count_i != "" and mentioned_count_i != "":
            inferred = mentioned_count_i < driver_count_i
        elif coverage_f != "":
            inferred = coverage_f < 0.999

        all_text = json.dumps(a, ensure_ascii=False).lower()
        explicit = "driver_omission" in all_text or "driver omission" in all_text or "missing driver" in all_text

        if not (explicit or inferred):
            continue

        eo = eo_by_id.get(eid, {})
        eo_drivers = driver_names_from_eo(eo)

        mentioned = a.get("mentioned_drivers") or a.get("covered_drivers") or a.get("matched_drivers") or []
        if isinstance(mentioned, str):
            mentioned = [x.strip() for x in re.split(r"[;,|]", mentioned) if x.strip()]
        if not isinstance(mentioned, list):
            mentioned = []

        miss = []
        if eo_drivers and mentioned:
            mset = {str(x).lower() for x in mentioned}
            miss = [x for x in eo_drivers if x.lower() not in mset]

        candidates.append({
            "event_id": eid,
            "sample_source": "20k robustness audit",
            "source_file": str(audit_file),
            "condition": "constrained_20k",
            "driver_count": driver_count_i,
            "mentioned_driver_count": mentioned_count_i,
            "driver_coverage": coverage_f if coverage_f == "" else round(coverage_f, 6),
            "review_language_flag": boolish(review),
            "mentioned_drivers": "; ".join(str(x) for x in mentioned),
            "eo_top_drivers_if_available": "; ".join(eo_drivers),
            "missing_drivers_if_available": "; ".join(miss),
            "narrative_excerpt": text_by_id.get(eid, ""),
            "manual_category": "",
            "manual_reviewer_note": ""
        })

    diag = {
        "mode": "20k_attempt",
        "eo_rows": len(eos),
        "narrative_rows": len(narrs),
        "audit_rows": len(audits),
        "candidate_rows_found": len(candidates),
        "examples": examples,
        "audit_first_row_keys": sorted(audits[0].keys()) if audits else []
    }
    OUT_DIAG.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    return candidates, audit_file

def try_h2():
    obj = read_json(H2_JSON)
    rows = obj.get("row_metrics", [])
    candidates = []
    for r in rows:
        eid = str(r.get("event_id", ""))
        u = r.get("unconstrained") or {}
        eo_drivers = []
        mentioned = u.get("mentioned_drivers") or []
        if not isinstance(mentioned, list):
            mentioned = []

        dcount = u.get("driver_count")
        mcount = u.get("mentioned_driver_count")
        cov = u.get("driver_coverage")
        review = u.get("review_language_flag")

        try:
            omitted = int(mcount) < int(dcount)
        except Exception:
            omitted = cov is not None and float(cov) < 0.999

        if not omitted:
            continue

        candidates.append({
            "event_id": eid,
            "sample_source": "H2 same-EO ablation fallback",
            "source_file": str(H2_JSON),
            "condition": "unconstrained_h2",
            "driver_count": dcount,
            "mentioned_driver_count": mcount,
            "driver_coverage": cov,
            "review_language_flag": boolish(review),
            "mentioned_drivers": "; ".join(str(x) for x in mentioned),
            "eo_top_drivers_if_available": "; ".join(eo_drivers),
            "missing_drivers_if_available": "",
            "narrative_excerpt": "",
            "manual_category": "",
            "manual_reviewer_note": ""
        })
    return candidates, H2_JSON

print("== Manual driver-omission spot-check generator ==")
print("Trying 20k robustness audit first...")

rows, src = try_20k()
if len(rows) >= TARGET_N:
    print(f"[OK] 20k candidates found: {len(rows)}")
    write_outputs(rows, "20k robustness audit", src)
else:
    print(f"[WARN] 20k candidates found: {len(rows)}; falling back to H2 same-EO ablation.")
    rows, src = try_h2()
    if len(rows) >= TARGET_N:
        print(f"[OK] H2 fallback candidates found: {len(rows)}")
        write_outputs(rows, "H2 same-EO ablation fallback", src)
    else:
        print(f"[REVIEW] H2 fallback candidates found: {len(rows)}; fewer than target {TARGET_N}.")
        if len(rows) > 0:
            write_outputs(rows, "H2 same-EO ablation fallback partial", src)
        else:
            raise SystemExit("No suitable omission rows found.")
