from __future__ import annotations

import json
import os
import re

from openai import OpenAI
from narratives.template_engine import generate_template_narrative


def _eo_payload(eo, top_k):
    top = eo.top_drivers if top_k <= 0 else eo.top_drivers[:top_k]
    return {
        "event_id": eo.event_id,
        "score": eo.score,
        "risk_band": eo.risk_band.value,
        "top_drivers": [
            {
                "name": d.name,
                "direction": d.direction,
            }
            for d in top
        ],
        "thin_file_flag": eo.thin_file_flag,
        "evidence_strength": eo.evidence_strength.value,
        "drift_status": eo.monitoring.drift_status.value,
        "recommended_action_class": eo.recommended_action_class.value,
    }


def _extract_text_from_response(resp):
    # Preferred path for newer OpenAI clients
    txt = getattr(resp, "output_text", None)
    if txt:
        return txt

    # Fallback for nested response objects
    try:
        chunks = []
        for item in resp.output:
            content = getattr(item, "content", None) or []
            for c in content:
                if getattr(c, "text", None):
                    chunks.append(c.text)
        if chunks:
            return "\n".join(chunks)
    except Exception:
        pass

    # Last resort
    return str(resp)


def _strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _validate_payload(payload, eo, top_k):
    required = ["summary", "drivers_used", "recommended_action_class"]
    for k in required:
        if k not in payload:
            return False, f"missing_key:{k}"

    if not isinstance(payload["drivers_used"], list):
        return False, "drivers_used_not_list"

    eo_top = eo.top_drivers if top_k <= 0 else eo.top_drivers[:top_k]
    eo_names = [d.name for d in eo_top]
    drivers_used = payload["drivers_used"]

    # Closed-world: drivers_used must be subset of EO top drivers
    if not set(drivers_used).issubset(set(eo_names)):
        return False, "drivers_used_not_subset_of_eo_top_drivers"

    expected_action = eo.recommended_action_class.value
    if str(payload["recommended_action_class"]).strip().lower() != expected_action.lower():
        return False, "recommended_action_mismatch"

    # Required disclosures
    text_blobs = []
    if payload.get("summary"):
        text_blobs.append(str(payload["summary"]))
    disclosures = payload.get("disclosures")
    if isinstance(disclosures, dict):
        text_blobs.extend([str(v) for v in disclosures.values() if v])
    elif disclosures:
        text_blobs.append(str(disclosures))

    combined = " ".join(text_blobs).lower()

    if eo.thin_file_flag or eo.evidence_strength.value.upper() == "LOW":
        if not any(x in combined for x in ["thin-file", "low evidence", "limited evidence", "evidence is limited"]):
            return False, "missing_thinfile_or_low_evidence_disclosure"

    if eo.monitoring.drift_status.value in {"WARN", "FAIL"}:
        if not any(x in combined for x in ["drift", "warn", "caution"]):
            return False, "missing_drift_disclosure"

    return True, "ok"


def _render_text(payload, eo, top_k):
    eo_top = eo.top_drivers if top_k <= 0 else eo.top_drivers[:top_k]
    used = payload.get("drivers_used", [])
    used_set = set(used)

    inc = [d.name for d in eo_top if d.name in used_set and d.direction == "+"]
    dec = [d.name for d in eo_top if d.name in used_set and d.direction == "-"]

    parts = []
    parts.append(f"Risk band: {eo.risk_band.value} (score {eo.score:.3f}).")

    summary = str(payload.get("summary", "")).strip()
    if summary:
        parts.append(summary if summary.endswith(".") else summary + ".")

    if inc:
        parts.append("Top risk-increasing drivers: " + ", ".join(inc) + ".")
    if dec:
        parts.append("Top risk-decreasing drivers: " + ", ".join(dec) + ".")

    disclosures = payload.get("disclosures")
    if isinstance(disclosures, dict):
        disclosure_text = " ".join([str(v).strip() for v in disclosures.values() if v]).strip()
        if disclosure_text:
            parts.append(disclosure_text if disclosure_text.endswith(".") else disclosure_text + ".")
    elif disclosures:
        disc = str(disclosures).strip()
        parts.append(disc if disc.endswith(".") else disc + ".")

    parts.append(f"Recommended action: {eo.recommended_action_class.value}.")
    return " ".join(parts)


def build_prompt(eo, persona, top_k):
    eo_payload = _eo_payload(eo, top_k=top_k)
    return f"""
You are generating a governance-ready fraud decisioning narrative.

Return ONLY valid JSON (no markdown, no prose outside JSON) with this exact shape:
{{
  "summary": "short narrative summary",
  "drivers_used": ["driver1", "driver2"],
  "recommended_action_class": "{eo.recommended_action_class.value}",
  "disclosures": {{
    "evidence": "if applicable",
    "drift": "if applicable"
  }}
}}

Hard constraints:
- Only use facts present in the EO payload below.
- drivers_used must be a subset of EO top_drivers names.
- recommended_action_class must equal EO recommended_action_class exactly.
- If thin_file_flag is true OR evidence_strength is LOW, include limited-evidence disclosure.
- If drift_status is WARN or FAIL, include drift/caution disclosure.
- Keep the JSON concise and deterministic.

Persona: {persona}

EO payload:
{json.dumps(eo_payload, ensure_ascii=False)}
""".strip()


def generate_llm_narrative(
    eo,
    *,
    persona="ops_triage",
    top_k=5,
    model=None,
    max_retries=2,
):
    model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI()

    prompt = build_prompt(eo, persona=persona, top_k=top_k)
    last_reason = "unknown"

    for attempt in range(max_retries + 1):
        resp = client.responses.create(
            model=model,
            input=prompt,
        )
        raw = _extract_text_from_response(resp)
        raw = _strip_code_fences(raw)

        try:
            payload = json.loads(raw)
        except Exception:
            last_reason = "json_parse_failed"
            continue

        ok, reason = _validate_payload(payload, eo, top_k=top_k)
        if not ok:
            last_reason = reason
            continue

        text = _render_text(payload, eo, top_k=top_k)

        return {
            "schema_version": "llm_narrative_v0",
            "persona": persona,
            "event_id": eo.event_id,
            "git_sha": eo.git_sha,
            "model": eo.model,
            "split_id": eo.split_id,
            "recommended_action_class": eo.recommended_action_class.value,
            "text": text,
            "llm_model": model,
            "validator_status": "accepted",
            "fallback_used": False,
            "validator_reason": "ok",
        }

    # Fallback to deterministic template if retries exhausted / validation failed
    fallback_text = generate_template_narrative(
        eo,
        persona=persona,
        top_k=top_k,
    )

    return {
        "schema_version": "llm_narrative_v0",
        "persona": persona,
        "event_id": eo.event_id,
        "git_sha": eo.git_sha,
        "model": eo.model,
        "split_id": eo.split_id,
        "recommended_action_class": eo.recommended_action_class.value,
        "text": fallback_text,
        "llm_model": model,
        "validator_status": "fallback",
        "fallback_used": True,
        "validator_reason": last_reason,
    }
