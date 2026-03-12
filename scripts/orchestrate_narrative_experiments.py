import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

# ---- Use Pydantic EO validation if available ----
try:
    from src.eo.evidence_object import EvidenceObject
    from pydantic import ValidationError
    USE_PYDANTIC_VALIDATION = True
except ImportError:
    USE_PYDANTIC_VALIDATION = False

try:
    import jsonschema
    SCHEMA_VALIDATION = True
except ImportError:
    SCHEMA_VALIDATION = False

# ---- Persona mappings ----
# .... amended for fixed jsonl files
PERSONA_FILES = {
    "engineering_debug": "eos_engineering_debug_thinfile_fixed.jsonl",
    "ops_triage": "eos_ops_triage_thinfile_fixed.jsonl",
    "governance_audit": "eos_governance_audit_thinfile_fixed.jsonl",
}

# ---- Output Narrative Schema ----
NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "persona": {"type": "string"},
        "summary": {"type": "string"},
        "drivers_used": {
            "type": "array",
            "items": {"type": "string"}
        },
        "driver_statements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "direction": {"type": "string"},
                    "text": {"type": "string"}
                },
                "required": ["name", "direction", "text"]
            }
        },
        "disclosures": {
            "type": "object"
        },
        "action_recommendation": {"type": "string"},
        "fallback_reason": {"type": "string"},
        "validator_attempt": {"type": "integer"},
    },
    "required": ["persona", "summary", "drivers_used", "driver_statements", "disclosures", "action_recommendation"],
}

LOGDIR = "./logs"
os.makedirs(LOGDIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGDIR, f"narrative_orchestration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)

RUN_ID = str(uuid.uuid4())
VERSION = "2026-03-12"

def generate_template_narrative(eo: Dict[str, Any], persona: str) -> Dict[str, Any]:
    top_drivers = eo.get('top_drivers', [])
    driver_statements = []
    for drv in top_drivers:
        sign = "increases" if drv['direction'] == "+" else "reduces"
        driver_statements.append(
            {"name": drv["name"], "direction": drv["direction"], "text": f"{drv['name']} {sign} risk."}
        )
    disclosures = {}
    if eo.get("monitoring", {}).get("drift_status") == "WARN":
        disclosures["drift"] = "Drift status WARN: treat score with caution."
    if eo.get("evidence_strength") == "LOW" or eo.get("thin_file_flag"):
        disclosures["evidence"] = "Thin-file / low coverage: limited supporting signals."
    # Use risk_band if available, else fallback to calibration_band, else UNKNOWN
    risk_band = eo.get('risk_band') or eo.get('calibration_band') or 'UNKNOWN'
    summary = f"Risk is {risk_band} (score {eo.get('score', 0):.6f})."
    summary += " " + " ".join([d['text'] for d in driver_statements])
    if disclosures:
        summary += " " + " ".join(disclosures.values())
    action = eo.get("recommended_action_class", "none")
    summary += f" Recommended action: {action}."
    return {
        "persona": persona,
        "summary": summary,
        "drivers_used": [d["name"] for d in driver_statements],
        "driver_statements": driver_statements,
        "disclosures": disclosures,
        "action_recommendation": action,
    }

def generate_constrained_llm_narrative(eo: Dict[str, Any], persona: str, engine_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Placeholder: Replace with real LLM call, or use template as fallback/default
    return generate_template_narrative(eo, persona)

def schema_valid(narr: Dict[str, Any]) -> bool:
    if SCHEMA_VALIDATION:
        try:
            jsonschema.validate(instance=narr, schema=NARRATIVE_SCHEMA)
            return True
        except Exception as e:
            logging.warning(f"Jsonschema validation failed: {e}")
            return False
    else:
        required = ["persona", "summary", "drivers_used", "driver_statements", "disclosures", "action_recommendation"]
        return all(k in narr for k in required)

def eo_consistent(narr: Dict[str, Any], eo: Dict[str, Any]) -> bool:
    eo_driver_names = [drv["name"] for drv in eo.get("top_drivers", [])]
    for drv in narr.get("driver_statements", []):
        if drv["name"] not in eo_driver_names:
            return False
        eo_drv = next((d for d in eo["top_drivers"] if d["name"] == drv["name"]), None)
        if eo_drv and drv["direction"] != eo_drv["direction"]:
            return False
    if narr.get("action_recommendation") != eo.get("recommended_action_class"):
        return False
    return True

def disclosures_ok(narr: Dict[str, Any], eo: Dict[str, Any]) -> bool:
    if eo.get("monitoring", {}).get("drift_status") == "WARN":
        if "drift" not in narr.get("disclosures", {}):
            return False
    if eo.get("evidence_strength") == "LOW" or eo.get("thin_file_flag"):
        if "evidence" not in narr.get("disclosures", {}):
            return False
    return True

def template_fallback(eo: Dict[str, Any], persona: str, reason: str) -> Dict[str, Any]:
    narr = generate_template_narrative(eo, persona)
    narr["fallback_reason"] = reason
    return narr

def tiered_validator(
    eo: Dict[str, Any], 
    persona: str, 
    engine: str,
    engine_args: Optional[Dict[str, Any]] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    for attempt in range(max_retries):
        if engine == "template":
            output = generate_template_narrative(eo, persona)
        elif engine == "llm":
            output = generate_constrained_llm_narrative(eo, persona, engine_args=engine_args)
        else:
            raise ValueError(f"Unknown engine: {engine}")
        if not schema_valid(output):
            logging.warning(f"Schema invalid ({persona}) EO {eo.get('event_id', 'unknown')}")
            return template_fallback(eo, persona, reason="schema_invalid")
        if not eo_consistent(output, eo):
            logging.warning(f"EO consistency fail ({persona}) EO {eo.get('event_id', 'unknown')}, attempt {attempt}")
            continue
        if not disclosures_ok(output, eo):
            logging.warning(f"Disclosure fail ({persona}) EO {eo.get('event_id', 'unknown')}, attempt {attempt}")
            continue
        output["validator_attempt"] = attempt+1
        output["validator_version"] = VERSION
        return output
    logging.warning(f"Validator retries exhausted ({persona}) EO {eo.get('event_id', 'unknown')}")
    return template_fallback(eo, persona, reason="validator_retry_exhausted")

def faithfulness_metrics(narr: Dict[str, Any], eo: Dict[str, Any]) -> Dict[str, Any]:
    eo_drivers = [drv["name"] for drv in eo.get("top_drivers", [])]
    narr_drivers = narr.get("drivers_used", [])
    overlap = sum(1 for n in narr_drivers if n in eo_drivers) / max(len(narr_drivers), 1)
    direction_match = all(
        drv["direction"] == next((d["direction"] for d in eo["top_drivers"] if d["name"] == drv["name"]), None)
        for drv in narr.get("driver_statements", [])
    )
    action_match = narr.get("action_recommendation") == eo.get("recommended_action_class")
    return {
        "overlap_at_K": overlap,
        "direction_accuracy": direction_match,
        "action_consistency": action_match,
    }

def thin_file_check(narr: Dict[str, Any], eo: Dict[str, Any]) -> Dict[str, Any]:
    valid = (
        (eo.get("thin_file_flag") or eo.get("evidence_strength") == "LOW") and
        narr.get("action_recommendation") == "step-up" and
        "evidence" in narr.get("disclosures", {})
    )
    return {"thin_file_action_valid": valid}

def main(persona: str = None, engine: str = "llm", output_dir: str = "./output", engine_args: Optional[Dict[str, Any]] = None):
    run_metadata = {
        "run_id": RUN_ID,
        "version": VERSION,
        "timestamp": datetime.now().isoformat(),
        "engine": engine,
        "engine_args": engine_args if engine_args else {},
    }

    if persona and persona not in PERSONA_FILES:
        logging.error(f"Unknown persona: {persona}. Must be one of {list(PERSONA_FILES.keys())}")
        sys.exit(1)
    personas = [persona] if persona else list(PERSONA_FILES.keys())

    os.makedirs(output_dir, exist_ok=True)
    eval_log = []
    for p in personas:
        eo_file = PERSONA_FILES[p]
        eo_path = os.path.join(".", eo_file)
        if not os.path.exists(eo_path):
            logging.warning(f"EO file for persona {p} not found: {eo_path}")
            continue
        output_path = os.path.join(output_dir, f"narratives_{p}_{engine}.jsonl")
        with open(eo_path, "r") as fin, open(output_path, "w") as fout:
            count = 0
            for idx, line in enumerate(fin):
                # ---- EO validation ----
                eo = None
                if USE_PYDANTIC_VALIDATION:
                    try:
                        eo_dict = json.loads(line)
                        eo_obj = EvidenceObject.model_validate(eo_dict)
                        eo = eo_obj.model_dump()  # Pydantic -> dict
                        if not isinstance(eo.get('top_drivers', None), list):
                            raise ValidationError("EO missing top_drivers (not a list)")
                    except ValidationError as ve:
                        logging.error(f"EO failed Pydantic validation at line {idx}: {ve}")
                        continue
                    except Exception as e:
                        logging.error(f"Malformed EO (not JSON object) at line {idx}: {line.strip()[:80]}, error: {e}")
                        continue
                else:
                    try:
                        eo = json.loads(line)
                        if not isinstance(eo, dict) or not isinstance(eo.get('top_drivers', None), list):
                            logging.error(f"EO missing top_drivers or is not a dict at line {idx}: {line.strip()[:80]}")
                            continue
                    except Exception as e:
                        logging.error(f"Malformed EO (not JSON object) at line {idx}: {line.strip()[:80]}, error: {e}")
                        continue
                # ---- Narrative and metrics ----
                narr = tiered_validator(eo, p, engine, engine_args=engine_args)
                metrics = faithfulness_metrics(narr, eo)
                metrics.update(thin_file_check(narr, eo))
                result = {
                    "eo_event_id": eo.get("event_id", f"{p}_item_{count}"),
                    "persona": p,
                    "engine": engine,
                    "run_metadata": run_metadata,
                    "narrative": narr,
                    "metrics": metrics,
                }
                fout.write(json.dumps(result) + "\n")
                eval_log.append(result)
                count += 1
            logging.info(f"Persona {p}: processed {count} EOs; Output written to {output_path}")
    eval_summary_path = os.path.join(output_dir, "eval_summary_all.json")
    with open(eval_summary_path, "w") as fsum:
        json.dump({
            "run_metadata": run_metadata,
            "eval_log": eval_log
        }, fsum, indent=2)
    print(f"Experiment run complete. Summary at {eval_summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Orchestrate EO → Narrative → Validator → Evaluation pipeline.")
    parser.add_argument("--persona", type=str, default=None, help="Persona to run (default: all)")
    parser.add_argument("--engine", type=str, default="llm", help="Narrative engine: 'llm' or 'template'")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--engine_args", type=str, default=None, help="JSON string with extra engine args")
    args = parser.parse_args()
    engine_args = json.loads(args.engine_args) if args.engine_args else {}
    main(
        persona=args.persona,
        engine=args.engine,
        output_dir=args.output_dir,
        engine_args=engine_args
    )
