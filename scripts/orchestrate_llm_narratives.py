from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eo.evidence_object import EvidenceObject
from narratives.template_engine import generate_template_narrative


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "narratives_ops_triage_llm.jsonl"


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_fallback_record(eo, persona, top_k, model_name, reason):
    text = generate_template_narrative(
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
        "text": text,
        "llm_model": model_name,
        "validator_status": "fallback",
        "fallback_used": True,
        "validator_reason": reason,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Generate LLM narratives with graceful fallback when no API key is available."
    )
    ap.add_argument("--eos-jsonl", default=str(DEFAULT_EOS_PATH))
    ap.add_argument("--out-jsonl", default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--persona", default="ops_triage", choices=["ops_triage"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    eos_path = Path(args.eos_jsonl)
    out_path = Path(args.out_jsonl)

    if not eos_path.exists():
        raise FileNotFoundError(f"EO file not found: {eos_path}")

    api_key = os.getenv("OPENAI_API_KEY")
    llm_enabled = bool(api_key and api_key.strip())

    llm_reason = None
    generate_llm_narrative = None

    if llm_enabled:
        try:
            from narratives.llm_engine import generate_llm_narrative as _generate_llm_narrative
            generate_llm_narrative = _generate_llm_narrative
        except Exception as e:
            llm_enabled = False
            llm_reason = f"llm_import_failed:{type(e).__name__}"
    else:
        llm_reason = "missing_openai_api_key"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Reading EO from:", eos_path)
    print("Writing LLM narratives to:", out_path)
    print("Model:", args.model)
    print("LLM mode:", "ENABLED" if llm_enabled else f"DISABLED -> fallback only ({llm_reason})")

    rows = 0
    accepted_rows = 0
    fallback_rows = 0

    with out_path.open("w", encoding="utf-8") as f:
        for obj in iter_jsonl(eos_path):
            eo = EvidenceObject.model_validate(obj)

            if llm_enabled:
                try:
                    out = generate_llm_narrative(
                        eo,
                        persona=args.persona,
                        top_k=int(args.top_k),
                        model=args.model,
                        max_retries=int(args.max_retries),
                    )
                except Exception as e:
                    out = build_fallback_record(
                        eo,
                        persona=args.persona,
                        top_k=int(args.top_k),
                        model_name=args.model,
                        reason=f"llm_runtime_error:{type(e).__name__}",
                    )
            else:
                out = build_fallback_record(
                    eo,
                    persona=args.persona,
                    top_k=int(args.top_k),
                    model_name=args.model,
                    reason=llm_reason,
                )

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

            rows += 1
            if out.get("validator_status") == "accepted":
                accepted_rows += 1
            if out.get("fallback_used"):
                fallback_rows += 1

            if rows == 1:
                print("\nSample narrative:")
                print(out.get("text", "")[:400], "...\n")
                print("Sample metadata:")
                print(
                    json.dumps(
                        {
                            "validator_status": out.get("validator_status"),
                            "fallback_used": out.get("fallback_used"),
                            "validator_reason": out.get("validator_reason"),
                        },
                        indent=2,
                    )
                )
                print()

            if args.limit and rows >= args.limit:
                break

    print("✅ Completed")
    print("Rows written:", rows)
    print("Accepted rows:", accepted_rows)
    print("Fallback rows:", fallback_rows)


if __name__ == "__main__":
    main()