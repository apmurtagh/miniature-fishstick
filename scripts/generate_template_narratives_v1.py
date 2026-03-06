from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from eo.evidence_object import EvidenceObject, DriftStatus, EvidenceStrength


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test.jsonl"
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "template_narratives_ops_triage.jsonl"


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_ops_triage(eo: EvidenceObject) -> str:
    """
    Deterministic, EO-grounded template narrative for ops triage persona.

    Closed-world constraints:
    - only reference EO fields
    - do not invent external facts
    """
    parts: list[str] = []
    parts.append(f"Risk band: {eo.risk_band.value} (score {eo.score:.3f}).")

    if eo.top_drivers:
        inc = [d for d in eo.top_drivers if d.direction == "+"]
        dec = [d for d in eo.top_drivers if d.direction == "-"]
        if inc:
            parts.append("Drivers increasing risk: " + ", ".join(d.name for d in inc[:3]) + ".")
        if dec:
            parts.append("Mitigating signals: " + ", ".join(d.name for d in dec[:2]) + ".")
    else:
        parts.append("Top drivers: not available (EO top_drivers is empty).")

    # Level-3 disclosure gate (proposal): uncertainty / thin-file / drift messaging
    disclosures: list[str] = []
    if eo.thin_file_flag or eo.evidence_strength == EvidenceStrength.LOW:
        disclosures.append("Evidence is limited (thin-file / low evidence strength).")
    if eo.monitoring.drift_status in {DriftStatus.WARN, DriftStatus.FAIL}:
        disclosures.append(f"Monitoring: drift status {eo.monitoring.drift_status.value}; treat with caution.")
    if disclosures:
        parts.append(" ".join(disclosures))

    # Must match EO recommended_action_class
    parts.append(f"Recommended action: {eo.recommended_action_class.value}.")

    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos-jsonl", default=str(DEFAULT_EOS_PATH))
    ap.add_argument("--out-jsonl", default=str(DEFAULT_OUT_PATH))
    ap.add_argument("--persona", default="ops_triage", choices=["ops_triage"])
    args = ap.parse_args()

    eos_path = Path(args.eos_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for obj in iter_jsonl(eos_path):
            eo = EvidenceObject.model_validate(obj)

            if args.persona == "ops_triage":
                text = render_ops_triage(eo)
            else:
                raise ValueError(f"Unsupported persona: {args.persona}")

            out = {
                "schema_version": "template_narrative_v0",
                "persona": args.persona,
                "event_id": eo.event_id,
                "git_sha": eo.git_sha,
                "model": eo.model,
                "split_id": eo.split_id,
                "recommended_action_class": eo.recommended_action_class.value,
                "text": text,
            }
            f.write(json.dumps(out, ensure_ascii=False))
            f.write("\n")
            n += 1

    print("Wrote:", out_path)
    print("Rows:", n)


if __name__ == "__main__":
    main()
