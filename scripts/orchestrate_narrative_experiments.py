from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from eo.evidence_object import EvidenceObject
from narratives.template_engine import generate_template_narrative


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_template_narratives(
    eos_jsonl: Path,
    out_jsonl: Path,
    *,
    persona: str,
    top_k: int,
) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for obj in iter_jsonl(eos_jsonl):
            eo = EvidenceObject.model_validate(obj)

            text = generate_template_narrative(
                eo,
                persona=persona,
                top_k=int(top_k),
            )

            out = {
                "schema_version": "template_narrative_v0",
                "persona": persona,
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

    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Orchestrate template narrative generation from enriched EO JSONL."
    )
    ap.add_argument(
        "--eos-jsonl",
        default=str(DEFAULT_EOS_PATH),
        help="Path to EO JSONL. Defaults to enriched EO output.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory to write narrative artefacts into.",
    )
    ap.add_argument(
        "--persona",
        default="ops_triage",
        choices=["ops_triage"],
        help="Narrative persona to generate.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many EO top_drivers to mention (0 = all available).",
    )
    args = ap.parse_args()

    eos_path = Path(args.eos_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eos_path.exists():
        raise FileNotFoundError(
            f"EO file not found: {eos_path}\n"
            f"Run the upstream steps first:\n"
            f"  PYTHONPATH=src python scripts/emit_eos_v1.py\n"
            f"  PYTHONPATH=src python scripts/attach_shap_drivers_v1.py"
        )

    out_path = out_dir / f"narratives_{args.persona}_template.jsonl"

    print(f"Reading EO from: {eos_path}")
    print(f"Writing narratives to: {out_path}")

    rows = write_template_narratives(
        eos_jsonl=eos_path,
        out_jsonl=out_path,
        persona=args.persona,
        top_k=int(args.top_k),
    )

    print("✅ Completed")
    print("Rows written:", rows)


if __name__ == "__main__":
    main()