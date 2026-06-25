from __future__ import annotations

import argparse
from pathlib import Path

from orchestrate_narrative_experiments import write_template_narratives


DEFAULT_RUN_DIR = Path("artifacts") / "baselines" / "lgbm_numeric_v1_subsample"

# Align with the enriched EO path now proven to work
DEFAULT_EOS_PATH = DEFAULT_RUN_DIR / "eos_test_with_drivers.jsonl"

# Keep the canonical output name
DEFAULT_OUT_PATH = DEFAULT_RUN_DIR / "narratives_ops_triage_template.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate deterministic template narratives from enriched EO JSONL."
    )
    ap.add_argument(
        "--eos-jsonl",
        default=str(DEFAULT_EOS_PATH),
        help="Path to EO JSONL. Defaults to enriched EO output.",
    )
    ap.add_argument(
        "--out-jsonl",
        default=str(DEFAULT_OUT_PATH),
        help="Path to output narrative JSONL.",
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
    out_path = Path(args.out_jsonl)

    if not eos_path.exists():
        raise FileNotFoundError(
            f"EO file not found: {eos_path}\n"
            f"Run upstream steps first:\n"
            f"  PYTHONPATH=src python scripts/emit_eos_v1.py\n"
            f"  PYTHONPATH=src python scripts/attach_shap_drivers_v1.py"
        )

    print(f"Reading EO from: {eos_path}")
    print(f"Writing output to: {out_path}")

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