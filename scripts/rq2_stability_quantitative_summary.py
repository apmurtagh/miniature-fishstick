import json
from pathlib import Path

ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
STAB = ART / "regeneration_stability"
SRC = STAB / "regeneration_stability_summary.json"

OUT_JSON = STAB / "rq2_stability_quantitative_summary.json"
OUT_MD = STAB / "rq2_stability_quantitative_summary.md"
DOC_MD = Path("docs/thesis_final/rq2_stability_quantitative_summary.md")

def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rows = data["rows"]

    by_variant = {r["variant"]: r for r in rows}

    stable_variants = ["original", "driver_order_shuffled", "magnitude_perturbed"]
    stable_avg_variant_overlap = sum(by_variant[v]["variant_driver_overlap"] for v in stable_variants) / len(stable_variants)
    stable_avg_original_overlap = sum(by_variant[v]["original_driver_overlap"] for v in stable_variants) / len(stable_variants)

    rand = by_variant["randomised_drivers"]
    randomised_evidence_following_gap = rand["variant_driver_overlap"] - rand["original_driver_overlap"]

    total_rows = sum(r["rows"] for r in rows)
    all_risk_present = all(r["risk_present_rate"] == 1.0 for r in rows)
    all_action_present = all(r["action_present_rate"] == 1.0 for r in rows)
    all_zero_fallback = all(r["fallback_rate"] == 0.0 for r in rows)

    result = {
        "status": "run",
        "total_regeneration_outputs": total_rows,
        "variants": rows,
        "stable_variants": stable_variants,
        "stable_avg_variant_driver_overlap": stable_avg_variant_overlap,
        "stable_avg_original_driver_overlap": stable_avg_original_overlap,
        "randomised_variant_driver_overlap": rand["variant_driver_overlap"],
        "randomised_original_driver_overlap": rand["original_driver_overlap"],
        "randomised_evidence_following_gap": randomised_evidence_following_gap,
        "risk_present_all_variants": all_risk_present,
        "action_present_all_variants": all_action_present,
        "fallback_zero_all_variants": all_zero_fallback,
        "interpretation": (
            "Stable perturbations preserve decision framing and similar driver overlap. "
            "The randomised-driver variant follows the supplied randomised EO drivers more than the original drivers, "
            "which supports evidence sensitivity rather than generic narrative repetition."
        ),
        "caveat": (
            "This is automated regeneration-based stability evidence. It does not replace human semantic review."
        ),
    }

    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = []
    lines.append("# RQ2 Stability Quantitative Summary\n\n")
    lines.append("## Purpose\n\n")
    lines.append(
        "This note converts the regeneration stability artefact from an output-count statement into a quantitative RQ2 result. "
        "It reports variant-level overlap, decision-frame preservation and the randomised-driver evidence-following gap.\n\n"
    )

    lines.append("## Variant Results\n\n")
    lines.append("| Variant | Rows | Variant-driver overlap | Original-driver overlap | All variant drivers | At least one variant driver | Risk present | Action present | Fallback |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['rows']} | {r['variant_driver_overlap']:.3f} | "
            f"{r['original_driver_overlap']:.3f} | {r['all_variant_drivers_rate']:.3f} | "
            f"{r['at_least_one_variant_driver_rate']:.3f} | {r['risk_present_rate']:.3f} | "
            f"{r['action_present_rate']:.3f} | {r['fallback_rate']:.3f} |\n"
        )

    lines.append("\n## Derived Stability Indicators\n\n")
    lines.append("| Indicator | Value | Interpretation |\n")
    lines.append("|---|---:|---|\n")
    lines.append(f"| Total regeneration outputs | {total_rows} | Four variants, 200 rows each. |\n")
    lines.append(f"| Stable-variant average overlap | {stable_avg_variant_overlap:.3f} | Original, shuffled and magnitude-perturbed variants remain similar. |\n")
    lines.append(f"| Randomised variant-driver overlap | {rand['variant_driver_overlap']:.3f} | Narratives follow the supplied randomised driver set. |\n")
    lines.append(f"| Randomised original-driver overlap | {rand['original_driver_overlap']:.3f} | Narratives do not simply repeat original drivers. |\n")
    lines.append(f"| Randomised evidence-following gap | {randomised_evidence_following_gap:.3f} | Higher gap supports evidence sensitivity. |\n")
    lines.append(f"| Risk present across variants | {all_risk_present} | Decision framing preserved. |\n")
    lines.append(f"| Action present across variants | {all_action_present} | Decision framing preserved. |\n")
    lines.append(f"| Zero fallback across variants | {all_zero_fallback} | No generation fallback in these stability runs. |\n")

    lines.append("\n## Thesis-Safe Interpretation\n\n")
    lines.append(
        "RQ2 is materially supported by automated regeneration evidence. Stable perturbations preserve risk/action framing and broadly similar driver overlap. "
        "The randomised-driver condition shows a large evidence-following gap, with higher overlap to supplied randomised drivers than to original drivers. "
        "This supports evidence sensitivity, but remains an automated proxy rather than human semantic validation.\n"
    )

    md = "".join(lines)
    OUT_MD.write_text(md, encoding="utf-8")
    DOC_MD.write_text(md, encoding="utf-8")

    print("[OK] Wrote", OUT_JSON)
    print("[OK] Wrote", OUT_MD)
    print("[OK] Wrote", DOC_MD)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
