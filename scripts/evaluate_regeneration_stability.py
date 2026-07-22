import json
from pathlib import Path
import csv

ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
STAB = ART / "regeneration_stability"

VARIANTS = [
    "original",
    "driver_order_shuffled",
    "magnitude_perturbed",
    "randomised_drivers",
]

OUT_CSV = STAB / "regeneration_stability_metrics.csv"
OUT_MD = STAB / "regeneration_stability_summary.md"
OUT_JSON = STAB / "regeneration_stability_summary.json"


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


def top_driver_names(eo):
    drivers = eo.get("top_drivers") or eo.get("drivers") or []
    out = []
    for d in drivers:
        if isinstance(d, dict):
            name = d.get("name") or d.get("feature") or d.get("driver")
        else:
            name = str(d)
        if name is not None:
            out.append(str(name))
    return out


def risk_band(eo):
    return str(eo.get("risk_band") or eo.get("calibration_band") or eo.get("score_band") or "")


def action_class(eo):
    return str(eo.get("recommended_action_class") or eo.get("recommended_action") or eo.get("action") or "")


def flatten_text(obj):
    parts = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            parts.append(str(x))

    walk(obj)
    return " ".join(parts)


def narrative_text(row):
    for k in ["narrative", "text", "summary", "output", "response", "llm_output", "template_narrative"]:
        if k in row:
            return flatten_text(row[k])
    return flatten_text(row)


def mentions(text, token):
    return bool(token) and str(token).lower() in text.lower()


def get_metrics(variant, original_eos):
    eos = read_jsonl(STAB / f"stability_{variant}_eos_200.jsonl")
    outs = read_jsonl(STAB / f"stability_{variant}_llm_narratives.jsonl")
    n = min(len(eos), len(outs), len(original_eos))

    if n == 0:
        return {
            "variant": variant,
            "rows": 0,
            "variant_driver_overlap": None,
            "original_driver_overlap": None,
            "all_variant_drivers_rate": None,
            "at_least_one_variant_driver_rate": None,
            "risk_present_rate": None,
            "action_present_rate": None,
            "fallback_rate": None,
        }

    variant_overlaps = []
    original_overlaps = []
    all_variant = []
    atleast_variant = []
    risk_ok = []
    action_ok = []
    fallback = []

    for eo, original_eo, out in zip(eos[:n], original_eos[:n], outs[:n]):
        text = narrative_text(out)

        variant_drivers = top_driver_names(eo)
        original_drivers = top_driver_names(original_eo)

        mentioned_variant = [d for d in variant_drivers if mentions(text, d)]
        mentioned_original = [d for d in original_drivers if mentions(text, d)]

        variant_overlaps.append(len(set(variant_drivers) & set(mentioned_variant)) / max(1, len(set(variant_drivers))))
        original_overlaps.append(len(set(original_drivers) & set(mentioned_original)) / max(1, len(set(original_drivers))))

        all_variant.append(set(variant_drivers).issubset(set(mentioned_variant)))
        atleast_variant.append(len(mentioned_variant) > 0)
        risk_ok.append(mentions(text, risk_band(eo)))
        action_ok.append(mentions(text, action_class(eo)))
        fallback.append(bool(out.get("fallback_used", False)))

    return {
        "variant": variant,
        "rows": n,
        "variant_driver_overlap": sum(variant_overlaps) / n,
        "original_driver_overlap": sum(original_overlaps) / n,
        "all_variant_drivers_rate": sum(all_variant) / n,
        "at_least_one_variant_driver_rate": sum(atleast_variant) / n,
        "risk_present_rate": sum(risk_ok) / n,
        "action_present_rate": sum(action_ok) / n,
        "fallback_rate": sum(fallback) / n,
    }


original_eos = read_jsonl(STAB / "stability_original_eos_200.jsonl")
rows = [get_metrics(v, original_eos) for v in VARIANTS]

with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

OUT_JSON.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def fmt(x):
    return "NA" if x is None else f"{x:.3f}"


md = []
md.append("# Regeneration-Based Stability Summary\n\n")
md.append("| Variant | Rows | Variant-driver overlap | Original-driver overlap | All variant drivers | At least one variant driver | Risk present | Action present | Fallback |\n")
md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")

for r in rows:
    md.append(
        f"| {r['variant']} | {r['rows']} | {fmt(r['variant_driver_overlap'])} | "
        f"{fmt(r['original_driver_overlap'])} | {fmt(r['all_variant_drivers_rate'])} | "
        f"{fmt(r['at_least_one_variant_driver_rate'])} | {fmt(r['risk_present_rate'])} | "
        f"{fmt(r['action_present_rate'])} | {fmt(r['fallback_rate'])} |\n"
    )

md.append(
    "\nInterpretation: This regeneration-based stability check compares generated narratives across original, "
    "driver-order-shuffled, magnitude-perturbed and randomised-driver EO variants. For shuffled and magnitude-perturbed variants, "
    "stable narratives should broadly preserve the same decision framing. For randomised-driver variants, a faithful narrative should "
    "follow the randomised EO driver set rather than the original driver set; therefore, high variant-driver overlap combined with lower "
    "original-driver overlap indicates sensitivity to the supplied EO evidence. This is stronger than the earlier output-only randomisation proxy.\n"
)

OUT_MD.write_text("".join(md), encoding="utf-8")

print("[OK] Regeneration stability evaluation complete")
print(f"[OK] CSV: {OUT_CSV}")
print(f"[OK] MD: {OUT_MD}")
print("".join(md))
