import json
import math
import random
from pathlib import Path

IN_JSON = Path("artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_unconstrained_vs_constrained_summary_100.json")
OUT_MD = IN_JSON.parent / "h2_paired_uncertainty_addendum.md"
OUT_JSON = IN_JSON.parent / "h2_paired_uncertainty_addendum.json"

N_BOOT = 10000
SEED = 20260722

def as_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "flagged", "present"}:
        return True
    if s in {"0", "false", "no", "n", "clean", "absent"}:
        return False
    return None

def exact_binom_two_sided(k, n):
    if n == 0:
        return None
    probs = []
    for i in range(n + 1):
        probs.append(math.comb(n, i) * (0.5 ** n))
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-15))

def mcnemar_exact(a, b):
    # a = constrained flag, b = unconstrained flag
    b01 = sum((x is False) and (y is True) for x, y in zip(a, b))
    b10 = sum((x is True) and (y is False) for x, y in zip(a, b))
    disc = b01 + b10
    p = exact_binom_two_sided(min(b01, b10), disc) if disc else None
    return b01, b10, disc, p

def bootstrap_ci(values):
    rng = random.Random(SEED)
    n = len(values)
    boots = []
    for _ in range(N_BOOT):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * N_BOOT)]
    hi = boots[int(0.975 * N_BOOT) - 1]
    return sum(values) / n, lo, hi

obj = json.loads(IN_JSON.read_text(encoding="utf-8"))
rows = obj.get("row_metrics", [])

if len(rows) != 100:
    raise SystemExit(f"Expected 100 row_metrics, found {len(rows)}")

pairs = []
for r in rows:
    event_id = r.get("event_id")
    c = r.get("constrained") or {}
    u = r.get("unconstrained") or {}

    c_cov = c.get("driver_coverage")
    u_cov = u.get("driver_coverage")

    if c_cov is None or u_cov is None:
        continue

    c_count = c.get("mentioned_driver_count")
    u_count = u.get("mentioned_driver_count")

    c_zero = (c_count == 0) if c_count is not None else (not as_bool(c.get("any_driver_mentioned")))
    u_zero = (u_count == 0) if u_count is not None else (not as_bool(u.get("any_driver_mentioned")))

    pairs.append({
        "event_id": event_id,
        "constrained_coverage": float(c_cov),
        "unconstrained_coverage": float(u_cov),
        "coverage_diff": float(c_cov) - float(u_cov),
        "constrained_zero_driver": bool(c_zero),
        "unconstrained_zero_driver": bool(u_zero),
        "constrained_review_flag": bool(as_bool(c.get("review_language_flag"))),
        "unconstrained_review_flag": bool(as_bool(u.get("review_language_flag"))),
    })

if len(pairs) != 100:
    raise SystemExit(f"Expected 100 usable pairs, found {len(pairs)}")

diffs = [p["coverage_diff"] for p in pairs]
mean_diff, ci_lo, ci_hi = bootstrap_ci(diffs)

cz = [p["constrained_zero_driver"] for p in pairs]
uz = [p["unconstrained_zero_driver"] for p in pairs]
z_b01, z_b10, z_disc, z_p = mcnemar_exact(cz, uz)

cr = [p["constrained_review_flag"] for p in pairs]
ur = [p["unconstrained_review_flag"] for p in pairs]
r_b01, r_b10, r_disc, r_p = mcnemar_exact(cr, ur)

summary = {
    "status": "complete",
    "source_file": str(IN_JSON),
    "n_pairs": len(pairs),
    "coverage": {
        "constrained_mean": sum(p["constrained_coverage"] for p in pairs) / len(pairs),
        "unconstrained_mean": sum(p["unconstrained_coverage"] for p in pairs) / len(pairs),
        "paired_mean_difference_constrained_minus_unconstrained": mean_diff,
        "paired_bootstrap_95_ci": [ci_lo, ci_hi],
        "n_boot": N_BOOT,
        "seed": SEED
    },
    "zero_driver_mcnemar": {
        "constrained_false_unconstrained_true": z_b01,
        "constrained_true_unconstrained_false": z_b10,
        "discordant_pairs": z_disc,
        "exact_two_sided_p": z_p
    },
    "review_language_mcnemar": {
        "constrained_false_unconstrained_true": r_b01,
        "constrained_true_unconstrained_false": r_b10,
        "discordant_pairs": r_disc,
        "exact_two_sided_p": r_p
    },
    "caveat": "Exploratory same-EO paired automated uncertainty check; not human semantic validation or population-level proof."
}

def fmt_p(p):
    if p is None:
        return "NA"
    if p < 0.0001:
        return f"{p:.3e}"
    return f"{p:.4f}"

md = f"""# H2 Paired Uncertainty Addendum

This addendum reuses the existing 100-row same-EO H2 constrained versus unconstrained ablation from `{IN_JSON.name}`. No additional LLM generation was performed.

## Coverage difference

- Paired rows: `{len(pairs)}`
- Mean constrained coverage: `{summary["coverage"]["constrained_mean"]:.3f}`
- Mean unconstrained coverage: `{summary["coverage"]["unconstrained_mean"]:.3f}`
- Paired mean difference, constrained minus unconstrained: `{mean_diff:.3f}`
- Paired bootstrap 95% CI: `[{ci_lo:.3f}, {ci_hi:.3f}]`
- Bootstrap resamples: `{N_BOOT}`
- Random seed: `{SEED}`

## Zero-driver row presence

- Constrained false / unconstrained true: `{z_b01}`
- Constrained true / unconstrained false: `{z_b10}`
- Discordant pairs: `{z_disc}`
- Exact McNemar-style two-sided p-value: `{fmt_p(z_p)}`

## Review-language flag presence

- Constrained false / unconstrained true: `{r_b01}`
- Constrained true / unconstrained false: `{r_b10}`
- Discordant pairs: `{r_disc}`
- Exact McNemar-style two-sided p-value: `{fmt_p(r_p)}`

## Interpretation and caveat

The paired check strengthens H2 as a bounded automated governance result: the constrained and unconstrained outputs are compared on the same 100 EOs, limiting input confounding. The check remains exploratory and automated. It is not human semantic validation, human utility evidence, or population-level proof across all prompts, models or deployment settings.
"""

OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
OUT_MD.write_text(md, encoding="utf-8")

print(md)
print("[OK] wrote", OUT_MD)
print("[OK] wrote", OUT_JSON)
