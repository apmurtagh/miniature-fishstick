# H2 Paired Uncertainty Addendum

This addendum reuses the existing 100-row same-EO H2 constrained versus unconstrained ablation from `h2_unconstrained_vs_constrained_summary_100.json`. No additional LLM generation was performed.

## Coverage difference

- Paired rows: `100`
- Mean constrained coverage: `0.690`
- Mean unconstrained coverage: `0.432`
- Paired mean difference, constrained minus unconstrained: `0.258`
- Paired bootstrap 95% CI: `[0.180, 0.336]`
- Bootstrap resamples: `10000`
- Random seed: `20260722`

## Zero-driver row presence

- Constrained false / unconstrained true: `27`
- Constrained true / unconstrained false: `0`
- Discordant pairs: `27`
- Exact McNemar-style two-sided p-value: `1.490e-08`

## Review-language flag presence

- Constrained false / unconstrained true: `64`
- Constrained true / unconstrained false: `1`
- Discordant pairs: `65`
- Exact McNemar-style two-sided p-value: `1.163e-16`

## Interpretation and caveat

The paired check strengthens H2 as a bounded automated governance result: the constrained and unconstrained outputs are compared on the same 100 EOs, limiting input confounding. The check remains exploratory and automated. It is not human semantic validation, human utility evidence, or population-level proof across all prompts, models or deployment settings.
