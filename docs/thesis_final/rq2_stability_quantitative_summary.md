# RQ2 Stability Quantitative Summary

## Purpose

This note converts the regeneration stability artefact from an output-count statement into a quantitative RQ2 result. It reports variant-level overlap, decision-frame preservation and the randomised-driver evidence-following gap.

## Variant Results

| Variant | Rows | Variant-driver overlap | Original-driver overlap | All variant drivers | At least one variant driver | Risk present | Action present | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 200 | 0.695 | 0.695 | 0.430 | 1.000 | 1.000 | 1.000 | 0.000 |
| driver_order_shuffled | 200 | 0.697 | 0.697 | 0.410 | 1.000 | 1.000 | 1.000 | 0.000 |
| magnitude_perturbed | 200 | 0.708 | 0.708 | 0.425 | 1.000 | 1.000 | 1.000 | 0.000 |
| randomised_drivers | 200 | 0.763 | 0.239 | 0.495 | 1.000 | 1.000 | 1.000 | 0.000 |

## Derived Stability Indicators

| Indicator | Value | Interpretation |
|---|---:|---|
| Total regeneration outputs | 800 | Four variants, 200 rows each. |
| Stable-variant average overlap | 0.700 | Original, shuffled and magnitude-perturbed variants remain similar. |
| Randomised variant-driver overlap | 0.763 | Narratives follow the supplied randomised driver set. |
| Randomised original-driver overlap | 0.239 | Narratives do not simply repeat original drivers. |
| Randomised evidence-following gap | 0.524 | Higher gap supports evidence sensitivity. |
| Risk present across variants | True | Decision framing preserved. |
| Action present across variants | True | Decision framing preserved. |
| Zero fallback across variants | True | No generation fallback in these stability runs. |

## Thesis-Safe Interpretation

RQ2 is materially supported by automated regeneration evidence. Stable perturbations preserve risk/action framing and broadly similar driver overlap. The randomised-driver condition shows a large evidence-following gap, with higher overlap to supplied randomised drivers than to original drivers. This supports evidence sensitivity, but remains an automated proxy rather than human semantic validation.
