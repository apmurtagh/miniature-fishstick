# Thin-File Masking Interpretation

## Purpose

This note clarifies how the completed feature-masking re-score should be interpreted and why median replacement can suppress scores.

## Completed Result

The exact model-ready `X_test` matrix was reconstructed over all 20,000 rows and reproduced original baseline predictions to numerical precision. The feature-masking re-score was then run on a 1,000-row sample under 0%, 30% and 60% masking conditions.

| Mask rate | Rows | Mean base score | Mean masked score | Mean abs score delta | Action change rate | Step-up/review rate | HIGH risk rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1,000 | 0.0405 | 0.0405 | 0.0000 | 0.0000 | 0.0220 | 0.0080 |
| 0.3 | 1,000 | 0.0405 | 0.0267 | 0.0269 | 0.0200 | 0.0070 | 0.0000 |
| 0.6 | 1,000 | 0.0405 | 0.0332 | 0.0280 | 0.0210 | 0.0110 | 0.0000 |

## Interpretation

Median replacement can make masked transactions look more typical. If high-risk signals are replaced by train-set medians, the model may assign lower risk rather than higher uncertainty. This explains why mean masked scores and HIGH-risk rates can fall under masking.

The result therefore demonstrates controlled score/action sensitivity to degraded feature availability. It does not prove that reduced evidence automatically produces more cautious step-up actions.

## Action Logic

The masking re-score primarily tests model score and action recalibration under inference-time feature degradation. It should be distinguished from narrative disclosure logic:

- model score/action movement is driven by the trained LightGBM model and threshold/action policy;
- evidence-strength disclosures are EO/narrative governance controls;
- production-style enrichment outage simulation would require coherent feature-group masking, EO regeneration and narrative regeneration.

## Thesis-Safe Claim

The completed masking re-score strengthens RQ5 beyond disclosure-only evidence, but remains a controlled inference-time robustness stress test rather than a production enrichment-outage simulation.
