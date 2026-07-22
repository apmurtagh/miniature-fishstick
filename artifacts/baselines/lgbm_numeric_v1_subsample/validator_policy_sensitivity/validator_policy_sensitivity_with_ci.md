# Validator Policy Sensitivity with 95% Confidence Intervals

## Purpose

This note adds Wilson 95% confidence intervals and a headline SVG figure to the validator policy sensitivity analysis.

| Policy | Accepted rows | Review / fallback / incomplete rows | Accepted rate | 95% CI | Interpretation |
|---|---:|---:|---:|---:|---|
| operations_summary | 20000 | 0 | 1.0000 | 0.9998 to 1.0000 | Suitable for concise operational summaries. Not equivalent to audit-complete rendering. |
| audit_complete_all_driver | 7994 | 12006 | 0.3997 | 0.3929 to 0.4065 | Represents stricter audit-complete evidence rendering under current validator taxonomy. |
| direction_proxy_confirmed_review | 6565 | 13435 | 0.3282 | 0.3218 to 0.3348 | Review-flag policy only. This is not a definitive semantic-faithfulness measure. |

## Figure

`artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_figure.svg`

## Thesis-Safe Interpretation

The policy rates are calculated over 20,000 rows and are therefore precise as automated policy rates. They are not human semantic validation or production approval.
