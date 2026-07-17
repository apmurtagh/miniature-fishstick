# Proxy Cohort Diagnostics Summary

## Purpose

This diagnostic inspects model score and outcome differences across available operational proxy cohorts. It is included to address the proposal's proxy-disparity theme without claiming protected-class fairness evidence.

## Scope Boundary

IEEE-CIS proxy fields such as product, card, address, email-domain and device attributes are operational fields, not protected-class labels. The results below are therefore proxy cohort diagnostics, not a fairness audit and not fairness-performance evidence.

## Overall Settings

- Rows evaluated: 20000
- Minimum group size for gap summaries: 100
- Top score threshold, 95th percentile: 0.144201
- Cohort fields evaluated: ProductCD, card4, card6, addr1, addr2, P_emaildomain, R_emaildomain, DeviceType

## Field-Level Gap Summary

| Field | Eligible groups | Mean score gap | Fraud-rate gap | High-score top-5% rate gap |
|---|---:|---:|---:|---:|
| ProductCD | 5 | 0.1086 | 0.1107 | 0.1778 |
| card4 | 5 | 0.0198 | 0.0548 | 0.0568 |
| card6 | 3 | 0.0424 | 0.0552 | 0.0758 |
| addr1 | 11 | 0.1109 | 0.1177 | 0.1877 |
| addr2 | 2 | 0.1028 | 0.1085 | 0.1654 |
| P_emaildomain | 11 | 0.1149 | 0.1295 | 0.1692 |
| R_emaildomain | 6 | 0.1062 | 0.1218 | 0.1838 |
| DeviceType | 3 | 0.0885 | 0.0984 | 0.1625 |

## Interpretation

Large gaps across operational cohorts should be treated as prompts for further portfolio-specific investigation. They do not identify protected-class disparity and do not prove unfairness or fairness. A production fairness review would require legally and ethically appropriate cohort definitions, governance approval, and calibrated fairness metrics.

## Artefacts

- Group-level CSV: `artifacts/baselines/lgbm_numeric_v1_subsample/proxy_cohort_diagnostics/proxy_cohort_diagnostics_by_group.csv`
- Machine-readable summary: `artifacts/baselines/lgbm_numeric_v1_subsample/proxy_cohort_diagnostics/proxy_cohort_diagnostics_summary.json`
