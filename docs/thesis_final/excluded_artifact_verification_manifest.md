# Excluded Artefact Verification Manifest

## Purpose

This manifest records existence, size, SHA-256 checksum and row-count information for key generated artefacts where those artefacts are present in the local checkout. It supports traceability while keeping large generated files out of Git.

| Artefact | Exists | Size bytes | Row count | SHA-256 | Note |
|---|---:|---:|---:|---|---|
| `artifacts/baselines/lgbm_numeric_v1_subsample/metrics.json` | True | 893 |  | 3797832c12b77bc5... | Available artefact or lightweight summary. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/test_predictions.csv` | True | 617048 | 20000 | 25ca5eaa89e407c6... | Available artefact or lightweight summary. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/model.txt` | True | 965497 |  | 7537c119fc0b02dd... | Generated/heavy artefact; intentionally excluded from Git unless explicitly lightweight. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/X_test.parquet` | True | 719048 |  | 908f4ee174b72a36... | Generated/heavy artefact; intentionally excluded from Git unless explicitly lightweight. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/model_ready_x_test_manifest.json` | True | 775 |  | b620d5a8981c76fa... | Available artefact or lightweight summary. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/narratives_ops_triage_template.jsonl` | True | 9117111 | 20000 | e6e1cdb477bf0f4f... | Generated/heavy artefact; intentionally excluded from Git unless explicitly lightweight. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/narratives_ops_triage_llm_5753rows_backup.jsonl` | True | 3963802 | 5753 | 15f4f4c8f7893410... | Generated/heavy artefact; intentionally excluded from Git unless explicitly lightweight. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/llm_20000_robustness/narratives_ops_triage_llm_20000_resume_safe.jsonl` | True | 13702160 | 20000 | 948d3cb641b34b4b... | Generated/heavy artefact; intentionally excluded from Git unless explicitly lightweight. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/llm_20000_robustness/llm_20000_attempt_audit.jsonl` | True | 9854089 | 20000 | e1c355f77c482b3c... | Generated/heavy artefact; intentionally excluded from Git unless explicitly lightweight. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/llm_20000_robustness/llm_20000_robustness_summary.json` | True | 324 |  | 1fca5ee1a032e023... | Available artefact or lightweight summary. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/feature_masking_rescore/feature_masking_rescore_summary.json` | True | 1762 |  | d9cb716832a66b42... | Available artefact or lightweight summary. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/semantic_validator_proxy/semantic_validator_proxy_summary.json` | True | 484 |  | 291266d1f8994ea1... | Available artefact or lightweight summary. |
| `artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_summary.json` | True | 1686 |  | f8b560b5f88999fe... | Available artefact or lightweight summary. |

## Interpretation

This manifest is not a substitute for storing all generated artefacts in Git. It records local verification metadata for the final submission state and helps reviewers understand which artefacts are lightweight, generated, heavy or externally retained.
