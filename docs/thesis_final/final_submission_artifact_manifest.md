# Final Submission Artefact Manifest

## Purpose

This manifest documents the final artefacts supporting the dissertation submission. It distinguishes source code, reproducibility artefacts, lightweight summaries and heavy generated outputs.

## Final thesis interpretation

The original 5,753-row LLM run is retained as historical/intermediate evidence. The final LLM scale evidence is the 20,000-row audited robustness run. The remaining LLM limitation is driver completeness, not sample size.

## Core artefacts

| Artefact | Purpose |
|---|---|
| `artifacts/baselines/lgbm_numeric_v1_subsample/metrics.json` | Baseline model metrics |
| `artifacts/baselines/lgbm_numeric_v1_subsample/test_predictions.csv` | Test predictions used for EO generation and evaluation |
| `artifacts/baselines/lgbm_numeric_v1_subsample/eos_test_with_drivers.jsonl` | Evidence Objects enriched with SHAP drivers |
| `artifacts/baselines/lgbm_numeric_v1_subsample/eos_test_with_drivers_with_transactiondt.jsonl` | EO artefact enriched with TransactionDT for drift diagnostics |
| `artifacts/baselines/lgbm_numeric_v1_subsample/narratives_ops_triage_template.jsonl` | Final 20,000-row deterministic template narrative output |
| `artifacts/baselines/lgbm_numeric_v1_subsample/narratives_ops_triage_llm_5753rows_backup.jsonl` | Original 5,753-row accepted LLM run retained as historical evidence |
| `artifacts/baselines/lgbm_numeric_v1_subsample/llm_20000_robustness/narratives_ops_triage_llm_20000_resume_safe.jsonl` | Final full 20,000-row constrained LLM robustness output |
| `artifacts/baselines/lgbm_numeric_v1_subsample/llm_20000_robustness/llm_20000_attempt_audit.jsonl` | Matched 20,000-row application-level audit log |

## Lightweight final summaries

| Artefact | Purpose |
|---|---|
| `llm_20000_robustness/llm_20000_robustness_summary.md` | Final 20,000-row LLM robustness summary |
| `regeneration_stability/regeneration_stability_summary.md` | Regeneration-based stability summary |
| `drift_metric_suite/drift_metric_suite_summary.md` | Offline production-style drift diagnostics |
| `feature_masking_rescore/thin_file_re_score_attempt_closure.md` | Thin-file re-score attempted-closure note |
| `portability_appendix/eo_protocol_portability_appendix.md` | Optional protocol-level portability appendix |

## Notebook

| Artefact | Purpose |
|---|---|
| `notebooks/governance_ready_fraud_decisioning_end_to_end_reproduction.ipynb` | End-to-end aligned reproduction and commentary notebook |

## Git policy

Large `.jsonl`, `.parquet`, raw data, model files and full generated outputs should generally remain untracked unless deliberately archived through a release package or external storage. Source scripts, notebooks, runbooks and lightweight summaries should be committed.

## Optional Cross-Domain and Graph Portability Extension

| Artefact | Purpose |
|---|---|
| `docs/thesis_final/cross_domain_graph_portability_extension.md` | Protocol-level cyber, crypto and entity-graph/GNN portability extension |
| `artifacts/baselines/lgbm_numeric_v1_subsample/cross_domain_graph_portability/cross_domain_graph_portability_summary.md` | Lightweight summary of the cross-domain portability extension |
| `artifacts/baselines/lgbm_numeric_v1_subsample/cross_domain_graph_portability/cross_domain_graph_portability_examples.jsonl` | Tiny illustrative EO examples for cyber, crypto and graph contexts |
| `artifacts/baselines/lgbm_numeric_v1_subsample/cross_domain_graph_portability/cross_domain_graph_portability_status.json` | Status metadata for the optional extension |

This optional extension addresses cyber portability, crypto portability and the entity-graph/GNN enhancement at protocol level only. It does not claim trained cross-domain model performance.

## Final Low-Effort Governance Uplifts

| Artefact | Purpose |
|---|---|
| `docs/thesis_final/final_governance_controls_runbook.md` | Drift threshold policy, production outage simulation design and graph/GNN checklist |
| `scripts/semantic_validator_proxy.py` | Lightweight rule-based semantic validator proxy |
| `artifacts/baselines/lgbm_numeric_v1_subsample/semantic_validator_proxy/semantic_validator_proxy_summary.md` | Semantic validator proxy result summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/semantic_validator_proxy/semantic_validator_proxy_summary.json` | Semantic validator proxy result metadata |

These artefacts strengthen non-human governance evidence without claiming human semantic validation, production monitoring or trained cross-domain model performance.

## Final Closure and Claim-Calibration Artefacts

| Artefact | Purpose |
|---|---|
| `docs/thesis_final/rq_hypothesis_disposition.md` | Explicitly closes RQs and hypotheses as supported, partially supported, operationalised or not directly tested |
| `docs/thesis_final/validator_acceptance_policy.md` | Clarifies clean acceptance, flagged acceptance, hard failure, fallback and audit-complete use |
| `docs/thesis_final/direction_proxy_interpretation.md` | Explains the 0.3282 direction-proxy result as a conservative review flag |
| `docs/thesis_final/feature_masking_sample_rationale.md` | Explains why the completed feature-masking re-score used a 1,000-row sample |

These artefacts close the remaining proposal-to-thesis interpretation gaps without adding unsupported claims.

## Final Metric and Interpretation Notes

| Artefact | Purpose |
|---|---|
| `docs/thesis_final/metric_definitions_and_interpretation.md` | Defines driver precision, driver coverage/recall, top-k overlap, direction proxy, action consistency and disclosure compliance |
| `docs/thesis_final/thin_file_masking_interpretation.md` | Interprets the completed 1,000-row feature-masking re-score and explains median-replacement score suppression |
| `docs/thesis_final/baseline_and_calibration_scope.md` | Clarifies that the LightGBM baseline is a sufficient testbed, not a state-of-the-art or production model claim |

These notes strengthen the final thesis interpretation without expanding unsupported empirical claims.

## Validator Policy Sensitivity Analysis

| Artefact | Purpose |
|---|---|
| `scripts/validator_policy_sensitivity.py` | Quantifies operations-summary, audit-complete and direction-proxy-confirmed validation policies |
| `docs/thesis_final/validator_policy_sensitivity.md` | Thesis-facing validator policy sensitivity summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_summary.md` | Artefact-side policy sensitivity summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_summary.json` | Machine-readable policy sensitivity result |

This analysis quantifies the central governance trade-off: scalable operations-summary acceptance versus stricter audit-complete evidence rendering.

## Final Submission Pack

| Artefact | Purpose |
|---|---|
| `FINAL_draft_SUBMISSION.md` | One-page final submission entry point for examiners, reviewers and future productisation |
| `docs/thesis_final/excluded_artifact_verification_manifest.md` | Human-readable checksum, size and row-count manifest for excluded/generated artefacts |
| `docs/thesis_final/excluded_artifact_verification_manifest.json` | Machine-readable checksum, size and row-count manifest |

This final submission pack improves discoverability and traceability while preserving Git hygiene for heavy generated artefacts.

## Validator Policy Sensitivity Figure and Confidence Intervals

| Artefact | Purpose |
|---|---|
| `docs/thesis_final/validator_policy_sensitivity_figure.md` | Thesis-facing policy sensitivity figure and CI note |
| `artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_figure.svg` | Dependency-free headline policy sensitivity SVG figure |
| `artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_with_ci.md` | Artefact-side policy sensitivity confidence interval summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_with_ci.json` | Machine-readable policy sensitivity confidence interval result |

This figure and CI summary make the central operations-summary versus audit-complete governance trade-off visually and statistically explicit.

## RQ2 Stability Quantitative Summary

| Artefact | Purpose |
|---|---|
| `scripts/rq2_stability_quantitative_summary.py` | Generates quantitative RQ2 stability summary from regeneration-stability artefacts |
| `docs/thesis_final/rq2_stability_quantitative_summary.md` | Thesis-facing RQ2 stability summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/regeneration_stability/rq2_stability_quantitative_summary.md` | Artefact-side RQ2 stability summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/regeneration_stability/rq2_stability_quantitative_summary.json` | Machine-readable RQ2 stability result |

This summary converts the 800-output regeneration experiment into quantitative evidence for RQ2, including the randomised-driver evidence-following gap.

## Proxy Cohort Diagnostics

| Artefact | Purpose |
|---|---|
| `scripts/proxy_cohort_diagnostics.py` | Generates operational proxy cohort diagnostics over available IEEE-CIS fields |
| `docs/thesis_final/proxy_cohort_diagnostics.md` | Thesis-facing proxy cohort diagnostics summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/proxy_cohort_diagnostics/proxy_cohort_diagnostics_summary.md` | Artefact-side proxy cohort diagnostics summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/proxy_cohort_diagnostics/proxy_cohort_diagnostics_summary.json` | Machine-readable proxy cohort diagnostics summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/proxy_cohort_diagnostics/proxy_cohort_diagnostics_by_group.csv` | Group-level proxy cohort diagnostic table |

This diagnostic addresses the proposal's proxy-disparity theme while explicitly avoiding protected-class fairness claims.

## H2 Unconstrained Ablation Pilot

| Artefact | Purpose |
|---|---|
| `docs/thesis_final/h2_unconstrained_ablation_pilot.md` | Thesis-facing bounded H2 ablation summary |
| `scripts/compare_h2_unconstrained_vs_constrained_100.py` | Same-row constrained vs unconstrained 100-row comparison script |
| `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_unconstrained_summary_100.md` | Unconstrained 100-row ablation summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_unconstrained_summary_100.json` | Machine-readable unconstrained 100-row ablation metrics |
| `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_unconstrained_vs_constrained_summary_100.md` | Same-row constrained vs unconstrained comparison summary |
| `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_unconstrained_vs_constrained_summary_100.json` | Machine-readable same-row H2 comparison |

This bounded 100-row automated ablation addresses the proposal's H2 / Condition C theme while remaining explicitly proxy-based and not human semantic validation.
