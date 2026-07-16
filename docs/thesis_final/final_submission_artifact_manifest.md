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
