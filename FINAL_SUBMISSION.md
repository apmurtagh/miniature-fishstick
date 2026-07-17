# Final MSc Thesis Submission Package

## Repository State

This repository supports the final MSc thesis submission for:

**Governance-Ready Fraud Decisioning for Transaction Monitoring: Evidence-Grounded Narrative Explanations with Tiered Guardrails**

| Item | Value |
|---|---|
| Final branch | `proposal-gap-uplift-20260713_100844` |
| Final tag | `thesis-final-v23-proxy-cohort-diagnostics-20260717` |
| Prior evidence tag | `thesis-final-v23-proxy-cohort-diagnostics-20260717` |
| Pull request | `#13 Final thesis reproducibility and governance artefacts` |
| Notebook | `notebooks/governance_ready_fraud_decisioning_end_to_end_reproduction.ipynb` |
| Artefact manifest | `docs/thesis_final/final_submission_artifact_manifest.md` |
| Governance runbook | `docs/thesis_final/final_governance_controls_runbook.md` |
| Final hygiene tests | `tests/test_final_thesis_hygiene.py` |

If PR #13 remains unmerged at submission, the immutable submission state is the named branch and final tag, not the default `main` branch.

## Central Thesis Finding

The thesis demonstrates a governance-controllable Evidence Object to narrative architecture for fraud decisioning.

The central empirical finding is the distinction between:

1. **Operations-summary acceptance**: concise LLM summaries can preserve risk/action framing and required disclosures at scale.
2. **Audit-complete evidence rendering**: complete top-driver evidence requires deterministic templates, stricter driver coverage enforcement, fallback, or incomplete-output labelling.

## Final Evidence Base

| Evidence area | Final status |
|---|---|
| Fraud baseline | LightGBM testbed with ROC-AUC 0.8687, PR-AUC 0.4594, Brier 0.0236 and 10-bin ECE 0.0043 |
| Template narratives | 20,000 deterministic template narratives |
| LLM robustness | 20,000 constrained LLM narratives and 20,000 matched audit records |
| LLM clean outputs | 7,994 accepted-clean outputs |
| LLM flagged outputs | 12,006 accepted-with-driver-omission outputs |
| Fallbacks | 0 fallback rows under operations-summary policy |
| Validator policy sensitivity | 20,000 operations-summary accepted; 7,994 audit-complete all-driver accepted; 6,565 direction-proxy-confirmed; SVG figure and Wilson 95% CIs added |
| Feature masking | Exact model-ready `X_test` reconstruction plus 1,000-row masking re-score |
| Stability | 800 regeneration outputs across perturbation variants; quantitative RQ2 stability summary added |
| Drift | Offline TransactionDT drift diagnostics |
| Optional portability | Cyber, crypto and graph/GNN extension at protocol level only |
| Proxy cohort diagnostics | Operational proxy cohort score/outcome diagnostics added; not a fairness audit or fairness-performance claim |

## Claim Boundaries

The thesis does **not** claim:

- human readability superiority;
- human simulatability or demonstrated analyst utility;
- causal reduction of over-trust;
- fairness performance;
- live production monitoring;
- cross-domain model transfer;
- production readiness;
- regulatory certification;
- audit-complete LLM generation by default.

## Final Test Command

Run:

```bash
PYTHONPATH=src pytest -q tests/test_final_thesis_hygiene.py
```

Expected result:

```text
11 passed
```

## Productisation Path

The most natural productisation path is:

1. configurable validator policy engine;
2. EO schema versioning;
3. deterministic evidence-complete fallback service;
4. flagged narrative review queue;
5. model card and narrative card generation;
6. production-style enrichment outage simulation;
7. dashboard for audit, drift and evidence-completeness monitoring.
