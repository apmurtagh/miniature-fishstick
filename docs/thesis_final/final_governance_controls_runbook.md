# Final Governance Controls Runbook

## Purpose

This runbook translates the final dissertation artefacts into practical governance controls. It does not claim production deployment. It documents how the offline evidence could be operationalised in a controlled implementation.

## 1. Candidate Drift Threshold Policy

The dissertation includes offline production-style drift diagnostics. The thresholds below are illustrative operational starting points and would require calibration on production data before deployment.

| Metric | Green | Amber | Red | Candidate response |
|---|---:|---:|---:|---|
| Score PSI | < 0.10 | 0.10 to 0.25 | > 0.25 | Review score distribution shift and recent feature movements |
| Score KS | < 0.05 | 0.05 to 0.10 | > 0.10 | Review temporal distribution shift |
| Action TVD | < 0.02 | 0.02 to 0.05 | > 0.05 | Review decision-policy/action stability |
| Evidence TVD | < 0.03 | 0.03 to 0.08 | > 0.08 | Review evidence-strength population mix |
| Top-driver JSD | < 0.02 | 0.02 to 0.05 | > 0.05 | Review explanation-driver stability |
| ECE | < 0.01 | 0.01 to 0.03 | > 0.03 | Trigger calibration review |

## 2. Thin-File Production Outage Simulation Design

The completed feature-masking re-score is an inference-time controlled stress test. A production-style enrichment outage simulation would extend it as follows:

1. Define coherent feature groups, such as identity, device, velocity, behavioural history and address/enrichment features.
2. Mask one feature group at a time, not only random feature proportions.
3. Re-score the exact model-ready `X_test` matrix.
4. Recompute risk bands and recommended actions.
5. Regenerate EOs using the masked-feature score/action result.
6. Regenerate template and LLM narratives.
7. Measure action movement, disclosure compliance, fallback rate and driver-coverage changes.
8. Compare results against operational capacity and customer-friction tolerances.

This would simulate enrichment degradation more realistically than random feature masking.

## 3. Semantic Validator Proxy

The thesis primarily uses automated proxy evidence. A lightweight semantic validator can add an additional non-human control layer by checking:

- risk band mention;
- recommended action mention;
- limited-evidence disclosure when evidence is LOW;
- risk-increasing language for positive drivers;
- mitigating language for negative drivers;
- absence of unsupported hard-action language under LOW evidence unless the EO action requires it.

This remains a proxy. It does not replace human semantic review.

## 4. Graph/GNN Evidence-Source Checklist

The optional entity-graph/GNN extension should be treated as an evidence-source adapter. Before using graph-derived evidence in narratives, check:

| Check | Requirement |
|---|---|
| Node identity | `event_id` and graph node ID are traceable |
| Edge semantics | Edge types are documented and domain-specific |
| Neighbourhood depth | Depth is finite and recorded |
| Graph features | Degree, centrality, community risk or similar features are logged |
| GNN score | If available, score source and model version are logged |
| Sparse graph flag | Sparse-neighbourhood or low-coverage condition is disclosed |
| Narrative grounding | Narrative uses only EO-provided graph facts |
| Action consistency | Action matches `recommended_action_class` |
| Disclosure | Sparse graph or limited attribution requires explicit disclosure |

## 5. Operational Boundary

These controls support governance readiness but do not prove production readiness. Human workflow testing, portfolio-specific threshold calibration, live monitoring and model-risk approval remain outside the dissertation scope.
