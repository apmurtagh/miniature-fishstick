# Research Question and Hypothesis Disposition

## Purpose

This document explicitly distinguishes implemented deliverables from directly tested research claims. It closes the remaining proposal-to-thesis interpretation gaps around H1, H2, RQ3, RQ4, fairness/proxy disparity and optional portability.

## Research Question Disposition

| Item | Final disposition | Evidence / caveat |
|---|---|---|
| RQ1 Faithfulness | Strongly supported for templates; partially supported for LLMs | Templates preserve EO evidence completely under implemented checks. LLMs preserve risk/action and at least one driver, but average top-k overlap is 0.688 and 12,006 of 20,000 outputs carry driver-omission flags. |
| RQ2 Stability | Materially supported by automated evidence | 800 regeneration outputs were generated across original, driver-order shuffled, magnitude-perturbed and randomised-driver variants. Human assessment was not conducted. |
| RQ3 Decision utility / simulatability | Not directly tested | No human simulatability study or raw-driver-list comparison was conducted. Automated governance proxies do not establish improved human understanding. |
| RQ4 Drift awareness / over-trust | Operationalised, not causally tested | Drift diagnostics and disclosure logic are implemented, but no with/without drift-message experiment measured over-trust reduction. |
| RQ5 Thin-file robustness | Materially strengthened under controlled proxy conditions | Exact model-ready X_test reconstruction and a 1,000-row feature-masking re-score were completed. This remains an inference-time stress test rather than a production enrichment-outage simulation. |
| RQ6 Portability | Addressed structurally | Cyber, crypto and graph/GNN mappings were added at EO protocol level only. No cross-domain model performance is claimed. |

## Hypothesis Disposition

| Hypothesis | Final disposition | Evidence / caveat |
|---|---|---|
| H1 Readability / clarity | Not directly tested | No human readability or preference study comparing constrained LLM and template narratives was conducted. The thesis claims LLM scalability and decision-frame preservation, not demonstrated human readability superiority. |
| H2 Constraint value | Not directly tested via unconstrained ablation | The proposed unconstrained 50-100 EO ablation was not reported as a final empirical result. Constraint value is supported indirectly through the validator taxonomy and bounded generation design, but the marginal effect of removing constraints remains future work. |
| H3 Drift messaging / over-trust | Not causally tested | Drift metrics and disclosure logic were implemented, but the thesis does not demonstrate that drift messaging reduces human over-trust. |
| H4 Thin-file | Partially supported / materially strengthened | Disclosure behaviour and controlled feature-masking evidence support thin-file robustness under proxy conditions. Production-style enrichment outage simulation remains future work. |

## Interpretation

The final thesis is strongest when framed as a governance-controllable explanation system rather than as proof of human readability, human simulatability, live production monitoring or cross-domain model transfer. This distinction preserves the contribution while avoiding overclaiming.
