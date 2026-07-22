# Validator Policy Sensitivity Summary

## Purpose

This analysis quantifies how the 20,000-row LLM robustness result changes under increasingly strict validation policies. It directly supports the distinction between operations-summary acceptance and audit-complete evidence rendering.

## Policy Results

| Policy | Accepted rows | Review / fallback / incomplete rows | Accepted rate | Interpretation |
|---|---:|---:|---:|---|
| operations_summary | 20000 | 0 | 1.0000 | Suitable for concise operational summaries. Not equivalent to audit-complete rendering. |
| audit_complete_all_driver | 7994 | 12006 | 0.3997 | Represents stricter audit-complete evidence rendering under current validator taxonomy. |
| direction_proxy_confirmed_review | 6565 | 13435 | 0.3282 | Review-flag policy only. This is not a definitive semantic-faithfulness measure. |

## Interpretation

The operations-summary policy accepts 20,000 non-fallback outputs, but this should not be confused with audit-complete evidence rendering. Under a stricter all-driver audit-complete policy, only 7,994 outputs are clean while 12,006 outputs require retry, deterministic evidence-complete fallback or explicit incomplete-output labelling. The direction-proxy-confirmed policy is more conservative again and should be interpreted as a review queue, not proof of semantic failure.

## Thesis-Safe Claim

The LLM path is scalable and suitable for operations-summary use under the implemented controls, but audit-complete use requires stricter driver coverage enforcement, fallback or explicit incomplete-output labelling.
