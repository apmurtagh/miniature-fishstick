# Validator Acceptance Policy Clarification

## Purpose

This document clarifies how the final 20,000-row LLM robustness run should be interpreted. It distinguishes clean acceptance, flagged acceptance and audit-complete acceptance.

## Acceptance States

| State | Meaning | Permitted use |
|---|---|---|
| Clean accepted | Risk band, recommended action, required disclosures and implemented validation checks pass without flags. | Operations summary; potentially audit use if full driver coverage also holds. |
| Accepted with validation flags | Output preserves core decision framing but carries a soft warning, typically driver omission. | Operations-summary use only; not audit-complete. |
| Hard failure | Schema failure, unsupported out-of-EO claim, action mismatch, missing mandatory disclosure or confirmed direction inversion. | Retry or deterministic fallback required. |
| Fallback | Deterministic template output used because LLM output failed acceptance. | Audit-safe deterministic fallback. |
| Audit-complete | Full top-k driver coverage, no unsupported facts, correct action and required disclosures. | Governance, audit or model-risk evidence. |

## Interpretation of Zero Fallbacks

The 20,000-row robustness run produced zero fallback rows. This should not be interpreted as zero validation concern. It means no row failed the operations-summary acceptance threshold.

The 12,006 outputs accepted with driver-omission flags demonstrate why an audit-complete policy must be stricter than an operations-summary policy. Under an audit-complete policy, flagged outputs should trigger retry, deterministic fallback or explicit incomplete-output labelling.

## Link to Thesis Claim

The final thesis therefore does not claim that constrained LLM narratives are audit-complete by default. It claims that an EO-bound LLM path can provide scalable operations summaries, while validator/audit instrumentation exposes when stricter evidence-complete rendering is required.
