# H2 Unconstrained vs Constrained Comparison, 100 Rows

## Purpose

This same-row exploratory comparison evaluates whether unconstrained natural-language generation differs from the constrained LLM path on the 100-row ablation sample. It is intended to address H2 as a bounded ablation, not as full-scale proof.

## Condition Summary

| Condition | Rows | Risk present | Action present | Any driver mentioned | Mean driver coverage | Zero-driver rows | Disclosure when required | Review-language flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constrained | 100 | 1.000 | 1.000 | 1.000 | 0.690 | 0 | 1.000 | 0.010 |
| unconstrained | 100 | 1.000 | 1.000 | 0.730 | 0.432 | 27 | 1.000 | 0.640 |

## Interpretation

The unconstrained pilot preserves risk/action framing and required disclosure in this small sample, but exhibits evidence compression through low mean driver coverage and multiple zero-driver rows. Review-language flags indicate that unconstrained prose may also introduce interpretive phrasing that requires review. This supports the thesis claim that fluency is not equivalent to audit-complete evidence rendering.

## Caveat

This is a bounded 100-row automated ablation. It should not be presented as human semantic validation. It is useful as proposal-aligned evidence that motivates constrained generation and validator policy, but not as a replacement for a larger ablation or human semantic review.
