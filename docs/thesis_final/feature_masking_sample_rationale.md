# Feature-Masking Sample-Size Rationale

## Purpose

This note explains why the completed feature-masking re-score was run on a 1,000-row sample rather than the full 20,000-row test matrix.

## Rationale

The first requirement was to recreate the exact model-ready X_test matrix. That was completed over the full 20,000 rows. The reconstructed matrix matched all 20,000 original prediction rows in order and reproduced baseline predictions to numerical precision, with maximum absolute prediction difference of 1.11e-16.

The subsequent masking re-score was run on a 1,000-row sample to keep the final uplift computationally bounded. The objective was not to re-estimate production performance, but to demonstrate controlled score/action recalibration under degraded feature availability.

## Final Interpretation

The feature-masking result is therefore reported as an inference-time robustness stress test. It strengthens RQ5 beyond disclosure-only evidence, but it is not a full-population production enrichment-outage simulation.

A future production-style extension would mask coherent enrichment groups, such as identity, device, velocity or behavioural history, regenerate EOs and narratives, and assess action movement and disclosure compliance end to end.
