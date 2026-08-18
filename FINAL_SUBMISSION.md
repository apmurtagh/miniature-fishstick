# Final Submission Entry Point

Final MSc Thesis Submission Package

This file is the final submission entry point for the MSc Data Science dissertation:

**Governance-Ready Fraud Decisioning for Transaction Monitoring: Evidence-Grounded Narrative Explanations with Tiered Guardrails**

Author: Andrew Murtagh  
Final submission date: 24 July 2026  
Repository: `apmurtagh/miniature-fishstick`

## 1. Repository State

### Complete evidential package

The complete final submission package is integrated on `main` at:

- Final v30 release/reference state: `cf638f9`
- Relevant merged pull requests:
  - PR #13: final reproducibility and governance artefacts
  - PR #14: H2 paired uncertainty addendum
  - PR #15: manual driver-omission spot-check
  - PR #16: final submission entry-point update

### Empirical source state

The authoritative empirical source checkpoint remains:

- Empirical source commit: `27a69ce`
- Empirical source tag: `thesis-final-v24-h2-ablation-pilot-20260717`

The v24 tag is the frozen empirical checkpoint for the bounded H2 ablation source artefacts. Later commits through `cf638f9` add documentation, notebook/reference alignment, release/IP-notice material, derived H2 paired uncertainty analysis and manual driver-omission error analysis using existing artefacts only. These later commits do not introduce additional LLM generation and do not alter the reported empirical metrics.

Note: `cf638f9` is the v30 thesis/release reference state. Later administrative hygiene commits, if any, do not alter the frozen empirical source state or reported empirical metrics.

## Policy Sensitivity Summary

- Operations-summary acceptance: `20,000 / 20,000`, accepted rate `1.0000`, Wilson 95% CI `0.9998 to 1.0000`.
- Audit-complete evidence rendering: validator-defined audit-complete acceptance `7,994 / 20,000`, accepted rate `0.3997`, Wilson 95% CI `0.3929 to 0.4065`.
- Driver-omission flagged rows: `12,006 / 20,000` (`60.03%`).
- Direction-proxy confirmed rows: `6,565 / 20,000`, accepted rate `0.3282`, Wilson 95% CI `0.3218 to 0.3348`.

## 2. Headline Empirical Results

- Baseline ROC-AUC: `0.8687`
- Baseline PR-AUC: `0.4594`
- Template narratives: `20,000`
- Constrained LLM robustness narratives: `20,000`
- Matched LLM audit records: `20,000`
- Operations-summary accepted rows: `20,000`
- Validator-defined audit-complete accepted rows: `7,994`
- Driver-omission flagged rows: `12,006`
- Direction-proxy confirmed rows: `6,565`
- H2 constrained driver coverage: `0.690`
- H2 unconstrained driver coverage: `0.432`
- H2 constrained zero-driver rows: `0`
- H2 unconstrained zero-driver rows: `27`
- H2 paired coverage difference, constrained minus unconstrained: `0.258`
- H2 paired bootstrap 95% CI: `[0.180, 0.336]`
- Manual driver-omission spot-check: `50` sampled omission-flagged narratives
- Manual spot-check result: `50/50` coded as salience compression
- Final hygiene tests: `12 passed`

## 3. Important Claim Boundaries

The dissertation is a governance-measurement and reproducibility contribution. It does not claim:

- human readability superiority;
- human simulatability;
- task utility;
- causal reduction of over-trust;
- protected-class fairness performance;
- production readiness;
- live monitoring readiness;
- external regulatory certification;
- independent human semantic validation.

The manual driver-omission spot-check is a manual error-analysis of existing artefacts only. It is not a human-subjects study, not a task-utility evaluation and not an independent semantic validation exercise.

## 4. Key Artefacts

Core submission and reproducibility artefacts include:

- `FINAL_SUBMISSION.md`
- `docs/thesis_final/final_submission_artifact_manifest.md`
- `docs/thesis_final/excluded_artifact_verification_manifest.md`
- `docs/thesis_final/excluded_artifact_verification_manifest.json`
- `notebooks/governance_ready_fraud_decisioning_end_to_end_reproduction.ipynb`
- `tests/test_final_thesis_hygiene.py`
- `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_unconstrained_vs_constrained_summary_100.md`
- `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_unconstrained_vs_constrained_summary_100.json`
- `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_paired_uncertainty_addendum.md`
- `artifacts/baselines/lgbm_numeric_v1_subsample/h2_unconstrained_ablation/h2_paired_uncertainty_addendum.json`
- `docs/thesis_final/manual_driver_omission_spot_check_appendix_c.md`
- `docs/thesis_final/manual_driver_omission_spot_check_completed_summary.md`
- `docs/thesis_final/manual_driver_omission_spot_check_completed_summary.json`
- `docs/thesis_final/manual_driver_omission_spot_check_sample_50_final.csv`
- `docs/thesis_final/manual_driver_omission_spot_check_integrity_audit.json`

## 5. Verification Command

To verify the final hygiene checks from the integrated repository state:

```bash
git checkout main
git pull --ff-only origin main
PYTHONPATH=src pytest -q tests/test_final_thesis_hygiene.py
```

Expected result:

```text
12 passed
```

To inspect the frozen empirical source checkpoint:

```bash
git checkout thesis-final-v24-h2-ablation-pilot-20260717
```

Return to the final integrated state with:

```bash
git checkout main
git pull --ff-only origin main
```

## 6. Reading Order for Examiners

Recommended reading path:

1. Final dissertation PDF/DOCX.
2. This final submission entry point.
3. Final submission artefact manifest.
4. Excluded artefact verification manifest.
5. End-to-end reproduction notebook.
6. H2 paired uncertainty addendum.
7. Manual driver-omission spot-check summary.
8. Hygiene tests.
