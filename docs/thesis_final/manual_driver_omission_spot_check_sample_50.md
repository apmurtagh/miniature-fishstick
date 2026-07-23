# Manual Driver-Omission Spot-Check Sample

- Sample source: `20k robustness audit`
- Source file: `/workspaces/miniature-fishstick/artifacts/baselines/lgbm_numeric_v1_subsample/llm_20000_robustness/llm_20000_attempt_audit.jsonl`
- Candidate omission rows found: `12006`
- Sample target: `50`
- Sample generated: `50`
- Random seed: `20260723`

This file provides a coding template. Complete `manual_category` and `manual_reviewer_note` in the CSV using the coding guide.

## Coding categories

- `A_TRUE_OMISSION`
- `B_LEXICAL_MISMATCH`
- `C_SALIENCE_COMPRESSION`
- `D_SUMMARY_LENGTH_COMPRESSION`
- `E_AMBIGUOUS_REVIEW`

## First 20 sampled rows

| # | event_id | condition | driver_coverage | mentioned / driver count | review flag |
|---:|---|---|---:|---|---|
| 1 | 3490679 | constrained_20k |  |  /  | False |
| 2 | 3493280 | constrained_20k |  |  /  | False |
| 3 | 3494053 | constrained_20k |  |  /  | False |
| 4 | 3495519 | constrained_20k |  |  /  | False |
| 5 | 3497141 | constrained_20k |  |  /  | False |
| 6 | 3498283 | constrained_20k |  |  /  | False |
| 7 | 3498653 | constrained_20k |  |  /  | False |
| 8 | 3505323 | constrained_20k |  |  /  | False |
| 9 | 3507034 | constrained_20k |  |  /  | False |
| 10 | 3508447 | constrained_20k |  |  /  | False |
| 11 | 3510284 | constrained_20k |  |  /  | False |
| 12 | 3511571 | constrained_20k |  |  /  | False |
| 13 | 3513701 | constrained_20k |  |  /  | False |
| 14 | 3518705 | constrained_20k |  |  /  | False |
| 15 | 3519467 | constrained_20k |  |  /  | False |
| 16 | 3519607 | constrained_20k |  |  /  | False |
| 17 | 3523093 | constrained_20k |  |  /  | False |
| 18 | 3523274 | constrained_20k |  |  /  | False |
| 19 | 3524686 | constrained_20k |  |  /  | False |
| 20 | 3525263 | constrained_20k |  |  /  | False |

## Claim boundary

This is a manual error-analysis spot-check, not a human evaluation. It should not be used to claim improved readability, human trust calibration, analyst utility, fairness, production readiness or independent semantic truth.