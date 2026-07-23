# Manual Driver-Omission Spot-Check Completed Summary

This manual spot-check contextualises automated driver-omission flags from the 20,000-row LLM robustness audit. It is not a human-subjects study and does not test readability, utility, trust calibration or analyst decision quality.

- Source population: `12,006` driver-omission-flagged rows
- Manual sample size: `50`
- No additional LLM generation was performed.

## Manual classification results

| Category | Count | Share |
|---|---:|---:|
| True omission | 0 | 0.0% |
| Lexical mismatch | 0 | 0.0% |
| Salience compression | 50 | 100.0% |
| Summary-length compression | 0 | 0.0% |
| Ambiguous / review required | 0 | 0.0% |

## Interpretation

All 50 sampled driver-omission flags were coded as salience compression. In these sampled cases, the narratives rendered a subset of EO top drivers while preserving the risk band, recommended action and relevant cautionary disclosure. This supports the thesis interpretation that operations-summary acceptance and validator-defined audit-complete rendering are distinct governance states: an output can be operationally useful while still failing the stricter all-driver audit-complete policy.

The result should not be interpreted as independent human semantic validation or evidence of user utility. It is a bounded manual error-analysis of existing artefacts only.

## Claim boundary

This spot-check does not establish improved readability, trust calibration, analyst utility, fairness, production readiness or regulatory audit completeness. It contextualises the automated omission flags and supports the need for independent human validation as future work.
