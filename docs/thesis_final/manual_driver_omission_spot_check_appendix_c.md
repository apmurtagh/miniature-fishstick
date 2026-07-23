# Appendix C. Manual Driver-Omission Spot-Check

## Purpose

This appendix reports a bounded manual error-analysis spot-check of driver-omission-flagged narratives from the 20,000-row LLM robustness audit. The purpose is to contextualise the 12,006 automated driver-omission flags by checking whether sampled flags reflect complete evidence omission, lexical mismatch, salience compression, summary-length compression or ambiguous review cases. No additional LLM generation was performed.

## Scope boundary

The spot-check is not a human-subjects study and does not test readability, analyst utility, trust calibration, fairness or production readiness. It is a manual review of existing artefacts only and should be interpreted as qualitative error analysis rather than independent semantic validation.

## Coding categories

- **True omission:** one or more EO top drivers are absent from the narrative and not reasonably paraphrased.
- **Lexical mismatch:** the narrative appears to refer to an EO driver using different wording, grouping or paraphrase, but the validator may not recognise it as coverage.
- **Salience compression:** the narrative foregrounds dominant drivers and omits weaker EO drivers while preserving risk/action framing.
- **Summary-length compression:** the narrative is concise and operational, so omission appears related to brevity rather than contradiction.
- **Ambiguous / review required:** the reviewer cannot confidently distinguish omission from paraphrase or compression.

## Results

| Category | Count | Share |
|---|---:|---:|
| True omission | 0 | 0.0% |
| Lexical mismatch | 0 | 0.0% |
| Salience compression | 50 | 100.0% |
| Summary-length compression | 0 | 0.0% |
| Ambiguous / review required | 0 | 0.0% |

## Interpretation

In the 50 sampled rows, all automated driver-omission flags were coded as salience compression. The reviewed narratives typically mentioned a subset of EO top drivers and omitted the remaining EO drivers, while preserving the risk band, recommended action and relevant cautionary disclosure. This result supports the thesis interpretation that operations-summary acceptance and validator-defined audit-complete rendering are distinct governance states. The narratives can be suitable as concise operational summaries while still falling short of the stricter all-driver audit-complete evidence-rendering policy.

## Claim boundary

The spot-check does not establish that users understand, trust or act on the narratives more effectively. It is not a human-subjects study, not a task-utility evaluation and not independent semantic validation. Independent human validation remains future work, as described in Appendix B.
