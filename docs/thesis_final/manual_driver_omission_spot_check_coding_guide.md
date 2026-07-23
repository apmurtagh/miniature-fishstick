# Manual Driver-Omission Spot-Check Coding Guide

## Purpose

This guide supports a bounded manual error-analysis spot-check of automated driver-omission flags. It is not a human-subjects study and does not test readability, utility, trust calibration or analyst decision quality.

## Coding categories

Use exactly one category per sampled row.

- A_TRUE_OMISSION: one or more EO top drivers are absent from the narrative and not reasonably paraphrased.
- B_LEXICAL_MISMATCH: the narrative appears to refer to an EO driver using different wording, grouping or paraphrase, but the validator does not recognise it as coverage.
- C_SALIENCE_COMPRESSION: the narrative foregrounds one or two dominant drivers and omits weaker EO drivers while preserving risk/action framing.
- D_SUMMARY_LENGTH_COMPRESSION: the narrative is concise and operational, so omission appears related to brevity or output economy rather than contradiction.
- E_AMBIGUOUS_REVIEW: the reviewer cannot confidently distinguish omission from paraphrase or compression using the available artefacts.

## Claim boundary

This spot-check contextualises automated omission flags. It does not establish independent semantic validity, human usability, fairness, production readiness or regulatory audit completeness.
