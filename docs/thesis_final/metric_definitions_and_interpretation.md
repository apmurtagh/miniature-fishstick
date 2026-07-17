# Metric Definitions and Interpretation

## Purpose

This note defines the main automated metrics used in the dissertation and clarifies their interpretation. It distinguishes driver precision from driver coverage/recall because this distinction is central to the operations-summary versus audit-complete finding.

## Driver Sets

For each event:

- `D_EO` is the ordered set of top-k drivers in the Evidence Object.
- `D_N` is the set of drivers mentioned in the narrative.

## Driver Precision

Driver precision measures whether drivers mentioned in the narrative are valid EO drivers:

`driver_precision = |D_N ∩ D_EO| / |D_N|`

A narrative can have high driver precision while still omitting many EO drivers. For example, mentioning one correct EO driver gives high precision but poor evidence completeness.

## Driver Coverage / Recall

Driver coverage, or driver recall, measures how much of the EO driver set appears in the narrative:

`driver_coverage = |D_N ∩ D_EO| / |D_EO|`

This is the more relevant metric for audit-complete rendering because audit evidence requires full or near-full EO driver coverage.

## Top-k Overlap

The dissertation reports average top-k overlap as an automated proxy for driver alignment. It should be interpreted as a driver-overlap proxy rather than full semantic faithfulness. Where driver omissions are observed, top-k overlap supports the conclusion that LLM narratives are evidence-compressive.

## Direction Proxy

The direction proxy checks whether signed EO drivers are accompanied by simple risk-increasing or risk-mitigating phrases. It is a conservative rule-based review flag, not a definitive semantic-faithfulness score.

A low direction-proxy rate can reflect:

- driver direction unstated;
- phrase dictionary false negatives;
- compressed or neutral summaries;
- ambiguous wording;
- genuine direction inversion.

## Action Consistency

Action consistency checks whether the narrative recommendation matches the EO `recommended_action_class`.

## Disclosure Compliance

Disclosure compliance checks whether LOW evidence, thin-file, sparse telemetry, limited attribution or drift-warning conditions are explicitly disclosed where required.

## Audit-Complete Interpretation

For audit-complete evidence rendering, driver coverage/recall, action consistency, disclosure compliance and absence of unsupported facts matter more than narrative fluency. LLM narratives may be valid operations summaries while failing audit-complete evidence coverage.
