# Direction Proxy Interpretation Note

## Purpose

The semantic validator proxy reported a direction-proxy OK rate of 0.3282. This document clarifies how that result should be interpreted.

## Result

The semantic proxy evaluated 20,000 LLM narratives. It found:

- risk band present proxy rate: 1.0000;
- action present proxy rate: 1.0000;
- any-driver-mentioned rate: 1.0000;
- disclosure present when required rate: 1.0000;
- direction proxy OK when applicable rate: 0.3282.

## Interpretation

The direction-proxy result is a conservative rule-based review flag, not a definitive semantic-faithfulness score. The proxy checks for simple risk-increasing or risk-mitigating phrases associated with signed EO drivers. A low rate can arise through several mechanisms:

1. Direction unstated: the narrative mentions a driver but does not explicitly say whether it increases or reduces risk.
2. Proxy false negative: the narrative uses wording outside the simple phrase dictionary.
3. Driver compression: the narrative summarises multiple drivers without repeating signed direction.
4. Ambiguous wording: the direction cannot be clearly inferred from the phrase.
5. Genuine direction error: the narrative actually reverses or misstates the driver direction.

## Recommended Future Review

A manual stratified review could classify a small sample into:

- correct direction;
- genuine inversion;
- direction unstated;
- proxy false negative;
- ambiguous wording.

Until that review is conducted, the 0.3282 result should be treated as a governance review signal, not as proof that two-thirds of LLM directions are semantically incorrect.

## Thesis-Safe Claim

The robust automated findings are that risk band, recommended action, at least one EO driver and required disclosures were preserved across the full 20,000-row LLM run. Full direction-level semantic faithfulness remains future work.
