from eo.evidence_object import EvidenceObject, DriftStatus, EvidenceStrength


def generate_template_narrative(
    eo: EvidenceObject,
    *,
    persona: str,
    top_k: int,
) -> str:
    """
    Entry point for deterministic template narrative generation.

    Enforces:
    - Closed-world grounding (EO only)
    - Persona-aware rendering
    """

    if persona == "ops_triage":
        return _render_ops_triage(eo, top_k=top_k)

    raise ValueError(f"Unsupported persona: {persona}")


def _render_ops_triage(eo: EvidenceObject, *, top_k: int) -> str:
    """
    Deterministic, EO-grounded template narrative for ops triage persona.

    Closed-world constraints:
    - Only reference EO fields
    - Do not invent external facts

    Policy:
    - Always mention EO top-K drivers (if present)
    - Preserve EO ordering
    - Group by sign (+/-) for readability
    """

    parts: list[str] = []

    # Risk summary
    parts.append(f"Risk band: {eo.risk_band.value} (score {eo.score:.3f}).")

    # Driver explanation
    if eo.top_drivers:
        k = int(top_k)
        top = eo.top_drivers if k <= 0 else eo.top_drivers[:k]

        inc = [d.name for d in top if d.direction == "+"]
        dec = [d.name for d in top if d.direction == "-"]

        if inc:
            parts.append("Top risk-increasing drivers: " + ", ".join(inc) + ".")
        if dec:
            parts.append("Top risk-decreasing drivers: " + ", ".join(dec) + ".")

        if not inc and not dec and top:
            parts.append("Top drivers: " + ", ".join(d.name for d in top) + ".")
    else:
        parts.append("Top drivers: not available (EO top_drivers is empty).")

    # Disclosures
    disclosures: list[str] = []

    if eo.thin_file_flag or eo.evidence_strength == EvidenceStrength.LOW:
        disclosures.append("Evidence is limited (thin-file / low evidence strength).")

    if eo.monitoring.drift_status in {DriftStatus.WARN, DriftStatus.FAIL}:
        disclosures.append(
            f"Monitoring: drift status {eo.monitoring.drift_status.value}; treat with caution."
        )

    if disclosures:
        parts.append(" ".join(disclosures))

    # Action
    parts.append(f"Recommended action: {eo.recommended_action_class.value}.")

    return " ".join(parts)
