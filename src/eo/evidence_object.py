"""
Evidence Object (EO) schema for governance-ready, evidence-grounded narratives.

EO is the closed-world "contract" between:
- model outputs (score/risk band/action),
- evidence (top drivers + signs),
- monitoring & evidence-strength metadata,
- and downstream narrative generation / validation.

Keep it strict (extra="forbid") so schema drift is explicit.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DriftStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


class EvidenceStrength(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class RiskBand(str, Enum):
    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class ActionClass(str, Enum):
    ALLOW = "allow"
    STEP_UP = "step-up"
    REVIEW = "review"
    BLOCK = "block"


class EvidenceSpan(BaseModel):
    """Optional structured reference for why a driver is salient (bin/value etc.)."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: Optional[str] = None
    bin: Optional[str] = None


class Driver(BaseModel):
    """
    Local driver representation.

    direction:
      '+' means increases risk; '-' means decreases risk relative to baseline.
    magnitude:
      comparable within an EO instance (use SHAP magnitude later).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    direction: Literal["+", "-"]
    magnitude: float = Field(ge=0.0)
    evidence_span: Optional[EvidenceSpan] = None


class Monitoring(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drift_status: DriftStatus = DriftStatus.OK
    data_quality_flags: list[str] = Field(default_factory=list)


class EvidenceObject(BaseModel):
    """
    EO v0.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "eo_v0"
    domain: str = "fraud"

    # Provenance
    created_utc: datetime
    git_sha: str = ""
    model: str
    split_id: str

    # Identifiers (IEEE-CIS uses TransactionID)
    event_id: str

    # Model outputs
    score: float = Field(ge=0.0, le=1.0)
    risk_band: RiskBand
    recommended_action_class: ActionClass

    # Evidence for explanations
    top_drivers: list[Driver] = Field(default_factory=list)

    # Coverage / robustness signals
    thin_file_flag: bool = False
    evidence_strength: EvidenceStrength = EvidenceStrength.MED
    data_coverage_score: float = Field(default=1.0, ge=0.0, le=1.0)

    # Monitoring
    monitoring: Monitoring = Field(default_factory=Monitoring)

    # Free-form metadata (debug only; keep minimal)
    meta: dict[str, Any] = Field(default_factory=dict)


def band_from_score(score: float) -> RiskBand:
    """
    Simple coarse banding. Replace with calibrated bands later.
    """
    if score >= 0.8:
        return RiskBand.HIGH
    if score >= 0.5:
        return RiskBand.MED
    return RiskBand.LOW


def evidence_strength_from_thin_flag(thin_file_flag: bool) -> EvidenceStrength:
    return EvidenceStrength.LOW if thin_file_flag else EvidenceStrength.MED


def action_from_band(
    band: RiskBand,
    *,
    evidence_strength: EvidenceStrength,
    drift_status: DriftStatus,
) -> ActionClass:
    """
    Conservative default action policy (placeholder).
    In WARN/LOW-evidence regimes, bias toward step-up vs hard actions.
    """
    if drift_status in {DriftStatus.WARN, DriftStatus.FAIL} or evidence_strength == EvidenceStrength.LOW:
        return ActionClass.STEP_UP if band in {RiskBand.MED, RiskBand.HIGH} else ActionClass.ALLOW

    if band == RiskBand.HIGH:
        return ActionClass.REVIEW
    if band == RiskBand.MED:
        return ActionClass.STEP_UP
    return ActionClass.ALLOW
