from typing import Dict, Any


def emit_evidence_object(
    prediction_id: str,
    score: float,
    drivers: list,
    metadata: Dict[str, Any]
):
    """
    Construct Evidence Object (EO).
    """

    eo = {
        "prediction_id": prediction_id,
        "score": float(score),
        "drivers": drivers,
        "metadata": metadata
    }

    return eo
