def validate_eo(eo: dict):
    """
    Basic EO validation.
    """
    assert "prediction_id" in eo
    assert "score" in eo
    assert "drivers" in eo

    assert isinstance(eo["drivers"], list)
    assert len(eo["drivers"]) > 0

    return True


def validate_narrative(narrative: dict, eo: dict):
    """
    Ensure narrative is grounded in EO drivers.
    """
    eo_features = {d["feature"] for d in eo["drivers"]}

    narrative_features = set(narrative.get("features_mentioned", []))

    # closed-world assumption
    assert narrative_features.issubset(eo_features)

    return True
