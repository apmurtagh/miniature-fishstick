def generate_template_narrative(eo: dict, persona: str):
    """
    Generate deterministic narrative from EO.
    """

    drivers = eo["drivers"]
    score = eo["score"]

    top_features = [d["feature"] for d in drivers]

    narrative_text = (
        f"Prediction score is {score:.3f}. "
        f"Key drivers include: {', '.join(top_features)}."
    )

    return {
        "persona": persona,
        "text": narrative_text,
        "features_mentioned": top_features
    }
