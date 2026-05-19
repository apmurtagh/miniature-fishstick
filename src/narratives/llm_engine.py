def generate_llm_narrative(eo: dict, persona: str, llm_client):
    """
    Generate narrative via LLM (expects existing client).
    """

    prompt = {
        "score": eo["score"],
        "drivers": eo["drivers"],
        "persona": persona
    }

    # You already have this logic somewhere → plug it in
    response = llm_client.generate(prompt)

    return {
        "persona": persona,
        "text": response["text"],
        "features_mentioned": response.get("features_mentioned", [])
    }
