from src.explain.shap_explainer import compute_shap_values, get_top_k_drivers
from src.eo.emitter import emit_evidence_object
from src.eo.validator import validate_eo, validate_narrative
from src.narratives.template_engine import generate_template_narrative
from src.eval.driver_metrics import overlap_at_k


def run_pipeline(model, X, feature_names):
    shap_values = compute_shap_values(model, X)

    results = []

    for i in range(len(X)):
        drivers = get_top_k_drivers(shap_values[i:i+1], feature_names)

        eo = emit_evidence_object(
            prediction_id=str(i),
            score=0.5,  # replace with actual model score
            drivers=drivers,
            metadata={}
        )

        validate_eo(eo)

        narrative = generate_template_narrative(eo, persona="ops_triage")

        validate_narrative(narrative, eo)

        metric = overlap_at_k(eo["drivers"], narrative["features_mentioned"])

        results.append({
            "eo": eo,
            "narrative": narrative,
            "overlap": metric
        })

    return results
