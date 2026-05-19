import shap
import pandas as pd


def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values for a trained model and dataset.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values


def get_top_k_drivers(shap_values, feature_names, k=5):
    """
    Extract top-K drivers by absolute SHAP importance.
    Returns list of dicts: [{feature, value, sign}]
    """
    import numpy as np

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:k]

    drivers = []
    for idx in top_idx:
        drivers.append({
            "feature": feature_names[idx],
            "importance": float(mean_abs[idx]),
            "sign": float(np.sign(shap_values[:, idx].mean()))
        })

    return drivers
