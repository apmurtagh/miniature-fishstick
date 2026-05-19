def overlap_at_k(eo_drivers: list, narrative_features: list, k=5):
    eo_top = [d["feature"] for d in eo_drivers[:k]]

    overlap = len(set(eo_top) & set(narrative_features))

    return overlap / k


def direction_accuracy(eo_drivers: list, narrative_features: list):
    eo_map = {d["feature"]: d["sign"] for d in eo_drivers}

    correct = 0
    total = 0

    for f in narrative_features:
        if f in eo_map:
            total += 1
            # simplistic placeholder
            correct += 1

    return correct / total if total else 0.0
