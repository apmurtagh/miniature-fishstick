import copy
import json
from pathlib import Path

import numpy as np

ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
OUT = ART / "regeneration_stability"
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = ART / "eos_test_with_drivers_with_transactiondt.jsonl"
if not SOURCE.exists():
    SOURCE = ART / "eos_test_with_drivers.jsonl"

N = 200
SEED = 42
REMOVE_KEYS = {"transaction_dt", "temporal", "temporal_window"}


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if k not in REMOVE_KEYS}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def get_drivers(eo):
    return eo.get("top_drivers") or eo.get("drivers") or []


def set_drivers(eo, new_drivers):
    if "top_drivers" in eo:
        eo["top_drivers"] = new_drivers
    else:
        eo["drivers"] = new_drivers


def event_id(eo):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in eo:
            return eo[k]
    return None


eos = read_jsonl(SOURCE)
rng = np.random.default_rng(SEED)

indices = np.linspace(0, len(eos) - 1, N, dtype=int)
base = [copy.deepcopy(eos[i]) for i in indices]

driver_universe = []
for eo in eos:
    for d in get_drivers(eo):
        if isinstance(d, dict):
            driver_universe.append(copy.deepcopy(d))

if not driver_universe:
    raise RuntimeError("No driver dictionaries found in source EO file")

variants = {
    "original": [],
    "driver_order_shuffled": [],
    "magnitude_perturbed": [],
    "randomised_drivers": [],
}

metadata = []

for local_idx, source_idx in enumerate(indices):
    eo = copy.deepcopy(eos[int(source_idx)])
    d0 = [copy.deepcopy(d) for d in get_drivers(eo)]
    if not d0:
        continue

    metadata.append({
        "local_index": local_idx,
        "source_index": int(source_idx),
        "event_id": event_id(eo),
        "top_driver_names_original": [
            str(d.get("name") or d.get("feature") or d.get("driver"))
            for d in d0
            if isinstance(d, dict)
        ],
    })

    eo_original = copy.deepcopy(eo)
    variants["original"].append(eo_original)

    eo_shuffle = copy.deepcopy(eo)
    d_shuffle = [copy.deepcopy(d) for d in d0]
    rng.shuffle(d_shuffle)
    set_drivers(eo_shuffle, d_shuffle)
    variants["driver_order_shuffled"].append(eo_shuffle)

    eo_mag = copy.deepcopy(eo)
    d_mag = [copy.deepcopy(d) for d in d0]
    for d in d_mag:
        if isinstance(d, dict):
            for key in ["magnitude", "value", "shap_value", "importance"]:
                if key in d:
                    try:
                        d[key] = float(d[key]) * float(rng.uniform(0.95, 1.05))
                    except Exception:
                        pass
    set_drivers(eo_mag, d_mag)
    variants["magnitude_perturbed"].append(eo_mag)

    eo_rand = copy.deepcopy(eo)
    k = len(d0)
    sampled_idx = rng.choice(len(driver_universe), size=k, replace=False)
    sampled = [copy.deepcopy(driver_universe[int(i)]) for i in sampled_idx]
    set_drivers(eo_rand, sampled)
    variants["randomised_drivers"].append(eo_rand)

for name, rows in variants.items():
    path = OUT / f"stability_{name}_eos_200.jsonl"
    write_jsonl(path, rows)
    print(f"[OK] {name}: {len(rows)} rows -> {path}")

meta_path = OUT / "regeneration_stability_metadata_200.jsonl"
with meta_path.open("w", encoding="utf-8") as f:
    for row in metadata:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

manifest = {
    "source": str(SOURCE),
    "n_base": len(variants["original"]),
    "seed": SEED,
    "metadata": str(meta_path),
    "variants": {
        name: str(OUT / f"stability_{name}_eos_200.jsonl")
        for name in variants
    },
    "note": "EO variant files are strict-schema compatible; temporal uplift fields are omitted for Pydantic validation."
}

(OUT / "regeneration_stability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("[OK] Manifest written")
