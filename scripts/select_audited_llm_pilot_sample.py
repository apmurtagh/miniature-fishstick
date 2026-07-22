import json
from pathlib import Path

import numpy as np
import pandas as pd


ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
OUT = ART / "audited_llm_pilot"
OUT.mkdir(parents=True, exist_ok=True)

EO_CANDIDATES = [
    ART / "eos_test_with_drivers_with_transactiondt.jsonl",
    ART / "eos_test_with_drivers.jsonl",
    ART / "eos_test.jsonl",
]

TARGET_N = 1000
SEED = 42


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def event_id(row):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in row and row[k] is not None:
            return row[k]
    return None


def risk_band(row):
    return str(
        row.get("risk_band")
        or row.get("calibration_band")
        or row.get("score_band")
        or row.get("band")
        or "UNKNOWN"
    )


def action_class(row):
    return str(
        row.get("recommended_action_class")
        or row.get("recommended_action")
        or row.get("action")
        or row.get("decision")
        or "UNKNOWN"
    )


def evidence_strength(row):
    return str(row.get("evidence_strength") or row.get("evidence_band") or "UNKNOWN").upper()


def temporal_window(row):
    return str(
        row.get("temporal_window")
        or row.get("temporal", {}).get("window")
        or "UNKNOWN"
    )


def thin_file(row):
    v = row.get("thin_file_flag")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"true", "1", "yes", "y"}
    return False


def top_driver_count(row):
    drivers = row.get("top_drivers") or row.get("drivers") or []
    return len(drivers)


eo_path = next((p for p in EO_CANDIDATES if p.exists()), None)
if not eo_path:
    raise FileNotFoundError(f"No EO file found among: {EO_CANDIDATES}")

eos = read_jsonl(eo_path)
if not eos:
    raise RuntimeError(f"EO file empty: {eo_path}")

records = []
for idx, eo in enumerate(eos):
    records.append({
        "row_index": idx,
        "event_id": event_id(eo),
        "risk_band": risk_band(eo),
        "recommended_action": action_class(eo),
        "evidence_strength": evidence_strength(eo),
        "temporal_window": temporal_window(eo),
        "thin_file_flag": thin_file(eo),
        "top_driver_count": top_driver_count(eo),
    })

df = pd.DataFrame(records)

# Stratify across the dimensions that matter for the proposal.
df["stratum"] = (
    df["risk_band"].astype(str)
    + "|"
    + df["recommended_action"].astype(str)
    + "|"
    + df["evidence_strength"].astype(str)
    + "|"
    + df["temporal_window"].astype(str)
)

rng = np.random.default_rng(SEED)

# First, take at least one from each non-empty stratum where possible.
selected_idx = []
for _, g in df.groupby("stratum", dropna=False):
    take = min(len(g), 1)
    chosen = rng.choice(g.index.to_numpy(), size=take, replace=False)
    selected_idx.extend(chosen.tolist())

remaining_n = max(0, TARGET_N - len(selected_idx))

remaining = df.drop(index=selected_idx)
if remaining_n > 0 and len(remaining) > 0:
    # Proportional top-up using all remaining records.
    take = min(remaining_n, len(remaining))
    chosen = rng.choice(remaining.index.to_numpy(), size=take, replace=False)
    selected_idx.extend(chosen.tolist())

sample = df.loc[selected_idx].copy()
sample = sample.sort_values(["temporal_window", "risk_band", "recommended_action", "evidence_strength", "row_index"])

# Write sample IDs and sampled EO file.
sample_csv = OUT / "audited_llm_pilot_sample_1000.csv"
sample_jsonl = OUT / "audited_llm_pilot_eos_1000.jsonl"
manifest_json = OUT / "audited_llm_pilot_manifest.json"

sample.to_csv(sample_csv, index=False)

selected_rows = set(sample["row_index"].astype(int).tolist())
with sample_jsonl.open("w", encoding="utf-8") as f:
    for idx, eo in enumerate(eos):
        if idx in selected_rows:
            f.write(json.dumps(eo, ensure_ascii=False) + "\n")

manifest = {
    "source_eo_file": str(eo_path),
    "target_n": TARGET_N,
    "actual_n": int(len(sample)),
    "seed": SEED,
    "outputs": {
        "sample_csv": str(sample_csv),
        "sample_jsonl": str(sample_jsonl),
        "manifest_json": str(manifest_json),
    },
    "counts": {
        "risk_band": sample["risk_band"].value_counts(dropna=False).to_dict(),
        "recommended_action": sample["recommended_action"].value_counts(dropna=False).to_dict(),
        "evidence_strength": sample["evidence_strength"].value_counts(dropna=False).to_dict(),
        "temporal_window": sample["temporal_window"].value_counts(dropna=False).to_dict(),
        "thin_file_flag": sample["thin_file_flag"].value_counts(dropna=False).to_dict(),
    },
    "strata_count": int(sample["stratum"].nunique()),
}

manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("[OK] Stratified audited LLM pilot sample created")
print(f"[OK] Source EO: {eo_path}")
print(f"[OK] Sample rows: {len(sample)}")
print(f"[OK] CSV: {sample_csv}")
print(f"[OK] JSONL: {sample_jsonl}")
print(f"[OK] Manifest: {manifest_json}")

print("")
print("== Sample distribution ==")
for col in ["risk_band", "recommended_action", "evidence_strength", "temporal_window", "thin_file_flag"]:
    print("")
    print(col)
    print(sample[col].value_counts(dropna=False).to_string())
