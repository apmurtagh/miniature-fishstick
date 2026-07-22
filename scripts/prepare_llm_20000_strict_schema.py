import json
from pathlib import Path

ART = Path("artifacts/baselines/lgbm_numeric_v1_subsample")
ROBUST = ART / "llm_20000_robustness"
ROBUST.mkdir(parents=True, exist_ok=True)

SOURCE_CANDIDATES = [
    ART / "eos_test_with_drivers_with_transactiondt.jsonl",
    ART / "eos_test_with_drivers.jsonl",
    ART / "eos_test.jsonl",
]

STRICT_OUT = ROBUST / "eos_20000_strict_schema.jsonl"
TEMPORAL_META = ROBUST / "eos_20000_temporal_metadata.jsonl"
MANIFEST = ROBUST / "llm_20000_robustness_manifest.json"

REMOVE_KEYS = {"transaction_dt", "temporal", "temporal_window"}


def event_id(eo):
    for k in ["event_id", "TransactionID", "transaction_id", "id"]:
        if k in eo:
            return eo[k]
    return None


source = next((p for p in SOURCE_CANDIDATES if p.exists()), None)
if not source:
    raise FileNotFoundError("No EO source file found")

n = 0
with source.open("r", encoding="utf-8") as fin, \
     STRICT_OUT.open("w", encoding="utf-8") as fout, \
     TEMPORAL_META.open("w", encoding="utf-8") as fmeta:

    for line in fin:
        if not line.strip():
            continue

        eo = json.loads(line)

        meta = {
            "row_index": n,
            "event_id": event_id(eo),
            "transaction_dt": eo.get("transaction_dt") or eo.get("temporal", {}).get("transaction_dt"),
            "temporal_window": eo.get("temporal_window") or eo.get("temporal", {}).get("window"),
        }
        fmeta.write(json.dumps(meta, ensure_ascii=False) + "\n")

        clean = {k: v for k, v in eo.items() if k not in REMOVE_KEYS}
        fout.write(json.dumps(clean, ensure_ascii=False) + "\n")

        n += 1

manifest = {
    "source_eo": str(source),
    "strict_schema_eos": str(STRICT_OUT),
    "temporal_metadata": str(TEMPORAL_META),
    "rows": n,
    "note": "Strict-schema EO file removes temporal uplift fields rejected by EvidenceObject Pydantic validation. Temporal metadata is retained in sidecar."
}
MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("[OK] Prepared strict-schema 20,000 EO file")
print(json.dumps(manifest, indent=2))
