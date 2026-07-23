import csv
from pathlib import Path

CSV_PATH = Path("docs/thesis_final/manual_driver_omission_spot_check_sample_50.csv")

CATEGORY_MAP = {
    "A": "A_TRUE_OMISSION",
    "B": "B_LEXICAL_MISMATCH",
    "C": "C_SALIENCE_COMPRESSION",
    "D": "D_SUMMARY_LENGTH_COMPRESSION",
    "E": "E_AMBIGUOUS_REVIEW",
}

CATEGORY_HELP = """
Categories:
  A = TRUE OMISSION
      One or more EO top drivers are absent from the narrative and not reasonably paraphrased.

  B = LEXICAL MISMATCH
      The narrative refers to the driver using different wording/grouping/paraphrase, but the validator may not count it.

  C = SALIENCE COMPRESSION
      Narrative foregrounds dominant drivers and omits weaker EO drivers while preserving risk/action framing.

  D = SUMMARY-LENGTH COMPRESSION
      Narrative is concise/operational and omission appears due to brevity rather than contradiction.

  E = AMBIGUOUS / REVIEW REQUIRED
      Cannot confidently distinguish omission from paraphrase or compression.
"""

def load_rows():
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def save_rows(rows, fieldnames):
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

rows = load_rows()
fieldnames = list(rows[0].keys())

print("== Manual Driver-Omission Spot-Check Coding ==")
print("Rows:", len(rows))
print(CATEGORY_HELP)

for idx, row in enumerate(rows, start=1):
    if row.get("manual_category"):
        continue

    print("\n" + "=" * 90)
    print(f"Row {idx} of {len(rows)}")
    print("event_id:", row.get("event_id"))
    print("driver_coverage:", row.get("driver_coverage"))
    print("review_language_flag:", row.get("review_language_flag"))
    print("\nEO top drivers:")
    print(row.get("eo_top_drivers_if_available") or "[blank]")
    print("\nMentioned drivers:")
    print(row.get("mentioned_drivers") or "[blank]")
    print("\nMissing drivers if available:")
    print(row.get("missing_drivers_if_available") or "[blank]")
    print("\nNarrative excerpt:")
    print(row.get("narrative_excerpt") or "[blank]")
    print(CATEGORY_HELP)

    while True:
        choice = input("Enter category A/B/C/D/E, S to skip, Q to quit: ").strip().upper()
        if choice == "Q":
            save_rows(rows, fieldnames)
            print("Saved progress. Exiting.")
            raise SystemExit(0)
        if choice == "S":
            print("Skipped row.")
            break
        if choice in CATEGORY_MAP:
            row["manual_category"] = CATEGORY_MAP[choice]
            note = input("Short reviewer note: ").strip()
            row["manual_reviewer_note"] = note
            save_rows(rows, fieldnames)
            print("Saved.")
            break
        print("Invalid choice. Use A, B, C, D, E, S or Q.")

save_rows(rows, fieldnames)
print("\nAll rows coded or skipped. Done.")
