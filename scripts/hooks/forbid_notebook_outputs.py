import json
import sys
from pathlib import Path

def notebook_has_outputs(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            return True
        if cell.get("execution_count") is not None:
            return True
    return False

def main(paths: list[str]) -> int:
    bad = []
    for p in paths:
        path = Path(p)

        # Safety: enforce the policy only for notebooks/ (pre-commit already filters,
        # but keep it defensive).
        if "notebooks" not in path.parts:
            continue
        if ("reports" in path.parts) and (path.parts[path.parts.index("notebooks") + 1] == "reports"):
            continue

        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append((p, f"Could not parse notebook JSON: {e}"))
            continue

        if notebook_has_outputs(nb):
            bad.append((p, "Notebook contains outputs and/or execution_count."))

    if bad:
        sys.stderr.write(
            "\nCommit blocked: notebooks with outputs must live under notebooks/reports/.\n"
            "For notebooks in notebooks/, strip outputs first.\n\n"
        )
        for p, msg in bad:
            sys.stderr.write(f"- {p}: {msg}\n")
        sys.stderr.write("\nTip: run `pre-commit run -a` to auto-strip most notebooks.\n")
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
