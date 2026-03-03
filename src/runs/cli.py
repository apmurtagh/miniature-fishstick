"""
Command-line interface for run management.

Usage
-----
    python -m runs create [--runs-dir PATH] [--params KEY=VAL ...]
    python -m runs finish <run_id> [--runs-dir PATH] [--metrics KEY=VAL ...]
    python -m runs list   [--runs-dir PATH]
"""

import argparse
import json
import sys
from pathlib import Path

from .config import _ENV_VAR, _DEFAULT_RUNS_DIR
from .index import append_to_index, read_index
from .manager import create_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_kv(pairs: list[str]) -> dict[str, str]:
    """Convert a list of ``KEY=VALUE`` strings to a dict."""
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(
                f"Expected KEY=VALUE, got: {pair!r}"
            )
        k, _, v = pair.partition("=")
        result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_create(args: argparse.Namespace) -> None:
    params = _parse_kv(args.params)
    manifest = create_run(
        runs_dir=args.runs_dir,
        params=params,
        capture_provenance=not args.no_provenance,
    )
    print(json.dumps({"run_id": manifest["run_id"], "run_dir": manifest["run_dir"]}))


def cmd_finish(args: argparse.Namespace) -> None:
    metrics = _parse_kv(args.metrics)
    row = {"run_id": args.run_id, **metrics}
    index_path = append_to_index(row, runs_dir=args.runs_dir)
    print(f"Appended run {args.run_id!r} to {index_path}")


def cmd_list(args: argparse.Namespace) -> None:
    rows = read_index(runs_dir=args.runs_dir)
    if not rows:
        print("(no runs in index)")
        return
    for row in rows:
        ts = row.get("timestamp", "")
        model = row.get("model", "")
        auc = row.get("auc_roc", "")
        print(f"{row['run_id']}  ts={ts}  model={model}  auc={auc}")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runs",
        description="Run-management CLI for the fraud-thesis project.",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        metavar="PATH",
        help=(
            f"Root directory for run artifacts.  "
            f"Overrides ${_ENV_VAR} and the compiled default "
            f"({_DEFAULT_RUNS_DIR!r})."
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- create ---
    p_create = sub.add_parser("create", help="Initialise a new run directory.")
    p_create.add_argument(
        "--params",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Key=value parameters to embed in the manifest.",
    )
    p_create.add_argument(
        "--no-provenance",
        action="store_true",
        help="Skip writing environment provenance files.",
    )
    p_create.set_defaults(func=cmd_create)

    # --- finish ---
    p_finish = sub.add_parser(
        "finish", help="Record metrics for a completed run in the index."
    )
    p_finish.add_argument("run_id", help="The run ID to record.")
    p_finish.add_argument(
        "--metrics",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Metric key=value pairs to record (e.g. auc_roc=0.92).",
    )
    p_finish.set_defaults(func=cmd_finish)

    # --- list ---
    p_list = sub.add_parser("list", help="List runs in the index.")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
