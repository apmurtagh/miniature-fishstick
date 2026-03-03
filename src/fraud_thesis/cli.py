from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fraud_thesis.run_manager import (
    append_to_index,
    capture_env_meta,
    capture_git_meta,
    create_run_dir,
    get_runs_dir,
    load_manifest,
    make_run_id,
    utc_now_compact,
    write_manifest,
)


def cmd_show_runs_dir(_: argparse.Namespace) -> int:
    print(str(get_runs_dir()))
    return 0


def cmd_init_run(args: argparse.Namespace) -> int:
    run_id = make_run_id(
        split_version=args.split_version,
        model=args.model,
        narrative_mode=args.narrative_mode,
        seed=args.seed,
        eo_schema_version=args.eo_schema_version,
        narrative_schema_version=args.narrative_schema_version,
        masking_rate=args.masking_rate,
        extra_tags=args.tag or (),
    )

    paths = create_run_dir(run_id, runs_dir=Path(args.runs_dir) if args.runs_dir else None)

    git_meta = capture_git_meta(paths.run_dir)
    capture_env_meta(paths.run_dir)

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": utc_now_compact(),
        "runs_dir": str((Path(args.runs_dir).expanduser().resolve() if args.runs_dir else get_runs_dir())),
        "git": git_meta,
        "params": {
            "split_version": args.split_version,
            "seed": args.seed,
            "model": args.model,
            "eo_schema_version": args.eo_schema_version,
            "narrative_mode": args.narrative_mode,
            "narrative_schema_version": args.narrative_schema_version,
            "masking_rate": args.masking_rate,
            "persona": args.persona,
            "tags": args.tag or [],
        },
        "artifacts": {},
    }

    write_manifest(paths.run_dir, manifest)

    # Print run_dir so callers can capture it.
    print(str(paths.run_dir))
    return 0


def cmd_finalize_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest = load_manifest(run_dir)

    metrics_path = args.metrics_path
    if args.metrics_json:
        out = run_dir / "eval" / "metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(args.metrics_json.strip() + "\n", encoding="utf-8")
        metrics_path = str(out.relative_to(run_dir))

    artifacts = manifest.setdefault("artifacts", {})
    if metrics_path:
        artifacts["metrics_path"] = metrics_path
    if args.eo_path:
        artifacts["eo_path"] = args.eo_path
    if args.narratives_path:
        artifacts["narratives_path"] = args.narratives_path
    if args.validator_stats_path:
        artifacts["validator_stats_path"] = args.validator_stats_path

    write_manifest(run_dir, manifest)

    git = manifest.get("git", {}) or {}
    params = manifest.get("params", {}) or {}

    runs_dir = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else get_runs_dir()

    row = {
        "run_id": manifest.get("run_id", ""),
        "timestamp_utc": manifest.get("timestamp_utc", ""),
        "git_commit": git.get("commit", ""),
        "git_dirty": git.get("dirty", ""),
        "run_dir": str(run_dir),
        "split_version": params.get("split_version", ""),
        "seed": params.get("seed", ""),
        "model": params.get("model", ""),
        "eo_schema_version": params.get("eo_schema_version", ""),
        "narrative_mode": params.get("narrative_mode", ""),
        "narrative_schema_version": params.get("narrative_schema_version", ""),
        "masking_rate": params.get("masking_rate", ""),
        "persona": params.get("persona", ""),
        "metrics_path": artifacts.get("metrics_path", ""),
    }

    index_path = append_to_index(runs_dir=runs_dir, row=row)
    print(str(index_path))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fraud-thesis")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("show-runs-dir", help="Print resolved runs directory")
    s.set_defaults(func=cmd_show_runs_dir)

    s = sub.add_parser("init-run", help="Create a new run directory and initial manifest")
    s.add_argument("--runs-dir", default=None, help="Override runs dir (else env/default)")
    s.add_argument("--split-version", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--model", required=True, help="Model name/version label, e.g. lgbm_v1")
    s.add_argument("--eo-schema-version", default=None)
    s.add_argument("--narrative-mode", required=True, help="e.g. templates | constrained | ablation")
    s.add_argument("--narrative-schema-version", default=None)
    s.add_argument("--masking-rate", type=float, default=None)
    s.add_argument("--persona", default=None)
    s.add_argument("--tag", action="append", help="Extra tag; may be passed multiple times")
    s.set_defaults(func=cmd_init_run)

    s = sub.add_parser("finalize-run", help="Update manifest artifacts and append runs_index.csv")
    s.add_argument("--runs-dir", default=None, help="Override runs dir (else env/default)")
    s.add_argument("--run-dir", required=True, help="Absolute path to run directory")
    s.add_argument("--metrics-path", default=None, help="Relative path (from run_dir) to metrics file")
    s.add_argument("--metrics-json", default=None, help="Raw JSON string written to eval/metrics.json")
    s.add_argument("--eo-path", default=None, help="Relative path from run_dir to EO output")
    s.add_argument("--narratives-path", default=None, help="Relative path from run_dir to narratives output")
    s.add_argument("--validator-stats-path", default=None, help="Relative path from run_dir to validator stats")
    s.set_defaults(func=cmd_finalize_run)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
EOF
