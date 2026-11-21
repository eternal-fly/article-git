#!/usr/bin/env python3
"""CLI for one graph-stream validity experiment or one saved plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from validity_core.graph_stream_validity import ValidityConfig, run_collision_ceiling, run_cross_order, run_invariance, run_padding_ablation, run_plan


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--plan", type=Path, help="Run an existing plan.json")
    command.add_argument("--dataset", choices=("BACE", "BBBP"))
    command.add_argument("--data-dir", type=Path)
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--experiment", choices=("invariance", "padding", "collision", "cross_order"))
    command.add_argument("--q", type=int, default=16)
    command.add_argument("--t", type=int, default=16)
    command.add_argument("--profile", "--mode", dest="profile", choices=("quick", "full"), default="full")
    command.add_argument("--no-qrc", action="store_true", help="CPU/GPU classical smoke without QRC")
    command.add_argument("--force", action="store_true")
    return command


def main() -> None:
    args = parser().parse_args()
    if args.plan:
        result = run_plan(args.plan, profile_override=args.profile)
    else:
        missing = [name for name in ("dataset", "data_dir", "output_dir", "experiment") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"Missing direct-run arguments: {', '.join(missing)}")
        config = ValidityConfig(
            dataset=args.dataset,
            data_dir=str(args.data_dir),
            output_dir=str(args.output_dir),
            experiment=args.experiment,
            q=args.q,
            t=args.t,
            profile=args.profile,
            run_qrc=not args.no_qrc,
            force=args.force,
        )
        runners = {
            "invariance": run_invariance,
            "padding": run_padding_ablation,
            "collision": run_collision_ceiling,
            "cross_order": run_cross_order,
        }
        result = runners[args.experiment](config)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
