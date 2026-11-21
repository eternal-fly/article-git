"""CLI for controlled QRC graph-event benchmarks.

Examples
--------
CPU smoke test:
    python run_controlled_graph_streams.py --profile smoke

Full two-GPU run:
    python run_controlled_graph_streams.py --profile full --devices cuda:0,cuda:1

Existing token replay:
    python run_controlled_graph_streams.py --profile quick --replay-cache CACHE.npz \
        --replay-name BACE --replay-t 16 --q 16
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from controlled_streams import RunConfig, available_devices, run_experiment, run_replay, task_settings


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "quick", "full"), default="full")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--n-groups", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated paired seeds")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--devices", type=str, default="auto", help="auto or comma-separated torch devices")
    parser.add_argument("--train-device", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods; default is all")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated exact setting names or families")
    parser.add_argument("--max-settings", type=int, default=None)
    parser.add_argument("--qrc-batch-size", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--replay-cache", type=Path, default=None)
    parser.add_argument("--replay-name", type=str, default="BACE")
    parser.add_argument("--replay-t", type=int, default=16)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    overrides = {"output_root": str(args.output_root.resolve())}
    if args.q is not None:
        overrides["q"] = args.q
    if args.n_groups is not None:
        overrides["n_groups"] = args.n_groups
    if args.seeds is not None:
        overrides["seeds"] = _parse_int_tuple(args.seeds)
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    devices = available_devices() if args.devices.lower() == "auto" else _parse_str_tuple(args.devices)
    overrides["devices"] = devices
    overrides["train_device"] = args.train_device or devices[0]
    if args.methods is not None:
        overrides["methods"] = _parse_str_tuple(args.methods)
    overrides["qrc_batch_size"] = args.qrc_batch_size
    overrides["force"] = bool(args.force)
    overrides["progress"] = not args.quiet
    cfg = RunConfig.for_profile(args.profile, **overrides)

    if args.replay_cache is not None:
        metrics, summary, paired = run_replay(cfg, args.replay_cache, args.replay_name, args.replay_t)
    else:
        selected = task_settings(args.profile)
        if args.tasks:
            requested = set(_parse_str_tuple(args.tasks))
            selected = [setting for setting in selected if setting.name in requested or setting.family in requested]
            if not selected:
                raise SystemExit(f"No settings matched --tasks={args.tasks!r}")
        if args.max_settings is not None:
            selected = selected[: args.max_settings]
        metrics, summary, paired = run_experiment(cfg, selected)
    print("\nCompleted metrics rows:", len(metrics))
    print(summary[["setting", "method", "test_auc_mean", "test_acc_mean"]].to_string(index=False))
    print("Paired-statistics rows:", len(paired))


if __name__ == "__main__":
    main()

