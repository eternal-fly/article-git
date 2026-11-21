#!/usr/bin/env python3
"""Run the Task 3 A-F observable and decoder ablations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from task3_core import (
    TASK_DIR,
    compute_esn_temporal_two_devices,
    compute_features_two_devices,
    diagnose_features,
    evaluate_all_variants,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "features", "evaluate", "diagnose"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=TASK_DIR / "config.json")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    feature_dir = TASK_DIR / "features"
    output_dir = TASK_DIR / "outputs"
    log_dir = TASK_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stages = []

    if not cfg.processed_dataset.exists():
        raise FileNotFoundError(cfg.processed_dataset)

    if args.phase in {"all", "features"}:
        for seed in cfg.reservoir_seeds:
            stage_start = time.perf_counter()
            path = compute_features_two_devices(
                cfg,
                seed,
                feature_dir,
                force=args.force_features,
            )
            elapsed = time.perf_counter() - stage_start
            stages.append(
                {
                    "stage": "features",
                    "seed": int(seed),
                    "path": str(path),
                    "devices": list(cfg.devices),
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                f"[features] seed={seed} -> {path} "
                f"({elapsed:.2f}s; {cfg.devices})",
                flush=True,
            )
            stage_start = time.perf_counter()
            esn_path = compute_esn_temporal_two_devices(
                cfg,
                seed,
                feature_dir,
                force=args.force_features,
            )
            elapsed = time.perf_counter() - stage_start
            stages.append(
                {
                    "stage": "esn_temporal_features",
                    "seed": int(seed),
                    "path": str(esn_path),
                    "devices": list(cfg.devices),
                    "elapsed_seconds": elapsed,
                }
            )
            print(
                f"[esn-temporal] seed={seed} -> {esn_path} "
                f"({elapsed:.2f}s; {cfg.devices})",
                flush=True,
            )

    if args.phase in {"all", "evaluate"}:
        stage_start = time.perf_counter()
        path = evaluate_all_variants(cfg, feature_dir, output_dir)
        stages.append(
            {
                "stage": "evaluate",
                "path": str(path),
                "elapsed_seconds": time.perf_counter() - stage_start,
            }
        )
        print(f"[evaluate] {path}", flush=True)

    if args.phase in {"all", "diagnose"}:
        stage_start = time.perf_counter()
        path = diagnose_features(cfg, feature_dir, output_dir)
        stages.append(
            {
                "stage": "diagnose",
                "path": str(path),
                "elapsed_seconds": time.perf_counter() - stage_start,
            }
        )
        print(f"[diagnose] {path}", flush=True)

    log = {
        "status": "completed",
        "phase": args.phase,
        "total_elapsed_seconds": time.perf_counter() - started,
        "stages": stages,
    }
    (log_dir / "last_run.json").write_text(
        json.dumps(log, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
