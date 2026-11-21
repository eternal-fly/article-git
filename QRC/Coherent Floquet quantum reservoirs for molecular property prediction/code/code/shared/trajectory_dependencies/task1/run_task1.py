#!/usr/bin/env python3
"""Run preprocessing, dual-GPU features, and affine evaluation for task 1."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from task1_core import (
    TASK_DIR,
    compute_features_dual_gpu,
    evaluate_features,
    load_config,
    preprocess_real_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "preprocess", "features", "evaluate"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=TASK_DIR / "config.json")
    parser.add_argument("--force-preprocess", action="store_true")
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
    stage_log = []

    if args.phase in {"all", "preprocess"}:
        stage_started = time.perf_counter()
        dataset_path = preprocess_real_dataset(cfg, force=args.force_preprocess)
        stage_log.append(
            {
                "stage": "preprocess",
                "path": str(dataset_path),
                "elapsed_seconds": time.perf_counter() - stage_started,
            }
        )
        print(f"[preprocess] {dataset_path}", flush=True)

    if args.phase in {"all", "features"}:
        if not cfg.processed_dataset.exists():
            raise FileNotFoundError(
                f"processed dataset missing: {cfg.processed_dataset}; "
                "run --phase preprocess first"
            )
        for method in ("qrc", "esn"):
            for seed in cfg.reservoir_seeds:
                stage_started = time.perf_counter()
                feature_path = compute_features_dual_gpu(
                    method,
                    seed,
                    cfg,
                    feature_dir,
                    force=args.force_features,
                )
                elapsed = time.perf_counter() - stage_started
                stage_log.append(
                    {
                        "stage": "features",
                        "method": method,
                        "seed": int(seed),
                        "path": str(feature_path),
                        "elapsed_seconds": elapsed,
                        "gpus": list(cfg.gpu_ids),
                    }
                )
                print(
                    f"[features] {method} seed={seed} -> {feature_path} "
                    f"({elapsed:.2f}s, GPUs={cfg.gpu_ids})",
                    flush=True,
                )

    if args.phase in {"all", "evaluate"}:
        stage_started = time.perf_counter()
        report_path = evaluate_features(cfg, feature_dir, output_dir)
        stage_log.append(
            {
                "stage": "evaluate",
                "path": str(report_path),
                "elapsed_seconds": time.perf_counter() - stage_started,
            }
        )
        print(f"[evaluate] {report_path}", flush=True)

    run_log = {
        "status": "completed",
        "phase": args.phase,
        "total_elapsed_seconds": time.perf_counter() - started,
        "stages": stage_log,
    }
    (log_dir / "last_run.json").write_text(
        json.dumps(run_log, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

