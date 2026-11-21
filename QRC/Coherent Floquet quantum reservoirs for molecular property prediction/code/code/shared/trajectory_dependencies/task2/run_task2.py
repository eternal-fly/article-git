#!/usr/bin/env python3
"""Run the 18-injection Transition1x QRC/ESN experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from task2_core import (
    TASK2_DIR,
    build_task2_dataset,
    compare_with_task1,
    compute_features_dual_gpu,
    evaluate_features,
    load_config,
)
from diagnose_task2 import main as run_feature_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", "preprocess", "features", "evaluate"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=TASK2_DIR / "config.json")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime_cfg = cfg.runtime_config()
    feature_dir = TASK2_DIR / "features"
    output_dir = TASK2_DIR / "outputs"
    log_dir = TASK2_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage_log = []

    if args.phase in {"all", "preprocess"}:
        stage_started = time.perf_counter()
        dataset_path = build_task2_dataset(
            cfg, force=args.force_preprocess
        )
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
                f"processed dataset missing: {cfg.processed_dataset}"
            )
        for method in ("qrc", "esn"):
            for seed in cfg.reservoir_seeds:
                stage_started = time.perf_counter()
                feature_path = compute_features_dual_gpu(
                    method,
                    seed,
                    runtime_cfg,
                    feature_dir,
                    force=args.force_features,
                )
                elapsed = time.perf_counter() - stage_started
                stage_log.append(
                    {
                        "stage": "features",
                        "method": method,
                        "seed": seed,
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
        report_path = evaluate_features(
            runtime_cfg, feature_dir, output_dir
        )
        comparison_path = compare_with_task1(output_dir)
        run_feature_diagnostics()
        diagnostics_path = output_dir / "feature_diagnostics_summary.csv"
        stage_log.append(
            {
                "stage": "evaluate",
                "path": str(report_path),
                "comparison": str(comparison_path),
                "diagnostics": str(diagnostics_path),
                "elapsed_seconds": time.perf_counter() - stage_started,
            }
        )
        print(f"[evaluate] {report_path}", flush=True)
        print(f"[compare] {comparison_path}", flush=True)
        print(f"[diagnose] {diagnostics_path}", flush=True)

    (log_dir / "last_run.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "phase": args.phase,
                "total_elapsed_seconds": time.perf_counter() - started,
                "q": cfg.q,
                "g": cfg.g,
                "W": cfg.W,
                "total_injections": cfg.total_injections,
                "stages": stage_log,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
