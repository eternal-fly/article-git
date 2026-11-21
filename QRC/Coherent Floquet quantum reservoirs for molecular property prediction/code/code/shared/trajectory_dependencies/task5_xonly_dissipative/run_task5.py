#!/usr/bin/env python3
"""Run the Transition1x X-only dissipative DTC-QRC/ESN comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from task5_core import TASK_DIR, build_dataset, compute_suffix_bank_dual_gpu, evaluate, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("all", "preprocess", "features", "evaluate"), default="all")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    feature_dir, output_dir = TASK_DIR / "features", TASK_DIR / "outputs"
    started, stages = time.perf_counter(), []
    if args.phase in {"all", "preprocess"}:
        path = build_dataset(cfg, force=args.force_preprocess)
        stages.append({"stage": "preprocess", "path": str(path)})
        print(f"[preprocess] {path}", flush=True)
    if args.phase in {"all", "features"}:
        if not cfg.processed_dataset.exists():
            raise FileNotFoundError(cfg.processed_dataset)
        for system in ("qrc", "esn"):
            for seed in cfg.reservoir_seeds:
                stage_started = time.perf_counter()
                path = compute_suffix_bank_dual_gpu(system, seed, cfg, feature_dir, force=args.force_features)
                elapsed = time.perf_counter() - stage_started
                stages.append({"stage": "features", "system": system, "seed": seed, "path": str(path), "elapsed_seconds": elapsed})
                print(f"[features] {system} seed={seed} -> {path} ({elapsed:.2f}s)", flush=True)
    if args.phase in {"all", "evaluate"}:
        path = evaluate(cfg, feature_dir, output_dir)
        stages.append({"stage": "evaluate", "path": str(path)})
        print(f"[evaluate] {path}", flush=True)
    (TASK_DIR / "logs").mkdir(exist_ok=True)
    (TASK_DIR / "logs" / "last_run.json").write_text(json.dumps({
        "status": "completed", "phase": args.phase,
        "total_elapsed_seconds": time.perf_counter() - started, "stages": stages,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

