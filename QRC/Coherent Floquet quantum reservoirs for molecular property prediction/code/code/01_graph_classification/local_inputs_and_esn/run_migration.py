#!/usr/bin/env python3
"""Resumable command-line runner for the complete migration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import traceback

from protocol import (
    DATASETS,
    LOG_ROOT,
    Q_VALUES,
    SEEDS,
    T_VALUES,
    TOKEN_VARIANTS,
    esn_widths,
)


def _choices(raw: str, allowed: tuple, cast=str) -> tuple:
    if raw.lower() == "all":
        return allowed
    values = tuple(cast(value.strip()) for value in raw.split(",") if value.strip())
    bad = sorted(set(values) - set(allowed))
    if bad:
        raise ValueError(f"unsupported values {bad}; allowed={allowed}")
    return values


def _device_chunks(tasks: list[tuple], devices: list[str]) -> list[tuple[str, list[tuple]]]:
    chunks = [[] for _ in devices]
    for index, task in enumerate(tasks):
        chunks[index % len(devices)].append(task)
    return [(device, chunk) for device, chunk in zip(devices, chunks) if chunk]


def _qrc_feature_worker(device: str, tasks: list[tuple], force: bool) -> list[dict]:
    from qrc_features import compute_all_t_suffix_banks

    rows = []
    for dataset, variant, q, seed in tasks:
        rows.extend(
            compute_all_t_suffix_banks(
                dataset, variant, q, seed, device=device, force=force
            )
        )
    return rows


def _qrc_feature_queue_worker(device: str, task_queue, force: bool) -> list[dict]:
    from qrc_features import compute_all_t_suffix_banks

    rows = []
    while True:
        task = task_queue.get()
        if task is None:
            break
        dataset, variant, q, seed = task
        rows.extend(
            compute_all_t_suffix_banks(
                dataset, variant, q, seed, device=device, force=force
            )
        )
    return rows


def _esn_feature_worker(device: str, tasks: list[tuple], force: bool) -> list[dict]:
    from esn_features import compute_esn_all_t

    rows = []
    for dataset, variant, q, width, seed in tasks:
        rows.extend(
            compute_esn_all_t(
                dataset, variant, q, width, seed, device=device, force=force
            )
        )
    return rows


def _esn_feature_queue_worker(device: str, task_queue, force: bool) -> list[dict]:
    from esn_features import compute_esn_all_t

    rows = []
    while True:
        task = task_queue.get()
        if task is None:
            break
        dataset, variant, q, width, seed = task
        rows.extend(
            compute_esn_all_t(
                dataset, variant, q, width, seed, device=device, force=force
            )
        )
    return rows


def _qrc_evaluation_worker(device: str, tasks: list[tuple], force: bool) -> list[dict]:
    from readouts import evaluate_qrc_setting

    rows = []
    for dataset, variant, q, t, seed in tasks:
        rows.extend(
            evaluate_qrc_setting(
                dataset, variant, q, t, seed, device=device, force=force
            )
        )
    return rows


def _qrc_evaluation_queue_worker(device: str, task_queue, force: bool) -> list[dict]:
    from readouts import evaluate_qrc_setting

    rows = []
    while True:
        task = task_queue.get()
        if task is None:
            break
        dataset, variant, q, t, seed = task
        rows.extend(
            evaluate_qrc_setting(
                dataset, variant, q, t, seed, device=device, force=force
            )
        )
    return rows


def _esn_evaluation_worker(device: str, tasks: list[tuple], force: bool) -> list[dict]:
    from esn_evaluation import evaluate_esn_width_group

    rows = []
    for dataset, variant, q, t, seed in tasks:
        rows.extend(
            evaluate_esn_width_group(
                dataset, variant, q, t, seed, device=device, force=force
            )
        )
    return rows


def _esn_evaluation_queue_worker(device: str, task_queue, force: bool) -> list[dict]:
    from esn_evaluation import evaluate_esn_width_group

    rows = []
    while True:
        task = task_queue.get()
        if task is None:
            break
        dataset, variant, q, t, seed = task
        rows.extend(
            evaluate_esn_width_group(
                dataset, variant, q, t, seed, device=device, force=force
            )
        )
    return rows


def _parallel_stage(worker, tasks: list[tuple], devices: list[str], force: bool) -> list[dict]:
    if not tasks:
        return []
    chunks = _device_chunks(tasks, devices)
    if len(chunks) == 1:
        return worker(chunks[0][0], chunks[0][1], force)
    context = mp.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=context) as pool:
        futures = [pool.submit(worker, device, chunk, force) for device, chunk in chunks]
        for future in as_completed(futures):
            rows.extend(future.result())
    return rows


def _queue_process_entry(worker, device: str, task_queue, result_queue, force: bool) -> None:
    try:
        result_queue.put({"ok": True, "device": device, "rows": worker(device, task_queue, force)})
    except Exception as error:
        result_queue.put(
            {
                "ok": False,
                "device": device,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )


def _parallel_queue_stage(worker, tasks: list[tuple], devices: list[str], force: bool) -> list[dict]:
    """Dynamic shared queue: a faster device automatically receives more jobs."""
    if not tasks:
        return []
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in devices:
        task_queue.put(None)
    processes = [
        context.Process(
            target=_queue_process_entry,
            args=(worker, device, task_queue, result_queue, force),
        )
        for device in devices
    ]
    for process in processes:
        process.start()
    messages = []
    try:
        for _ in processes:
            messages.append(result_queue.get())
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    failures = [message for message in messages if not message["ok"]]
    if failures:
        raise RuntimeError(f"{failures[0]['error']}\n{failures[0]['traceback']}")
    rows = []
    for message in messages:
        rows.extend(message["rows"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("tokens", "qrc-features", "qrc-evaluate", "esn-features", "esn-evaluate", "summarize", "audit"),
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--q", default="all")
    parser.add_argument("--t", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--widths", default="all", help="all, matched, or comma-separated integers")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    datasets = _choices(args.datasets, DATASETS, str)
    variants = _choices(args.variants, TOKEN_VARIANTS, str)
    q_values = _choices(args.q, Q_VALUES, int)
    t_values = _choices(args.t, T_VALUES, int)
    seeds = _choices(args.seeds, SEEDS, int)
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices:
        raise ValueError("at least one device is required")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage == "tokens":
        from splits import build_splits
        from tokens import build_token_cache

        rows = []
        for dataset in datasets:
            for variant in variants:
                path, metadata = build_token_cache(dataset, variant, force=args.force)
                rows.append({"path": str(path.resolve()), **metadata})
            build_splits(dataset, force=args.force)
    elif args.stage == "qrc-features":
        tasks = [
            (dataset, variant, q, seed)
            for dataset in datasets
            for variant in variants
            for q in q_values
            for seed in seeds
        ]
        # Larger Q is dispatched first to balance the two persistent GPU workers.
        tasks.sort(key=lambda row: (-row[2], row[0], row[1], row[3]))
        rows = _parallel_queue_stage(
            _qrc_feature_queue_worker, tasks, devices, args.force
        )
    elif args.stage == "qrc-evaluate":
        tasks = [
            (dataset, variant, q, t, seed)
            for dataset in datasets
            for variant in variants
            for q in q_values
            for t in t_values
            for seed in seeds
        ]
        rows = _parallel_queue_stage(
            _qrc_evaluation_queue_worker, tasks, devices, args.force
        )
    elif args.stage == "esn-features":
        if args.widths == "matched":
            width_map = {q: (q * (q + 1) // 2,) for q in q_values}
        elif args.widths == "all":
            width_map = {q: esn_widths(q) for q in q_values}
        else:
            requested = tuple(int(value) for value in args.widths.split(","))
            width_map = {q: requested for q in q_values}
        # Build each exact recurrent matrix once before CUDA workers can race for it.
        from esn_features import build_or_load_recurrent

        for width in sorted({width for values in width_map.values() for width in values}):
            for seed in seeds:
                build_or_load_recurrent(width, seed, force=args.force)
        tasks = [
            (dataset, variant, q, width, seed)
            for dataset in datasets
            for variant in variants
            for q in q_values
            for width in width_map[q]
            for seed in seeds
        ]
        tasks.sort(key=lambda row: (-row[3], row[0], row[1], row[2], row[4]))
        rows = _parallel_queue_stage(
            _esn_feature_queue_worker, tasks, devices, args.force
        )
    elif args.stage == "esn-evaluate":
        if args.widths != "all":
            raise ValueError("vectorized ESN evaluation requires --widths all")
        tasks = [
            (dataset, variant, q, t, seed)
            for dataset in datasets
            for variant in variants
            for q in q_values
            for t in t_values
            for seed in seeds
        ]
        rows = _parallel_queue_stage(
            _esn_evaluation_queue_worker, tasks, devices, args.force
        )
    elif args.stage == "summarize":
        from summarize import summarize_all

        rows = [summarize_all(strict=args.strict)]
    else:
        from audit import run_audit

        rows = [run_audit(strict=args.strict)]
    status = {"stage": args.stage, "rows": len(rows), "completed": True}
    (LOG_ROOT / f"{args.stage}_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
