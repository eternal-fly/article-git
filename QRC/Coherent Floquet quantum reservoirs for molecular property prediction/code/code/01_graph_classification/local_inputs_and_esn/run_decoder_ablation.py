#!/usr/bin/env python3
"""Resumable SVR and regularized-MLP decoder ablation runner."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from decoder_ablation import (
    DECODER_PROTOCOL,
    DECODERS,
    SYSTEMS,
    decoder_result_dir,
    evaluate_case,
    selected_case_rows,
)
from protocol import PROTOCOL_VERSION, RESULT_ROOT, SEEDS


ABLATION_ROOT = RESULT_ROOT / "decoder_ablation"


def _svr_worker(row: dict, force: bool) -> dict:
    return evaluate_case(row, "SVR_RBF", device="cpu", force=force)


def run_svr(cases: list[dict], workers: int, force: bool) -> list[dict]:
    context = mp.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(_svr_worker, case, force) for case in cases]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"SVR {completed}/{len(futures)}", flush=True)
    return rows


def _mlp_process_entry(device: str, task_queue, result_queue, force: bool) -> None:
    try:
        rows = []
        while True:
            case = task_queue.get()
            if case is None:
                break
            payload = evaluate_case(case, "REG_MLP", device=device, force=force)
            rows.append(payload)
            print(
                f"[{device}] MLP {payload['dataset']}/{payload['token_variant']}/"
                f"{payload['system']}/seed{payload['seed']}",
                flush=True,
            )
        result_queue.put({"ok": True, "device": device, "rows": rows})
    except Exception as error:
        result_queue.put(
            {
                "ok": False,
                "device": device,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )


def run_mlp(cases: list[dict], devices: list[str], force: bool) -> list[dict]:
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for case in cases:
        task_queue.put(case)
    for _ in devices:
        task_queue.put(None)
    processes = [
        context.Process(
            target=_mlp_process_entry,
            args=(device, task_queue, result_queue, force),
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
    return [row for message in messages for row in message["rows"]]


def collect(strict: bool = True) -> pd.DataFrame:
    cases = selected_case_rows()
    rows = []
    missing = []
    for case in cases.to_dict("records"):
        for decoder in DECODERS:
            path = decoder_result_dir(
                case["dataset"],
                case["token_variant"],
                case["system"],
                decoder,
                case["seed"],
            ) / "metrics.json"
            if not path.exists():
                missing.append(str(path))
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("protocol_version") != PROTOCOL_VERSION
                or payload.get("decoder_protocol") != DECODER_PROTOCOL
            ):
                raise RuntimeError(f"stale decoder result {path}")
            rows.append(payload)
    if strict and missing:
        raise FileNotFoundError(
            f"missing {len(missing)} decoder results; first={missing[0]}"
        )
    frame = pd.DataFrame(rows)
    if len(frame) and frame.duplicated(
        ["dataset", "token_variant", "system", "decoder", "seed"]
    ).any():
        raise RuntimeError("duplicate decoder result keys")
    return frame


def summarize(strict: bool = True) -> dict:
    learned = collect(strict=strict)
    cases = selected_case_rows()
    affine = cases.rename(
        columns={
            "affine_test_auc": "test_auc",
            "affine_test_acc": "test_acc",
            "affine_test_brier": "test_brier",
        }
    ).copy()
    affine["decoder"] = "AFFINE_ORIGINAL"
    affine["protocol_version"] = PROTOCOL_VERSION
    affine["decoder_protocol"] = DECODER_PROTOCOL
    columns = sorted(set(learned.columns) | set(affine.columns))
    combined = pd.concat(
        [learned.reindex(columns=columns), affine.reindex(columns=columns)],
        ignore_index=True,
    )
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    learned.to_csv(ABLATION_ROOT / "new_decoder_metrics_all.csv", index=False)
    combined.to_csv(ABLATION_ROOT / "metrics_with_affine_baseline.csv", index=False)
    summary = (
        combined.groupby(
            ["dataset", "token_variant", "system", "decoder"], as_index=False
        )
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_brier_mean=("test_brier", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    summary.to_csv(ABLATION_ROOT / "decoder_summary.csv", index=False)

    affine_key = affine[
        ["dataset", "token_variant", "system", "seed", "test_auc"]
    ].rename(columns={"test_auc": "affine_test_auc"})
    decoder_delta = learned.merge(
        affine_key,
        on=["dataset", "token_variant", "system", "seed"],
        validate="many_to_one",
    )
    decoder_delta["decoder_minus_affine_test_auc"] = (
        decoder_delta.test_auc - decoder_delta.affine_test_auc
    )
    decoder_delta.to_csv(
        ABLATION_ROOT / "paired_decoder_vs_affine_by_seed.csv", index=False
    )
    (
        decoder_delta.groupby(
            ["dataset", "token_variant", "system", "decoder"], as_index=False
        )
        .agg(
            decoder_auc_mean=("test_auc", "mean"),
            affine_auc_mean=("affine_test_auc", "mean"),
            mean_delta_auc=("decoder_minus_affine_test_auc", "mean"),
            std_delta_auc=("decoder_minus_affine_test_auc", "std"),
            favourable_seeds=(
                "decoder_minus_affine_test_auc", lambda x: int(np.sum(x > 0))
            ),
            n_seeds=("seed", "nunique"),
        )
    ).to_csv(ABLATION_ROOT / "paired_decoder_vs_affine_summary.csv", index=False)

    pair_specs = (
        ("QRC_OPEN", "QRC_CLOSED"),
        ("QRC_OPEN", "ESN_MATCHED"),
        ("QRC_OPEN", "ESN_BEST_WIDTH"),
        ("QRC_CLOSED", "ESN_MATCHED"),
        ("QRC_CLOSED", "ESN_BEST_WIDTH"),
    )
    paired_rows = []
    for decoder in ("AFFINE_ORIGINAL",) + DECODERS:
        selected = combined[combined.decoder == decoder]
        for left, right in pair_specs:
            lhs = selected[selected.system == left]
            rhs = selected[selected.system == right]
            merged = lhs.merge(
                rhs,
                on=["dataset", "token_variant", "seed", "decoder"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            for _, row in merged.iterrows():
                paired_rows.append(
                    {
                        "dataset": row.dataset,
                        "token_variant": row.token_variant,
                        "decoder": decoder,
                        "comparison": f"{left}_MINUS_{right}",
                        "seed": int(row.seed),
                        "left_test_auc": float(row.test_auc_left),
                        "right_test_auc": float(row.test_auc_right),
                        "delta_test_auc": float(
                            row.test_auc_left - row.test_auc_right
                        ),
                    }
                )
    pairs = pd.DataFrame(paired_rows)
    pairs.to_csv(ABLATION_ROOT / "paired_system_comparisons_by_seed.csv", index=False)
    (
        pairs.groupby(
            ["dataset", "token_variant", "decoder", "comparison"], as_index=False
        )
        .agg(
            left_auc_mean=("left_test_auc", "mean"),
            right_auc_mean=("right_test_auc", "mean"),
            mean_delta_auc=("delta_test_auc", "mean"),
            std_delta_auc=("delta_test_auc", "std"),
            favourable_seeds=("delta_test_auc", lambda x: int(np.sum(x > 0))),
            n_seeds=("seed", "nunique"),
        )
    ).to_csv(ABLATION_ROOT / "paired_system_comparisons_summary.csv", index=False)
    status = {
        "protocol_version": PROTOCOL_VERSION,
        "decoder_protocol": DECODER_PROTOCOL,
        "new_decoder_rows": len(learned),
        "affine_rows": len(affine),
        "combined_rows": len(combined),
    }
    (ABLATION_ROOT / "summary_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )
    return status


def audit(strict: bool = True) -> dict:
    frame = collect(strict=strict)
    expected = 2 * 2 * len(SYSTEMS) * len(DECODERS) * len(SEEDS)
    errors = []
    if len(frame) != expected:
        errors.append(f"result rows {len(frame)} != {expected}")
    if len(frame):
        if not np.isfinite(frame[["test_auc", "test_acc", "test_brier"]]).all().all():
            errors.append("non-finite test metrics")
        counts = frame.groupby(
            ["dataset", "token_variant", "system", "decoder"]
        ).seed.nunique()
        if not (counts == len(SEEDS)).all():
            errors.append("incomplete seed groups")
    for case in selected_case_rows().to_dict("records"):
        for decoder in DECODERS:
            destination = decoder_result_dir(
                case["dataset"],
                case["token_variant"],
                case["system"],
                decoder,
                case["seed"],
            )
            common = (
                destination / "metrics.json",
                destination / "validation_screen.csv",
                destination / "test_predictions.csv",
            )
            decoder_files = (
                (destination / "model.joblib",)
                if decoder == "SVR_RBF"
                else (
                    destination / "model.pt",
                    destination / "scaler.joblib",
                    destination / "training_history.csv",
                )
            )
            for artifact in common + decoder_files:
                if not artifact.exists() or artifact.stat().st_size == 0:
                    errors.append(f"missing or empty artifact {artifact}")
            screen_path = destination / "validation_screen.csv"
            if screen_path.exists():
                screen = pd.read_csv(screen_path)
                expected_candidates = 9 if decoder == "SVR_RBF" else 3
                if len(screen) != expected_candidates:
                    errors.append(
                        f"candidate rows {len(screen)} != {expected_candidates}: "
                        f"{screen_path}"
                    )
            if decoder == "REG_MLP":
                history_path = destination / "training_history.csv"
                if history_path.exists():
                    history = pd.read_csv(history_path)
                    expected_history = 3 * 500
                    if len(history) != expected_history:
                        errors.append(
                            f"history rows {len(history)} != {expected_history}: "
                            f"{history_path}"
                        )
            predictions_path = destination / "test_predictions.csv"
            if predictions_path.exists():
                predictions = pd.read_csv(predictions_path)
                if predictions.empty or predictions.sample_index.duplicated().any():
                    errors.append(f"invalid test predictions {predictions_path}")
    errors = sorted(set(errors))
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "decoder_protocol": DECODER_PROTOCOL,
        "expected_rows": expected,
        "valid_rows": len(frame),
        "errors": errors,
    }
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
    (ABLATION_ROOT / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if strict and errors:
        raise AssertionError(errors[0])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("svr", "mlp", "summarize", "audit"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    cases = selected_case_rows().to_dict("records")
    if args.stage == "svr":
        rows = run_svr(cases, max(1, int(args.workers)), args.force)
    elif args.stage == "mlp":
        devices = [value.strip() for value in args.devices.split(",") if value.strip()]
        if not devices:
            raise ValueError("at least one MLP device is required")
        rows = run_mlp(cases, devices, args.force)
    elif args.stage == "summarize":
        rows = [summarize(strict=args.strict)]
    else:
        rows = [audit(strict=args.strict)]
    print(
        json.dumps(
            {"stage": args.stage, "completed": True, "rows": len(rows)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
