#!/usr/bin/env python3
"""Process all ten five-task hardware archives and analyze them locally."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import run_local_analysis as common


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "downloaded_five_tasks"
OUTPUT_DIR = ROOT / "local_results_five_tasks"
DATASET_PATH = ROOT / "data" / "bace_bal400_canonical_v3_q6_t6.npz"
IDEAL_PATHS = {
    1: ROOT / "data" / "ideal_bace_bal400_canonical_v3_q6_t6_l1_seed0.npz",
    3: ROOT / "data" / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz",
}
BACKENDS = ("baihua", "shenglian")
TASKS = {
    "01_q6_t6_l3_batch01": {"layers": 3, "fold_factor": 1},
    "02_q6_t6_l3_batch02": {"layers": 3, "fold_factor": 1},
    "03_q6_t6_l3_batch03": {"layers": 3, "fold_factor": 1},
    "04_q6_t6_l1_base_batch01": {"layers": 1, "fold_factor": 1},
    "05_q6_t6_l1_fold3_batch01": {"layers": 1, "fold_factor": 3},
}


def load_ideal(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as package:
        return {
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "token_hashes": package["token_hashes"].astype(str),
            "features": package["features"].astype(np.float32),
        }


def locate_manifest(backend: str, task_key: str) -> Path:
    archive_name = f"{backend}_{task_key}_cloud_results.zip"
    candidates = sorted(DOWNLOAD_DIR.glob(f"**/{archive_name}"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {archive_name} under {DOWNLOAD_DIR}, "
            f"got {candidates}"
        )
    destination = DOWNLOAD_DIR / "extracted" / backend / task_key
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(candidates[0]) as handle:
        handle.extractall(destination)
    manifests = sorted(
        destination.glob(f"**/{backend}_{task_key}_job_manifest.jsonl")
    )
    if len(manifests) != 1:
        raise FileNotFoundError(
            f"{backend}/{task_key}: expected one manifest, got {manifests}"
        )
    return manifests[0]


def build_hardware_features(
    backend: str,
    task_key: str,
    bundle: dict,
) -> Path:
    expected = TASKS[task_key]
    manifest = locate_manifest(backend, task_key)
    records = common.read_jsonl(manifest)
    if len(records) != 400:
        raise RuntimeError(
            f"{backend}/{task_key}: manifest has {len(records)}/400 records"
        )
    row_by_index = {
        int(original): row
        for row, original in enumerate(bundle["indices"].tolist())
    }
    rows = []
    raw_dir = manifest.parent / f"raw_counts_{backend}_{task_key}"
    for record in records:
        if int(record["layers"]) != int(expected["layers"]):
            raise AssertionError(f"{backend}/{task_key}: layer mismatch")
        if int(record["fold_factor"]) != int(expected["fold_factor"]):
            raise AssertionError(f"{backend}/{task_key}: fold mismatch")
        raw_name = record.get("raw_file_name")
        if not raw_name:
            raise RuntimeError(
                f"{backend}/{task_key}: missing raw filename for "
                f"id={record['original_index']}"
            )
        raw_path = raw_dir / raw_name
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        counts = common.extract_counts(result)
        if not counts:
            raise RuntimeError(f"{backend}/{task_key}: no counts in {raw_path}")
        original = int(record["original_index"])
        dataset_row = row_by_index[original]
        if str(record["token_hash"]) != str(
            bundle["token_hashes"][dataset_row]
        ):
            raise AssertionError(
                f"{backend}/{task_key}: token hash mismatch for {original}"
            )
        rows.append(
            (
                dataset_row,
                original,
                common.counts_to_features(counts, record),
                int(round(sum(float(value) for value in counts.values()))),
                str(record["tid"]),
                int(record["submission_rank"]),
                int(record.get("global_submission_rank", record["submission_rank"])),
            )
        )
    rows.sort(key=lambda value: value[0])
    if [value[1] for value in rows] != bundle["indices"].astype(int).tolist():
        raise AssertionError(f"{backend}/{task_key}: dataset order reconstruction failed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"hardware_features_{backend}_{task_key}.npz"
    np.savez_compressed(
        output,
        indices=np.asarray([value[1] for value in rows], dtype=np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        features=np.stack([value[2] for value in rows]).astype(np.float32),
        shots=np.asarray([value[3] for value in rows], dtype=np.int64),
        tids=np.asarray([value[4] for value in rows], dtype=str),
        submission_ranks=np.asarray([value[5] for value in rows], dtype=np.int64),
        global_submission_ranks=np.asarray(
            [value[6] for value in rows], dtype=np.int64
        ),
        backend=np.asarray(backend),
        task_key=np.asarray(task_key),
        layers=np.asarray(expected["layers"], dtype=np.int64),
        fold_factor=np.asarray(expected["fold_factor"], dtype=np.int64),
        source_manifest=np.asarray(str(manifest.resolve())),
    )
    print(f"[saved] {output}")
    return output


def analyze(
    bundle: dict,
    ideals: dict[int, dict],
    feature_paths: dict[tuple[str, str], Path],
) -> None:
    hardware = {}
    for key, path in feature_paths.items():
        with np.load(path, allow_pickle=False) as package:
            if not np.array_equal(package["indices"], bundle["indices"]):
                raise AssertionError(f"{key}: index mismatch")
            hardware[key] = package["features"].astype(np.float32)

    rows = []
    predictions = []
    for seed in common.SEEDS:
        split = common.split_rows(bundle, seed)
        ideal_states = {}
        ideal_fit_rows = {}
        ideal_scores_by_layer = {}
        ideal_metrics_by_layer = {}
        test = split["test"]
        for layers, ideal in ideals.items():
            state, fit = common.train_ideal_linear_head(
                ideal["features"],
                bundle["labels"],
                split,
                seed,
            )
            ideal_states[layers] = state
            ideal_fit_rows[layers] = fit
            ideal_scores = common.logits(ideal["features"][test], state)
            ideal_scores_by_layer[layers] = ideal_scores
            ideal_metrics_by_layer[layers] = common.metrics(
                bundle["labels"][test],
                ideal_scores,
            )

        for task_key, task in TASKS.items():
            layers = int(task["layers"])
            ideal_scores = ideal_scores_by_layer[layers]
            ideal_metrics = ideal_metrics_by_layer[layers]
            for backend in BACKENDS:
                hardware_features = hardware[(backend, task_key)]

                # Primary end-to-end comparison: every feature domain receives
                # the same linear-readout architecture, split, seed, and
                # validation selection rule, but its readout is fitted on that
                # domain's own training features.
                hardware_state, hardware_fit = common.train_ideal_linear_head(
                    hardware_features,
                    bundle["labels"],
                    split,
                    seed,
                )
                hardware_scores = common.logits(
                    hardware_features[test],
                    hardware_state,
                )
                hardware_metrics = common.metrics(
                    bundle["labels"][test],
                    hardware_scores,
                )

                # Secondary zero-calibration diagnostic: apply the
                # ideal-feature readout unchanged to hardware features. This
                # measures coordinate preservation, not final task utility.
                frozen_scores = common.logits(
                    hardware_features[test],
                    ideal_states[layers],
                )
                frozen_metrics = common.metrics(
                    bundle["labels"][test],
                    frozen_scores,
                )

                rows.append(
                    {
                        "backend": backend,
                        "task_key": task_key,
                        "layers": layers,
                        "fold_factor": int(task["fold_factor"]),
                        "seed": seed,
                        "n_test": len(test),
                        "ideal_best_epoch": ideal_fit_rows[layers]["best_epoch"],
                        "ideal_val_auc": ideal_fit_rows[layers]["val_auc"],
                        "ideal_val_loss": ideal_fit_rows[layers]["val_loss"],
                        "ideal_val_acc": ideal_fit_rows[layers]["val_acc"],
                        "ideal_test_auc": ideal_metrics["auc"],
                        "ideal_test_loss": ideal_metrics["loss"],
                        "ideal_test_acc": ideal_metrics["acc"],
                        "hardware_best_epoch": hardware_fit["best_epoch"],
                        "hardware_val_auc": hardware_fit["val_auc"],
                        "hardware_val_loss": hardware_fit["val_loss"],
                        "hardware_val_acc": hardware_fit["val_acc"],
                        "hardware_trained_test_auc": hardware_metrics["auc"],
                        "hardware_trained_test_loss": hardware_metrics["loss"],
                        "hardware_trained_test_acc": hardware_metrics["acc"],
                        "hardware_minus_ideal_auc": (
                            hardware_metrics["auc"] - ideal_metrics["auc"]
                        ),
                        "frozen_ideal_head_hardware_test_auc": (
                            frozen_metrics["auc"]
                        ),
                        "frozen_ideal_head_hardware_test_loss": (
                            frozen_metrics["loss"]
                        ),
                        "frozen_ideal_head_hardware_test_acc": (
                            frozen_metrics["acc"]
                        ),
                        # Retained as an explicitly deprecated compatibility
                        # alias for earlier local audit scripts.
                        "frozen_hardware_test_auc": frozen_metrics["auc"],
                        "frozen_hardware_test_loss": frozen_metrics["loss"],
                        "frozen_hardware_test_acc": frozen_metrics["acc"],
                        "frozen_transfer_minus_ideal_auc": (
                            frozen_metrics["auc"] - ideal_metrics["auc"]
                        ),
                        "readout": "single torch.nn.Linear(d,1) readout layer",
                        "primary_protocol": (
                            "domain-specific linear readout fitted on each "
                            "ideal/hardware training feature matrix"
                        ),
                        "diagnostic_protocol": (
                            "ideal-feature readout frozen and transferred "
                            "unchanged to hardware"
                        ),
                        "selection_rule": common.SELECTION_RULE,
                    }
                )
                predictions.extend(
                    {
                        "backend": backend,
                        "task_key": task_key,
                        "seed": seed,
                        "original_index": int(bundle["indices"][row]),
                        "label": int(bundle["labels"][row]),
                        "ideal_score": float(ideal_score),
                        "hardware_trained_score": float(hardware_score),
                        "frozen_ideal_head_hardware_score": float(frozen_score),
                    }
                    for row, ideal_score, hardware_score, frozen_score in zip(
                        test,
                        ideal_scores,
                        hardware_scores,
                        frozen_scores,
                    )
                )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(
        OUTPUT_DIR / "five_task_ideal_vs_hardware_metrics_by_seed.csv",
        index=False,
    )
    metrics[
        [
            "backend",
            "task_key",
            "layers",
            "fold_factor",
            "seed",
            "n_test",
            "ideal_test_auc",
            "ideal_test_acc",
            "hardware_trained_test_auc",
            "hardware_trained_test_acc",
            "hardware_minus_ideal_auc",
            "hardware_best_epoch",
            "hardware_val_auc",
            "hardware_val_loss",
            "hardware_val_acc",
            "readout",
            "selection_rule",
        ]
    ].to_csv(
        OUTPUT_DIR / "five_task_end_to_end_linear_readout_by_seed.csv",
        index=False,
    )
    metrics[
        [
            "backend",
            "task_key",
            "layers",
            "fold_factor",
            "seed",
            "n_test",
            "ideal_test_auc",
            "ideal_test_acc",
            "frozen_ideal_head_hardware_test_auc",
            "frozen_ideal_head_hardware_test_acc",
            "frozen_transfer_minus_ideal_auc",
            "diagnostic_protocol",
            "selection_rule",
        ]
    ].to_csv(
        OUTPUT_DIR / "five_task_zero_calibration_transfer_by_seed.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "five_task_ideal_vs_hardware_test_scores.csv",
        index=False,
    )
    task_summary = (
        metrics.groupby(
            ["backend", "task_key", "layers", "fold_factor"],
            as_index=False,
        )
        .agg(
            nested_readout_seed_count=("seed", "nunique"),
            ideal_auc_mean=("ideal_test_auc", "mean"),
            ideal_auc_sd_nested=("ideal_test_auc", "std"),
            ideal_acc_mean=("ideal_test_acc", "mean"),
            ideal_acc_sd_nested=("ideal_test_acc", "std"),
            hardware_auc_mean=("hardware_trained_test_auc", "mean"),
            hardware_auc_sd_nested=("hardware_trained_test_auc", "std"),
            hardware_acc_mean=("hardware_trained_test_acc", "mean"),
            hardware_acc_sd_nested=("hardware_trained_test_acc", "std"),
            hardware_minus_ideal_auc_mean=("hardware_minus_ideal_auc", "mean"),
            frozen_transfer_auc_mean=(
                "frozen_ideal_head_hardware_test_auc",
                "mean",
            ),
            frozen_transfer_auc_sd_nested=(
                "frozen_ideal_head_hardware_test_auc",
                "std",
            ),
            frozen_transfer_acc_mean=(
                "frozen_ideal_head_hardware_test_acc",
                "mean",
            ),
            frozen_transfer_acc_sd_nested=(
                "frozen_ideal_head_hardware_test_acc",
                "std",
            ),
            frozen_transfer_minus_ideal_auc_mean=(
                "frozen_transfer_minus_ideal_auc",
                "mean",
            ),
        )
    )
    task_summary["independent_device_repeat_unit"] = np.where(
        task_summary["layers"] == 3,
        "task_key/batch_id",
        "paired L1 acquisition batch",
    )
    task_summary.to_csv(
        OUTPUT_DIR / "five_task_summary.csv",
        index=False,
    )

    l3 = metrics[metrics["layers"] == 3].copy()
    l3_nested = (
        l3.groupby(["backend", "seed"], as_index=False)
        .agg(
            batch_count=("task_key", "nunique"),
            hardware_auc_mean_across_batches=(
                "hardware_trained_test_auc",
                "mean",
            ),
            hardware_auc_sd_across_batches=(
                "hardware_trained_test_auc",
                "std",
            ),
            frozen_transfer_auc_mean_across_batches=(
                "frozen_ideal_head_hardware_test_auc",
                "mean",
            ),
            frozen_transfer_auc_sd_across_batches=(
                "frozen_ideal_head_hardware_test_auc",
                "std",
            ),
        )
    )
    l3_nested.to_csv(
        OUTPUT_DIR / "l3_three_batch_robustness_by_nested_seed.csv",
        index=False,
    )
    l3_batch = (
        l3.groupby(["backend", "task_key"], as_index=False)
        .agg(
            nested_seed_auc_mean=("hardware_trained_test_auc", "mean"),
            nested_seed_auc_sd=("hardware_trained_test_auc", "std"),
            frozen_transfer_auc_mean=(
                "frozen_ideal_head_hardware_test_auc",
                "mean",
            ),
            frozen_transfer_auc_sd=(
                "frozen_ideal_head_hardware_test_auc",
                "std",
            ),
        )
    )
    l3_batch.to_csv(
        OUTPUT_DIR / "l3_acquisition_batch_level_auc.csv",
        index=False,
    )

    base = metrics[
        metrics["task_key"] == "04_q6_t6_l1_base_batch01"
    ][["backend", "seed", "hardware_trained_test_auc"]].rename(
        columns={"hardware_trained_test_auc": "l1_base_auc"}
    )
    fold = metrics[
        metrics["task_key"] == "05_q6_t6_l1_fold3_batch01"
    ][["backend", "seed", "hardware_trained_test_auc"]].rename(
        columns={"hardware_trained_test_auc": "l1_fold3_auc"}
    )
    l1_pair = base.merge(fold, on=["backend", "seed"], validate="one_to_one")
    l1_pair["fold3_minus_base_auc"] = (
        l1_pair["l1_fold3_auc"] - l1_pair["l1_base_auc"]
    )
    l1_pair.to_csv(
        OUTPUT_DIR / "l1_base_fold3_paired_by_nested_seed.csv",
        index=False,
    )
    l1_pair.groupby("backend", as_index=False).agg(
        nested_seed_count=("seed", "nunique"),
        base_auc_mean=("l1_base_auc", "mean"),
        fold3_auc_mean=("l1_fold3_auc", "mean"),
        fold3_minus_base_auc_mean=("fold3_minus_base_auc", "mean"),
        fold3_minus_base_auc_sd_nested=("fold3_minus_base_auc", "std"),
    ).to_csv(
        OUTPUT_DIR / "l1_base_fold3_paired_summary.csv",
        index=False,
    )

    frozen_base = metrics[
        metrics["task_key"] == "04_q6_t6_l1_base_batch01"
    ][["backend", "seed", "frozen_ideal_head_hardware_test_auc"]].rename(
        columns={"frozen_ideal_head_hardware_test_auc": "l1_base_auc"}
    )
    frozen_fold = metrics[
        metrics["task_key"] == "05_q6_t6_l1_fold3_batch01"
    ][["backend", "seed", "frozen_ideal_head_hardware_test_auc"]].rename(
        columns={"frozen_ideal_head_hardware_test_auc": "l1_fold3_auc"}
    )
    frozen_pair = frozen_base.merge(
        frozen_fold,
        on=["backend", "seed"],
        validate="one_to_one",
    )
    frozen_pair["fold3_minus_base_auc"] = (
        frozen_pair["l1_fold3_auc"] - frozen_pair["l1_base_auc"]
    )
    frozen_pair.to_csv(
        OUTPUT_DIR
        / "l1_base_fold3_zero_calibration_transfer_by_nested_seed.csv",
        index=False,
    )
    print(task_summary.to_string(index=False))


def main() -> None:
    common.DATASET_PATH = DATASET_PATH
    bundle = common.load_dataset()
    ideals = {layers: load_ideal(path) for layers, path in IDEAL_PATHS.items()}
    for layers, ideal in ideals.items():
        if not np.array_equal(ideal["indices"], bundle["indices"]):
            raise AssertionError(f"L{layers}: ideal index mismatch")
        if not np.array_equal(ideal["token_hashes"], bundle["token_hashes"]):
            raise AssertionError(f"L{layers}: ideal token hash mismatch")
    feature_paths = {
        (backend, task_key): build_hardware_features(
            backend,
            task_key,
            bundle,
        )
        for backend in BACKENDS
        for task_key in TASKS
    }
    analyze(bundle, ideals, feature_paths)


if __name__ == "__main__":
    main()
