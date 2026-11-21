#!/usr/bin/env python3
"""End-to-end offline validation of the complete five-task local analysis."""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "local"))

import run_complete_five_task_analysis as analysis  # noqa: E402
import run_local_analysis as common  # noqa: E402
from validate_local_pipeline import synthetic_counts  # noqa: E402


BACKEND_CONFIG = {
    "baihua": {
        "chain": [72, 73, 74, 75, 76, 77],
        "classical_num_bits": 156,
        "shrink": 0.82,
    },
    "shenglian": {
        "chain": [27, 34, 41, 48, 54, 61],
        "classical_num_bits": 84,
        "shrink": 0.88,
    },
}


def make_archive(
    root: Path,
    backend: str,
    task_key: str,
    bundle: dict,
    ideal_features: np.ndarray,
) -> None:
    task = analysis.TASKS[task_key]
    backend_config = BACKEND_CONFIG[backend]
    result_name = f"{backend}_{task_key}_cloud_results"
    result_root = root / "staging" / backend / task_key / result_name
    raw_dir = result_root / f"raw_counts_{backend}_{task_key}"
    raw_dir.mkdir(parents=True)
    seed = {
        "01_q6_t6_l3_batch01": 31001,
        "02_q6_t6_l3_batch02": 31002,
        "03_q6_t6_l3_batch03": 31003,
        "04_q6_t6_l1_base_batch01": 41001,
        "05_q6_t6_l1_fold3_batch01": 41001,
    }[task_key]
    order = np.random.default_rng(seed).permutation(len(bundle["indices"]))
    records = []
    for rank, row_value in enumerate(order):
        row = int(row_value)
        original = int(bundle["indices"][row])
        raw_name = (
            f"rank{rank:05d}_id{original:05d}"
            f"_tid-fake-{backend}-{task_key}-{rank}.json"
        )
        shrink = float(backend_config["shrink"])
        if int(task["fold_factor"]) == 3:
            shrink *= 0.92
        counts = synthetic_counts(
            ideal_features[row, :6],
            backend_config["chain"],
            backend_config["classical_num_bits"],
            shrink,
            full=bool(row % 2),
        )
        (raw_dir / raw_name).write_text(
            json.dumps(
                {"data": {"counts": counts}, "status": "Finished"},
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "row": row,
                "submission_rank": rank,
                "global_submission_rank": (
                    2 * rank
                    + (
                        1
                        if task_key == "05_q6_t6_l1_fold3_batch01"
                        else 0
                    )
                    if int(task["layers"]) == 1
                    else rank
                ),
                "original_index": original,
                "q": 6,
                "layers": int(task["layers"]),
                "fold_factor": int(task["fold_factor"]),
                "measured_clbits": backend_config["chain"],
                "classical_num_bits": backend_config["classical_num_bits"],
                "token_hash": str(bundle["token_hashes"][row]),
                "tid": f"fake-{backend}-{task_key}-{rank}",
                "raw_file_name": raw_name,
                "status": "CountsDownloaded",
            }
        )
    manifest = result_root / f"{backend}_{task_key}_job_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    archive = root / "downloaded" / f"{backend}_{task_key}_cloud_results.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(result_root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(result_root.parent))


def main() -> None:
    common.DATASET_PATH = analysis.DATASET_PATH
    bundle = common.load_dataset()
    ideals = {
        layers: analysis.load_ideal(path)
        for layers, path in analysis.IDEAL_PATHS.items()
    }
    with tempfile.TemporaryDirectory(
        prefix="q6t6-five-task-local-validation-"
    ) as name:
        root = Path(name)
        for backend in analysis.BACKENDS:
            for task_key, task in analysis.TASKS.items():
                make_archive(
                    root,
                    backend,
                    task_key,
                    bundle,
                    ideals[int(task["layers"])]["features"],
                )
        analysis.DOWNLOAD_DIR = root / "downloaded"
        analysis.OUTPUT_DIR = root / "local_results"
        analysis.main()
        feature_files = list(
            analysis.OUTPUT_DIR.glob("hardware_features_*.npz")
        )
        if len(feature_files) != 10:
            raise AssertionError(f"Expected 10 feature files, got {len(feature_files)}")
        metrics = pd.read_csv(
            analysis.OUTPUT_DIR
            / "five_task_ideal_vs_hardware_metrics_by_seed.csv"
        )
        summary = pd.read_csv(
            analysis.OUTPUT_DIR / "five_task_summary.csv"
        )
        l3_batches = pd.read_csv(
            analysis.OUTPUT_DIR / "l3_acquisition_batch_level_auc.csv"
        )
        l1_pair = pd.read_csv(
            analysis.OUTPUT_DIR / "l1_base_fold3_paired_by_nested_seed.csv"
        )
        if len(metrics) != 100:
            raise AssertionError(f"Expected 100 metric rows, got {len(metrics)}")
        if len(summary) != 10:
            raise AssertionError(f"Expected 10 summary rows, got {len(summary)}")
        if len(l3_batches) != 6:
            raise AssertionError(f"Expected 6 L3 batch rows, got {len(l3_batches)}")
        if len(l1_pair) != 20:
            raise AssertionError(f"Expected 20 L1 pair rows, got {len(l1_pair)}")
        numeric = metrics[
            [
                "ideal_test_auc",
                "hardware_trained_test_auc",
                "hardware_minus_ideal_auc",
                "frozen_ideal_head_hardware_test_auc",
                "frozen_transfer_minus_ideal_auc",
            ]
        ].to_numpy()
        legacy_aliases = metrics[
            [
                "frozen_hardware_test_auc",
                "frozen_hardware_minus_ideal_auc",
            ]
        ].to_numpy()
        if not np.isfinite(numeric).all() or not np.isfinite(legacy_aliases).all():
            raise AssertionError("Nonfinite complete-pipeline metric")
        print(
            "PASS complete local pipeline: 10 raw-result ZIPs -> "
            "10 feature NPZs -> 100 task/backend/seed rows -> "
            "three-batch and L1-pair summaries"
        )


if __name__ == "__main__":
    main()
