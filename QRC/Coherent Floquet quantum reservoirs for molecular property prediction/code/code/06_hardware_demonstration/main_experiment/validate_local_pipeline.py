#!/usr/bin/env python3
"""End-to-end offline validation of the local raw-count analysis pipeline.

The test creates temporary Baihua/Shenglian result archives that follow the
cloud-notebook contract. It mixes compact and full-register count keys, then
runs the real local feature reconstruction and linear-readout analysis.
Nothing is written into the manuscript result directories.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANALYSIS_PATH = HERE / "local" / "run_local_analysis.py"
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


def load_analysis_module():
    spec = importlib.util.spec_from_file_location("local_analysis", ANALYSIS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ANALYSIS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bit_key(bits: np.ndarray, chain: list[int], classical_n: int, full: bool) -> str:
    if not full:
        return "".join(str(int(value)) for value in bits[::-1])
    register = ["0"] * classical_n
    for logical, classical_bit in enumerate(chain):
        register[classical_n - 1 - classical_bit] = str(int(bits[logical]))
    return "".join(register)


def synthetic_counts(
    ideal_z: np.ndarray,
    chain: list[int],
    classical_n: int,
    shrink: float,
    full: bool,
) -> dict[str, int]:
    probabilities_one = np.clip((1.0 - shrink * ideal_z) / 2.0, 0.03, 0.97)
    states = (
        np.arange(64, dtype=np.uint64)[:, None]
        >> np.arange(6, dtype=np.uint64)[None, :]
    ) & 1
    probabilities = np.prod(
        np.where(states == 1, probabilities_one, 1.0 - probabilities_one),
        axis=1,
    )
    expected = probabilities * 2048
    integer = np.floor(expected).astype(int)
    remainder = 2048 - int(integer.sum())
    if remainder:
        order = np.argsort(expected - integer)[::-1]
        integer[order[:remainder]] += 1
    return {
        bit_key(bits, chain, classical_n, full): int(count)
        for bits, count in zip(states, integer)
        if count > 0
    }


def make_backend_archive(
    temporary_root: Path,
    backend: str,
    dataset: dict,
    ideal_features: np.ndarray,
) -> Path:
    config = BACKEND_CONFIG[backend]
    result_name = f"{backend}_q6t6l3_cloud_results"
    result_root = temporary_root / "staging" / result_name
    raw_dir = result_root / f"raw_counts_{backend}_q6t6l3"
    raw_dir.mkdir(parents=True)
    order = np.random.default_rng(20260728).permutation(len(dataset["indices"]))
    records = []
    for rank, row_value in enumerate(order):
        row = int(row_value)
        original = int(dataset["indices"][row])
        raw_name = f"rank{rank:05d}_id{original:05d}_tid-fake-{backend}-{rank}.json"
        counts = synthetic_counts(
            ideal_features[row, :6],
            config["chain"],
            config["classical_num_bits"],
            config["shrink"],
            full=bool(row % 2),
        )
        payload = {"data": {"counts": counts}, "status": "Finished"}
        (raw_dir / raw_name).write_text(
            json.dumps(payload, separators=(",", ":")), encoding="utf-8"
        )
        records.append(
            {
                "row": row,
                "submission_rank": rank,
                "original_index": original,
                "q": 6,
                "measured_clbits": config["chain"],
                "classical_num_bits": config["classical_num_bits"],
                "token_hash": str(dataset["token_hashes"][row]),
                "tid": f"fake-{backend}-{rank}",
                "raw_file_name": raw_name,
                "status": "CountsDownloaded",
            }
        )
    manifest = result_root / f"{backend}_q6t6l3_job_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    archive = temporary_root / "downloaded" / f"{backend}_q6t6l3_cloud_results.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(result_root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(result_root.parent))
    return archive


def main() -> None:
    analysis = load_analysis_module()
    dataset = analysis.load_dataset()
    ideal = analysis.load_ideal()
    with tempfile.TemporaryDirectory(prefix="q6t6l3-local-validation-") as name:
        temporary_root = Path(name)
        for backend in BACKEND_CONFIG:
            make_backend_archive(
                temporary_root,
                backend,
                dataset,
                ideal["features"],
            )
        analysis.DOWNLOAD_DIR = temporary_root / "downloaded"
        analysis.OUTPUT_DIR = temporary_root / "local_results"
        analysis.main()
        metrics_path = (
            analysis.OUTPUT_DIR / "ideal_vs_hardware_metrics_by_seed.csv"
        )
        summary_path = analysis.OUTPUT_DIR / "ideal_vs_hardware_summary.csv"
        metrics = pd.read_csv(metrics_path)
        summary = pd.read_csv(summary_path)
        if len(metrics) != 20:
            raise AssertionError(f"Expected 20 metric rows, got {len(metrics)}")
        if len(summary) != 2:
            raise AssertionError(f"Expected 2 summary rows, got {len(summary)}")
        if not np.isfinite(
            metrics[
                [
                    "ideal_test_auc",
                    "frozen_hardware_test_auc",
                    "task_score_pearson",
                    "task_score_spearman",
                ]
            ].to_numpy()
        ).all():
            raise AssertionError("Nonfinite metric in local pipeline output")
        for backend in BACKEND_CONFIG:
            feature_path = (
                analysis.OUTPUT_DIR / f"hardware_features_{backend}_q6t6l3.npz"
            )
            with np.load(feature_path, allow_pickle=False) as package:
                if package["features"].shape != (400, 21):
                    raise AssertionError(package["features"].shape)
                if not np.all(package["shots"] == 2048):
                    raise AssertionError(f"{backend}: unexpected synthetic shot count")
        print(
            "PASS local end-to-end validation: two raw-result ZIPs -> "
            "(400,21) features -> 20 seed/backend rows -> 2-row summary"
        )


if __name__ == "__main__":
    main()
