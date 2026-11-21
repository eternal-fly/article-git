"""Task 2: expand six molecular images into 18 hierarchical injections."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TASK2_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK2_DIR.parent
TASK1_DIR = TRANSITION_DIR / "task1"
if str(TASK1_DIR) not in sys.path:
    sys.path.insert(0, str(TASK1_DIR))

from task1_core import (  # noqa: E402
    Task1Config,
    compute_features_dual_gpu,
    evaluate_features,
)


@dataclass(frozen=True)
class Task2Config:
    source_processed_dataset: Path
    processed_dataset: Path
    train_count: int
    val_count: int
    test_count: int
    sample_seed: int
    q: int
    source_images: int
    hierarchical_stages: tuple[str, ...]
    injections_per_image: int
    total_injections: int
    target_image_index: int
    g: float
    W: float
    layers: int
    reservoir_seeds: tuple[int, ...]
    gpu_ids: tuple[int, int]

    @property
    def output_dim(self) -> int:
        return int(self.q * (self.q + 1) // 2)

    @property
    def total_count(self) -> int:
        return int(self.train_count + self.val_count + self.test_count)

    def runtime_config(self) -> Task1Config:
        """Return the interface consumed by the shared GPU/evaluation code."""
        return Task1Config(
            raw_h5=self.source_processed_dataset,
            processed_dataset=self.processed_dataset,
            train_count=self.train_count,
            val_count=self.val_count,
            test_count=self.test_count,
            sample_seed=self.sample_seed,
            q=self.q,
            prefix_length=self.total_injections,
            target_image_index=self.target_image_index,
            g=self.g,
            W=self.W,
            layers=self.layers,
            reservoir_seeds=self.reservoir_seeds,
            gpu_ids=self.gpu_ids,
        )


def load_config(path: str | Path | None = None) -> Task2Config:
    config_path = TASK2_DIR / "config.json" if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    counts = payload["sample_counts"]
    qrc = payload["qrc"]
    cfg = Task2Config(
        source_processed_dataset=(
            TASK2_DIR / payload["source_processed_dataset"]
        ).resolve(),
        processed_dataset=(TASK2_DIR / payload["processed_dataset"]).resolve(),
        train_count=int(counts["train"]),
        val_count=int(counts["val"]),
        test_count=int(counts["test"]),
        sample_seed=int(payload["sample_seed"]),
        q=int(payload["num_qubits"]),
        source_images=int(payload["source_images"]),
        hierarchical_stages=tuple(payload["hierarchical_stages"]),
        injections_per_image=int(payload["injections_per_image"]),
        total_injections=int(payload["total_injections"]),
        target_image_index=int(payload["target_image_index"]),
        g=float(qrc["g"]),
        W=float(qrc["W"]),
        layers=int(qrc["layers_per_injection"]),
        reservoir_seeds=tuple(int(value) for value in payload["reservoir_seeds"]),
        gpu_ids=tuple(int(value) for value in payload["gpus_per_method"]),
    )
    if cfg.injections_per_image != 3:
        raise ValueError("task2 hierarchical encoder requires exactly three stages")
    if cfg.total_injections != cfg.source_images * cfg.injections_per_image:
        raise ValueError("total_injections must equal source_images * stages")
    if cfg.total_injections < 16:
        raise ValueError("task2 must contain at least 16 injections")
    if cfg.q != 8 or cfg.g != 0.84 or cfg.W != 0.6:
        raise ValueError("task2 locks Q=8, g=0.84, and W=0.6")
    return cfg


def expand_hierarchical_injections(
    source_tokens: np.ndarray,
    q: int,
) -> np.ndarray:
    """Create identity -> geometry -> motion stages for every source image.

    The three source fields for each pair are:
      0. chemical pair/role identity
      1. normalized distance
      2. distance increment

    Stages are cumulative so later stages refine the same chemical event rather
    than introducing interpolated molecular geometries.
    """
    source = np.asarray(source_tokens, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 3 * int(q):
        raise ValueError(
            f"source_tokens must have shape [N,T,{3 * int(q)}], got {source.shape}"
        )
    structured = source.reshape(len(source), source.shape[1], int(q), 3)
    expanded = np.zeros(
        (len(source), 3 * source.shape[1], int(q), 3),
        dtype=np.float32,
    )
    expanded[:, 0::3, :, 0] = structured[:, :, :, 0]
    expanded[:, 1::3, :, :2] = structured[:, :, :, :2]
    expanded[:, 2::3, :, :] = structured
    return expanded.reshape(len(source), 3 * source.shape[1], 3 * int(q))


def build_task2_dataset(cfg: Task2Config, force: bool = False) -> Path:
    if cfg.processed_dataset.exists() and not force:
        return cfg.processed_dataset
    if not cfg.source_processed_dataset.exists():
        raise FileNotFoundError(cfg.source_processed_dataset)

    with np.load(cfg.source_processed_dataset, allow_pickle=False) as source:
        source_tokens = source["tokens"].astype(np.float32)
        if source_tokens.shape != (
            cfg.total_count,
            cfg.source_images,
            3 * cfg.q,
        ):
            raise RuntimeError(
                f"unexpected task1 token shape {source_tokens.shape}; expected "
                f"{(cfg.total_count, cfg.source_images, 3 * cfg.q)}"
            )
        expanded = expand_hierarchical_injections(source_tokens, cfg.q)
        arrays = {
            key: source[key]
            for key in (
                "reaction_ids",
                "formulas",
                "splits",
                "num_atoms",
                "atomic_numbers",
                "target_delta_rho",
                "target_delta_energy",
                "pair_slots",
                "role_ids",
                "changed_pair_mask",
                "valid_pair_mask",
                "relative_energy_paths",
            )
        }

    expected_splits = {
        "train": cfg.train_count,
        "val": cfg.val_count,
        "test": cfg.test_count,
    }
    observed_splits = {
        split: int(np.sum(arrays["splits"] == split))
        for split in ("train", "val", "test")
    }
    if observed_splits != expected_splits:
        raise RuntimeError(
            f"source split counts changed: {observed_splits} != {expected_splits}"
        )

    cfg.processed_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.processed_dataset,
        dataset_version=np.asarray(
            "transition1x-task2-hierarchical-injections-v1"
        ),
        source_processed_dataset=np.asarray(str(cfg.source_processed_dataset)),
        tokens=expanded,
        **arrays,
        q=np.asarray(cfg.q, dtype=np.int64),
        source_images=np.asarray(cfg.source_images, dtype=np.int64),
        injections_per_image=np.asarray(
            cfg.injections_per_image, dtype=np.int64
        ),
        total_injections=np.asarray(cfg.total_injections, dtype=np.int64),
        target_image_index=np.asarray(cfg.target_image_index, dtype=np.int64),
        sample_seed=np.asarray(cfg.sample_seed, dtype=np.int64),
    )
    metadata = {
        "dataset_version": "transition1x-task2-hierarchical-injections-v1",
        "source_processed_dataset": str(cfg.source_processed_dataset),
        "processed_dataset": str(cfg.processed_dataset),
        "sample_counts": observed_splits,
        "q": cfg.q,
        "g": cfg.g,
        "W": cfg.W,
        "layers_per_injection": cfg.layers,
        "source_images": cfg.source_images,
        "injections_per_image": cfg.injections_per_image,
        "total_injections": cfg.total_injections,
        "hierarchical_stages": list(cfg.hierarchical_stages),
        "stage_semantics": [
            "[identity, 0, 0]",
            "[identity, geometry, 0]",
            "[identity, geometry, motion]",
        ],
        "uses_interpolated_geometries": False,
        "uses_bfs": False,
        "target_image_index": cfg.target_image_index,
    }
    (cfg.processed_dataset.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cfg.processed_dataset


def compare_with_task1(task2_output_dir: Path) -> Path:
    task1_metrics = TRANSITION_DIR / "task1" / "outputs" / "summary.csv"
    task2_metrics = task2_output_dir / "summary.csv"
    if not task1_metrics.exists() or not task2_metrics.exists():
        raise FileNotFoundError("task1/task2 summary.csv is required for comparison")

    def read_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return {
                (row["method"], row["metric"]): row
                for row in csv.DictReader(handle)
            }

    before = read_summary(task1_metrics)
    after = read_summary(task2_metrics)
    rows: list[dict[str, Any]] = []
    for method in ("qrc", "esn"):
        for metric in (
            "rho_mae_all_pairs",
            "rho_mae_changed_pairs",
            "energy_mae_eV",
        ):
            task1_mean = float(before[(method, metric)]["mean"])
            task2_mean = float(after[(method, metric)]["mean"])
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "task1_T6_mean": task1_mean,
                    "task2_T18_mean": task2_mean,
                    "task2_minus_task1": task2_mean - task1_mean,
                    "relative_change_percent": (
                        100.0 * (task2_mean / task1_mean - 1.0)
                    ),
                }
            )

    path = task2_output_dir / "task1_vs_task2.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "TASK2_DIR",
    "Task2Config",
    "build_task2_dataset",
    "compare_with_task1",
    "compute_features_dual_gpu",
    "evaluate_features",
    "expand_hierarchical_injections",
    "load_config",
]
