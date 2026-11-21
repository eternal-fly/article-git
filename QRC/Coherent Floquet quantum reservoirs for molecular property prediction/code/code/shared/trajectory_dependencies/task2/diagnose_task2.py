#!/usr/bin/env python3
"""Compare T=6 and T=18 reservoir geometry on training reactions only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


TASK2_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK2_DIR.parent
SEEDS = (0, 10, 20)
METHODS = ("qrc", "esn")


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(axis=0)) / np.maximum(
        values.std(axis=0), 1e-12
    )


def diagnostics(
    features: np.ndarray,
    target_rho: np.ndarray,
    target_energy: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(features, dtype=np.float64)
    standardized = standardize(values)
    covariance = np.cov(standardized, rowvar=False)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    probabilities = eigenvalues / np.maximum(eigenvalues.sum(), 1e-12)
    nonzero = probabilities > 0
    effective_rank = float(
        np.exp(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    )
    participation_ratio = float(
        eigenvalues.sum() ** 2 / np.maximum(np.sum(eigenvalues**2), 1e-12)
    )
    correlation = np.corrcoef(values, rowvar=False)
    off_diagonal = np.abs(
        correlation[np.triu_indices_from(correlation, k=1)]
    )

    rho_correlation = (
        standardized.T @ standardize(target_rho) / (len(values) - 1)
    )
    energy_correlation = (
        standardized.T
        @ standardize(np.asarray(target_energy)[:, None])
        / (len(values) - 1)
    )
    return {
        "mean_feature_std": float(values.std(axis=0).mean()),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "mean_abs_offdiag_correlation": float(off_diagonal.mean()),
        "mean_abs_feature": float(np.mean(np.abs(values))),
        "saturation_fraction_abs_gt_0_95": float(
            np.mean(np.abs(values) > 0.95)
        ),
        "rho_mean_max_abs_feature_correlation": float(
            np.mean(np.max(np.abs(rho_correlation), axis=0))
        ),
        "rho_cross_correlation_frobenius": float(
            np.linalg.norm(rho_correlation)
        ),
        "energy_max_abs_feature_correlation": float(
            np.max(np.abs(energy_correlation))
        ),
        "energy_cross_correlation_l2": float(
            np.linalg.norm(energy_correlation)
        ),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    task_specs = {
        "task1_T6": (
            TRANSITION_DIR
            / "data/processed/task1/transition1x_task1_q8_t6_n3000.npz",
            TRANSITION_DIR / "task1/features",
        ),
        "task2_T18": (
            TRANSITION_DIR
            / "data/processed/task2/transition1x_task2_q8_t18_n3000.npz",
            TASK2_DIR / "features",
        ),
    }
    rows = []
    for task_name, (dataset_path, feature_dir) in task_specs.items():
        with np.load(dataset_path, allow_pickle=False) as data:
            train = data["splits"] == "train"
            target_rho = data["target_delta_rho"][train]
            target_energy = data["target_delta_energy"][train]
        for method in METHODS:
            for seed in SEEDS:
                with np.load(
                    feature_dir / f"{method}_seed{seed:03d}.npz",
                    allow_pickle=False,
                ) as package:
                    features = package["features"][train]
                rows.append(
                    {
                        "task": task_name,
                        "method": method,
                        "seed": seed,
                        **diagnostics(features, target_rho, target_energy),
                    }
                )

    metric_names = list(rows[0])[3:]
    summary_rows = []
    for task_name in task_specs:
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["task"] == task_name and row["method"] == method
            ]
            summary = {"task": task_name, "method": method}
            for metric in metric_names:
                values = np.asarray([row[metric] for row in selected])
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sample_std"] = float(
                    values.std(ddof=1)
                )
            summary_rows.append(summary)

    output_dir = TASK2_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "feature_diagnostics_by_seed.csv", rows)
    write_rows(output_dir / "feature_diagnostics_summary.csv", summary_rows)
    (output_dir / "feature_diagnostics.json").write_text(
        json.dumps(
            {
                "scope": "training reactions only",
                "tasks": list(task_specs),
                "rows": summary_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_dir / "feature_diagnostics_summary.csv")


if __name__ == "__main__":
    main()
