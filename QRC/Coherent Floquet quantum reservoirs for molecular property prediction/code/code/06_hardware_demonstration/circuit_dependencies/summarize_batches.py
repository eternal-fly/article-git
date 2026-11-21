#!/usr/bin/env python3
"""Aggregate downloaded batches without treating readout seeds as device repeats."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path, nargs="?", default=Path("experiment"))
    parser.add_argument("--output", type=Path, default=Path("batch_level_summary.csv"))
    parser.add_argument(
        "--device-output",
        type=Path,
        default=Path("independent_l3_device_summary.csv"),
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=Path("l1_fold_pair_descriptive_summary.csv"),
    )
    parser.add_argument(
        "--shot-output",
        type=Path,
        default=Path("shot_budget_batch_summary.csv"),
    )
    args = parser.parse_args()
    paths = sorted(args.experiment_root.glob("*/runs/*/linear_readout_metrics_by_seed.csv"))
    if not paths:
        raise FileNotFoundError("No completed linear_readout_metrics_by_seed.csv files found")
    table = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    batch = (
        table.groupby(["batch_id", "variant", "fold_factor"], as_index=False)
        .agg(
            readout_seed_count=("seed", "nunique"),
            ideal_auc=("ideal_test_auc", "mean"),
            frozen_hardware_auc=("frozen_ideal_to_hardware_test_auc", "mean"),
            frozen_delta_auc=("frozen_delta_auc", "mean"),
            task_score_pearson=("task_score_pearson", "mean"),
            task_score_spearman=("task_score_spearman", "mean"),
            hardware_refit_auc=("hardware_refit_test_auc", "mean"),
        )
    )
    batch.to_csv(args.output, index=False)
    print(batch.to_string(index=False))
    print(f"[saved] {args.output}")

    repeated_l3 = batch[
        batch["batch_id"].astype(str).str.match(r"^q6_t6_l3_batch\d+$")
        & batch["variant"].eq("base")
    ].copy()
    if not repeated_l3.empty:
        device = pd.DataFrame(
            [
                {
                    "independent_unit": "hardware batch_id",
                    "nested_seed_policy": "average 10 seeds within batch first",
                    "n_independent_batches": repeated_l3["batch_id"].nunique(),
                    "ideal_auc_mean": repeated_l3["ideal_auc"].mean(),
                    "ideal_auc_between_batch_sd": repeated_l3["ideal_auc"].std(),
                    "hardware_auc_mean": repeated_l3["frozen_hardware_auc"].mean(),
                    "hardware_auc_between_batch_sd": repeated_l3[
                        "frozen_hardware_auc"
                    ].std(),
                    "delta_auc_mean": repeated_l3["frozen_delta_auc"].mean(),
                    "delta_auc_between_batch_sd": repeated_l3[
                        "frozen_delta_auc"
                    ].std(),
                    "delta_auc_min": repeated_l3["frozen_delta_auc"].min(),
                    "delta_auc_max": repeated_l3["frozen_delta_auc"].max(),
                    "task_score_pearson_mean": repeated_l3[
                        "task_score_pearson"
                    ].mean(),
                    "task_score_pearson_between_batch_sd": repeated_l3[
                        "task_score_pearson"
                    ].std(),
                }
            ]
        )
        device.to_csv(args.device_output, index=False)
        print(device.to_string(index=False))
        print(f"[saved] {args.device_output}")

    fold = table[
        table["batch_id"].astype(str).str.startswith("q6_t6_l1_fold_pair_batch")
    ].copy()
    if set(fold["variant"].unique()) >= {"base", "fold3"}:
        columns = [
            "frozen_ideal_to_hardware_test_auc",
            "frozen_delta_auc",
            "task_score_pearson",
            "task_score_spearman",
        ]
        paired = fold.pivot(
            index=["batch_id", "seed"],
            columns="variant",
            values=columns,
        )
        rows = []
        for metric in columns:
            delta = paired[(metric, "fold3")] - paired[(metric, "base")]
            rows.append(
                {
                    "comparison": "fold3-minus-base",
                    "metric": metric,
                    "hardware_batch_count": int(
                        paired.index.get_level_values("batch_id").nunique()
                    ),
                    "nested_readout_seed_count": int(delta.notna().sum()),
                    "nested_seed_difference_mean": float(delta.mean()),
                    "nested_seed_difference_sd": float(delta.std()),
                    "inference_status": (
                        "descriptive within-batch control; not independent-device inference"
                    ),
                }
            )
        fold_summary = pd.DataFrame(rows)
        fold_summary.to_csv(args.fold_output, index=False)
        print(fold_summary.to_string(index=False))
        print(f"[saved] {args.fold_output}")

    shot_paths = sorted(args.experiment_root.glob("*/runs/*/shot_bootstrap_draws.csv"))
    if shot_paths:
        shot = pd.concat([pd.read_csv(path) for path in shot_paths], ignore_index=True)
        shot_batch = (
            shot.groupby(["batch_id", "variant", "shot_budget"], as_index=False)
            .agg(
                nested_draw_count=("delta_auc", "size"),
                frozen_hardware_auc=("frozen_hardware_auc", "mean"),
                frozen_delta_auc=("delta_auc", "mean"),
            )
        )
        shot_batch["protocol_group"] = shot_batch["batch_id"].where(
            ~shot_batch["batch_id"].astype(str).str.match(r"^q6_t6_l3_batch\d+$"),
            "q6_t6_l3_independent_batches",
        )
        shot_summary = (
            shot_batch.groupby(
                ["protocol_group", "variant", "shot_budget"], as_index=False
            )
            .agg(
                independent_batch_count=("batch_id", "nunique"),
                hardware_auc_mean=("frozen_hardware_auc", "mean"),
                hardware_auc_between_batch_sd=("frozen_hardware_auc", "std"),
                delta_auc_mean=("frozen_delta_auc", "mean"),
                delta_auc_between_batch_sd=("frozen_delta_auc", "std"),
            )
        )
        shot_summary.to_csv(args.shot_output, index=False)
        print(shot_summary.to_string(index=False))
        print(f"[saved] {args.shot_output}")


if __name__ == "__main__":
    main()
