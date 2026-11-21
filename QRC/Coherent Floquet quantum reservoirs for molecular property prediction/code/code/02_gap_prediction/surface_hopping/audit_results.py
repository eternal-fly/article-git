#!/usr/bin/env python3
"""Audit data provenance, protocol coverage, selection and completed outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from a01_core import SPLIT_NAMES, TARGET_NAMES, feature_path, load_config, verify_raw_data


def main() -> None:
    cfg = load_config(); verify_raw_data(cfg)
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        assert package["tokens"].shape == (2376, 20, 45)
        split, trajectory = package["split"], package["trajectory_id"]
        groups = [set(trajectory[split == index]) for index in range(3)]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert [len(group) for group in groups] == [178, 59, 60]
    for system in ("qrc", "esn"):
        for seed in cfg.seeds:
            with np.load(feature_path(system, seed), allow_pickle=False) as package:
                bank = package["suffixes"]
                assert bank.shape == (20, 2376, 360) and np.isfinite(bank).all()
                assert int(package["real_inputs_per_step"]) == 30
                assert int(package["repeated_inputs_per_step"]) == 0
    strict = cfg.processed_path.parents[2] / "outputs" / "strict_decoder_only"
    metrics = pd.read_csv(strict / "reservoir_metrics_all_fixed_axes.csv")
    summary = pd.read_csv(strict / "reservoir_summary_all_fixed_axes.csv")
    paired = pd.read_csv(strict / "qrc_minus_esn_paired_all_fixed_axes.csv")
    optima = pd.read_csv(strict / "qrc_p_scan_optima_by_fixed_decoder_readout.csv")
    expected_metrics = (len(cfg.p_grid) + 1) * len(cfg.seeds) * 2 * 2 * len(TARGET_NAMES)
    expected_summary = (len(cfg.p_grid) + 1) * 2 * 2 * len(TARGET_NAMES)
    expected_paired = len(cfg.p_grid) * 2 * 2 * len(TARGET_NAMES)
    assert len(metrics) == expected_metrics
    assert len(summary) == expected_summary and len(paired) == expected_paired
    assert set(metrics.seed) == set(cfg.seeds) == {0, 10, 20, 30, 40, 50}
    assert np.allclose(
        np.sort(metrics.p.unique().astype(float)), np.sort(np.asarray(cfg.p_grid))
    )
    tiny_existing = np.asarray([0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])
    uniform = np.round(np.arange(0.05, 1.001, 0.05), 2)
    formal_grid = np.asarray(sorted(set(tiny_existing) | set(uniform)))
    assert len(cfg.p_grid) == 27 and np.allclose(cfg.p_grid, formal_grid)
    assert not any(np.isclose(metrics.p.to_numpy(float), p).any()
                   for p in (0.16, 0.32, 0.64))
    assert set(metrics[metrics.system == "esn"].p) == {0.0}
    assert set(metrics.readout) == {"final", "tm3"}
    assert set(metrics.decoder) == {"linear", "rbf_svr"}
    assert metrics.decoder_selection_rule.str.contains("physical axes fixed", regex=False).all()
    numeric_metrics = [
        "validation_mae_ev", "mae_ev", "rmse_ev", "r2"
    ]
    assert np.isfinite(metrics[numeric_metrics].to_numpy(float)).all()
    assert set(paired.num_seeds) == {6} and set(paired.num_test_trajectories) == {60}
    assert len(optima) == 2 * 2 * len(TARGET_NAMES)
    assert len(optima[["readout", "decoder", "target"]].drop_duplicates()) == len(optima)
    assert set(optima.p_star_test_grid_argmin).issubset(set(cfg.p_grid))
    assert set(optima.num_p_values_scanned) == {len(cfg.p_grid)}
    assert set(optima.num_seeds) == {6} and set(optima.num_test_trajectories) == {60}
    assert optima.selection_scope.str.contains("descriptive argmin", regex=False).all()
    assert optima.multiplicity_control.str.contains("over all p", regex=False).all()
    for low, center, high in (
        ("pointwise_95ci_low_ev", "mean_qrc_minus_esn_trajectory_mae_ev", "pointwise_95ci_high_ev"),
        ("simultaneous_95band_low_ev", "mean_qrc_minus_esn_trajectory_mae_ev", "simultaneous_95band_high_ev"),
        ("closed_pointwise_95ci_low_ev", "mean_p_star_minus_closed_trajectory_mae_ev", "closed_pointwise_95ci_high_ev"),
        ("closed_simultaneous_95band_low_ev", "mean_p_star_minus_closed_trajectory_mae_ev", "closed_simultaneous_95band_high_ev"),
    ):
        assert np.isfinite(optima[[low, center, high]].to_numpy(float)).all()
        assert (optima[low] <= optima[center]).all() and (optima[center] <= optima[high]).all()
    print(
        "A01 audit passed: official data, leakage-free split, twelve complete "
        "QRC/ESN banks, six seeds, a fixed full-stream ESN, and decoder-only "
        f"validation over all {len(cfg.p_grid)} QRC physical-axis values on the "
        "tiny-p plus uniform-0.05 formal grid."
    )


if __name__ == "__main__":
    main()
