#!/usr/bin/env python3
"""Descriptive analysis of the uniform p=0.50,...,1.00 extension.

This script never selects a deployable p. It reports the grid minimum, the
first mean/pointwise-significant QRC advantage, and any rebound toward p=1 for
each already-fixed decoder/readout/target slice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_strict_axes import (
    _crossed_bootstrap,
    _crossed_bootstrap_curves,
    _exact_sign_flip_p,
    _stable_seed,
    _trajectory_errors,
)


ROOT = Path(__file__).resolve().parent
STRICT_ROOT = ROOT / "outputs" / "strict_decoder_only"
HIGH_P_GRID = np.linspace(0.5, 1.0, 11)


def _on_uniform_grid(values: pd.Series) -> np.ndarray:
    source = values.to_numpy(float)
    return np.logical_or.reduce([
        np.isclose(source, p) for p in HIGH_P_GRID
    ])


def _first_or_nan(frame: pd.DataFrame, mask: pd.Series) -> float:
    selected = frame.loc[mask, "p"]
    return float(selected.iloc[0]) if len(selected) else float("nan")


def main() -> None:
    paired = pd.read_csv(
        STRICT_ROOT / "qrc_minus_esn_paired_all_fixed_axes.csv"
    )
    metrics = pd.read_csv(
        STRICT_ROOT / "reservoir_metrics_all_fixed_axes.csv"
    )
    paired = paired.loc[_on_uniform_grid(paired.p)].copy()
    qrc = metrics[(metrics.system == "qrc") & _on_uniform_grid(metrics.p)].copy()
    esn = metrics[(metrics.system == "esn") & np.isclose(metrics.p, 0.0)].copy()
    keys = ["seed", "readout", "decoder", "target"]
    seed_effects = qrc.merge(
        esn[keys + ["mae_ev"]].rename(columns={"mae_ev": "esn_mae_ev"}),
        on=keys,
        validate="many_to_one",
    )
    seed_effects["qrc_minus_esn_mae_ev"] = (
        seed_effects.mae_ev - seed_effects.esn_mae_ev
    )
    seed_effects = seed_effects.sort_values(keys + ["p"]).reset_index(drop=True)

    rows = []
    group_keys = ["decoder", "readout", "target"]
    for identity, curve in paired.groupby(group_keys, sort=True):
        curve = curve.sort_values("p").reset_index(drop=True)
        if len(curve) != len(HIGH_P_GRID) or not np.allclose(
            curve.p, HIGH_P_GRID
        ):
            raise RuntimeError((identity, curve.p.tolist()))
        effect = curve.mean_qrc_minus_esn_trajectory_mae_ev.to_numpy(float)
        best_index = int(np.argmin(effect))
        best = curve.iloc[best_index]
        slice_seed = seed_effects[
            (seed_effects.decoder == identity[0])
            & (seed_effects.readout == identity[1])
            & (seed_effects.target == identity[2])
        ]
        per_seed_best = (
            slice_seed.sort_values(["seed", "qrc_minus_esn_mae_ev", "p"])
            .groupby("seed", sort=True).head(1)
        )
        counts = {
            f"{float(p):.2f}": int(count)
            for p, count in per_seed_best.p.value_counts().sort_index().items()
        }
        best_seed = slice_seed[np.isclose(slice_seed.p, float(best.p))]
        rebound = float(effect[-1] - effect[best_index])

        rebound_curves = []
        reference_ids = None
        for p in HIGH_P_GRID:
            per_seed = []
            for seed in sorted(slice_seed.seed.unique()):
                ids, p_error = _trajectory_errors(
                    "qrc", int(seed), float(p), identity[1], identity[0], identity[2]
                )
                p1_ids, p1_error = _trajectory_errors(
                    "qrc", int(seed), 1.0, identity[1], identity[0], identity[2]
                )
                if not np.array_equal(ids, p1_ids):
                    raise RuntimeError("high-p/p=1 trajectory mismatch")
                if reference_ids is None:
                    reference_ids = ids
                elif not np.array_equal(reference_ids, ids):
                    raise RuntimeError("seed trajectory mismatch")
                per_seed.append(p1_error - p_error)
            rebound_curves.append(np.stack(per_seed))
        rebound_curves = np.stack(rebound_curves)
        selected_rebound = rebound_curves[best_index]
        rebound_boot = _crossed_bootstrap(
            selected_rebound,
            np.random.default_rng(_stable_seed("dense-high-p-rebound", *identity)),
            20_000,
        )
        rebound_boot_curves = _crossed_bootstrap_curves(
            rebound_curves,
            np.random.default_rng(
                _stable_seed("dense-high-p-rebound-simultaneous", *identity)
            ),
            20_000,
        )
        rebound_curve_mean = rebound_curves.mean(axis=(1, 2))
        rebound_critical = float(np.quantile(
            np.max(
                np.abs(rebound_boot_curves - rebound_curve_mean[None, :]), axis=1
            ),
            0.95,
        ))
        rebound_seed_means = selected_rebound.mean(axis=1)
        rows.append({
            "decoder": identity[0],
            "readout": identity[1],
            "target": identity[2],
            "descriptive_best_p_high_uniform_grid": float(best.p),
            "best_mean_qrc_minus_esn_trajectory_mae_ev": float(effect[best_index]),
            "best_pointwise_95ci_low_ev": float(best.crossed_bootstrap_95ci_low_ev),
            "best_pointwise_95ci_high_ev": float(best.crossed_bootstrap_95ci_high_ev),
            "first_mean_below_zero_p": _first_or_nan(curve, curve.mean_qrc_minus_esn_trajectory_mae_ev < 0),
            "first_pointwise_95ci_below_zero_p": _first_or_nan(curve, curve.crossed_bootstrap_95ci_high_ev < 0),
            "effect_at_p_0p50_ev": float(effect[0]),
            "effect_at_p_1p00_ev": float(effect[-1]),
            "p1_rebound_from_grid_minimum_ev": rebound,
            "p1_rebound_pointwise_95ci_low_ev": float(np.quantile(rebound_boot, 0.025)),
            "p1_rebound_pointwise_95ci_high_ev": float(np.quantile(rebound_boot, 0.975)),
            "p1_rebound_high_p_scanwide_95band_low_ev": float(rebound - rebound_critical),
            "p1_rebound_high_p_scanwide_95band_high_ev": float(rebound + rebound_critical),
            "p1_rebound_scanwide_critical_ev": rebound_critical,
            "p1_worse_than_grid_minimum_seed_fraction": float(
                np.mean(rebound_seed_means > 0)
            ),
            "p1_vs_grid_minimum_exact_two_sided_seed_sign_flip_p": (
                _exact_sign_flip_p(rebound_seed_means)
            ),
            "grid_minimum_is_interior": bool(0 < best_index < len(curve) - 1),
            "p1_is_worse_than_grid_minimum": bool(rebound > 0),
            "qrc_better_seed_fraction_at_grid_minimum": float(
                np.mean(best_seed.qrc_minus_esn_mae_ev < 0)
            ),
            "per_seed_high_p_best_counts_json": json.dumps(counts, sort_keys=True),
            "num_uniform_p_values": len(HIGH_P_GRID),
            "num_seeds": int(best.num_seeds),
            "num_test_trajectories": int(best.num_test_trajectories),
            "scope": "descriptive complete high-p grid; never validation/deployment selection",
        })
    summary = pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)
    summary.to_csv(STRICT_ROOT / "dense_high_p_turning_summary.csv", index=False)
    seed_effects.to_csv(STRICT_ROOT / "dense_high_p_seed_effects.csv", index=False)
    source_root = ROOT / "figures" / "strict_six_seed" / "source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(source_root / "hopping_dense_high_p_turning_summary.csv", index=False)
    seed_effects.to_csv(source_root / "hopping_dense_high_p_seed_effects.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
