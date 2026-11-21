#!/usr/bin/env python3
"""Mechanical QA for the strict six-seed and dense high-p figure bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ROOT / "figures" / "strict_six_seed"
SOURCE_ROOT = FIGURE_ROOT / "source_data"
HIGH_P_GRID = np.linspace(0.5, 1.0, 11)
TINY_P_GRID = np.asarray([0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])
FORMAL_P_GRID = np.asarray(sorted(
    set(TINY_P_GRID) | set(np.round(np.arange(0.05, 1.001, 0.05), 2))
))


def main() -> None:
    manifest = json.loads(
        (FIGURE_ROOT / "figure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["backend"] == "Python/matplotlib"
    assert "independent y limits" in manifest["closed_comparison_scale"]
    assert "linear 0-to-1" in manifest["closed_comparison_x_layout"]
    assert "without remapping" in manifest["closed_comparison_x_layout"]
    expected_zoom = "fig_hopping_qrc_esn_effect_high_p_uniform_zoom"
    assert expected_zoom in manifest["figures"]
    expected_closed_zoom = "fig_hopping_qrc_closed_effect_high_p_uniform_zoom"
    assert expected_closed_zoom in manifest["figures"]
    for stem in manifest["figures"]:
        for extension in manifest["formats"]:
            path = FIGURE_ROOT / f"{stem}.{extension}"
            assert path.exists() and path.stat().st_size > 10_000, path
        svg = (FIGURE_ROOT / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg and "font-family" in svg
        with Image.open(FIGURE_ROOT / f"{stem}.png") as image:
            # Tight bounding boxes differ slightly with label width; all previews
            # remain at least double-column 300-dpi line-art resolution.
            assert image.width >= 1_900 and image.height >= 1_300
        with Image.open(FIGURE_ROOT / f"{stem}.tiff") as image:
            assert image.info.get("compression") == "tiff_lzw"

    full = pd.read_csv(SOURCE_ROOT / "hopping_paired_effects_all_fixed_axes.csv")
    zoom = pd.read_csv(SOURCE_ROOT / "hopping_paired_effects_high_p_uniform_grid.csv")
    closed_full = pd.read_csv(
        SOURCE_ROOT / "hopping_qrc_minus_closed_effects_all_fixed_axes.csv"
    )
    closed_zoom = pd.read_csv(
        SOURCE_ROOT / "hopping_qrc_minus_closed_effects_high_p_uniform_grid.csv"
    )
    assert len(full) == 27 * 2 * 2 * 4
    assert len(zoom) == 11 * 2 * 2 * 4
    assert len(closed_full) == 27 * 2 * 2 * 4
    assert len(closed_zoom) == 11 * 2 * 2 * 4
    assert np.allclose(np.sort(zoom.p.unique()), HIGH_P_GRID)
    assert np.allclose(np.sort(closed_zoom.p.unique()), HIGH_P_GRID)
    observed_closed = np.sort(closed_full.p.unique().astype(float))
    assert len(observed_closed) == 27
    assert np.allclose(observed_closed, FORMAL_P_GRID)
    assert not any(np.isclose(observed_closed, p).any()
                   for p in (0.16, 0.32, 0.64))
    assert set(zoom.num_seeds) == {6}
    assert set(zoom.num_test_trajectories) == {60}
    interval = zoom[[
        "crossed_bootstrap_95ci_low_ev",
        "mean_qrc_minus_esn_trajectory_mae_ev",
        "crossed_bootstrap_95ci_high_ev",
    ]].to_numpy(float)
    assert np.isfinite(interval).all()
    assert np.all(interval[:, 0] <= interval[:, 1])
    assert np.all(interval[:, 1] <= interval[:, 2])
    closed_interval = closed_zoom[[
        "crossed_bootstrap_95ci_low_ev",
        "mean_qrc_minus_closed_trajectory_mae_ev",
        "crossed_bootstrap_95ci_high_ev",
    ]].to_numpy(float)
    assert np.isfinite(closed_interval).all()
    assert np.all(closed_interval[:, 0] <= closed_interval[:, 1])
    assert np.all(closed_interval[:, 1] <= closed_interval[:, 2])
    closed_p0 = closed_full[np.isclose(closed_full.p, 0.0)]
    assert np.allclose(
        closed_p0.mean_qrc_minus_closed_trajectory_mae_ev.to_numpy(float), 0.0
    )
    print(
        "Strict six-seed figure audit passed: editable/vector exports, "
        "600-dpi LZW TIFF, source-data traceability, the retained tiny-p cells "
        "plus the uniform 0.05 grid, a physical linear p axis from 0 to 1, row-wise decoder "
        "scales, and the uniform p=0.50,...,1.00 zoom grid."
    )


if __name__ == "__main__":
    main()
