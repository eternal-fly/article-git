#!/usr/bin/env python3
"""Audit completed Task-5 caches and evaluation artifacts without retraining."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from task5_core import METHODS, TASK_DIR, load_config, mix_suffix_bank


def main() -> None:
    cfg = load_config()
    feature_dir, output_dir = TASK_DIR / "features", TASK_DIR / "outputs"
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        tokens = data["tokens"].reshape(cfg.total_count, cfg.total_injections, cfg.q, 3)
        assert tokens.shape == (3000, 18, 8, 3)
        assert np.count_nonzero(tokens[..., 1:]) == 0
        assert dict(zip(*np.unique(data["splits"], return_counts=True))) == {
            "test": 287, "train": 2488, "val": 225,
        }
    feature_checks = []
    for system in ("qrc", "esn"):
        for seed in cfg.reservoir_seeds:
            path = feature_dir / f"{system}_x_suffixes_seed{seed:03d}.npz"
            with np.load(path, allow_pickle=False) as package:
                suffixes = package["suffixes"]
                metadata = json.loads(str(package["gpu_parts"].item()))
                stored_w = float(package["W"])
            assert suffixes.shape == (6, 3000, 36)
            assert np.isfinite(suffixes).all()
            assert np.isclose(stored_w, cfg.W)
            assert {entry["gpu_id"] for entry in metadata} == {0, 1}
            assert sum(entry["num_samples"] for entry in metadata) == 3000
            np.testing.assert_array_equal(mix_suffix_bank(suffixes, 0.0), suffixes[0])
            if system == "qrc":
                assert float(np.max(np.abs(suffixes))) <= 1.0001
                assert metadata[0]["parameter_summary"] == metadata[1]["parameter_summary"]
            feature_checks.append({
                "system": system, "seed": seed, "shape": list(suffixes.shape),
                "gpus": [0, 1], "finite": True,
            })
    metrics = pd.read_csv(output_dir / "metrics_by_seed.csv")
    assert len(metrics) == len(METHODS) * len(cfg.reservoir_seeds)
    assert set(metrics.method) == set(METHODS)
    assert set(metrics.rho_selected_p).issubset(set(cfg.p_grid))
    assert set(metrics.energy_selected_p).issubset(set(cfg.p_grid))
    closed = metrics[metrics.method.str.endswith("closed")]
    assert set(closed.rho_selected_p) == {0.0}
    assert set(closed.energy_selected_p) == {0.0}
    bootstrap = pd.read_csv(output_dir / "paired_bootstrap.csv")
    assert len(bootstrap) == 18
    assert set(bootstrap.bootstrap_unit) == {"reaction", "formula_cluster"}
    assert len(list((output_dir / "predictions").glob("*.npz"))) == 12
    report = {
        "status": "PASS",
        "qrc_parameters": {"g": cfg.g, "W": cfg.W, "layers": cfg.layers},
        "x_only": {"shape": [3000, 18, 8, 3], "nonzero_yz": 0},
        "features": feature_checks,
        "metrics_rows": len(metrics),
        "prediction_archives": 12,
        "bootstrap_rows": len(bootstrap),
    }
    path = output_dir / "audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
