#!/usr/bin/env python3
"""Audit Q12-XZ p-scan inputs, feature banks, and result grid."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from task6_core import TASK_DIR, load_config, recover_source_values


def main() -> None:
    cfg = load_config()
    with np.load(cfg.source_processed_dataset, allow_pickle=False) as source:
        original = source["tokens"]
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        tokens = data["tokens"]
        structured = tokens.reshape(3000, 6, 12, 3)
        assert tokens.shape == (3000, 6, 36)
        assert np.count_nonzero(structured[..., 1]) == 0
        assert np.array_equal(recover_source_values(tokens), original)
        assert str(data["injection_axis"].item()) == "XZ"
    feature_checks = []
    for system in ("qrc", "esn"):
        for seed in cfg.reservoir_seeds:
            path = TASK_DIR / "features" / f"{system}_x_suffixes_seed{seed:03d}.npz"
            with np.load(path, allow_pickle=False) as package:
                suffixes = package["suffixes"]
                metadata = json.loads(str(package["gpu_parts"].item()))
                assert str(package["injection_axis"].item()) == "XZ"
                assert np.isclose(float(package["W"]), 0.5)
            assert suffixes.shape == (6, 3000, 78)
            assert np.isfinite(suffixes).all()
            assert {row["gpu_id"] for row in metadata} == {0, 1}
            if system == "qrc":
                assert float(np.max(np.abs(suffixes))) <= 1.0001
                assert metadata[0]["parameter_summary"] == metadata[1]["parameter_summary"]
            feature_checks.append({"system": system, "seed": seed, "shape": list(suffixes.shape), "gpus": [0, 1]})
    output = TASK_DIR / "outputs"
    scan = pd.read_csv(output / "p_scan_by_seed.csv")
    expected = {(system, seed, p) for system in ("qrc", "esn") for seed in cfg.reservoir_seeds for p in cfg.p_grid}
    actual = set(scan[["system", "seed", "p"]].itertuples(index=False, name=None))
    assert len(scan) == 90 and actual == expected
    assert set(scan.feature_dim) == {78}
    selected = pd.read_csv(output / "validation_selected_by_seed.csv")
    assert len(selected) == 6
    assert set(selected.rho_selected_p).issubset(set(cfg.p_grid))
    assert set(selected.energy_selected_p).issubset(set(cfg.p_grid))
    report = {
        "status": "PASS",
        "input": {"shape": [3000, 6, 12, 3], "nonzero_y": 0, "source_recovery_max_error": 0.0},
        "qrc": {"Q": 12, "g": 0.84, "W": 0.5, "L": 3, "feature_dim": 78},
        "feature_banks": feature_checks,
        "p_grid": list(cfg.p_grid), "scan_rows": len(scan), "selected_rows": len(selected),
    }
    path = output / "audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

