#!/usr/bin/env python3
"""Audit TM3 feature identity, frozen-p decoder comparison, and result coverage."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from task7_core import TASK_DIR, load_config


def main() -> None:
    cfg = load_config()
    checks = []
    for system in ("qrc", "esn"):
        for seed in cfg.task6.reservoir_seeds:
            path = TASK_DIR / "features" / f"{system}_tm3_suffixes_seed{seed:03d}.npz"
            with np.load(path, allow_pickle=False) as package:
                suffixes = package["suffixes"]
                error = float(package["final_block_max_abs_error"])
                metadata = json.loads(str(package["gpu_parts"].item()))
                assert suffixes.shape == (6, 3000, 234)
                assert np.isfinite(suffixes).all()
                assert error <= 2e-5
                assert int(package["temporal_nodes"]) == 3
                assert int(package["observables_per_node"]) == 78
                assert str(package["injection_axis"].item()) == "XZ"
                assert np.isclose(float(package["W"]), 0.5)
                assert {row["gpu_id"] for row in metadata} == {0, 1}
            checks.append({"system": system, "seed": seed, "shape": [6, 3000, 234], "final_block_max_abs_error": error})
    output = TASK_DIR / "outputs"
    metrics = pd.read_csv(output / "metrics_by_seed.csv")
    expected = {
        (representation, decoder, seed)
        for representation in ("qrc_final", "qrc_tm3", "esn_final", "esn_tm3")
        for decoder in ("linear", "svr")
        for seed in cfg.task6.reservoir_seeds
    }
    actual = set(metrics[["representation", "decoder", "seed"]].itertuples(index=False, name=None))
    assert len(metrics) == 24 and actual == expected
    for _, pair in metrics.groupby(["representation", "seed"]):
        assert len(pair) == 2
        assert pair.rho_selected_p.nunique() == 1
        assert pair.energy_selected_p.nunique() == 1
    assert set(metrics.rho_selected_p).issubset(set(cfg.task6.p_grid))
    assert set(metrics.energy_selected_p).issubset(set(cfg.task6.p_grid))
    assert np.isfinite(metrics[["rho_mae_all_pairs", "rho_mae_changed_pairs", "energy_mae_eV"]]).all().all()
    summary = pd.read_csv(output / "summary.csv")
    assert len(summary) == 8 and set(summary.num_seeds) == {3}
    report = {
        "status": "PASS",
        "feature_checks": checks,
        "metrics_rows": len(metrics),
        "summary_rows": len(summary),
        "frozen_p_across_decoders": True,
        "seeds": list(cfg.task6.reservoir_seeds),
    }
    path = output / "audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

