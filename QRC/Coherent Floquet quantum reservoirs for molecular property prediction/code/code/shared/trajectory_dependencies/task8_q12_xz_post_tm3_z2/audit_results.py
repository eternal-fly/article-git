#!/usr/bin/env python3
"""Audit the post-input TM3 timing and combined decoder comparison."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from task8_core import TASK_DIR, load_config


def main() -> None:
    cfg = load_config()
    checks = []
    for system in ("qrc", "esn"):
        for seed in cfg.task6.reservoir_seeds:
            path = TASK_DIR / "features" / f"{system}_post_tm3_suffixes_seed{seed:03d}.npz"
            with np.load(path, allow_pickle=False) as package:
                suffixes = package["suffixes"]
                error = float(package["t0_final_max_abs_error"])
                metadata = json.loads(str(package["gpu_parts"].item()))
                assert suffixes.shape == (6, 3000, 234)
                assert np.isfinite(suffixes).all()
                assert error <= 2e-5
                assert np.array_equal(package["sample_cycles"], np.asarray([0, 1, 2]))
                assert str(package["observable_family"].item()) == "Z+ZZ"
                assert str(package["input_during_readout"].item()) == "none"
                assert str(package["reset_during_readout"].item()) == "none"
                assert {row["gpu_id"] for row in metadata} == {0, 1}
            checks.append({"system": system, "seed": seed, "shape": [6, 3000, 234], "t0_final_max_abs_error": error})

    output = TASK_DIR / "outputs"
    metrics = pd.read_csv(output / "metrics_by_seed.csv")
    representations = (
        "qrc_final", "qrc_internal_tm3", "qrc_post_tm3",
        "esn_final", "esn_internal_tm3", "esn_post_tm3",
    )
    expected = {
        (representation, decoder, seed)
        for representation in representations
        for decoder in ("linear", "svr")
        for seed in cfg.task6.reservoir_seeds
    }
    actual = set(metrics[["representation", "decoder", "seed"]].itertuples(index=False, name=None))
    assert len(metrics) == 36 and actual == expected
    for _, pair in metrics.groupby(["representation", "seed"]):
        assert len(pair) == 2
        assert pair.rho_selected_p.nunique() == 1
        assert pair.energy_selected_p.nunique() == 1
    old = pd.read_csv(cfg.source_task7_output_dir / "metrics_by_seed.csv")
    old["representation"] = old["representation"].replace({
        "qrc_tm3": "qrc_internal_tm3", "esn_tm3": "esn_internal_tm3",
    })
    inherited = metrics[metrics.readout != "post_tm3"].sort_values(["representation", "decoder", "seed"])
    old = old.sort_values(["representation", "decoder", "seed"])
    for column in ("rho_mae_all_pairs", "rho_mae_changed_pairs", "energy_mae_eV"):
        assert np.array_equal(inherited[column].to_numpy(), old[column].to_numpy())
    report = {
        "status": "PASS",
        "feature_checks": checks,
        "post_input_sampling": {"cycles": [0, 1, 2], "observable": "Z+ZZ", "new_inputs": 0, "readout_resets": 0},
        "metrics_rows": len(metrics),
        "frozen_p_across_decoders": True,
        "task7_reference_rows_unchanged": True,
    }
    path = output / "audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

