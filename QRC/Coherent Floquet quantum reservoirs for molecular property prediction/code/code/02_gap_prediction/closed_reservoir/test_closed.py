from pathlib import Path

import numpy as np

from closed_core import SOURCE_TASK, SYSTEMS, feature_path, load_config


def test_closed_contract():
    cfg = load_config()
    assert cfg.q == 15 and cfg.output_dim == 120
    assert cfg.seeds == (0, 10, 20, 30, 40, 50)
    assert cfg.g == 0.84 and cfg.W == 0.5 and cfg.layers == 3
    assert cfg.histories == (1, 2, 4, 8, 12, 16, 20)
    assert cfg.payload["qrc"]["reset_probability"] == 0.0
    assert cfg.payload["qrc"]["readout"] == "final"


def test_source_dataset_contract():
    cfg = load_config()
    with np.load(cfg.dataset_path, allow_pickle=False) as package:
        assert package["tokens"].shape == (2376, 20, 45)
        assert package["targets"].shape == (2376, 4)
        assert set(package["split"].tolist()) == {0, 1, 2}


def test_system_names_are_unique():
    assert len(SYSTEMS) == len(set(SYSTEMS)) == 4


def test_w05_feature_metadata_and_qrc_suffix_identity():
    cfg = load_config()
    for system in SYSTEMS:
        for seed in cfg.seeds:
            with np.load(feature_path(system, seed), allow_pickle=False) as package:
                assert np.isclose(float(package["W"]), 0.5)
    for seed in cfg.seeds:
        source = SOURCE_TASK / "features" / f"qrc_post_tm3_suffixes_seed{seed:03d}.npz"
        with np.load(source, allow_pickle=False) as source_package, np.load(
            feature_path("qrc", seed), allow_pickle=False
        ) as closed_package:
            assert np.array_equal(
                closed_package["features"][-1],
                source_package["suffixes"][0, :, : cfg.output_dim],
            )
