from __future__ import annotations

import numpy as np
import pytest

from a01_core import (
    PAIR_INDEX, TARGET_NAMES, build_dataset, load_config, reset_suffix_weights,
    verify_raw_data,
)
from evaluate_strict_axes import _crossed_bootstrap_curves, evaluate_shard


def test_official_raw_contract_and_hash():
    cfg = load_config()
    audit = verify_raw_data(cfg)
    assert audit["verified"] and audit["shapes"]["trajid"] == [62031]
    assert audit["atoms"] == ["C", "C", "H", "H", "H", "H"]


def test_processed_trajectory_split_and_targets():
    cfg = load_config(); build_dataset(cfg)
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        assert package["tokens"].shape == (297 * 8, 20, 45)
        assert package["pair_windows"].shape == (297 * 8, 20, 15, 2)
        assert package["targets"].shape == (297 * 8, 4)
        assert tuple(package["target_names"]) == TARGET_NAMES
        assert not bool(package["source_generated"])
        trajectory, split = package["trajectory_id"], package["split"]
        groups = [set(trajectory[split == index]) for index in range(3)]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        unique, counts = np.unique(trajectory, return_counts=True)
        assert len(unique) == 297 and np.all(counts == 8)


def test_q15_xz_mapping_and_readout_dimensions():
    cfg = load_config(); build_dataset(cfg)
    assert cfg.seeds == (0, 10, 20, 30, 40, 50)
    assert PAIR_INDEX.shape == (15, 2)
    assert (cfg.output_dim, cfg.tm3_dim) == (120, 360)
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        tokens = package["tokens"].reshape(-1, 20, 15, 3)
        assert np.count_nonzero(tokens[..., 1]) == 0
        assert np.count_nonzero(tokens[..., 0]) > 0
        assert np.count_nonzero(tokens[..., 2]) > 0


def test_exact_reset_weights_for_complete_p_scan():
    cfg = load_config()
    tiny_existing = np.asarray([0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])
    uniform = np.round(np.arange(0.05, 1.001, 0.05), 2)
    expected = np.asarray(sorted(set(tiny_existing) | set(uniform)))
    assert len(cfg.p_grid) == 27
    np.testing.assert_allclose(cfg.p_grid, expected)
    assert not any(np.isclose(p, [0.16, 0.32, 0.64]).any() for p in cfg.p_grid)
    for p in cfg.p_grid:
        weights = reset_suffix_weights(cfg.history_steps, p)
        assert np.isclose(weights.sum(), 1.0) and np.all(weights >= 0)
    np.testing.assert_array_equal(reset_suffix_weights(20, 0), [1] + [0] * 19)
    np.testing.assert_array_equal(reset_suffix_weights(20, 1), [0] * 19 + [1])


def test_standard_esn_is_not_given_the_qrc_reset_axis():
    with pytest.raises(ValueError, match="standard ESN"):
        evaluate_shard("esn", 0, 0.1, "final")


def test_scan_bootstrap_uses_one_shared_resample_across_p():
    base = np.arange(24, dtype=float).reshape(3, 8)
    curves = np.stack((base, base + 1.25, base - 2.5))
    draws = _crossed_bootstrap_curves(
        curves, np.random.default_rng(1234), draws=137
    )
    assert draws.shape == (137, 3)
    np.testing.assert_allclose(draws[:, 1] - draws[:, 0], 1.25)
    np.testing.assert_allclose(draws[:, 2] - draws[:, 0], -2.5)
