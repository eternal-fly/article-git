from __future__ import annotations

import numpy as np

from protocol import P_VALUES, Q_VALUES, SEEDS, T_VALUES, W, esn_widths, readout_dim
from qrc_features import mix_suffix_bank, refresh_suffix_weights


def test_frozen_contract():
    assert W == 0.5
    assert SEEDS == (0, 10, 20, 30, 40, 50)
    assert Q_VALUES == (12, 16)
    assert T_VALUES == (2, 4, 6, 8, 10, 12, 16, 20, 24, 30)
    assert readout_dim(12) == 78
    assert readout_dim(16) == 136
    assert esn_widths(12) == (78, 500, 1000, 2000)
    assert esn_widths(16) == (136, 500, 1000, 2000)


def test_exact_reset_weights():
    for t in T_VALUES:
        for p in P_VALUES:
            weights = refresh_suffix_weights(t, p)
            assert len(weights) == t + 1
            assert np.isclose(weights.sum(), 1.0, atol=1e-12)
            assert np.all(weights >= 0.0)


def test_closed_suffix_identity():
    bank = np.random.default_rng(4).normal(size=(8, 11, 17)).astype(np.float32)
    assert np.array_equal(mix_suffix_bank(bank, 0.0), bank[0])

