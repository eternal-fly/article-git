from __future__ import annotations

import numpy as np

from hamiltonian_qrc import compute_suffix_banks, mix_suffix_bank, refresh_suffix_weights


def test_refresh_weights_and_p0_identity() -> None:
    weights = refresh_suffix_weights(4, 0.10)
    assert np.isclose(weights.sum(), 1.0)
    bank = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    assert np.array_equal(mix_suffix_bank(bank, 0.0), bank[0])


def test_direct_hamiltonian_cpu_smoke() -> None:
    rng = np.random.default_rng(7)
    tokens = rng.uniform(-0.5, 0.5, size=(5, 4, 6)).astype(np.float32)
    banks, metadata = compute_suffix_banks(
        tokens, q=2, t_values=(2, 4), seed=0, device="cpu", batch_size=5
    )
    assert banks[2].shape == (2, 5, 3)
    assert banks[4].shape == (4, 5, 3)
    assert metadata["input_preprocessing_in_reservoir"] == "none"
    assert metadata["micro_time_steps"] == 1
    assert metadata["interaction_scale"] == 1.0
    assert np.max(np.abs(banks[4])) <= 1.00001


def test_interaction_scale_zero_changes_parameter_identity() -> None:
    rng = np.random.default_rng(11)
    tokens = rng.uniform(-0.3, 0.3, size=(4, 3, 9)).astype(np.float32)
    full, full_meta = compute_suffix_banks(
        tokens, q=3, t_values=(3,), seed=0, device="cpu", batch_size=4,
        interaction_scale=1.0,
    )
    off, off_meta = compute_suffix_banks(
        tokens, q=3, t_values=(3,), seed=0, device="cpu", batch_size=4,
        interaction_scale=0.0,
    )
    assert full_meta["parameter_sha256"] != off_meta["parameter_sha256"]
    assert off_meta["interaction_scale"] == 0.0
    assert not np.allclose(full[3], off[3])
