import numpy as np

from evaluate_ablation import mix_bace_bank, mix_hopping_bank
from mechanism_features import connected_zz_diagnostics, simulate_sequence_features


def test_reset_mixtures_have_p0_identity() -> None:
    rng = np.random.default_rng(3)
    bace = rng.normal(size=(4, 5, 6)).astype(np.float32)
    hopping = rng.normal(size=(4, 5, 18)).astype(np.float32)
    assert np.array_equal(mix_bace_bank(bace, 0.0), bace[0])
    assert np.array_equal(mix_hopping_bank(hopping, 0.0, "tm3"), hopping[0])


def test_connected_diagnostic_for_factorized_features() -> None:
    rng = np.random.default_rng(4)
    z = rng.uniform(-1, 1, size=(7, 3))
    zz = np.stack([z[:, 0] * z[:, 1], z[:, 0] * z[:, 2], z[:, 1] * z[:, 2]], axis=1)
    rms, maximum = connected_zz_diagnostics(np.concatenate((z, zz), axis=1), 3)
    assert rms < 1e-14
    assert maximum < 1e-14


def test_j0_statevector_remains_product() -> None:
    rng = np.random.default_rng(5)
    tokens = rng.normal(scale=0.2, size=(3, 2, 9)).astype(np.float32)
    features, diagnostics = simulate_sequence_features(
        tokens,
        q=3,
        seed=0,
        j_scale=0.0,
        dephasing_strength=0.0,
        trajectories=1,
        device="cpu",
        batch_size=3,
        angle_scale=1.0,
        tm3=False,
        suffix_start=0,
    )
    assert features.shape == (3, 6)
    assert diagnostics["connected_zz_rms"] < 2e-6
    assert diagnostics["connected_zz_max_abs"] < 5e-6

