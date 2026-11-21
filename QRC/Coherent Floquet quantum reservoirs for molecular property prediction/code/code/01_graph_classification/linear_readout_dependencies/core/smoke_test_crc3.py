"""Fast invariant checks for the no-PCA CRC3 z/zz baseline."""

from __future__ import annotations

import numpy as np

from crc3_baseline import (
    compute_crc3_state,
    compute_crc3_z_zz_features,
    crc3_z_zz_dim,
    z_zz_readout,
)


def main() -> None:
    rng = np.random.default_rng(123)
    n, t, q, layers = 7, 5, 4, 3
    tokens = rng.normal(size=(n, t, 3 * q)).astype(np.float32)

    state, _ = compute_crc3_state(tokens, q=q, layers=layers, seed=20)
    features, _ = compute_crc3_z_zz_features(tokens, q=q, layers=layers, seed=20)
    repeated, _ = compute_crc3_z_zz_features(tokens, q=q, layers=layers, seed=20)
    changed, _ = compute_crc3_z_zz_features(tokens, q=q, layers=layers, seed=30)

    assert state.shape == (n, q, 3)
    assert features.shape == (n, crc3_z_zz_dim(q))
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, repeated)
    assert not np.array_equal(features, changed)
    np.testing.assert_allclose(features, z_zz_readout(state), rtol=0, atol=0)

    z = state[:, :, 2]
    np.testing.assert_allclose(features[:, :q], z, rtol=0, atol=0)
    cursor = q
    for i in range(q):
        for j in range(i + 1, q):
            np.testing.assert_allclose(features[:, cursor], z[:, i] * z[:, j], rtol=0, atol=0)
            cursor += 1
    assert cursor == crc3_z_zz_dim(q)

    try:
        compute_crc3_z_zz_features(tokens[:, :, :-1], q=q, layers=layers, seed=20)
    except ValueError:
        pass
    else:
        raise AssertionError("CRC3 accepted an input width different from 3Q")

    print("CRC3 z/zz smoke test: PASS")


if __name__ == "__main__":
    main()
