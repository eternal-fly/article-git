"""Fast invariant checks for the shared standard ESN baseline."""

from __future__ import annotations

import numpy as np

from esn_baseline import compute_esn_standard_features, esn_standard_dim


def main() -> None:
    rng = np.random.default_rng(17)
    n, t, q = 9, 5, 4
    tokens = rng.normal(size=(n, t, 3 * q)).astype(np.float32)
    first, params = compute_esn_standard_features(tokens, q, 3, 20, device="cpu")
    repeated, _ = compute_esn_standard_features(tokens, q, 3, 20, device="cpu")
    changed, _ = compute_esn_standard_features(tokens, q, 3, 30, device="cpu")
    assert first.shape == (n, esn_standard_dim(q))
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, changed)
    radius = np.max(np.abs(np.linalg.eigvals(params["recurrent"].astype(np.float64))))
    assert abs(float(radius) - 0.90) < 1e-4
    try:
        compute_esn_standard_features(tokens[:, :, :-1], q, 3, 20, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("ESN accepted an input width different from 3Q")
    print("Standard ESN smoke test: PASS")


if __name__ == "__main__":
    main()
