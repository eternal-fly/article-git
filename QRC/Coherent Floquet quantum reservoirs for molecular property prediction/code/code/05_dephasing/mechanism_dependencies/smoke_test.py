#!/usr/bin/env python3
"""CPU smoke test; it does not read datasets or start a publication scan."""

import numpy as np

from mechanism_study import (
    CRC3_Z_ZZ_PROTOCOL,
    StudyPoint,
    compute_crc3_z_zz_features,
    compute_features,
    effective_rank,
    fixed_information_tokens,
)


def main() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(5, 3, 12)).astype(np.float32)
    x = fixed_information_tokens(source, input_qubits=2, total_qubits=3, events=3)
    assert x.shape == (5, 3, 9)
    assert np.all(x[:, :, 6:] == 0)
    crc3, _ = compute_crc3_z_zz_features(x, q=3, layers=3, seed=0)
    assert crc3.shape == (5, 6) and np.isfinite(crc3).all()
    assert CRC3_Z_ZZ_PROTOCOL.startswith("crc3_z_zz_matched")
    for variant in ("full", "J0", "no_disorder", "event_dephase", "mean_field", "clifford_reservoir_proxy"):
        point = StudyPoint("BACE", "smoke", 3, 2, 3, variant=variant)
        features, _ = compute_features(x, point, seed=0, device="cpu", batch_size=2, dephase_trajectories=2)
        assert features.shape == (5, 6), (variant, features.shape)
        assert np.isfinite(features).all()
        assert effective_rank(features) >= 0
    print("mechanism smoke: PASS")


if __name__ == "__main__":
    main()
