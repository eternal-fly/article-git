#!/usr/bin/env python3
"""Small CUDA identity test; writes only Q=3/N=8 smoke caches."""

from __future__ import annotations

import sys

import numpy as np
import torch

from protocol import G, LAYERS, QRC_BACKEND, W
from qrc_features import (
    compute_all_t_suffix_banks,
    load_qrc_bank,
    mix_suffix_bank,
)


if str(QRC_BACKEND) not in sys.path:
    sys.path.insert(0, str(QRC_BACKEND))

from gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible")
    compute_all_t_suffix_banks(
        "BACE",
        "canonical_general",
        3,
        0,
        device="cuda:0",
        batch_size=8,
        force=True,
        max_samples=8,
    )
    bank = load_qrc_bank("BACE", "canonical_general", 3, 12, 0, n_samples=8)
    with np.load(
        "cache/tokens/BACE_canonical_general_Q24_T30.npz", allow_pickle=False
    ) as archive:
        tokens = archive["tokens"][:8, :12, :9].astype(np.float32)
    cfg = DTCQRCConfig(
        num_qubits=3,
        features_per_qubit=3,
        g=G,
        W_scale=W,
        layers_per_injection=LAYERS,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    reference, _ = compute_qrc_features_gpu(
        torch.from_numpy(tokens),
        cfg,
        seed=0,
        batch_size=8,
        device="cuda:0",
        progress=False,
    )
    error = float(np.max(np.abs(bank[0] - reference.numpy())))
    if error != 0.0 or not np.array_equal(mix_suffix_bank(bank, 0.0), bank[0]):
        raise AssertionError(f"closed/open p=0 identity failed: {error}")
    print(
        {
            "cuda_devices": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0),
            "W": W,
            "closed_suffix_identity_max_error": error,
        }
    )


if __name__ == "__main__":
    main()

