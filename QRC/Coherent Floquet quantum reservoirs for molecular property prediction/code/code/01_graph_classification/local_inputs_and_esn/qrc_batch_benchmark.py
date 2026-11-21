#!/usr/bin/env python3
"""Benchmark safe Q16 batch sizes and verify feature identity."""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from protocol import G, LAYERS, QRC_BACKEND, W


sys.path.insert(0, str(QRC_BACKEND))
from gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu  # noqa: E402


def main() -> None:
    with np.load(
        "cache/tokens/BACE_canonical_general_Q24_T30.npz", allow_pickle=False
    ) as archive:
        tokens = archive["tokens"][:128, :30, :48].astype(np.float32)
    cfg = DTCQRCConfig(
        num_qubits=16,
        features_per_qubit=3,
        g=G,
        W_scale=W,
        layers_per_injection=LAYERS,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    outputs = {}
    for batch_size in (8, 32):
        started = time.time()
        features, _ = compute_qrc_features_gpu(
            torch.from_numpy(tokens),
            cfg,
            seed=0,
            batch_size=batch_size,
            device="cuda:0",
            progress=False,
        )
        outputs[batch_size] = features.numpy()
        print({"batch_size": batch_size, "seconds": time.time() - started}, flush=True)
    error = float(np.max(np.abs(outputs[8] - outputs[32])))
    print({"max_absolute_error": error}, flush=True)
    if error > 1e-6:
        raise AssertionError(error)


if __name__ == "__main__":
    main()

