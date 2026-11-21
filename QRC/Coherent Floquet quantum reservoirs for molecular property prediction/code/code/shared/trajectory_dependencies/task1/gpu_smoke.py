#!/usr/bin/env python3
"""Small dual-GPU smoke test for both task-1 reservoirs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from task1_core import Task1Config, compute_features_dual_gpu


def main() -> None:
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset = root / "smoke.npz"
        np.savez_compressed(
            dataset,
            tokens=rng.normal(size=(32, 3, 24)).astype(np.float32),
        )
        cfg = Task1Config(
            raw_h5=root / "unused.h5",
            processed_dataset=dataset,
            train_count=16,
            val_count=8,
            test_count=8,
            q=8,
            prefix_length=3,
            target_image_index=3,
            reservoir_seeds=(0,),
            gpu_ids=(0, 1),
        )
        feature_dir = root / "features"
        for method in ("qrc", "esn"):
            path = compute_features_dual_gpu(
                method, seed=0, cfg=cfg, feature_dir=feature_dir, force=True
            )
            with np.load(path, allow_pickle=False) as package:
                shape = tuple(package["features"].shape)
            if shape != (32, 36):
                raise RuntimeError(f"{method}: unexpected smoke shape {shape}")
            print(f"[smoke] {method}: {shape}, GPUs=(0,1)")


if __name__ == "__main__":
    main()

