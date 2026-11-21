#!/usr/bin/env python3
"""Verify that both task2 methods split 18-step data across both GPUs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from task2_core import compute_features_dual_gpu, load_config


def main() -> None:
    cfg = load_config()
    runtime_cfg = cfg.runtime_config()
    rng = np.random.default_rng(92)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        dataset_path = root / "smoke.npz"
        np.savez_compressed(
            dataset_path,
            tokens=rng.normal(
                size=(32, cfg.total_injections, 3 * cfg.q)
            ).astype(np.float32),
        )
        runtime_cfg = type(runtime_cfg)(
            **{
                **runtime_cfg.__dict__,
                "processed_dataset": dataset_path,
                "reservoir_seeds": (0,),
            }
        )
        for method in ("qrc", "esn"):
            path = compute_features_dual_gpu(
                method, 0, runtime_cfg, root / "features", force=True
            )
            with np.load(path, allow_pickle=False) as package:
                shape = package["features"].shape
                if shape != (32, cfg.output_dim):
                    raise RuntimeError(f"{method}: unexpected shape {shape}")
            print(
                f"[smoke] {method}: {shape}, injections={cfg.total_injections}, "
                f"GPUs={cfg.gpu_ids}",
                flush=True,
            )


if __name__ == "__main__":
    main()
