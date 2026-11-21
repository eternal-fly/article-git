#!/usr/bin/env python3
"""Compare the migrated affine head with one immutable manuscript run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from readouts import fit_linear


def main() -> None:
    old_root = Path("../13_confirmatory_molecular_sweep/outputs/BACE/Q12/T12")
    with np.load(old_root / "features/qrc_seed000.npz", allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
    with np.load(
        "cache/tokens/BACE_canonical_general_Q24_T30.npz", allow_pickle=False
    ) as archive:
        labels = archive["labels"].astype(np.int64)
    split = json.loads(Path("cache/splits/BACE/seed000.json").read_text(encoding="utf-8"))
    migrated, _, _ = fit_linear(
        features,
        labels,
        split,
        seed=0,
        device="cuda:0",
        artifact_dir=Path("/tmp/qrc16_readout_identity_gpu"),
    )
    historical = json.loads(
        (old_root / "runs/qrc_seed000_metrics.json").read_text(encoding="utf-8")
    )
    keys = ("best_epoch", "val_auc", "val_loss", "val_acc", "test_auc", "test_loss", "test_acc")
    print(
        {
            key: {
                "migrated": migrated[key],
                "historical": historical[key],
                "absolute_difference": abs(float(migrated[key]) - float(historical[key])),
            }
            for key in keys
        }
    )


if __name__ == "__main__":
    main()

