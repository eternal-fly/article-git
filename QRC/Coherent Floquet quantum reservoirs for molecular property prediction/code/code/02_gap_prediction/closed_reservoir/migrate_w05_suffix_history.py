#!/usr/bin/env python3
"""Migrate the strict closed-history protocol from legacy W=0.6 to W=0.5.

QRC history h is read exactly from the main W=0.5 suffix bank at index 20-h.
Classical control features and shard predictions do not depend on W; only their
protocol metadata are migrated. Every replaced artifact is copied to a
deprecated_W0p6 directory before the atomic replacement.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "shnitsel_a01_surface_hopping"
FEATURES = ROOT / "features"
STRICT = ROOT / "outputs" / "strict_decoder_only"
HISTORIES = (1, 2, 4, 8, 12, 16, 20)
SEEDS = (0, 10, 20, 30, 40, 50)
SYSTEMS = ("qrc", "esn_standard", "esn_second_order_recurrent", "fp_cvr")
OUTPUT_DIM = 120
TARGET_W = 0.5
PROTOCOL_VERSION = "shnitsel-a01-q15-closed-W0p5-fixed-history-six-seed-v3"


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    partial = path.with_name(path.stem + ".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _backup(path: Path, relative_root: Path, backup_root: Path) -> None:
    destination = backup_root / path.relative_to(relative_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        return {name: package[name] for name in package.files}


def migrate_features() -> None:
    backup_root = FEATURES / "deprecated_W0p6"
    for system in SYSTEMS:
        for seed in SEEDS:
            path = FEATURES / f"{system}_closed_history_grid_seed{seed:03d}.npz"
            _backup(path, FEATURES, backup_root)
            if system == "qrc":
                source_path = SOURCE / "features" / f"qrc_post_tm3_suffixes_seed{seed:03d}.npz"
                with np.load(source_path, allow_pickle=False) as source:
                    bank = source["suffixes"].astype(np.float32)
                    parameter_summary = str(source["parameter_summary"].item())
                if bank.shape != (20, 2376, 360):
                    raise AssertionError((source_path, bank.shape))
                indices = np.asarray([20 - history for history in HISTORIES], dtype=np.int16)
                grid = bank[indices, :, :OUTPUT_DIM].astype(np.float32)
                arrays = {
                    "histories": np.asarray(HISTORIES, dtype=np.int16),
                    "features": grid,
                    "system": np.asarray("qrc"),
                    "seed": np.int64(seed),
                    "p": np.float32(0.0),
                    "closed_unitary": np.asarray(True),
                    "initialization": np.asarray("|0>^Q once at the start of each history suffix"),
                    "reset_between_frames": np.asarray(False),
                    "readout": np.asarray("final"),
                    "observable_family": np.asarray("Z+ZZ"),
                    "g": np.float32(0.84),
                    "W": np.float32(TARGET_W),
                    "layers": np.int16(3),
                    "elapsed_seconds": np.float64(0.0),
                    "parameter_summary": np.asarray(parameter_summary),
                    "source_suffix_bank": np.asarray(str(source_path.resolve())),
                    "source_suffix_indices": indices,
                    "protocol_version": np.asarray(PROTOCOL_VERSION),
                }
                if not np.array_equal(grid[-1], bank[0, :, :OUTPUT_DIM]):
                    raise AssertionError(f"T20 suffix mismatch: seed {seed}")
            else:
                arrays = _load_arrays(path)
                arrays["W"] = np.float32(TARGET_W)
                arrays["protocol_version"] = np.asarray(PROTOCOL_VERSION)
            _atomic_npz(path, arrays)


def migrate_control_shards() -> None:
    shard_root = STRICT / "shards"
    backup_root = STRICT / "deprecated_W0p6"
    for system in SYSTEMS[1:]:
        for seed in SEEDS:
            for history in HISTORIES:
                stem = shard_root / f"{system}_seed{seed:03d}_T{history:02d}"
                csv_path = stem.with_suffix(".csv")
                npz_path = stem.with_suffix(".npz")
                _backup(csv_path, STRICT, backup_root)
                _backup(npz_path, STRICT, backup_root)
                frame = pd.read_csv(csv_path)
                frame["W"] = TARGET_W
                frame["protocol_version"] = PROTOCOL_VERSION
                _atomic_csv(frame, csv_path)
                arrays = _load_arrays(npz_path)
                arrays["protocol_version"] = np.asarray(PROTOCOL_VERSION)
                _atomic_npz(npz_path, arrays)


def backup_aggregates_and_qrc_shards() -> None:
    backup_root = STRICT / "deprecated_W0p6"
    for path in STRICT.glob("*.csv"):
        _backup(path, STRICT, backup_root)
    for seed in SEEDS:
        for history in HISTORIES:
            stem = STRICT / "shards" / f"qrc_seed{seed:03d}_T{history:02d}"
            for suffix in (".csv", ".npz"):
                _backup(stem.with_suffix(suffix), STRICT, backup_root)


def main() -> None:
    backup_aggregates_and_qrc_shards()
    migrate_features()
    migrate_control_shards()
    manifest = {
        "status": "features-and-control-metadata-migrated",
        "source": "main W=0.5 QRC suffix banks",
        "histories": list(HISTORIES),
        "suffix_indices": [20 - history for history in HISTORIES],
        "seeds": list(SEEDS),
        "W": TARGET_W,
        "protocol_version": PROTOCOL_VERSION,
        "next_step": "force all qrc fixed-history shards, then aggregate",
    }
    path = STRICT / "w05_migration_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
