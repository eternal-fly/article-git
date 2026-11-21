"""Paired six-seed scaffold splits shared by every representation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from protocol import DATASETS, PROTOCOL_VERSION, SEEDS, split_path, token_cache_path


ROOT = Path(__file__).resolve().parent
LINEAR_ROOT = ROOT.parent / "2_linear_readout"
if str(LINEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(LINEAR_ROOT))

from core.linear_readout import scaffold_split_indices  # noqa: E402


def build_splits(dataset: str, *, force: bool = False) -> list[dict]:
    dataset = dataset.upper()
    if dataset not in DATASETS:
        raise ValueError(dataset)
    with np.load(token_cache_path(dataset, "canonical_general"), allow_pickle=False) as archive:
        smiles = archive["smiles"].astype(str)
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    rows = []
    for seed in SEEDS:
        destination = split_path(dataset, seed)
        if destination.exists() and not force:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if (
                payload.get("protocol_version") == PROTOCOL_VERSION
                and payload.get("token_sha256") == token_sha
                and int(payload.get("seed", -1)) == seed
            ):
                rows.append(payload)
                continue
        train, val, test = scaffold_split_indices(smiles, 0.80, 0.10, seed)
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset": dataset,
            "seed": int(seed),
            "split_type": "stable_scaffold_seeded_tie_shuffle",
            "token_sha256": token_sha,
            "train_indices": train.tolist(),
            "val_indices": val.tolist(),
            "test_indices": test.tolist(),
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "class_counts": {
                name: {str(c): int(np.sum(labels[index] == c)) for c in (0, 1)}
                for name, index in (("train", train), ("val", val), ("test", test))
            },
        }
        for name, index in (("validation", val), ("test", test)):
            if len(np.unique(labels[index])) != 2:
                raise RuntimeError(f"{dataset} seed={seed} has one-class {name} split")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(payload)
    return rows


def build_all_splits(*, force: bool = False) -> list[dict]:
    return [row for dataset in DATASETS for row in build_splits(dataset, force=force)]

