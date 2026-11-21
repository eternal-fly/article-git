"""Fail-closed audit of reused molecular tokens, splits, and injection gates."""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

from protocol import (
    DATASETS,
    HAMILTONIAN_BACKEND,
    MANIFEST_ROOT,
    SEEDS,
    TOKEN_VARIANTS,
    source_split_path,
    source_token_path,
    validate_protocol,
)


if str(HAMILTONIAN_BACKEND) not in sys.path:
    sys.path.insert(0, str(HAMILTONIAN_BACKEND))

from hamiltonian_dtc_qrc import injection_audit  # noqa: E402


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def audit_inputs() -> dict:
    protocol_counts = validate_protocol()
    tokens_summary = []
    dataset_lengths: dict[str, int] = {}
    for dataset in DATASETS:
        reference_labels = None
        reference_smiles = None
        for variant in TOKEN_VARIANTS:
            path = source_token_path(dataset, variant)
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                tokens = archive["tokens"].astype(np.float32, copy=False)
                labels = archive["labels"].astype(np.int64, copy=False)
                smiles = archive["smiles"].astype(str)
                stored_sha = str(archive["tokens_sha256"].item())
            if tokens.ndim != 3 or tokens.shape[1:] != (30, 72):
                raise AssertionError((path, tokens.shape))
            if not np.isfinite(tokens).all() or tokens.min() < -1 or tokens.max() > 1:
                raise AssertionError(f"invalid token range in {path}")
            if array_sha256(tokens) != stored_sha:
                raise AssertionError(f"token hash mismatch in {path}")
            if reference_labels is None:
                reference_labels, reference_smiles = labels.copy(), smiles.copy()
            elif not np.array_equal(labels, reference_labels) or not np.array_equal(
                smiles, reference_smiles
            ):
                raise AssertionError(f"variant rows differ for {dataset}")
            dataset_lengths[dataset] = len(tokens)
            tokens_summary.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "path": str(path.resolve()),
                    "shape": list(tokens.shape),
                    "sha256": stored_sha,
                    "min": float(tokens.min()),
                    "max": float(tokens.max()),
                }
            )

    split_summary = []
    for dataset in DATASETS:
        covered = set(range(dataset_lengths[dataset]))
        for seed in SEEDS:
            path = source_split_path(dataset, seed)
            payload = json.loads(path.read_text(encoding="utf-8"))
            partitions = {
                name: set(int(v) for v in payload[key])
                for name, key in (
                    ("train", "train_indices"),
                    ("validation", "val_indices"),
                    ("test", "test_indices"),
                )
            }
            if any(a & b for i, a in enumerate(partitions.values()) for b in list(partitions.values())[i + 1 :]):
                raise AssertionError(f"overlapping split: {path}")
            if set().union(*partitions.values()) != covered:
                raise AssertionError(f"incomplete split: {path}")
            split_summary.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sizes": {key: len(value) for key, value in partitions.items()},
                }
            )

    example = np.asarray(
        [[0.25, -0.40, 0.70], [-0.35, 0.15, 0.55], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    gate_audit = injection_audit(example)
    if gate_audit["max_unitarity_error"] > 1e-12:
        raise AssertionError(gate_audit)
    return {
        "status": "pass",
        "protocol_counts": protocol_counts,
        "tokens": tokens_summary,
        "splits": split_summary,
        "hamiltonian_gate": gate_audit,
    }


if __name__ == "__main__":
    result = audit_inputs()
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    (MANIFEST_ROOT / "input_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
