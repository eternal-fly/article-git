#!/usr/bin/env python3
"""Build the canonical-v3 balanced-400 hardware bundle and matched ideal features.

The generated token bundle is intentionally independent of the historical
``simulation/experiment`` bundles.  It crops the manuscript mainline
canonical-v3 cache to Q=T=6 and retains only local event fields.  Labels are
used only for balanced sampling and downstream scaffold splits, never for
token construction or traversal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
QRC_ROOT = HERE.parents[1]
LINEAR_ROOT = QRC_ROOT / "FINAL" / "2_linear_readout"
CANONICAL_CACHE = (
    QRC_ROOT
    / "FINAL"
    / "7_mechanism_and_controls"
    / "cache"
    / "BACE_general_canonical_v3_Q24_T30.npz"
)
CANONICAL_META = CANONICAL_CACHE.with_suffix(".json")
OUTPUT_DIR = HERE.parent / "data"
DATASET_OUTPUT = OUTPUT_DIR / "bace_bal400_canonical_v3_q6_t6.npz"
IDEAL_L1_OUTPUT = OUTPUT_DIR / "ideal_bace_bal400_canonical_v3_q6_t6_l1_seed0.npz"
IDEAL_L3_OUTPUT = OUTPUT_DIR / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz"

Q = 6
T = 6
TOKEN_DIM = 3 * Q
BALANCED_N = 400
BALANCED_SEED = 42
SPLIT_SEEDS = tuple(range(0, 100, 10))
PROTOCOL_VERSION = "2026-07-28-mainline-canonical-v3-q6t6l3-cloud-acquisition-v2"
EXPECTED_SCHEMA = "local-bfs-canonical-isomeric-reparse-stable-v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value: np.ndarray) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def _load_splitter():
    sys.path.insert(0, str(LINEAR_ROOT))
    from core.linear_readout import scaffold_split_indices

    return scaffold_split_indices


def _class_counts(labels: np.ndarray, positions: np.ndarray) -> list[int]:
    return np.bincount(labels[positions], minlength=2).astype(int).tolist()


def build_dataset(output_path: Path = DATASET_OUTPUT) -> Path:
    if not CANONICAL_CACHE.is_file() or not CANONICAL_META.is_file():
        raise FileNotFoundError(
            "The authoritative canonical-v3 cache is missing. Expected "
            f"{CANONICAL_CACHE} and {CANONICAL_META}."
        )
    cache_metadata = json.loads(CANONICAL_META.read_text(encoding="utf-8"))
    if cache_metadata.get("schema_version") != EXPECTED_SCHEMA:
        raise AssertionError(
            f"Expected schema {EXPECTED_SCHEMA!r}, got "
            f"{cache_metadata.get('schema_version')!r}."
        )

    with np.load(CANONICAL_CACHE, allow_pickle=False) as package:
        full_tokens = package["tokens"].astype(np.float32, copy=False)
        labels = package["labels"].astype(np.int64, copy=False)
        smiles = package["smiles"].astype(str)
        event_counts = package["event_counts"].astype(np.int16, copy=False)
        source_row_indices = package["source_row_indices"].astype(np.int32, copy=False)
        schema = scalar_text(package["token_schema_version"])

    if schema != EXPECTED_SCHEMA:
        raise AssertionError(f"Cache scalar schema mismatch: {schema!r}")
    if full_tokens.shape[1] < T or full_tokens.shape[2] < TOKEN_DIM:
        raise AssertionError(f"Canonical cache is too small: {full_tokens.shape}")

    rng = np.random.default_rng(BALANCED_SEED)
    selected_parts = []
    for label in (0, 1):
        pool = np.flatnonzero(labels == label)
        if len(pool) < BALANCED_N // 2:
            raise RuntimeError(f"Class {label} has only {len(pool)} rows.")
        selected_parts.append(
            rng.choice(pool, size=BALANCED_N // 2, replace=False).astype(np.int64)
        )
    selected = np.concatenate(selected_parts)
    rng.shuffle(selected)

    selected_labels = labels[selected]
    selected_smiles = smiles[selected]
    selected_tokens = np.ascontiguousarray(
        full_tokens[selected, :T, :TOKEN_DIM], dtype=np.float32
    )
    selected_event_counts = np.minimum(event_counts[selected], T).astype(np.int16)
    token_hashes = np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in selected_tokens
        ],
        dtype="<U64",
    )

    splitter = _load_splitter()
    split_payload: dict[str, dict] = {}
    for seed in SPLIT_SEEDS:
        train, val, test = splitter(
            selected_smiles,
            train_ratio=0.80,
            val_ratio=0.10,
            seed=seed,
        )
        for name, positions in (("train", train), ("val", val), ("test", test)):
            if len(np.unique(selected_labels[positions])) != 2:
                raise RuntimeError(f"seed={seed} {name} is not class-valid")
        split_payload[str(seed)] = {
            "train_idx": selected[train].astype(int).tolist(),
            "val_idx": selected[val].astype(int).tolist(),
            "test_idx": selected[test].astype(int).tolist(),
            "train_class_counts": _class_counts(selected_labels, train),
            "val_class_counts": _class_counts(selected_labels, val),
            "test_class_counts": _class_counts(selected_labels, test),
            "split_type": "80/10/10 stable scaffold; seeded tie shuffle",
            "indices_are_canonical_cache_row_indices": True,
        }

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "BACE",
        "description": "Balanced-400 canonical-v3 local-event bundle for Q6/T6 hardware experiments.",
        "token_schema_version": EXPECTED_SCHEMA,
        "token_prior": "task-independent local chemical-event fields",
        "traversal": "canonical-v3 static canonical serialization",
        "global_descriptors": False,
        "task_guided_traversal": False,
        "labels_used_for_token_construction": False,
        "labels_used_for_balanced_sampling": True,
        "balanced_sampling": {
            "n": BALANCED_N,
            "per_class": BALANCED_N // 2,
            "seed": BALANCED_SEED,
            "without_replacement": True,
        },
        "shape": list(selected_tokens.shape),
        "class_counts": np.bincount(selected_labels, minlength=2).astype(int).tolist(),
        "split_seeds": list(SPLIT_SEEDS),
        "source_cache": str(CANONICAL_CACHE.resolve()),
        "source_cache_sha256": sha256_file(CANONICAL_CACHE),
        "source_metadata_sha256": sha256_file(CANONICAL_META),
        "source_dataset_fingerprint": cache_metadata.get("fingerprint"),
        "padding_policy": "typed PAD events are injected through the full fixed T horizon",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        indices=selected.astype(np.int64),
        source_row_indices=source_row_indices[selected].astype(np.int32),
        labels=selected_labels.astype(np.int64),
        smiles=np.asarray(selected_smiles, dtype=str),
        tokens=selected_tokens,
        event_counts=selected_event_counts,
        token_hashes=token_hashes,
        split_indices_json=np.asarray(json.dumps(split_payload, sort_keys=True)),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        protocol_version=np.asarray(PROTOCOL_VERSION),
        token_schema_version=np.asarray(EXPECTED_SCHEMA),
    )
    validate_dataset(output_path)
    print(f"[saved] {output_path}")
    return output_path


def validate_dataset(path: Path = DATASET_OUTPUT) -> dict:
    with np.load(path, allow_pickle=False) as package:
        required = {
            "indices",
            "source_row_indices",
            "labels",
            "smiles",
            "tokens",
            "event_counts",
            "token_hashes",
            "split_indices_json",
            "metadata_json",
            "protocol_version",
            "token_schema_version",
        }
        missing = required.difference(package.files)
        if missing:
            raise KeyError(f"{path}: missing keys {sorted(missing)}")
        indices = package["indices"].astype(np.int64)
        labels = package["labels"].astype(np.int64)
        tokens = package["tokens"].astype(np.float32)
        token_hashes = package["token_hashes"].astype(str)
        splits = json.loads(scalar_text(package["split_indices_json"]))
        metadata = json.loads(scalar_text(package["metadata_json"]))
        schema = scalar_text(package["token_schema_version"])

    if tokens.shape != (BALANCED_N, T, TOKEN_DIM):
        raise AssertionError(tokens.shape)
    if np.bincount(labels, minlength=2).tolist() != [200, 200]:
        raise AssertionError(np.bincount(labels, minlength=2))
    if len(np.unique(indices)) != BALANCED_N or not np.isfinite(tokens).all():
        raise AssertionError("Duplicate indices or non-finite tokens")
    if schema != EXPECTED_SCHEMA or metadata["token_schema_version"] != EXPECTED_SCHEMA:
        raise AssertionError("Token schema mismatch")
    recalculated = np.asarray(
        [
            hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            for row in tokens
        ]
    )
    if not np.array_equal(recalculated, token_hashes):
        raise AssertionError("Token hashes do not match token bytes")
    expected = set(indices.astype(int).tolist())
    for seed, split in splits.items():
        merged = set(split["train_idx"] + split["val_idx"] + split["test_idx"])
        if merged != expected:
            raise AssertionError(f"seed={seed}: split does not cover selected rows")
        if any(
            set(split[left]).intersection(split[right])
            for left, right in (("train_idx", "val_idx"), ("train_idx", "test_idx"), ("val_idx", "test_idx"))
        ):
            raise AssertionError(f"seed={seed}: split overlap")
    result = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "shape": list(tokens.shape),
        "class_counts": [200, 200],
        "split_count": len(splits),
        "token_schema_version": schema,
    }
    print(json.dumps(result, indent=2))
    return result


def build_ideal_features(dataset_path: Path = DATASET_OUTPUT) -> list[Path]:
    sys.path.insert(0, str(LINEAR_ROOT))
    import torch
    from core.gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu

    with np.load(dataset_path, allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32)
        indices = package["indices"].astype(np.int64)
        labels = package["labels"].astype(np.int64)
        token_hashes = package["token_hashes"].astype(str)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    outputs = []
    for layers, output_path in ((3, IDEAL_L3_OUTPUT),):
        cfg = DTCQRCConfig(
            num_qubits=Q,
            features_per_qubit=3,
            g=0.84,
            W_scale=0.6,
            layers_per_injection=layers,
            use_tanh_before_encoding=True,
            token_dim_policy="strict",
            complex_dtype="complex64",
        )
        features, params = compute_qrc_features_gpu(
            torch.from_numpy(tokens),
            cfg=cfg,
            seed=0,
            batch_size=128,
            device=device,
            progress=True,
        )
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "feature_source": "matched ideal statevector",
            "dataset_file": str(dataset_path.resolve()),
            "dataset_sha256": sha256_file(dataset_path),
            "q": Q,
            "t": T,
            "layers": layers,
            "g": 0.84,
            "W": 0.6,
            "reservoir_seed": 0,
            "J_distribution": "Uniform(-0.9,0.9)",
            "interaction_block": "CX-RZ(J_i)-CX",
            "readout": "Z+all_ZZ",
            "feature_dim": Q * (Q + 1) // 2,
            "simulator": "FINAL/2_linear_readout/core/gpu_qrc_statevector.py",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            indices=indices,
            labels=labels,
            token_hashes=token_hashes,
            features=features.numpy().astype(np.float32),
            J=params["J"].astype(np.float32),
            h=params["h"].astype(np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        with np.load(output_path, allow_pickle=False) as check:
            if check["features"].shape != (BALANCED_N, Q * (Q + 1) // 2):
                raise AssertionError(check["features"].shape)
            if not np.isfinite(check["features"]).all():
                raise AssertionError("Non-finite ideal features")
        print(f"[saved] {output_path}")
        outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dataset", "ideal", "all", "validate"),
        default="all",
    )
    args = parser.parse_args()
    if args.mode in {"dataset", "all"}:
        build_dataset()
    if args.mode in {"ideal", "all"}:
        if not DATASET_OUTPUT.exists():
            build_dataset()
        build_ideal_features()
    if args.mode == "validate":
        validate_dataset()


if __name__ == "__main__":
    main()
