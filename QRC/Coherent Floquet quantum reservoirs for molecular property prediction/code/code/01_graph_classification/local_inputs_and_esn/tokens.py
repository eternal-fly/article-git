"""Source-generated canonical-v3 local and global-prior molecular tokens."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from protocol import (
    DATASETS,
    GLOBAL_TOKEN_SCHEMA_VERSION,
    MAX_T,
    PROTOCOL_VERSION,
    TOKEN_SCHEMA_VERSION,
    TOKEN_VARIANTS,
    data_root,
    token_cache_path,
    token_metadata_path,
)


ROOT = Path(__file__).resolve().parent
FINAL_ROOT = ROOT.parent
for path in (FINAL_ROOT / "2_linear_readout", FINAL_ROOT / "5_graph_stream_validity"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.general_tokens import (  # noqa: E402
    TASK_SLOT_INDICES,
    task_feature_names,
    tokenize_mol,
)
from validity_core.canonical_tokens import canonical_reparse  # noqa: E402


def _dataset_spec(dataset: str) -> tuple[Path, str, str]:
    root = data_root(dataset)
    if dataset == "BACE":
        return root / "bace" / "raw" / "bace.csv", "mol", "Class"
    if dataset == "BBBP":
        return root / "bbbp" / "raw" / "BBBP.csv", "smiles", "p_np"
    raise ValueError(dataset)


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _dataset_fingerprint(smiles: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    for smi, label in zip(smiles.astype(str), labels.astype(np.int64)):
        digest.update(smi.encode("utf-8"))
        digest.update(bytes([int(label)]))
    return digest.hexdigest()


def _atomic_write_npz(path: Path, **payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, **payload)
    partial.replace(path)


def build_token_cache(dataset: str, variant: str, *, force: bool = False) -> tuple[Path, dict]:
    dataset = dataset.upper()
    if dataset not in DATASETS or variant not in TOKEN_VARIANTS:
        raise ValueError((dataset, variant))
    cache_path = token_cache_path(dataset, variant)
    metadata_path = token_metadata_path(dataset, variant)
    expected_schema = (
        TOKEN_SCHEMA_VERSION if variant == "canonical_general" else GLOBAL_TOKEN_SCHEMA_VERSION
    )
    if cache_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol_version") == PROTOCOL_VERSION
            and metadata.get("schema_version") == expected_schema
            and metadata.get("variant") == variant
        ):
            with np.load(cache_path, allow_pickle=False) as archive:
                if _sha256_array(archive["tokens"]) == metadata.get("tokens_sha256"):
                    return cache_path, metadata

    csv_path, smiles_column, label_column = _dataset_spec(dataset)
    frame = pd.read_csv(csv_path)
    task_aware = variant == "canonical_global_prior"
    token_rows: list[np.ndarray] = []
    labels: list[int] = []
    smiles: list[str] = []
    event_counts: list[int] = []
    source_indices: list[int] = []
    invalid_smiles = 0
    invalid_labels = 0
    for source_index, row in frame.iterrows():
        try:
            label = int(float(row[label_column]))
        except (TypeError, ValueError):
            invalid_labels += 1
            continue
        if label not in (0, 1):
            invalid_labels += 1
            continue
        try:
            mol, canonical = canonical_reparse(str(row[smiles_column]))
        except ValueError:
            invalid_smiles += 1
            continue
        encoded, count = tokenize_mol(
            mol, dataset, task_aware=task_aware, max_events=MAX_T
        )
        token_rows.append(encoded)
        labels.append(label)
        smiles.append(canonical)
        event_counts.append(count)
        source_indices.append(int(source_index))
    if not token_rows:
        raise RuntimeError(f"No valid rows in {csv_path}")

    tokens = np.stack(token_rows).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    canonical_smiles = np.asarray(smiles)
    counts = np.asarray(event_counts, dtype=np.int16)
    task_slots = np.asarray(TASK_SLOT_INDICES, dtype=np.int16)
    real_event_mask = np.arange(MAX_T)[None, :] < counts[:, None]
    padding_mask = ~real_event_mask
    if variant == "canonical_general" and np.any(tokens[:, :, task_slots] != 0.0):
        raise AssertionError("general task slots are not all zero")
    if np.any(tokens[:, :, task_slots][padding_mask] != 0.0):
        raise AssertionError("padding task slots are not all zero")
    if task_aware:
        real_task = tokens[:, :, task_slots][real_event_mask]
        if len(real_task) == 0 or np.all(real_task == 0.0):
            raise AssertionError("global-prior task slots contain no signal")
        # The fixed molecule-level panel must be repeated exactly on all real events.
        for sample_index, count in enumerate(counts):
            sample_task_slots = tokens[sample_index, : int(count)][:, task_slots]
            if count and not np.array_equal(
                sample_task_slots,
                np.repeat(sample_task_slots[:1], int(count), axis=0),
            ):
                raise AssertionError(f"global prior is not repeated for sample {sample_index}")

    tokens_sha = _sha256_array(tokens)
    labels_sha = _sha256_array(y)
    smiles_sha = hashlib.sha256("\n".join(smiles).encode("utf-8")).hexdigest()
    _atomic_write_npz(
        cache_path,
        tokens=tokens,
        labels=y,
        smiles=canonical_smiles,
        event_counts=counts,
        source_row_indices=np.asarray(source_indices, dtype=np.int32),
        token_schema_version=np.asarray(expected_schema),
        token_variant=np.asarray(variant),
        tokens_sha256=np.asarray(tokens_sha),
    )
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "variant": variant,
        "schema_version": expected_schema,
        "traversal": "canonical isomeric SMILES -> reparse -> root 0 -> ascending-index BFS",
        "source_generator": str(Path(__file__).resolve()),
        "source_csv": str(csv_path.resolve()),
        "source_rows": int(len(frame)),
        "valid_labelled_rows": int(len(y)),
        "invalid_smiles_rows": int(invalid_smiles),
        "invalid_label_rows": int(invalid_labels),
        "shape": list(tokens.shape),
        "class_counts": {str(c): int(np.sum(y == c)) for c in (0, 1)},
        "dataset_fingerprint": _dataset_fingerprint(canonical_smiles, y),
        "tokens_sha256": tokens_sha,
        "labels_sha256": labels_sha,
        "smiles_sha256": smiles_sha,
        "task_slot_indices_zero_based": task_slots.tolist(),
        "task_feature_names": list(task_feature_names(dataset)),
        "accessible_task_feature_names_by_q": {
            str(q): [
                name
                for slot, name in zip(task_slots, task_feature_names(dataset))
                if int(slot) < 3 * q
            ]
            for q in (12, 16)
        },
        "task_slot_policy": (
            "all zero"
            if variant == "canonical_general"
            else "fixed label-free whole-molecule descriptors repeated on every real event; padding zero"
        ),
        "label_used_by_tokenizer": False,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return cache_path, metadata


def build_all_token_caches(*, force: bool = False) -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for variant in TOKEN_VARIANTS:
            path, metadata = build_token_cache(dataset, variant, force=force)
            rows.append({"path": str(path.resolve()), **metadata})
    return rows
