"""Core data, encoding, dual-GPU, and evaluation logic for Transition task 1."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TASK_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = TASK_DIR.parents[1]
LINEAR_ROOT = WORKSPACE_ROOT / "BFS-QRC" / "FINAL" / "2_linear_readout"
QRC_CORE_DIR = LINEAR_ROOT / "core"
if str(QRC_CORE_DIR) not in sys.path:
    # Import the two backend modules directly. Importing the `core` package
    # executes its broad __init__ and unnecessarily requires RDKit.
    sys.path.insert(0, str(QRC_CORE_DIR))

ENERGY_KEY = "wB97x_6-31G(d).energy"
PAIR_TYPES = [
    (1, 1),
    (1, 6),
    (1, 7),
    (1, 8),
    (6, 6),
    (6, 7),
    (6, 8),
    (7, 7),
    (7, 8),
    (8, 8),
]
COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}
ROLE_CHANGED = 0
ROLE_ACTIVE_H = 1
ROLE_CONTEXT = 2
ROLE_DEFORMING = 3
ROLE_STABLE = 4
ROLE_NAMES = {
    ROLE_CHANGED: "changed",
    ROLE_ACTIVE_H: "active_h",
    ROLE_CONTEXT: "context",
    ROLE_DEFORMING: "deforming",
    ROLE_STABLE: "stable",
}
ALPHA_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class Task1Config:
    raw_h5: Path
    processed_dataset: Path
    train_count: int = 2488
    val_count: int = 225
    test_count: int = 287
    sample_seed: int = 20260725
    q: int = 8
    prefix_length: int = 6
    target_image_index: int = 6
    g: float = 0.84
    W: float = 0.60
    layers: int = 3
    reservoir_seeds: tuple[int, ...] = (0, 10, 20)
    gpu_ids: tuple[int, int] = (0, 1)

    @property
    def total_count(self) -> int:
        return int(self.train_count + self.val_count + self.test_count)

    @property
    def output_dim(self) -> int:
        return int(self.q * (self.q + 1) // 2)


@dataclass(frozen=True)
class PairEncoding:
    tokens: np.ndarray
    rho: np.ndarray
    pair_slots: np.ndarray
    role_ids: np.ndarray
    changed_mask: np.ndarray
    valid_mask: np.ndarray


def load_config(path: str | Path | None = None) -> Task1Config:
    config_path = TASK_DIR / "config.json" if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    counts = payload["sample_counts"]
    qrc = payload["qrc"]
    return Task1Config(
        raw_h5=(TASK_DIR / payload["raw_h5"]).resolve(),
        processed_dataset=(TASK_DIR / payload["processed_dataset"]).resolve(),
        train_count=int(counts["train"]),
        val_count=int(counts["val"]),
        test_count=int(counts["test"]),
        sample_seed=int(payload["sample_seed"]),
        q=int(payload["num_qubits"]),
        prefix_length=int(payload["prefix_length"]),
        target_image_index=int(payload["target_image_index"]),
        g=float(qrc["g"]),
        W=float(qrc["W"]),
        layers=int(qrc["layers_per_injection"]),
        reservoir_seeds=tuple(int(value) for value in payload["reservoir_seeds"]),
        gpu_ids=tuple(int(value) for value in payload["gpus_per_method"]),
    )


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    differences = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=-1))


def normalized_distance(
    distance: float, atomic_number_i: int, atomic_number_j: int
) -> float:
    denominator = (
        COVALENT_RADII[int(atomic_number_i)]
        + COVALENT_RADII[int(atomic_number_j)]
    )
    return float(distance) / float(denominator)


def endpoint_bonds(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    threshold: float = 1.25,
) -> set[tuple[int, int]]:
    distances = pairwise_distances(positions)
    bonds: set[tuple[int, int]] = set()
    for i, j in combinations(range(len(atomic_numbers)), 2):
        rho = normalized_distance(
            distances[i, j], int(atomic_numbers[i]), int(atomic_numbers[j])
        )
        if rho <= float(threshold):
            bonds.add((i, j))
    return bonds


def select_pair_slots_no_bfs(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select fixed pair slots by chemical role; no graph traversal is used."""
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int16)
    positions = np.asarray(positions, dtype=np.float64)
    reactant_bonds = endpoint_bonds(atomic_numbers, positions[0])
    product_bonds = endpoint_bonds(atomic_numbers, positions[-1])
    changed_pairs = reactant_bonds.symmetric_difference(product_bonds)
    union_bonds = reactant_bonds.union(product_bonds)
    reaction_atoms = {atom for pair in changed_pairs for atom in pair}
    active_hydrogens = {
        atom for atom in reaction_atoms if int(atomic_numbers[atom]) == 1
    }

    endpoint_matrix = np.stack(
        [pairwise_distances(positions[0]), pairwise_distances(positions[-1])]
    )
    records: list[tuple[tuple[int, int], int, float, float, tuple[int, int]]] = []
    for pair in combinations(range(len(atomic_numbers)), 2):
        i, j = pair
        rho_start = normalized_distance(
            endpoint_matrix[0, i, j], atomic_numbers[i], atomic_numbers[j]
        )
        rho_end = normalized_distance(
            endpoint_matrix[1, i, j], atomic_numbers[i], atomic_numbers[j]
        )
        endpoint_change = abs(rho_end - rho_start)
        pair_type = tuple(sorted((int(atomic_numbers[i]), int(atomic_numbers[j]))))
        if pair in changed_pairs:
            role = ROLE_CHANGED
        elif (i in active_hydrogens or j in active_hydrogens):
            role = ROLE_ACTIVE_H
        elif pair in union_bonds and (i in reaction_atoms or j in reaction_atoms):
            role = ROLE_CONTEXT
        elif endpoint_change >= 0.08:
            role = ROLE_DEFORMING
        else:
            role = ROLE_STABLE
        records.append((pair, role, endpoint_change, min(rho_start, rho_end), pair_type))

    records.sort(
        key=lambda record: (
            record[1],
            record[4],
            -record[2],
            record[3],
            record[0][0],
            record[0][1],
        )
    )
    slots = np.full((int(q), 2), -1, dtype=np.int16)
    roles = np.full(int(q), -1, dtype=np.int8)
    changed_mask = np.zeros(int(q), dtype=bool)
    for slot, (pair, role, _, _, _) in enumerate(records[: int(q)]):
        slots[slot] = pair
        roles[slot] = int(role)
        changed_mask[slot] = pair in changed_pairs
    return slots, roles, changed_mask


def pair_role_code(
    atomic_number_i: int,
    atomic_number_j: int,
    role: int,
) -> float:
    pair_type = tuple(sorted((int(atomic_number_i), int(atomic_number_j))))
    pair_index = PAIR_TYPES.index(pair_type)
    combined_index = int(role) * len(PAIR_TYPES) + pair_index
    max_index = (max(ROLE_NAMES) + 1) * len(PAIR_TYPES) - 1
    return float(-0.95 + 1.90 * combined_index / max_index)


def encode_pair_event_path(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    q: int,
) -> PairEncoding:
    """Encode a 10-image path using stable atom-pair identities and no BFS."""
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int16)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1] != len(atomic_numbers):
        raise ValueError("positions must have shape [T,N,3] matching atomic_numbers")
    unsupported = set(map(int, atomic_numbers)) - set(COVALENT_RADII)
    if unsupported:
        raise ValueError(f"unsupported atomic numbers: {sorted(unsupported)}")

    slots, roles, changed_mask = select_pair_slots_no_bfs(
        atomic_numbers, positions, q
    )
    valid_mask = slots[:, 0] >= 0
    rho = np.zeros((len(positions), int(q)), dtype=np.float32)
    tokens = np.zeros((len(positions), int(q), 3), dtype=np.float32)
    for time_index, frame in enumerate(positions):
        distances = pairwise_distances(frame)
        for slot, (i, j) in enumerate(slots):
            if i < 0:
                continue
            value = normalized_distance(
                distances[i, j], atomic_numbers[i], atomic_numbers[j]
            )
            rho[time_index, slot] = value
            increment = (
                0.0
                if time_index == 0
                else value - float(rho[time_index - 1, slot])
            )
            tokens[time_index, slot] = (
                pair_role_code(atomic_numbers[i], atomic_numbers[j], int(roles[slot])),
                np.clip((value - 1.5) / 1.5, -1.0, 1.0),
                np.clip(increment / 0.4, -1.0, 1.0),
            )
    return PairEncoding(
        tokens=tokens,
        rho=rho,
        pair_slots=slots,
        role_ids=roles,
        changed_mask=changed_mask,
        valid_mask=valid_mask,
    )


def list_split_reactions(
    handle: h5py.File, split: str
) -> list[tuple[str, str]]:
    root = handle[split]
    return [
        (str(formula), str(reaction_id))
        for formula in sorted(root.keys())
        for reaction_id in sorted(root[formula].keys())
    ]


def read_final_band(
    group: h5py.Group,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only the last final band, avoiding all earlier saved NEB bands."""
    position_dataset = group["positions"]
    energy_dataset = group[ENERGY_KEY]
    n_saved = int(position_dataset.shape[0])
    if n_saved < 10 or (n_saved != 10 and (n_saved - 10) % 8):
        raise ValueError(f"invalid flattened Transition1x path length: {n_saved}")
    if n_saved == 10:
        positions = np.asarray(position_dataset[:], dtype=np.float32)
        energies = np.asarray(energy_dataset[:], dtype=np.float64)
    else:
        positions = np.concatenate(
            [
                np.asarray(group["reactant"]["positions"][0], dtype=np.float32)[None],
                np.asarray(position_dataset[-8:], dtype=np.float32),
                np.asarray(group["product"]["positions"][0], dtype=np.float32)[None],
            ],
            axis=0,
        )
        energies = np.concatenate(
            [
                np.asarray(
                    [group["reactant"][ENERGY_KEY][0]], dtype=np.float64
                ),
                np.asarray(energy_dataset[-8:], dtype=np.float64),
                np.asarray([group["product"][ENERGY_KEY][0]], dtype=np.float64),
            ]
        )
    atomic_numbers = np.asarray(group["atomic_numbers"][:], dtype=np.int16)
    return atomic_numbers, positions, energies


def preprocess_real_dataset(cfg: Task1Config, force: bool = False) -> Path:
    """Sample official splits and build the reusable no-BFS event dataset."""
    if cfg.processed_dataset.exists() and not force:
        return cfg.processed_dataset
    if not cfg.raw_h5.exists():
        raise FileNotFoundError(cfg.raw_h5)

    requested = {
        "train": int(cfg.train_count),
        "val": int(cfg.val_count),
        "test": int(cfg.test_count),
    }
    records: list[dict[str, Any]] = []
    with h5py.File(cfg.raw_h5, "r") as handle:
        for split_index, split in enumerate(("train", "val", "test")):
            keys = list_split_reactions(handle, split)
            if len(keys) < requested[split]:
                raise RuntimeError(
                    f"split {split} has {len(keys)} reactions, "
                    f"fewer than requested {requested[split]}"
                )
            selection_rng = np.random.default_rng(
                int(cfg.sample_seed) + 1009 * split_index
            )
            candidate_indices = selection_rng.permutation(len(keys))
            accepted_in_split = 0
            for formula, reaction_id in (
                keys[index] for index in candidate_indices
            ):
                group = handle[split][formula][reaction_id]
                try:
                    atomic_numbers, positions, energies = read_final_band(group)
                    encoded = encode_pair_event_path(atomic_numbers, positions, cfg.q)
                except (KeyError, ValueError, FloatingPointError) as error:
                    records.append(
                        {
                            "status": "rejected",
                            "split": split,
                            "formula": formula,
                            "reaction_id": reaction_id,
                            "reason": str(error),
                        }
                    )
                    continue
                target_index = int(cfg.target_image_index)
                records.append(
                    {
                        "status": "accepted",
                        "split": split,
                        "formula": formula,
                        "reaction_id": reaction_id,
                        "num_atoms": len(atomic_numbers),
                        "atomic_numbers": atomic_numbers,
                        "positions": positions,
                        "energies": energies,
                        "tokens": encoded.tokens,
                        "rho": encoded.rho,
                        "pair_slots": encoded.pair_slots,
                        "role_ids": encoded.role_ids,
                        "changed_mask": encoded.changed_mask,
                        "valid_mask": encoded.valid_mask,
                        "target_delta_rho": (
                            encoded.rho[target_index]
                            - encoded.rho[target_index - 1]
                        ),
                        "target_delta_energy": float(
                            energies[target_index] - energies[target_index - 1]
                        ),
                    }
                )
                accepted_in_split += 1
                if accepted_in_split == requested[split]:
                    break
            if accepted_in_split != requested[split]:
                raise RuntimeError(
                    f"split {split}: accepted {accepted_in_split} of "
                    f"{requested[split]} requested reactions after exhausting "
                    f"{len(keys)} candidates"
                )

    accepted = [record for record in records if record["status"] == "accepted"]
    accepted_counts = {
        split: sum(record["split"] == split for record in accepted)
        for split in ("train", "val", "test")
    }
    if accepted_counts != requested:
        raise RuntimeError(
            f"accepted counts differ from request: "
            f"accepted={accepted_counts}, requested={requested}"
        )

    max_atoms = max(int(record["num_atoms"]) for record in accepted)
    num_samples = len(accepted)
    atomic_numbers_padded = np.zeros((num_samples, max_atoms), dtype=np.int16)
    for index, record in enumerate(accepted):
        atomic_numbers_padded[index, : record["num_atoms"]] = record["atomic_numbers"]

    cfg.processed_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.processed_dataset,
        dataset_version=np.asarray("transition1x-task1-pair-events-v1"),
        source_h5=np.asarray(str(cfg.raw_h5)),
        reaction_ids=np.asarray([record["reaction_id"] for record in accepted]),
        formulas=np.asarray([record["formula"] for record in accepted]),
        splits=np.asarray([record["split"] for record in accepted]),
        num_atoms=np.asarray([record["num_atoms"] for record in accepted], dtype=np.int16),
        atomic_numbers=atomic_numbers_padded,
        tokens=np.stack(
            [
                record["tokens"][: cfg.prefix_length].reshape(
                    cfg.prefix_length, 3 * cfg.q
                )
                for record in accepted
            ]
        ).astype(np.float32),
        target_delta_rho=np.stack(
            [record["target_delta_rho"] for record in accepted]
        ).astype(np.float32),
        target_delta_energy=np.asarray(
            [record["target_delta_energy"] for record in accepted], dtype=np.float64
        ),
        pair_slots=np.stack([record["pair_slots"] for record in accepted]),
        role_ids=np.stack([record["role_ids"] for record in accepted]),
        changed_pair_mask=np.stack(
            [record["changed_mask"] for record in accepted]
        ),
        valid_pair_mask=np.stack([record["valid_mask"] for record in accepted]),
        relative_energy_paths=np.stack(
            [record["energies"] - record["energies"][0] for record in accepted]
        ).astype(np.float64),
        q=np.asarray(cfg.q, dtype=np.int64),
        prefix_length=np.asarray(cfg.prefix_length, dtype=np.int64),
        target_image_index=np.asarray(cfg.target_image_index, dtype=np.int64),
        sample_seed=np.asarray(cfg.sample_seed, dtype=np.int64),
    )
    write_preprocess_manifest(records, cfg.processed_dataset.parent / "manifest.csv")
    metadata = {
        "dataset_version": "transition1x-task1-pair-events-v1",
        "source_h5": str(cfg.raw_h5),
        "processed_dataset": str(cfg.processed_dataset),
        "sample_counts": accepted_counts,
        "sample_seed": cfg.sample_seed,
        "q": cfg.q,
        "prefix_length": cfg.prefix_length,
        "target_image_index": cfg.target_image_index,
        "uses_bfs": False,
        "pair_selection_uses": "reactant and product endpoints",
        "targets": ["next_pair_distance_increment", "next_DFT_energy_increment"],
    }
    download_metadata_path = cfg.raw_h5.with_suffix(
        cfg.raw_h5.suffix + ".download.json"
    )
    if download_metadata_path.exists():
        download_metadata = json.loads(
            download_metadata_path.read_text(encoding="utf-8")
        )
        metadata["source_size_bytes"] = download_metadata["size_bytes"]
        metadata["source_sha256"] = download_metadata["sha256"]
    (cfg.processed_dataset.parent / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return cfg.processed_dataset


def write_preprocess_manifest(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "split",
        "formula",
        "reaction_id",
        "num_atoms",
        "num_valid_pairs",
        "num_changed_pairs",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "status": record["status"],
                    "split": record["split"],
                    "formula": record["formula"],
                    "reaction_id": record["reaction_id"],
                    "num_atoms": record.get("num_atoms", ""),
                    "num_valid_pairs": (
                        int(np.sum(record["valid_mask"]))
                        if "valid_mask" in record
                        else ""
                    ),
                    "num_changed_pairs": (
                        int(np.sum(record["changed_mask"]))
                        if "changed_mask" in record
                        else ""
                    ),
                    "reason": record.get("reason", ""),
                }
            )


def _gpu_feature_worker(
    method: str,
    seed: int,
    gpu_id: int,
    indices: np.ndarray,
    dataset_path: str,
    output_path: str,
    cfg_payload: dict[str, Any],
    error_queue: mp.Queue,
) -> None:
    """Compute one method/seed chunk on one GPU and write an indexed part."""
    try:
        import torch

        torch.cuda.set_device(int(gpu_id))
        with np.load(dataset_path, allow_pickle=False) as data:
            tokens = np.ascontiguousarray(data["tokens"][indices], dtype=np.float32)
        q = int(cfg_payload["q"])
        layers = int(cfg_payload["layers"])
        started = time.perf_counter()

        if method == "qrc":
            from gpu_qrc_statevector import (
                DTCQRCConfig,
                compute_qrc_features_gpu,
            )

            qrc_cfg = DTCQRCConfig(
                num_qubits=q,
                features_per_qubit=3,
                g=float(cfg_payload["g"]),
                W_scale=float(cfg_payload["W"]),
                layers_per_injection=layers,
                use_tanh_before_encoding=True,
                token_dim_policy="strict",
                complex_dtype="complex64",
            )
            features, params = compute_qrc_features_gpu(
                torch.from_numpy(tokens),
                qrc_cfg,
                seed=int(seed),
                batch_size=int(cfg_payload["qrc_batch_size"]),
                device=f"cuda:{gpu_id}",
                progress=False,
            )
            result = features.numpy().astype(np.float32)
            parameter_summary = {
                "J": params["J"].tolist(),
                "h": params["h"].tolist(),
            }
        elif method == "esn":
            from esn_baseline import (
                ESN_STANDARD_PROTOCOL,
                compute_esn_standard_features,
            )

            result, params = compute_esn_standard_features(
                tokens,
                q=q,
                layers=layers,
                seed=int(seed),
                device=f"cuda:{gpu_id}",
            )
            parameter_summary = {
                "protocol": ESN_STANDARD_PROTOCOL,
                "spectral_radius": float(params["spectral_radius"]),
                "input_scale": float(params["input_scale"]),
                "leak_rate": float(params["leak_rate"]),
            }
        else:
            raise ValueError(f"unsupported method: {method}")

        elapsed = time.perf_counter() - started
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            indices=np.asarray(indices, dtype=np.int64),
            features=result,
            method=np.asarray(method),
            seed=np.asarray(seed, dtype=np.int64),
            gpu_id=np.asarray(gpu_id, dtype=np.int64),
            elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
            parameter_summary=np.asarray(json.dumps(parameter_summary)),
        )
    except Exception:
        error_queue.put(
            {
                "method": method,
                "seed": int(seed),
                "gpu_id": int(gpu_id),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def compute_features_dual_gpu(
    method: str,
    seed: int,
    cfg: Task1Config,
    feature_dir: Path,
    force: bool = False,
) -> Path:
    """Split one method across both GPUs, then merge indexed feature chunks."""
    method = str(method)
    feature_dir.mkdir(parents=True, exist_ok=True)
    combined_path = feature_dir / f"{method}_seed{int(seed):03d}.npz"
    if combined_path.exists() and not force:
        return combined_path
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        num_samples = int(len(data["tokens"]))
    index_parts = np.array_split(np.arange(num_samples, dtype=np.int64), len(cfg.gpu_ids))
    part_paths = [
        feature_dir
        / f"{method}_seed{int(seed):03d}_gpu{int(gpu_id)}_part.npz"
        for gpu_id in cfg.gpu_ids
    ]
    context = mp.get_context("spawn")
    error_queue: mp.Queue = context.Queue()
    cfg_payload = {
        "q": cfg.q,
        "layers": cfg.layers,
        "g": cfg.g,
        "W": cfg.W,
        "qrc_batch_size": 256 if cfg.q <= 8 else 24,
    }
    processes = []
    for gpu_id, indices, part_path in zip(cfg.gpu_ids, index_parts, part_paths):
        process = context.Process(
            target=_gpu_feature_worker,
            args=(
                method,
                int(seed),
                int(gpu_id),
                indices,
                str(cfg.processed_dataset),
                str(part_path),
                cfg_payload,
                error_queue,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    if errors or any(process.exitcode != 0 for process in processes):
        raise RuntimeError(
            f"dual-GPU {method} seed {seed} failed:\n"
            + "\n".join(error["traceback"] for error in errors)
        )

    features = np.empty((num_samples, cfg.output_dim), dtype=np.float32)
    part_metadata = []
    for part_path in part_paths:
        with np.load(part_path, allow_pickle=False) as part:
            indices = part["indices"]
            values = part["features"]
            if values.shape != (len(indices), cfg.output_dim):
                raise RuntimeError(
                    f"{part_path.name}: invalid feature shape {values.shape}"
                )
            features[indices] = values
            part_metadata.append(
                {
                    "gpu_id": int(part["gpu_id"]),
                    "num_samples": int(len(indices)),
                    "elapsed_seconds": float(part["elapsed_seconds"]),
                    "parameter_summary": json.loads(
                        str(part["parameter_summary"].item())
                    ),
                }
            )
    if not np.isfinite(features).all():
        raise RuntimeError(f"{method} seed {seed}: non-finite features")
    np.savez_compressed(
        combined_path,
        features=features,
        method=np.asarray(method),
        seed=np.asarray(seed, dtype=np.int64),
        q=np.asarray(cfg.q, dtype=np.int64),
        output_dim=np.asarray(cfg.output_dim, dtype=np.int64),
        gpu_parts=np.asarray(json.dumps(part_metadata)),
    )
    return combined_path


def _masked_errors(
    target: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    errors = np.asarray(prediction - target)[mask]
    return (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(errors**2))),
    )


def fit_ridge_with_validation(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    validation_metric,
) -> tuple[np.ndarray, float]:
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    scaler = StandardScaler().fit(features[train])
    train_features = scaler.transform(features[train])
    val_features = scaler.transform(features[val])
    best_alpha = None
    best_score = float("inf")
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=float(alpha)).fit(train_features, target[train])
        score = float(validation_metric(target[val], model.predict(val_features), val))
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    train_val = train | val
    final_scaler = StandardScaler().fit(features[train_val])
    final_model = Ridge(alpha=float(best_alpha)).fit(
        final_scaler.transform(features[train_val]), target[train_val]
    )
    prediction = final_model.predict(final_scaler.transform(features[test]))
    return np.asarray(prediction), float(best_alpha)


def evaluate_features(cfg: Task1Config, feature_dir: Path, output_dir: Path) -> Path:
    """Apply identical validation-selected affine heads to QRC and ESN features."""
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        splits = data["splits"]
        reaction_ids = data["reaction_ids"]
        pair_slots = data["pair_slots"]
        valid_mask = data["valid_pair_mask"]
        changed_mask = data["changed_pair_mask"]
        target_rho = data["target_delta_rho"].astype(np.float64)
        target_energy = data["target_delta_energy"].astype(np.float64)

    test = splits == "test"
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method in ("qrc", "esn"):
        for seed in cfg.reservoir_seeds:
            feature_path = feature_dir / f"{method}_seed{int(seed):03d}.npz"
            with np.load(feature_path, allow_pickle=False) as package:
                features = package["features"].astype(np.float64)

            def distance_val_metric(y_true, y_pred, val_rows):
                mask = valid_mask[val_rows]
                return np.mean(np.abs(y_pred - y_true)[mask])

            rho_prediction, rho_alpha = fit_ridge_with_validation(
                features,
                target_rho,
                splits,
                distance_val_metric,
            )

            def energy_val_metric(y_true, y_pred, _):
                return np.mean(np.abs(y_pred - y_true))

            energy_prediction, energy_alpha = fit_ridge_with_validation(
                features,
                target_energy,
                splits,
                energy_val_metric,
            )
            all_mae, all_rmse = _masked_errors(
                target_rho[test], rho_prediction, valid_mask[test]
            )
            reactive_mask = valid_mask[test] & changed_mask[test]
            changed_mae, changed_rmse = _masked_errors(
                target_rho[test], rho_prediction, reactive_mask
            )
            energy_error = energy_prediction - target_energy[test]
            row = {
                "method": method,
                "seed": int(seed),
                "feature_dim": int(features.shape[1]),
                "rho_alpha": rho_alpha,
                "energy_alpha": energy_alpha,
                "rho_mae_all_pairs": all_mae,
                "rho_rmse_all_pairs": all_rmse,
                "rho_mae_changed_pairs": changed_mae,
                "rho_rmse_changed_pairs": changed_rmse,
                "energy_mae_eV": float(np.mean(np.abs(energy_error))),
                "energy_rmse_eV": float(np.sqrt(np.mean(energy_error**2))),
            }
            rows.append(row)

            for local_index, sample_index in enumerate(np.flatnonzero(test)):
                prediction_rows.append(
                    {
                        "reaction_id": str(reaction_ids[sample_index]),
                        "method": method,
                        "seed": int(seed),
                        "target_delta_energy_eV": float(target_energy[sample_index]),
                        "predicted_delta_energy_eV": float(
                            energy_prediction[local_index]
                        ),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_by_seed.csv"
    _write_dict_rows(metrics_path, rows)
    _write_dict_rows(output_dir / "energy_test_predictions.csv", prediction_rows)

    summary_rows = []
    for method in ("qrc", "esn"):
        method_rows = [row for row in rows if row["method"] == method]
        for metric in (
            "rho_mae_all_pairs",
            "rho_mae_changed_pairs",
            "energy_mae_eV",
        ):
            values = np.asarray([row[metric] for row in method_rows], dtype=float)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)),
                    "num_seeds": int(len(values)),
                }
            )
    _write_dict_rows(output_dir / "summary.csv", summary_rows)

    qrc_by_seed = {row["seed"]: row for row in rows if row["method"] == "qrc"}
    esn_by_seed = {row["seed"]: row for row in rows if row["method"] == "esn"}
    paired_rows = []
    for seed in cfg.reservoir_seeds:
        paired_rows.append(
            {
                "seed": int(seed),
                "qrc_minus_esn_rho_mae_all_pairs": (
                    qrc_by_seed[seed]["rho_mae_all_pairs"]
                    - esn_by_seed[seed]["rho_mae_all_pairs"]
                ),
                "qrc_minus_esn_rho_mae_changed_pairs": (
                    qrc_by_seed[seed]["rho_mae_changed_pairs"]
                    - esn_by_seed[seed]["rho_mae_changed_pairs"]
                ),
                "qrc_minus_esn_energy_mae_eV": (
                    qrc_by_seed[seed]["energy_mae_eV"]
                    - esn_by_seed[seed]["energy_mae_eV"]
                ),
            }
        )
    _write_dict_rows(output_dir / "paired_differences.csv", paired_rows)

    report = {
        "status": "completed",
        "dataset": str(cfg.processed_dataset),
        "num_samples": int(len(splits)),
        "split_counts": {
            split: int(np.sum(splits == split))
            for split in ("train", "val", "test")
        },
        "q": cfg.q,
        "prefix_length": cfg.prefix_length,
        "target_image_index": cfg.target_image_index,
        "feature_dim": cfg.output_dim,
        "reservoir_seeds": list(cfg.reservoir_seeds),
        "gpu_contract": {
            "qrc": list(cfg.gpu_ids),
            "esn": list(cfg.gpu_ids),
            "strategy": "sample-axis split within every method and seed",
        },
        "uses_bfs": False,
        "summary": summary_rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report_path


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
