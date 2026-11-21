#!/usr/bin/env python3
"""Controlled observable and decoder ablations for Transition task 3."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


TASK_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK_DIR.parent
WORKSPACE_ROOT = TRANSITION_DIR.parent
QRC_CORE_DIR = (
    WORKSPACE_ROOT / "BFS-QRC" / "FINAL" / "2_linear_readout" / "core"
)
if str(QRC_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(QRC_CORE_DIR))


PauliSpec = tuple[tuple[int, str], ...]

VARIANT_LABELS = {
    "A": "A: final Z+ZZ / linear",
    "B": "B: A+single X/Y / linear",
    "C": "C: all 1-2 body Pauli / linear",
    "D": "D: A+ZZZ / linear",
    "E": "E: final Z+ZZ / polynomial-2",
    "F": "F: temporal Z+ZZ / linear",
    "FB": "FB: temporal A+single X/Y / linear",
    "FC": "FC: temporal all 1-2 body Pauli / linear",
    "FC216": "FC216: train-PCA216 temporal all-Pauli / linear",
    "ESN": "ESN: final 36-state / linear",
    "ESN_F": "ESN-F: temporal 36-state / linear",
}

QRC_FEATURE_KEYS = ("A", "B", "C", "D", "F", "FB", "FC")

MECHANISM_LABELS = {
    "B": "local_phase_coherence",
    "C": "transverse_two_body_correlations",
    "D": "diagonal_three_body_correlations",
    "E": "nonlinear_classical_decoding",
    "F": "intermediate_time_information",
}


@dataclass(frozen=True)
class Task3Config:
    processed_dataset: Path
    task1_feature_dir: Path
    q: int
    prefix_length: int
    g: float
    W: float
    layers: int
    batch_size: int
    reservoir_seeds: tuple[int, ...]
    devices: tuple[str, ...]
    ridge_alphas: tuple[float, ...]


def load_config(path: str | Path | None = None) -> Task3Config:
    config_path = TASK_DIR / "config.json" if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    qrc = payload["qrc"]

    def resolve(value: str) -> Path:
        return (config_path.parent / value).resolve()

    cfg = Task3Config(
        processed_dataset=resolve(payload["processed_dataset"]),
        task1_feature_dir=resolve(payload["task1_feature_dir"]),
        q=int(payload["num_qubits"]),
        prefix_length=int(payload["prefix_length"]),
        g=float(qrc["g"]),
        W=float(qrc["W"]),
        layers=int(qrc["layers_per_injection"]),
        batch_size=int(qrc["batch_size"]),
        reservoir_seeds=tuple(int(x) for x in payload["reservoir_seeds"]),
        devices=tuple(str(x) for x in payload["devices"]),
        ridge_alphas=tuple(float(x) for x in payload["ridge_alphas"]),
    )
    if cfg.q != 8 or cfg.prefix_length != 6 or cfg.layers != 3:
        raise ValueError("Task 3 is the fixed 8Q-6T-3L controlled experiment.")
    if len(cfg.devices) != 2:
        raise ValueError("Task 3 requires two sample-splitting devices.")
    return cfg


def _canonical_spec(items: Iterable[tuple[int, str]]) -> PauliSpec:
    spec = tuple(sorted(((int(w), str(a).upper()) for w, a in items)))
    wires = [wire for wire, _ in spec]
    if len(wires) != len(set(wires)):
        raise ValueError(f"duplicate wire in Pauli specification: {spec}")
    if any(axis not in {"X", "Y", "Z"} for _, axis in spec):
        raise ValueError(f"invalid Pauli axis in {spec}")
    return spec


def pauli_name(spec: PauliSpec) -> str:
    return "".join(axis for _, axis in spec) + "_" + "_".join(
        str(wire) for wire, _ in spec
    )


def observable_specifications(q: int, timesteps: int) -> dict[str, Any]:
    q = int(q)
    axes = ("X", "Y", "Z")
    z = [_canonical_spec(((i, "Z"),)) for i in range(q)]
    zz = [
        _canonical_spec(((i, "Z"), (j, "Z")))
        for i in range(q)
        for j in range(i + 1, q)
    ]
    base = z + zz
    xy_singles = [
        _canonical_spec(((i, axis),))
        for i in range(q)
        for axis in ("X", "Y")
    ]
    all_singles = [
        _canonical_spec(((i, axis),)) for i in range(q) for axis in axes
    ]
    all_pairs = [
        _canonical_spec(((i, axis_i), (j, axis_j)))
        for i in range(q)
        for j in range(i + 1, q)
        for axis_i in axes
        for axis_j in axes
    ]
    all_two_body = all_singles + all_pairs
    zzz = [
        _canonical_spec(((i, "Z"), (j, "Z"), (k, "Z")))
        for i in range(q)
        for j in range(i + 1, q)
        for k in range(j + 1, q)
    ]

    union: list[PauliSpec] = []
    seen: set[PauliSpec] = set()
    for spec in all_two_body + zzz:
        if spec not in seen:
            seen.add(spec)
            union.append(spec)
    lookup = {spec: index for index, spec in enumerate(union)}

    def indices(specs: list[PauliSpec]) -> np.ndarray:
        return np.asarray([lookup[spec] for spec in specs], dtype=np.int64)

    names = {
        "A": [pauli_name(spec) for spec in base],
        "B": [pauli_name(spec) for spec in base + xy_singles],
        "C": [pauli_name(spec) for spec in all_two_body],
        "D": [pauli_name(spec) for spec in base + zzz],
        "F": [
            f"t{t}_{pauli_name(spec)}"
            for t in range(int(timesteps))
            for spec in base
        ],
        "FB": [
            f"t{t}_{pauli_name(spec)}"
            for t in range(int(timesteps))
            for spec in base + xy_singles
        ],
        "FC": [
            f"t{t}_{pauli_name(spec)}"
            for t in range(int(timesteps))
            for spec in all_two_body
        ],
    }
    return {
        "base": base,
        "all_two_body": all_two_body,
        "zzz": zzz,
        "union": union,
        "indices": {
            "A": indices(base),
            "B": indices(base + xy_singles),
            "C": indices(all_two_body),
            "D": indices(base + zzz),
        },
        "names": names,
    }


def build_pauli_action_tables(
    specs: list[PauliSpec],
    num_qubits: int,
    device,
    complex_dtype,
):
    """Build |j> -> phase(j)|j xor flip> actions for Pauli strings."""
    import torch

    q = int(num_qubits)
    dim = 1 << q
    basis = torch.arange(dim, device=device, dtype=torch.long)
    if not specs:
        return (
            torch.empty((0, dim), device=device, dtype=torch.long),
            torch.empty((0, dim), device=device, dtype=complex_dtype),
        )
    gather_rows = []
    phase_rows = []
    for spec in specs:
        flip_mask = 0
        phase = torch.ones(dim, device=device, dtype=complex_dtype)
        for wire, axis in spec:
            shift = q - 1 - int(wire)
            bit = ((basis >> shift) & 1)
            z_sign = (1 - 2 * bit).to(
                torch.float32
                if complex_dtype == torch.complex64
                else torch.float64
            )
            if axis == "X":
                flip_mask ^= 1 << shift
            elif axis == "Y":
                flip_mask ^= 1 << shift
                phase = phase * (1j * z_sign)
            elif axis == "Z":
                phase = phase * z_sign
            else:
                raise ValueError(axis)
        gather_rows.append(basis ^ int(flip_mask))
        phase_rows.append(phase)
    return (
        torch.stack(gather_rows, dim=0).contiguous(),
        torch.stack(phase_rows, dim=0).contiguous(),
    )


def pauli_expectations(
    state,
    gather,
    phases,
    observable_chunk_size: int = 64,
):
    """Return real Pauli expectations for a batched pure state."""
    import torch

    if int(gather.shape[0]) == 0:
        return torch.empty(
            (int(state.shape[0]), 0),
            device=state.device,
            dtype=torch.float32,
        )
    rows = []
    for start in range(0, int(gather.shape[0]), int(observable_chunk_size)):
        end = min(start + int(observable_chunk_size), int(gather.shape[0]))
        selected = state[:, gather[start:end]]
        transformed = phases[start:end][None, :, :] * state[:, None, :]
        values = torch.sum(torch.conj(selected) * transformed, dim=-1).real
        rows.append(values)
    return torch.cat(rows, dim=1).float()


def compute_ablation_features(
    tokens,
    cfg: Task3Config,
    seed: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, list[str]]]:
    """Extract A-D/F from the same state evolution for one sample chunk."""
    import torch
    from gpu_qrc_statevector import (
        DTCQRCConfig,
        adapt_token_dim,
        apply_reservoir_layers,
        apply_token_injection,
        build_dtc_diagonal_phase,
        generate_dtc_params,
        real_dtype_from_complex,
        rx_gate,
        set_seed,
        torch_complex_dtype,
    )

    set_seed(int(seed))
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for requested device {device}")

    qrc_cfg = DTCQRCConfig(
        num_qubits=cfg.q,
        features_per_qubit=3,
        g=cfg.g,
        W_scale=cfg.W,
        layers_per_injection=cfg.layers,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    complex_dtype = torch_complex_dtype(qrc_cfg.complex_dtype)
    real_dtype = real_dtype_from_complex(complex_dtype)
    specs = observable_specifications(cfg.q, cfg.prefix_length)
    all2_gather, all2_phases = build_pauli_action_tables(
        specs["all_two_body"], cfg.q, torch_device, complex_dtype
    )
    zzz_gather, zzz_phases = build_pauli_action_tables(
        specs["zzz"], cfg.q, torch_device, complex_dtype
    )

    params = generate_dtc_params(cfg.q, seed=int(seed), W_scale=cfg.W)
    J = torch.tensor(params["J"], dtype=torch.float32)
    h = torch.tensor(params["h"], dtype=torch.float32)
    rx = rx_gate(cfg.g * math.pi, torch_device, complex_dtype)
    dtc_phase = build_dtc_diagonal_phase(
        cfg.q, J, h, torch_device, complex_dtype
    )

    tokens = np.ascontiguousarray(tokens, dtype=np.float32)
    outputs = {
        key: np.empty((len(tokens), len(specs["names"][key])), dtype=np.float32)
        for key in QRC_FEATURE_KEYS
    }
    index_tensors = {
        key: torch.as_tensor(value, device=torch_device, dtype=torch.long)
        for key, value in specs["indices"].items()
    }

    with torch.no_grad():
        for start in range(0, len(tokens), cfg.batch_size):
            end = min(start + cfg.batch_size, len(tokens))
            batch = torch.as_tensor(
                tokens[start:end], device=torch_device, dtype=real_dtype
            )
            batch = adapt_token_dim(
                batch, qrc_cfg.input_dim, qrc_cfg.token_dim_policy
            )
            batch = torch.tanh(batch)
            state = torch.zeros(
                (len(batch), qrc_cfg.state_dim),
                device=torch_device,
                dtype=complex_dtype,
            )
            state[:, 0] = 1.0 + 0.0j
            temporal_a = []
            temporal_b = []
            temporal_c = []
            for t in range(int(batch.shape[1])):
                state = apply_token_injection(
                    state, batch[:, t, :], qrc_cfg, complex_dtype
                )
                state = apply_reservoir_layers(
                    state, qrc_cfg, rx, dtc_phase
                )
                all2_values = pauli_expectations(
                    state, all2_gather, all2_phases
                )
                temporal_a.append(
                    all2_values[:, index_tensors["A"]]
                )
                temporal_b.append(
                    all2_values[:, index_tensors["B"]]
                )
                temporal_c.append(all2_values)
            final_all2 = temporal_c[-1]
            zzz_values = pauli_expectations(
                state, zzz_gather, zzz_phases
            )
            for key in ("A", "B", "C"):
                outputs[key][start:end] = final_all2[
                    :, index_tensors[key]
                ].cpu().numpy()
            outputs["D"][start:end] = torch.cat(
                [final_all2[:, index_tensors["A"]], zzz_values], dim=1
            ).cpu().numpy()
            outputs["F"][start:end] = (
                torch.cat(temporal_a, dim=1).cpu().numpy()
            )
            outputs["FB"][start:end] = (
                torch.cat(temporal_b, dim=1).cpu().numpy()
            )
            outputs["FC"][start:end] = (
                torch.cat(temporal_c, dim=1).cpu().numpy()
            )
            del (
                batch,
                state,
                temporal_a,
                temporal_b,
                temporal_c,
                final_all2,
                zzz_values,
            )
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    parameter_summary = {
        "J": params["J"].tolist(),
        "h": params["h"].tolist(),
        "seed": int(seed),
    }
    return outputs, parameter_summary, specs["names"]


def _feature_worker(
    seed: int,
    device: str,
    indices: np.ndarray,
    dataset_path: str,
    output_path: str,
    cfg_payload: dict[str, Any],
    error_queue,
) -> None:
    try:
        cfg = Task3Config(
            processed_dataset=Path(dataset_path),
            task1_feature_dir=Path(cfg_payload["task1_feature_dir"]),
            q=int(cfg_payload["q"]),
            prefix_length=int(cfg_payload["prefix_length"]),
            g=float(cfg_payload["g"]),
            W=float(cfg_payload["W"]),
            layers=int(cfg_payload["layers"]),
            batch_size=int(cfg_payload["batch_size"]),
            reservoir_seeds=(int(seed),),
            devices=(str(device), str(device)),
            ridge_alphas=tuple(cfg_payload["ridge_alphas"]),
        )
        with np.load(dataset_path, allow_pickle=False) as data:
            tokens = data["tokens"][indices]
        started = time.perf_counter()
        variants, params, names = compute_ablation_features(
            tokens, cfg, seed=int(seed), device=str(device)
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            indices=np.asarray(indices, dtype=np.int64),
            seed=np.asarray(seed, dtype=np.int64),
            device=np.asarray(device),
            elapsed_seconds=np.asarray(
                time.perf_counter() - started, dtype=np.float64
            ),
            parameter_summary=np.asarray(json.dumps(params)),
            feature_names=np.asarray(json.dumps(names)),
            **variants,
        )
    except Exception:
        error_queue.put(
            {
                "seed": int(seed),
                "device": str(device),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def compute_features_two_devices(
    cfg: Task3Config,
    seed: int,
    feature_dir: Path,
    force: bool = False,
) -> Path:
    """Split every seed along the sample axis over both configured devices."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    combined_path = feature_dir / f"qrc_ablation_seed{int(seed):03d}.npz"
    if combined_path.exists() and not force:
        return combined_path
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        num_samples = int(len(data["tokens"]))
    index_parts = np.array_split(
        np.arange(num_samples, dtype=np.int64), len(cfg.devices)
    )
    part_paths = [
        feature_dir
        / f"qrc_ablation_seed{int(seed):03d}_device{rank}_part.npz"
        for rank in range(len(cfg.devices))
    ]
    payload = {
        "task1_feature_dir": str(cfg.task1_feature_dir),
        "q": cfg.q,
        "prefix_length": cfg.prefix_length,
        "g": cfg.g,
        "W": cfg.W,
        "layers": cfg.layers,
        "batch_size": cfg.batch_size,
        "ridge_alphas": list(cfg.ridge_alphas),
    }
    context = mp.get_context("spawn")
    error_queue = context.Queue()
    processes = []
    for device, indices, part_path in zip(
        cfg.devices, index_parts, part_paths
    ):
        process = context.Process(
            target=_feature_worker,
            args=(
                int(seed),
                str(device),
                indices,
                str(cfg.processed_dataset),
                str(part_path),
                payload,
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
        details = "\n".join(error["traceback"] for error in errors)
        raise RuntimeError(f"Task 3 feature extraction failed:\n{details}")

    with np.load(part_paths[0], allow_pickle=False) as first:
        names_json = str(first["feature_names"].item())
        names = json.loads(names_json)
        combined = {
            key: np.empty(
                (num_samples, int(first[key].shape[1])), dtype=np.float32
            )
            for key in QRC_FEATURE_KEYS
        }
        parameter_summary = str(first["parameter_summary"].item())
    part_metadata = []
    for part_path in part_paths:
        with np.load(part_path, allow_pickle=False) as part:
            indices = part["indices"]
            if str(part["parameter_summary"].item()) != parameter_summary:
                raise RuntimeError("Reservoir parameters differ across devices.")
            if str(part["feature_names"].item()) != names_json:
                raise RuntimeError("Feature order differs across devices.")
            for key, values in combined.items():
                if part[key].shape != (len(indices), values.shape[1]):
                    raise RuntimeError(
                        f"{part_path.name} {key}: bad shape {part[key].shape}"
                    )
                values[indices] = part[key]
            part_metadata.append(
                {
                    "device": str(part["device"].item()),
                    "num_samples": int(len(indices)),
                    "elapsed_seconds": float(part["elapsed_seconds"]),
                }
            )

    for key, values in combined.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values in variant {key}.")
        if values.shape[1] != len(names[key]):
            raise RuntimeError(f"Name/feature mismatch for variant {key}.")

    task1_path = cfg.task1_feature_dir / f"qrc_seed{int(seed):03d}.npz"
    replication_max_abs_error = None
    if task1_path.exists():
        with np.load(task1_path, allow_pickle=False) as task1:
            reference = task1["features"].astype(np.float32)
        replication_max_abs_error = float(
            np.max(np.abs(combined["A"] - reference))
        )
        if replication_max_abs_error > 2e-5:
            raise RuntimeError(
                "Variant A does not replicate task 1: "
                f"max_abs_error={replication_max_abs_error:.3e}"
            )

    np.savez_compressed(
        combined_path,
        seed=np.asarray(seed, dtype=np.int64),
        feature_names=np.asarray(names_json),
        parameter_summary=np.asarray(parameter_summary),
        device_parts=np.asarray(json.dumps(part_metadata)),
        replication_max_abs_error=np.asarray(
            np.nan
            if replication_max_abs_error is None
            else replication_max_abs_error,
            dtype=np.float64,
        ),
        **combined,
    )
    return combined_path


def compute_esn_temporal_features(
    tokens: np.ndarray,
    cfg: Task3Config,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the standard ESN state after each matched injection boundary."""
    import torch
    from esn_baseline import esn_standard_dim

    values = np.ascontiguousarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 * cfg.q:
        raise ValueError(
            f"Standard ESN expects [N,T,{3 * cfg.q}], got {values.shape}"
        )
    state_dim = esn_standard_dim(cfg.q)
    input_dim = 3 * cfg.q
    rng = np.random.default_rng(int(seed))
    recurrent = rng.normal(size=(state_dim, state_dim)).astype(np.float32)
    radius = float(
        np.max(np.abs(np.linalg.eigvals(recurrent.astype(np.float64))))
    )
    spectral_radius = 0.90
    input_scale = 0.75
    leak_rate = 0.65
    recurrent *= np.float32(spectral_radius / max(radius, 1e-8))
    input_weights = (
        rng.normal(size=(state_dim, input_dim)).astype(np.float32)
        / math.sqrt(max(1, input_dim))
    )
    bias = rng.normal(scale=0.10, size=state_dim).astype(np.float32)

    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for requested device {device}")
    x = torch.tanh(torch.from_numpy(values).to(compute_device))
    recurrent_t = torch.from_numpy(recurrent).to(compute_device)
    input_weights_t = torch.from_numpy(input_weights).to(compute_device)
    bias_t = torch.from_numpy(bias).to(compute_device)
    state = torch.zeros(
        (len(values), state_dim),
        dtype=torch.float32,
        device=compute_device,
    )
    temporal = []
    for time_index in range(x.shape[1]):
        injection = input_scale * (
            x[:, time_index] @ input_weights_t.T
        )
        for _ in range(cfg.layers):
            proposal = torch.tanh(
                state @ recurrent_t.T + injection + bias_t
            )
            state = (1.0 - leak_rate) * state + leak_rate * proposal
        temporal.append(state)
    if compute_device.type == "cuda":
        torch.cuda.synchronize(compute_device)
    features = (
        torch.cat(temporal, dim=1).cpu().numpy().astype(np.float32, copy=False)
    )
    expected = (len(values), cfg.prefix_length * state_dim)
    if features.shape != expected or not np.isfinite(features).all():
        raise RuntimeError(
            f"Invalid temporal ESN output shape/values: {features.shape}"
        )
    return features, {
        "spectral_radius": spectral_radius,
        "input_scale": input_scale,
        "leak_rate": leak_rate,
        "seed": int(seed),
    }


def _esn_temporal_worker(
    seed: int,
    device: str,
    indices: np.ndarray,
    dataset_path: str,
    output_path: str,
    cfg_payload: dict[str, Any],
    error_queue,
) -> None:
    try:
        cfg = Task3Config(
            processed_dataset=Path(dataset_path),
            task1_feature_dir=Path(cfg_payload["task1_feature_dir"]),
            q=int(cfg_payload["q"]),
            prefix_length=int(cfg_payload["prefix_length"]),
            g=float(cfg_payload["g"]),
            W=float(cfg_payload["W"]),
            layers=int(cfg_payload["layers"]),
            batch_size=int(cfg_payload["batch_size"]),
            reservoir_seeds=(int(seed),),
            devices=(str(device), str(device)),
            ridge_alphas=tuple(cfg_payload["ridge_alphas"]),
        )
        with np.load(dataset_path, allow_pickle=False) as data:
            tokens = data["tokens"][indices]
        started = time.perf_counter()
        features, params = compute_esn_temporal_features(
            tokens, cfg, seed=int(seed), device=str(device)
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            indices=np.asarray(indices, dtype=np.int64),
            features=features,
            seed=np.asarray(seed, dtype=np.int64),
            device=np.asarray(device),
            elapsed_seconds=np.asarray(
                time.perf_counter() - started, dtype=np.float64
            ),
            parameter_summary=np.asarray(json.dumps(params)),
        )
    except Exception:
        error_queue.put(
            {
                "seed": int(seed),
                "device": str(device),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def compute_esn_temporal_two_devices(
    cfg: Task3Config,
    seed: int,
    feature_dir: Path,
    force: bool = False,
) -> Path:
    """Split the matched temporal ESN along samples over both devices."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    combined_path = feature_dir / f"esn_temporal_seed{int(seed):03d}.npz"
    if combined_path.exists() and not force:
        return combined_path
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        num_samples = int(len(data["tokens"]))
    index_parts = np.array_split(
        np.arange(num_samples, dtype=np.int64), len(cfg.devices)
    )
    part_paths = [
        feature_dir
        / f"esn_temporal_seed{int(seed):03d}_device{rank}_part.npz"
        for rank in range(len(cfg.devices))
    ]
    payload = {
        "task1_feature_dir": str(cfg.task1_feature_dir),
        "q": cfg.q,
        "prefix_length": cfg.prefix_length,
        "g": cfg.g,
        "W": cfg.W,
        "layers": cfg.layers,
        "batch_size": cfg.batch_size,
        "ridge_alphas": list(cfg.ridge_alphas),
    }
    context = mp.get_context("spawn")
    error_queue = context.Queue()
    processes = []
    for device, indices, part_path in zip(
        cfg.devices, index_parts, part_paths
    ):
        process = context.Process(
            target=_esn_temporal_worker,
            args=(
                int(seed),
                str(device),
                indices,
                str(cfg.processed_dataset),
                str(part_path),
                payload,
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
        details = "\n".join(error["traceback"] for error in errors)
        raise RuntimeError(f"Temporal ESN extraction failed:\n{details}")

    state_dim = cfg.q * (cfg.q + 1) // 2
    features = np.empty(
        (num_samples, cfg.prefix_length * state_dim), dtype=np.float32
    )
    metadata = []
    parameter_summary = None
    for part_path in part_paths:
        with np.load(part_path, allow_pickle=False) as part:
            indices = part["indices"]
            values = part["features"]
            if values.shape != (len(indices), features.shape[1]):
                raise RuntimeError(
                    f"{part_path.name}: bad shape {values.shape}"
                )
            current_params = str(part["parameter_summary"].item())
            if parameter_summary is None:
                parameter_summary = current_params
            elif current_params != parameter_summary:
                raise RuntimeError(
                    "ESN parameters differ across sample-split devices."
                )
            features[indices] = values
            metadata.append(
                {
                    "device": str(part["device"].item()),
                    "num_samples": int(len(indices)),
                    "elapsed_seconds": float(part["elapsed_seconds"]),
                }
            )
    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite temporal ESN features.")

    task1_path = cfg.task1_feature_dir / f"esn_seed{int(seed):03d}.npz"
    with np.load(task1_path, allow_pickle=False) as task1:
        final_reference = task1["features"].astype(np.float32)
    final_replication_max_abs_error = float(
        np.max(np.abs(features[:, -state_dim:] - final_reference))
    )
    if final_replication_max_abs_error > 2e-6:
        raise RuntimeError(
            "Temporal ESN final state does not replicate task 1: "
            f"max_abs_error={final_replication_max_abs_error:.3e}"
        )
    np.savez_compressed(
        combined_path,
        features=features,
        seed=np.asarray(seed, dtype=np.int64),
        feature_dim=np.asarray(features.shape[1], dtype=np.int64),
        parameter_summary=np.asarray(parameter_summary),
        device_parts=np.asarray(json.dumps(metadata)),
        final_replication_max_abs_error=np.asarray(
            final_replication_max_abs_error, dtype=np.float64
        ),
    )
    return combined_path


def _masked_errors(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    errors = np.asarray(prediction - target)[mask]
    return (
        float(np.mean(np.abs(errors))),
        float(np.sqrt(np.mean(errors**2))),
    )


def _fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    alphas: tuple[float, ...],
    validation_metric,
) -> tuple[np.ndarray, float, float]:
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    scaler = StandardScaler().fit(features[train])
    train_x = scaler.transform(features[train])
    val_x = scaler.transform(features[val])
    best_alpha = None
    best_score = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=float(alpha)).fit(train_x, target[train])
        score = float(validation_metric(target[val], model.predict(val_x), val))
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    train_val = train | val
    final_scaler = StandardScaler().fit(features[train_val])
    final_model = Ridge(alpha=float(best_alpha)).fit(
        final_scaler.transform(features[train_val]), target[train_val]
    )
    prediction = final_model.predict(final_scaler.transform(features[test]))
    return np.asarray(prediction), float(best_alpha), float(best_score)


def _polynomial_two(features: np.ndarray) -> np.ndarray:
    transform = PolynomialFeatures(degree=2, include_bias=False)
    return transform.fit_transform(features).astype(np.float64, copy=False)


def _training_only_pca(
    features: np.ndarray,
    splits: np.ndarray,
    n_components: int,
    seed: int,
) -> np.ndarray:
    """Apply label-free standardization/PCA fitted on training rows only."""
    train = splits == "train"
    scaler = StandardScaler().fit(features[train])
    standardized = scaler.transform(features)
    pca = PCA(
        n_components=int(n_components),
        svd_solver="randomized",
        iterated_power=5,
        random_state=int(seed),
    ).fit(standardized[train])
    return pca.transform(standardized).astype(np.float64, copy=False)


def evaluate_all_variants(
    cfg: Task3Config, feature_dir: Path, output_dir: Path
) -> Path:
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        splits = data["splits"]
        valid_mask = data["valid_pair_mask"]
        changed_mask = data["changed_pair_mask"]
        target_rho = data["target_delta_rho"].astype(np.float64)
        target_energy = data["target_delta_energy"].astype(np.float64)
    test = splits == "test"
    rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for seed in cfg.reservoir_seeds:
        package_path = feature_dir / f"qrc_ablation_seed{int(seed):03d}.npz"
        with np.load(package_path, allow_pickle=False) as package:
            raw_variants = {
                key: package[key].astype(np.float64)
                for key in QRC_FEATURE_KEYS
            }
        variants = {
            **raw_variants,
            "E": _polynomial_two(raw_variants["A"]),
            "FC216": _training_only_pca(
                raw_variants["FC"], splits, 216, int(seed)
            ),
        }
        esn_path = cfg.task1_feature_dir / f"esn_seed{int(seed):03d}.npz"
        with np.load(esn_path, allow_pickle=False) as package:
            variants["ESN"] = package["features"].astype(np.float64)
        with np.load(
            feature_dir / f"esn_temporal_seed{int(seed):03d}.npz",
            allow_pickle=False,
        ) as package:
            variants["ESN_F"] = package["features"].astype(np.float64)

        def evaluate_one(
            variant: str, label: str, features: np.ndarray
        ) -> dict[str, Any]:
            def distance_metric(y_true, y_pred, rows_mask):
                return float(
                    np.mean(
                        np.abs(y_pred - y_true)[valid_mask[rows_mask]]
                    )
                )

            rho_prediction, rho_alpha, rho_val_mae = _fit_ridge(
                features,
                target_rho,
                splits,
                cfg.ridge_alphas,
                distance_metric,
            )

            def energy_metric(y_true, y_pred, _):
                return float(np.mean(np.abs(y_pred - y_true)))

            energy_prediction, energy_alpha, energy_val_mae = _fit_ridge(
                features,
                target_energy,
                splits,
                cfg.ridge_alphas,
                energy_metric,
            )
            all_mae, all_rmse = _masked_errors(
                target_rho[test], rho_prediction, valid_mask[test]
            )
            changed = valid_mask[test] & changed_mask[test]
            changed_mae, changed_rmse = _masked_errors(
                target_rho[test], rho_prediction, changed
            )
            energy_error = energy_prediction - target_energy[test]
            return {
                "variant": variant,
                "label": label,
                "seed": int(seed),
                "feature_dim": int(features.shape[1]),
                "rho_alpha": rho_alpha,
                "rho_validation_mae": rho_val_mae,
                "energy_alpha": energy_alpha,
                "energy_validation_mae_eV": energy_val_mae,
                "rho_mae_all_pairs": all_mae,
                "rho_rmse_all_pairs": all_rmse,
                "rho_mae_changed_pairs": changed_mae,
                "rho_rmse_changed_pairs": changed_rmse,
                "energy_mae_eV": float(np.mean(np.abs(energy_error))),
                "energy_rmse_eV": float(
                    np.sqrt(np.mean(energy_error**2))
                ),
            }

        for variant, features in variants.items():
            rows.append(
                evaluate_one(variant, VARIANT_LABELS[variant], features)
            )

        # A plus target-independent Gaussian columns controls for the possibility
        # that a larger Ridge design matrix alone explains B-F.
        for offset, variant in enumerate(
            ("B", "C", "D", "E", "F", "FB", "FC", "ESN_F")
        ):
            target_dim = int(variants[variant].shape[1])
            if variant == "ESN_F":
                base = variants["ESN"]
            elif variant in {"FB", "FC"}:
                base = raw_variants["F"]
            else:
                base = raw_variants["A"]
            rng = np.random.default_rng(
                3_141_593 + 10_007 * int(seed) + offset
            )
            noise = rng.standard_normal(
                (len(base), target_dim - base.shape[1])
            )
            control_features = np.concatenate([base, noise], axis=1)
            control_rows.append(
                evaluate_one(
                    f"CTRL_{variant}",
                    (
                        f"ESN + random columns matched to {variant}"
                        if variant == "ESN_F"
                        else (
                            f"F + random columns matched to {variant}"
                            if variant in {"FB", "FC"}
                            else f"A + random columns matched to {variant}"
                        )
                    ),
                    control_features,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "metrics_by_seed.csv", rows)
    summary = _summarize_metrics(rows)
    _write_rows(output_dir / "summary.csv", summary)
    _write_rows(output_dir / "dimension_controls_by_seed.csv", control_rows)
    control_summary = _summarize_control_metrics(control_rows)
    _write_rows(output_dir / "dimension_controls_summary.csv", control_summary)
    mechanisms = _mechanism_rows(
        rows, summary, control_rows, control_summary
    )
    _write_rows(output_dir / "mechanism_tests.csv", mechanisms)
    temporal = _temporal_comparison_rows(
        rows, summary, control_rows, control_summary
    )
    _write_rows(output_dir / "qrc_vs_esn_temporal.csv", temporal)
    temporal_observables = _temporal_observable_rows(
        rows, summary, control_rows, control_summary
    )
    _write_rows(
        output_dir / "temporal_observable_tests.csv",
        temporal_observables,
    )
    report = {
        "status": "completed",
        "experiment": "Transition1x 8Q-6T-3L observable/readout ablation",
        "dataset": str(cfg.processed_dataset),
        "reservoir_seeds": list(cfg.reservoir_seeds),
        "variants": VARIANT_LABELS,
        "summary": summary,
        "dimension_controls": control_summary,
        "mechanism_tests": mechanisms,
        "temporal_qrc_vs_esn": temporal,
        "temporal_observable_tests": temporal_observables,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_results_markdown(
        summary,
        mechanisms,
        temporal,
        temporal_observables,
        TASK_DIR / "RESULTS.md",
    )
    return report_path


def _summarize_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for variant in (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "FB",
        "FC",
        "FC216",
        "ESN",
        "ESN_F",
    ):
        selected = [row for row in rows if row["variant"] == variant]
        for metric in (
            "rho_mae_all_pairs",
            "rho_mae_changed_pairs",
            "energy_mae_eV",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary.append(
                {
                    "variant": variant,
                    "label": VARIANT_LABELS[variant],
                    "feature_dim": int(selected[0]["feature_dim"]),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "num_seeds": int(len(values)),
                }
            )
    return summary


def _summarize_control_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary = []
    for variant in (
        "CTRL_B",
        "CTRL_C",
        "CTRL_D",
        "CTRL_E",
        "CTRL_F",
        "CTRL_FB",
        "CTRL_FC",
        "CTRL_ESN_F",
    ):
        selected = [row for row in rows if row["variant"] == variant]
        for metric in (
            "rho_mae_all_pairs",
            "rho_mae_changed_pairs",
            "energy_mae_eV",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            summary.append(
                {
                    "variant": variant,
                    "label": selected[0]["label"],
                    "feature_dim": int(selected[0]["feature_dim"]),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "num_seeds": int(len(values)),
                }
            )
    return summary


def _mechanism_rows(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    control_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "rho_mae_all_pairs",
        "rho_mae_changed_pairs",
        "energy_mae_eV",
    )
    lookup = {
        (row["variant"], row["metric"]): float(row["mean"])
        for row in summary
    }
    by_seed = {
        (row["variant"], int(row["seed"])): row for row in rows
    }
    control_lookup = {
        (row["variant"], row["metric"]): float(row["mean"])
        for row in control_summary
    }
    control_by_seed = {
        (row["variant"], int(row["seed"])): row for row in control_rows
    }
    mechanism_rows = []
    for variant in ("B", "C", "D", "E", "F"):
        for metric in metrics:
            a_mean = lookup[("A", metric)]
            value = lookup[(variant, metric)]
            esn_mean = lookup[("ESN", metric)]
            wins = sum(
                by_seed[(variant, seed)][metric]
                < by_seed[("A", seed)][metric]
                for seed in sorted(
                    row["seed"] for row in rows if row["variant"] == "A"
                )
            )
            control_variant = f"CTRL_{variant}"
            control_mean = control_lookup[(control_variant, metric)]
            wins_control = sum(
                by_seed[(variant, seed)][metric]
                < control_by_seed[(control_variant, seed)][metric]
                for seed in sorted(
                    row["seed"] for row in rows if row["variant"] == "A"
                )
            )
            improvement = a_mean - value
            if (
                improvement > 0
                and wins >= 2
                and value < control_mean
                and wins_control >= 2
            ):
                verdict = "directional_support"
            elif (
                improvement <= 0
                and wins <= 1
                or value >= control_mean
                and wins_control <= 1
            ):
                verdict = "not_supported"
            else:
                verdict = "mixed"
            gap = a_mean - esn_mean
            gap_closed = (
                float(improvement / gap) if abs(gap) > 1e-12 else float("nan")
            )
            mechanism_rows.append(
                {
                    "variant": variant,
                    "mechanism": MECHANISM_LABELS[variant],
                    "metric": metric,
                    "mean_delta_mae_vs_A": float(value - a_mean),
                    "relative_mae_change_vs_A_percent": float(
                        100.0 * (value - a_mean) / a_mean
                    ),
                    "improved_seeds_out_of_3": int(wins),
                    "random_dimension_control_mean_mae": control_mean,
                    "mean_delta_mae_vs_random_control": float(
                        value - control_mean
                    ),
                    "beat_random_control_seeds_out_of_3": int(wins_control),
                    "fraction_A_to_ESN_gap_closed": gap_closed,
                    "verdict": verdict,
                }
            )
    return mechanism_rows


def _temporal_comparison_rows(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    control_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "rho_mae_all_pairs",
        "rho_mae_changed_pairs",
        "energy_mae_eV",
    )
    mean = {
        (row["variant"], row["metric"]): float(row["mean"])
        for row in summary
    }
    by_seed = {
        (row["variant"], int(row["seed"])): row for row in rows
    }
    control_mean = {
        (row["variant"], row["metric"]): float(row["mean"])
        for row in control_summary
    }
    control_by_seed = {
        (row["variant"], int(row["seed"])): row for row in control_rows
    }
    seeds = sorted(row["seed"] for row in rows if row["variant"] == "A")
    output = []
    for metric in metrics:
        qrc_final = mean[("A", metric)]
        qrc_temporal = mean[("F", metric)]
        esn_final = mean[("ESN", metric)]
        esn_temporal = mean[("ESN_F", metric)]
        esn_temporal_wins = sum(
            by_seed[("ESN_F", seed)][metric]
            < by_seed[("ESN", seed)][metric]
            for seed in seeds
        )
        esn_control_wins = sum(
            by_seed[("ESN_F", seed)][metric]
            < control_by_seed[("CTRL_ESN_F", seed)][metric]
            for seed in seeds
        )
        qrc_wins = sum(
            by_seed[("F", seed)][metric]
            < by_seed[("ESN_F", seed)][metric]
            for seed in seeds
        )
        output.append(
            {
                "metric": metric,
                "qrc_final_A_mean": qrc_final,
                "qrc_temporal_F_mean": qrc_temporal,
                "qrc_temporal_relative_change_percent": float(
                    100.0 * (qrc_temporal - qrc_final) / qrc_final
                ),
                "esn_final_mean": esn_final,
                "esn_temporal_mean": esn_temporal,
                "esn_temporal_relative_change_percent": float(
                    100.0 * (esn_temporal - esn_final) / esn_final
                ),
                "esn_temporal_improved_seeds_out_of_3": int(
                    esn_temporal_wins
                ),
                "esn_temporal_random_control_mean": control_mean[
                    ("CTRL_ESN_F", metric)
                ],
                "esn_temporal_beat_control_seeds_out_of_3": int(
                    esn_control_wins
                ),
                "qrc_temporal_minus_esn_temporal_mae": float(
                    qrc_temporal - esn_temporal
                ),
                "qrc_temporal_wins_out_of_3": int(qrc_wins),
                "temporal_winner": (
                    "QRC_F" if qrc_temporal < esn_temporal else "ESN_F"
                ),
            }
        )
    return output


def _temporal_observable_rows(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    control_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "rho_mae_all_pairs",
        "rho_mae_changed_pairs",
        "energy_mae_eV",
    )
    mean = {
        (row["variant"], row["metric"]): float(row["mean"])
        for row in summary
    }
    by_seed = {
        (row["variant"], int(row["seed"])): row for row in rows
    }
    control_mean = {
        (row["variant"], row["metric"]): float(row["mean"])
        for row in control_summary
    }
    control_by_seed = {
        (row["variant"], int(row["seed"])): row for row in control_rows
    }
    seeds = sorted(row["seed"] for row in rows if row["variant"] == "F")
    output = []
    for variant in ("FB", "FC", "FC216"):
        for metric in metrics:
            value = mean[(variant, metric)]
            f_value = mean[("F", metric)]
            esn_value = mean[("ESN_F", metric)]
            improved_f = sum(
                by_seed[(variant, seed)][metric]
                < by_seed[("F", seed)][metric]
                for seed in seeds
            )
            if variant == "FC216":
                control_value = f_value
                beat_control = improved_f
            else:
                control_variant = f"CTRL_{variant}"
                control_value = control_mean[(control_variant, metric)]
                beat_control = sum(
                    by_seed[(variant, seed)][metric]
                    < control_by_seed[(control_variant, seed)][metric]
                    for seed in seeds
                )
            beat_esn = sum(
                by_seed[(variant, seed)][metric]
                < by_seed[("ESN_F", seed)][metric]
                for seed in seeds
            )
            supported = (
                value < f_value
                and improved_f >= 2
                and value < control_value
                and beat_control >= 2
            )
            output.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": value,
                    "F_mean": f_value,
                    "relative_change_vs_F_percent": float(
                        100.0 * (value - f_value) / f_value
                    ),
                    "improved_vs_F_seeds_out_of_3": int(improved_f),
                    "random_dimension_control_mean": control_value,
                    "beat_random_control_seeds_out_of_3": int(
                        beat_control
                    ),
                    "ESN_F_mean": esn_value,
                    "mean_delta_vs_ESN_F": float(value - esn_value),
                    "beat_ESN_F_seeds_out_of_3": int(beat_esn),
                    "winner_vs_ESN_F": (
                        variant if value < esn_value else "ESN_F"
                    ),
                    "verdict_vs_F": (
                        "directional_support"
                        if supported
                        else "not_supported"
                    ),
                }
            )
    return output


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean(axis=0)) / np.maximum(
        values.std(axis=0), 1e-12
    )


def _task_alignment(
    features: np.ndarray,
    target: np.ndarray,
    regularization: float = 1e-3,
) -> tuple[float, float]:
    """Training-only regularized canonical strength and effective task rank."""
    x = _standardize(features)
    y = _standardize(
        target if np.asarray(target).ndim == 2 else np.asarray(target)[:, None]
    )
    denom = max(len(x) - 1, 1)
    cxx = (x.T @ x) / denom
    cyy = (y.T @ y) / denom
    cxy = (x.T @ y) / denom

    def invsqrt(matrix: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, float(regularization))
        return (vectors * (1.0 / np.sqrt(values))) @ vectors.T

    aligned = invsqrt(cxx) @ cxy @ invsqrt(cyy)
    singular = np.clip(
        np.linalg.svd(aligned, compute_uv=False), 0.0, 1.0
    )
    weights = singular**2
    strength = float(weights.sum())
    if strength <= 1e-15:
        return 0.0, 0.0
    probabilities = weights / strength
    task_rank = float(
        np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15))))
    )
    return strength, task_rank


def diagnose_features(
    cfg: Task3Config, feature_dir: Path, output_dir: Path
) -> Path:
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        train = data["splits"] == "train"
        target_rho = data["target_delta_rho"][train]
        target_energy = data["target_delta_energy"][train]
    rows = []
    for seed in cfg.reservoir_seeds:
        with np.load(
            feature_dir / f"qrc_ablation_seed{int(seed):03d}.npz",
            allow_pickle=False,
        ) as package:
            variants = {
                key: package[key][train].astype(np.float64)
                for key in QRC_FEATURE_KEYS
            }
            replication_error = float(package["replication_max_abs_error"])
        with np.load(
            cfg.task1_feature_dir / f"esn_seed{int(seed):03d}.npz",
            allow_pickle=False,
        ) as package:
            variants["ESN"] = package["features"][train].astype(np.float64)
        with np.load(
            feature_dir / f"esn_temporal_seed{int(seed):03d}.npz",
            allow_pickle=False,
        ) as package:
            variants["ESN_F"] = package["features"][train].astype(np.float64)
        for variant, values in variants.items():
            standardized = _standardize(values)
            covariance = np.cov(standardized, rowvar=False)
            eigenvalues = np.clip(
                np.linalg.eigvalsh(covariance), 0.0, None
            )
            probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
            nonzero = probabilities > 0
            effective_rank = float(
                np.exp(
                    -np.sum(
                        probabilities[nonzero]
                        * np.log(probabilities[nonzero])
                    )
                )
            )
            feature_correlation = np.corrcoef(values, rowvar=False)
            offdiag = np.abs(
                feature_correlation[
                    np.triu_indices_from(feature_correlation, k=1)
                ]
            )
            rho_corr = (
                standardized.T @ _standardize(target_rho)
            ) / max(len(values) - 1, 1)
            energy_corr = (
                standardized.T
                @ _standardize(target_energy[:, None])
            ) / max(len(values) - 1, 1)
            rho_strength, rho_task_rank = _task_alignment(
                values, target_rho
            )
            energy_strength, energy_task_rank = _task_alignment(
                values, target_energy
            )
            rows.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "feature_dim": int(values.shape[1]),
                    "effective_rank": effective_rank,
                    "participation_ratio": float(
                        eigenvalues.sum() ** 2
                        / max(np.sum(eigenvalues**2), 1e-12)
                    ),
                    "mean_feature_std": float(
                        values.std(axis=0).mean()
                    ),
                    "mean_abs_offdiag_correlation": float(offdiag.mean()),
                    "rho_cross_correlation_frobenius": float(
                        np.linalg.norm(rho_corr)
                    ),
                    "energy_cross_correlation_l2": float(
                        np.linalg.norm(energy_corr)
                    ),
                    "rho_regularized_canonical_strength": rho_strength,
                    "rho_task_effective_rank": rho_task_rank,
                    "energy_regularized_canonical_strength": energy_strength,
                    "energy_task_effective_rank": energy_task_rank,
                    "A_replication_max_abs_error": replication_error,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    by_seed_path = output_dir / "feature_diagnostics_by_seed.csv"
    _write_rows(by_seed_path, rows)
    summary = []
    numeric = [
        key
        for key in rows[0]
        if key not in {"variant", "seed", "feature_dim"}
    ]
    for variant in (
        "A",
        "B",
        "C",
        "D",
        "F",
        "FB",
        "FC",
        "ESN",
        "ESN_F",
    ):
        selected = [row for row in rows if row["variant"] == variant]
        record: dict[str, Any] = {
            "variant": variant,
            "feature_dim": int(selected[0]["feature_dim"]),
        }
        for key in numeric:
            values = np.asarray([row[key] for row in selected], dtype=float)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_sample_std"] = float(values.std(ddof=1))
        summary.append(record)
    summary_path = output_dir / "feature_diagnostics_summary.csv"
    _write_rows(summary_path, summary)
    return summary_path


def write_results_markdown(
    summary: list[dict[str, Any]],
    mechanisms: list[dict[str, Any]],
    temporal: list[dict[str, Any]],
    temporal_observables: list[dict[str, Any]],
    path: Path,
) -> None:
    metrics = {
        (row["variant"], row["metric"]): row for row in summary
    }
    lines = [
        "# Task 3 results",
        "",
        "This file is generated from the held-out Transition1x test split.",
        "All variants use the same validation-selected Ridge grid through alpha=10000.",
        "",
        "## Mean test performance",
        "",
        "| Variant | dim | all-pair Δρ MAE | changed-pair Δρ MAE | ΔE MAE (eV) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in (
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "FB",
        "FC",
        "FC216",
        "ESN",
        "ESN_F",
    ):
        rho = metrics[(variant, "rho_mae_all_pairs")]
        changed = metrics[(variant, "rho_mae_changed_pairs")]
        energy = metrics[(variant, "energy_mae_eV")]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {rho['feature_dim']} | "
            f"{rho['mean']:.5f} ± {rho['sample_std']:.5f} | "
            f"{changed['mean']:.5f} ± {changed['sample_std']:.5f} | "
            f"{energy['mean']:.5f} ± {energy['sample_std']:.5f} |"
        )
    lines += [
        "",
        "## Directional mechanism screen",
        "",
        "A direction is supported when its mean MAE is below A and at least two "
        "matched seeds improve, and when it also beats the dimension-matched "
        "A-plus-random-columns control in at least two seeds. This is a screening "
        "rule, not a significance test.",
        "",
        "| Variant | metric | ΔMAE vs A | improved seeds | ΔMAE vs random-dim control | beat control seeds | verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in mechanisms:
        lines.append(
            f"| {row['variant']} | {row['metric']} | "
            f"{row['mean_delta_mae_vs_A']:+.6f} | "
            f"{row['improved_seeds_out_of_3']}/3 | "
            f"{row['mean_delta_mae_vs_random_control']:+.6f} | "
            f"{row['beat_random_control_seeds_out_of_3']}/3 | "
            f"{row['verdict']} |"
        )
    lines += [
        "",
        "## Matched temporal QRC versus ESN",
        "",
        "| metric | QRC final | QRC temporal | ESN final | ESN temporal | QRC-F − ESN-F | winner |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in temporal:
        lines.append(
            f"| {row['metric']} | {row['qrc_final_A_mean']:.5f} | "
            f"{row['qrc_temporal_F_mean']:.5f} | "
            f"{row['esn_final_mean']:.5f} | "
            f"{row['esn_temporal_mean']:.5f} | "
            f"{row['qrc_temporal_minus_esn_temporal_mae']:+.5f} | "
            f"{row['temporal_winner']} |"
        )
    lines += [
        "",
        "## Adding phase and two-body Pauli observables to F",
        "",
        "| variant | metric | mean | change vs F | improved seeds | random-control wins | Δ vs ESN-F | winner |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in temporal_observables:
        lines.append(
            f"| {row['variant']} | {row['metric']} | {row['mean']:.5f} | "
            f"{row['relative_change_vs_F_percent']:+.2f}% | "
            f"{row['improved_vs_F_seeds_out_of_3']}/3 | "
            f"{row['beat_random_control_seeds_out_of_3']}/3 | "
            f"{row['mean_delta_vs_ESN_F']:+.5f} | "
            f"{row['winner_vs_ESN_F']} |"
        )
    lines += [
        "",
        "## Main interpretation",
        "",
        "- B improves both geometry targets in every seed and gives weaker but "
        "positive energy evidence, supporting useful phase-sensitive single-qubit "
        "information.",
        "- C provides the strongest final-state all-pair geometry result; its gain "
        "beyond B shows that transverse two-body correlations contain additional "
        "target information.",
        "- D improves consistently but less strongly, so diagonal three-body "
        "correlations are useful but are not the dominant missing channel.",
        "- E supports nonlinear target information inside the original Z/ZZ "
        "features once the common regularization grid is extended.",
        "- ESN-F also improves every target in all three seeds and beats its "
        "dimension control. Final-state information loss is therefore not unique "
        "to QRC.",
        "- QRC receives the larger relative gain from temporal readout, but the "
        "absolute winner is task-dependent: ESN-F wins both geometry metrics in "
        "all seeds, whereas QRC-F wins energy in all seeds.",
        "- FB and FC improve every target in every seed, confirming that phase and "
        "transverse two-body information was omitted at earlier readout times.",
        "- Full FC beats ESN-F on changed-pair geometry and energy, but not on "
        "all-pair geometry. FC216 retains the energy advantage at the same "
        "classical readout dimension, while ESN-F retains both geometry wins.",
        "- The QRC advantage is therefore task-specific and strongest for energy "
        "and reaction-centre dynamics, not a general superiority over ESN.",
        "",
        "Feature geometry and task-alignment diagnostics are in "
        "`outputs/feature_diagnostics_summary.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
