"""Statevector feature extraction for coherence and interaction ablations."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from protocol import (
    BACE_Q,
    BACE_T,
    G,
    HAMILTONIAN_ROOT,
    HOPPING_ANGLE_SCALE,
    HOPPING_Q,
    HOPPING_ROOT,
    HOPPING_T,
    INJECTION_PROTOCOL,
    LAYERS,
    OBSERVABLE_FAMILY,
    PROTOCOL_VERSION,
    VARIANT_BY_NAME,
    W,
    bace_token_path,
    feature_path,
)


if str(HAMILTONIAN_ROOT) not in sys.path:
    sys.path.insert(0, str(HAMILTONIAN_ROOT))

from hamiltonian_dtc_qrc import (  # noqa: E402
    DTCQRCConfig,
    _basis_z_signs,
    _complex_dtype,
    _real_dtype,
    apply_1q_gate,
    apply_hamiltonian_injection,
    build_dtc_diagonal_phase,
    build_observable_signs,
    generate_dtc_params,
    rx_gate,
    set_seed,
)


def _sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _cycle(
    state: torch.Tensor,
    cfg: DTCQRCConfig,
    rx: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    for wire in range(cfg.num_qubits):
        state = apply_1q_gate(state, rx, wire, cfg.num_qubits)
    return state * phase[None, :]


def _apply_dephasing_trajectory(
    state: torch.Tensor,
    z_basis: torch.Tensor,
    strength: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one Pauli-Z trajectory for the event-wise dephasing channel.

    A Z flip with probability lambda/2 multiplies each single-qubit coherence
    by 1-lambda after ensemble averaging. At lambda=1 this is complete Z-basis
    dephasing. The channel is independently sampled for every sample/qubit.
    """
    if strength <= 0.0:
        return state
    flips = torch.rand(
        (len(state), z_basis.shape[1]),
        generator=generator,
        device=state.device,
    ) < (float(strength) / 2.0)
    for wire in range(z_basis.shape[1]):
        state = torch.where(
            flips[:, wire, None],
            state * z_basis[:, wire][None, :],
            state,
        )
    return state


def connected_zz_diagnostics(features: np.ndarray, q: int) -> tuple[float, float]:
    values = np.asarray(features, dtype=np.float64)
    width = q * (q + 1) // 2
    if values.ndim != 2 or values.shape[1] < width:
        raise ValueError(values.shape)
    final = values[:, :width]
    z = final[:, :q]
    zz = final[:, q:]
    products = np.stack(
        [z[:, i] * z[:, j] for i in range(q) for j in range(i + 1, q)],
        axis=1,
    )
    connected = zz - products
    return float(np.sqrt(np.mean(connected**2))), float(np.max(np.abs(connected)))


@torch.inference_mode()
def simulate_sequence_features(
    tokens: np.ndarray,
    *,
    q: int,
    seed: int,
    j_scale: float,
    dephasing_strength: float,
    trajectories: int,
    device: str,
    batch_size: int,
    angle_scale: float,
    tm3: bool,
    suffix_start: int,
) -> tuple[np.ndarray, dict]:
    source = np.ascontiguousarray(tokens, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 3 * int(q):
        raise ValueError(f"expected [N,T,{3*q}], received {source.shape}")
    if not 0.0 <= float(dephasing_strength) <= 1.0:
        raise ValueError(dephasing_strength)
    if int(trajectories) < 1:
        raise ValueError(trajectories)

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for {device}")
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    set_seed(int(seed))
    cfg = DTCQRCConfig(
        num_qubits=int(q),
        features_per_qubit=3,
        g=G,
        W_scale=W,
        layers_per_injection=LAYERS,
        complex_dtype="complex64",
    )
    params = generate_dtc_params(q, int(seed), W)
    scaled_j = np.asarray(params["J"] * np.float32(j_scale), dtype=np.float32)
    h = np.asarray(params["h"], dtype=np.float32)
    complex_dtype = _complex_dtype(cfg.complex_dtype)
    real_dtype = _real_dtype(complex_dtype)
    rx = rx_gate(cfg.g * math.pi, torch_device, complex_dtype)
    phase = build_dtc_diagonal_phase(
        cfg,
        torch.from_numpy(scaled_j),
        torch.from_numpy(h),
        torch_device,
        complex_dtype,
    )
    observable_signs = build_observable_signs(q, torch_device, real_dtype)
    z_basis = _basis_z_signs(q, torch_device, real_dtype)
    width = cfg.output_dim
    snapshots = 3 if tm3 else 1
    output = np.empty((len(source), snapshots * width), dtype=np.float32)
    cpu_tokens = torch.from_numpy(source * np.float32(angle_scale))
    started = time.perf_counter()

    for batch_start in range(0, len(source), int(batch_size)):
        batch_end = min(batch_start + int(batch_size), len(source))
        values = cpu_tokens[batch_start:batch_end].to(
            torch_device, dtype=real_dtype, non_blocking=True
        )
        accumulated = torch.zeros(
            (len(values), snapshots * width),
            device=torch_device,
            dtype=real_dtype,
        )
        trajectory_count = 1 if dephasing_strength == 0 else int(trajectories)
        for trajectory in range(trajectory_count):
            state = torch.zeros(
                (len(values), cfg.state_dim),
                device=torch_device,
                dtype=complex_dtype,
            )
            state[:, 0] = 1.0
            random_seed = (
                104729 * (int(seed) + 1)
                + 1009 * (int(suffix_start) + 1)
                + 37 * (batch_start + 1)
                + trajectory
            )
            generator = torch.Generator(device=torch_device).manual_seed(random_seed)
            for event in range(values.shape[1]):
                state = apply_hamiltonian_injection(
                    state, values[:, event], cfg, complex_dtype
                )
                for _ in range(cfg.layers_per_injection):
                    state = _cycle(state, cfg, rx, phase)
                state = _apply_dephasing_trajectory(
                    state, z_basis, dephasing_strength, generator
                )
            observed = []
            probabilities = torch.abs(state).square().real
            observed.append(probabilities @ observable_signs.T)
            if tm3:
                for _ in range(2):
                    state = _cycle(state, cfg, rx, phase)
                    probabilities = torch.abs(state).square().real
                    observed.append(probabilities @ observable_signs.T)
            accumulated += torch.stack(observed, dim=1).reshape(len(values), -1)
        output[batch_start:batch_end] = (
            accumulated / float(trajectory_count)
        ).float().cpu().numpy()
        del values, accumulated
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    rms, maximum = connected_zz_diagnostics(output, q)
    return output, {
        "elapsed_seconds": time.perf_counter() - started,
        "connected_zz_rms": rms,
        "connected_zz_max_abs": maximum,
        "J": scaled_j.tolist(),
        "h": h.tolist(),
    }


def _load_task_tokens(task: str) -> tuple[np.ndarray, dict]:
    if task == "bace":
        path = bace_token_path()
        with np.load(path, allow_pickle=False) as package:
            tokens = package["tokens"].astype(np.float32)[:, :BACE_T, : 3 * BACE_Q]
            token_hash = str(package["tokens_sha256"].item())
        return np.ascontiguousarray(tokens), {
            "q": BACE_Q,
            "t": BACE_T,
            "angle_scale": 1.0,
            "tm3": False,
            "batch_size": 32,
            "source": str(path),
            "token_sha256": token_hash,
        }
    if task == "hopping":
        processed = HOPPING_ROOT / "data" / "processed" / "a01_windows.npz"
        with np.load(processed, allow_pickle=False) as package:
            tokens = package["tokens"].astype(np.float32)
        if tokens.shape[1:] != (HOPPING_T, 3 * HOPPING_Q):
            raise ValueError(tokens.shape)
        return np.ascontiguousarray(tokens), {
            "q": HOPPING_Q,
            "t": HOPPING_T,
            "angle_scale": HOPPING_ANGLE_SCALE,
            "tm3": True,
            "batch_size": 64,
            "source": str(processed),
            "token_sha256": _sha256(tokens),
        }
    raise ValueError(task)


def compute_suffix_bank(
    task: str,
    variant_name: str,
    seed: int,
    device: str,
    *,
    force: bool = False,
    batch_size: int | None = None,
) -> Path:
    if variant_name == "coherent":
        raise ValueError("coherent banks are reused from the completed main protocols")
    variant = VARIANT_BY_NAME[variant_name]
    destination = feature_path(task, variant_name, seed)
    if destination.exists() and not force:
        return destination
    tokens, task_meta = _load_task_tokens(task)
    banks, rms_values, max_values, elapsed = [], [], [], 0.0
    for suffix_start in range(task_meta["t"]):
        features, diagnostics = simulate_sequence_features(
            tokens[:, suffix_start:],
            q=task_meta["q"],
            seed=int(seed),
            j_scale=variant.j_scale,
            dephasing_strength=variant.dephasing_strength,
            trajectories=variant.trajectories,
            device=device,
            batch_size=int(batch_size or task_meta["batch_size"]),
            angle_scale=task_meta["angle_scale"],
            tm3=task_meta["tm3"],
            suffix_start=suffix_start,
        )
        banks.append(features)
        rms_values.append(diagnostics["connected_zz_rms"])
        max_values.append(diagnostics["connected_zz_max_abs"])
        elapsed += diagnostics["elapsed_seconds"]
        print(
            f"[{task} {variant_name} seed={seed} {device}] "
            f"suffix {suffix_start + 1}/{task_meta['t']}",
            flush=True,
        )
    bank = np.stack(banks).astype(np.float32)
    expected_width = task_meta["q"] * (task_meta["q"] + 1) // 2
    expected = (
        task_meta["t"], len(tokens), expected_width * (3 if task_meta["tm3"] else 1)
    )
    if bank.shape != expected or not np.isfinite(bank).all():
        raise RuntimeError(f"invalid bank {bank.shape} != {expected}")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "task": task,
        "variant": variant.name,
        "j_scale": variant.j_scale,
        "dephasing_strength": variant.dephasing_strength,
        "dephasing_trajectories": variant.trajectories,
        "seed": int(seed),
        "q": task_meta["q"],
        "t": task_meta["t"],
        "g": G,
        "W": W,
        "layers": LAYERS,
        "angle_scale": task_meta["angle_scale"],
        "readout": "post-input TM3" if task_meta["tm3"] else "final",
        "observable_family": OBSERVABLE_FAMILY,
        "injection_protocol": INJECTION_PROTOCOL,
        "source": task_meta["source"],
        "token_sha256": task_meta["token_sha256"],
        "elapsed_seconds": elapsed,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        suffixes=bank,
        connected_zz_rms_by_suffix=np.asarray(rms_values, dtype=np.float64),
        connected_zz_max_abs_by_suffix=np.asarray(max_values, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        J=np.asarray(diagnostics["J"], dtype=np.float32),
        h=np.asarray(diagnostics["h"], dtype=np.float32),
    )
    partial.replace(destination)
    destination.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return destination

