"""Direct Hamiltonian-setting closed/open DTC-QRC feature extraction.

The input event has exactly ``3Q`` scalar components.  Qubit i receives the
triple (x_i, y_i, z_i) through one exponential
``exp[-i(x_i X + y_i Y + z_i Z)/2]``.  There is no U3 decomposition, tanh, PCA,
learned projection, or micro-time-step expansion in this module.
"""

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
    CHANNEL_PROTOCOL,
    FEATURES_PER_QUBIT,
    G,
    HAMILTONIAN_BACKEND,
    INJECTION_PROTOCOL,
    LAYERS,
    OBSERVABLES,
    PROTOCOL_VERSION,
    W,
    readout_dim,
)


if str(HAMILTONIAN_BACKEND) not in sys.path:
    sys.path.insert(0, str(HAMILTONIAN_BACKEND))

from hamiltonian_dtc_qrc import (  # noqa: E402
    DTCQRCConfig,
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


def adaptive_batch_size(q: int) -> int:
    if q >= 20:
        return 1
    if q >= 18:
        return 2
    if q >= 16:
        return 8
    if q >= 13:
        return 16
    if q >= 9:
        return 32
    return 128


def refresh_suffix_weights(sequence_length: int, probability: float) -> np.ndarray:
    """Exact final-state weights for a reset after every event."""
    t = int(sequence_length)
    p = float(probability)
    if t < 1 or not 0.0 <= p < 1.0:
        raise ValueError((t, p))
    survival = 1.0 - p
    weights = [survival**t]
    weights.extend(p * survival ** (t - start) for start in range(1, t))
    weights.append(p)
    result = np.asarray(weights, dtype=np.float64)
    if not np.isclose(result.sum(), 1.0, atol=1e-12):
        raise AssertionError(result.sum())
    return result


def mix_suffix_bank(suffixes: np.ndarray, probability: float) -> np.ndarray:
    source = np.asarray(suffixes, dtype=np.float64)
    if source.ndim != 3 or source.shape[0] < 1:
        raise ValueError(source.shape)
    weights = refresh_suffix_weights(source.shape[0], probability)
    mixed = np.tensordot(weights[:-1], source, axes=(0, 0))
    # The reset anchor |0...0> has expectation +1 for every Z and ZZ.
    mixed += weights[-1]
    return mixed.astype(np.float32)


def _one_dtc_cycle(
    state: torch.Tensor,
    cfg: DTCQRCConfig,
    rx: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    for wire in range(cfg.num_qubits):
        state = apply_1q_gate(state, rx, wire, cfg.num_qubits)
    return state * phase[None, :]


def _parameter_sha256(parameters: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(parameters):
        value = np.ascontiguousarray(parameters[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def compute_suffix_banks(
    tokens: np.ndarray,
    *,
    q: int,
    t_values: tuple[int, ...],
    seed: int,
    device: str,
    g: float = G,
    W_scale: float = W,
    layers: int = LAYERS,
    interaction_scale: float = 1.0,
    batch_size: int | None = None,
) -> tuple[dict[int, np.ndarray], dict]:
    """Compute final Z+ZZ suffix banks for several prefixes in one pass.

    For target T, output[T] has shape ``[T,N,Q(Q+1)/2]``.  Row s is the
    final feature after evolving events ``s..T-1`` from the reset state.  It is
    sufficient to obtain both p=0 and any fixed open p exactly.
    """
    q = int(q)
    targets = tuple(sorted({int(t) for t in t_values}))
    if not targets or targets[0] < 1:
        raise ValueError(targets)
    source = np.ascontiguousarray(tokens, dtype=np.float32)
    expected_dim = FEATURES_PER_QUBIT * q
    if source.ndim != 3 or source.shape[-1] != expected_dim:
        raise ValueError(f"expected [N,T,{expected_dim}], got {source.shape}")
    if source.shape[1] < targets[-1]:
        raise ValueError(f"tokens expose T={source.shape[1]}, requested {targets[-1]}")

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable for {device}")
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    set_seed(int(seed))
    cfg = DTCQRCConfig(
        num_qubits=q,
        features_per_qubit=FEATURES_PER_QUBIT,
        g=float(g),
        W_scale=float(W_scale),
        layers_per_injection=int(layers),
        complex_dtype="complex64",
    )
    if not 0.0 <= float(interaction_scale) <= 1.0:
        raise ValueError(f"interaction_scale must lie in [0,1], got {interaction_scale}")
    params = generate_dtc_params(q, int(seed), float(W_scale))
    params = dict(params)
    params["J"] = np.asarray(
        params["J"] * np.float32(interaction_scale), dtype=np.float32
    )
    J = torch.from_numpy(params["J"])
    h = torch.from_numpy(params["h"])
    complex_dtype = _complex_dtype(cfg.complex_dtype)
    real_dtype = _real_dtype(complex_dtype)
    signs = build_observable_signs(q, torch_device, real_dtype)
    rx = rx_gate(cfg.g * math.pi, torch_device, complex_dtype)
    phase = build_dtc_diagonal_phase(cfg, J, h, torch_device, complex_dtype)
    width = readout_dim(q)
    output = {
        t: np.empty((t, len(source), width), dtype=np.float32) for t in targets
    }
    cpu_tokens = torch.from_numpy(source[:, : targets[-1]])
    batch_size = int(batch_size or adaptive_batch_size(q))
    started = time.time()
    target_set = set(targets)

    for suffix_start in range(targets[-1]):
        active_targets = {t for t in target_set if suffix_start < t}
        if not active_targets:
            continue
        for batch_start in range(0, len(source), batch_size):
            batch_end = min(batch_start + batch_size, len(source))
            values = cpu_tokens[batch_start:batch_end].to(
                device=torch_device, dtype=real_dtype, non_blocking=True
            )
            state = torch.zeros(
                (len(values), cfg.state_dim),
                device=torch_device,
                dtype=complex_dtype,
            )
            state[:, 0] = 1.0
            for event in range(suffix_start, targets[-1]):
                # Direct Hamiltonian setting: no tanh and no Euler/U3 sequence.
                state = apply_hamiltonian_injection(
                    state, values[:, event], cfg, complex_dtype
                )
                for _ in range(cfg.layers_per_injection):
                    state = _one_dtc_cycle(state, cfg, rx, phase)
                end = event + 1
                if end in active_targets:
                    probabilities = torch.abs(state).square().real
                    output[end][suffix_start, batch_start:batch_end] = (
                        probabilities @ signs.T
                    ).float().cpu().numpy()
            del state, values
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    for t, bank in output.items():
        if not np.array_equal(mix_suffix_bank(bank, 0.0), bank[0]):
            raise AssertionError(f"p=0 identity failed at T={t}")
        if not np.isfinite(bank).all():
            raise AssertionError(f"non-finite feature at T={t}")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "injection_protocol": INJECTION_PROTOCOL,
        "channel_protocol": CHANNEL_PROTOCOL,
        "observables": OBSERVABLES,
        "q": q,
        "t_values": list(targets),
        "seed": int(seed),
        "g": float(g),
        "W": float(W_scale),
        "layers": int(layers),
        "interaction_scale": float(interaction_scale),
        "feature_dim": width,
        "input_dim_per_event": expected_dim,
        "input_preprocessing_in_reservoir": "none",
        "micro_time_steps": 1,
        "parameter_sha256": _parameter_sha256(params),
        "device": str(torch_device),
        "batch_size": batch_size,
        "elapsed_seconds": time.time() - started,
    }
    return output, metadata


def save_suffix_bank(path: str | Path, bank: np.ndarray, metadata: dict, t: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, "t": int(t)}
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        suffixes=np.asarray(bank, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True)),
        protocol_version=np.asarray(PROTOCOL_VERSION),
        injection_protocol=np.asarray(INJECTION_PROTOCOL),
        q=np.int16(metadata["q"]),
        t=np.int16(t),
        seed=np.int32(metadata["seed"]),
        g=np.float32(metadata["g"]),
        W=np.float32(metadata["W"]),
    )
    partial.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def load_suffix_bank(
    path: str | Path,
    *,
    q: int,
    t: int,
    seed: int,
    g: float,
    W_scale: float,
    n_samples: int | None = None,
    expected_metadata: dict | None = None,
) -> tuple[np.ndarray, dict]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        bank = archive["suffixes"].astype(np.float32, copy=False)
        metadata = json.loads(str(archive["metadata_json"].item()))
        identity = {
            "protocol_version": str(archive["protocol_version"].item()),
            "injection_protocol": str(archive["injection_protocol"].item()),
            "q": int(archive["q"]),
            "t": int(archive["t"]),
            "seed": int(archive["seed"]),
            "g": float(archive["g"]),
            "W": float(archive["W"]),
        }
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "injection_protocol": INJECTION_PROTOCOL,
        "q": int(q),
        "t": int(t),
        "seed": int(seed),
        "g": float(np.float32(g)),
        "W": float(np.float32(W_scale)),
    }
    if identity != expected:
        raise ValueError(f"stale Hamiltonian bank {path}: {identity} != {expected}")
    if expected_metadata:
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise ValueError(f"metadata mismatch in {path}: {mismatch}")
    expected_shape = (int(t), n_samples or bank.shape[1], readout_dim(q))
    if bank.shape != expected_shape or not np.isfinite(bank).all():
        raise ValueError(f"invalid Hamiltonian bank {path}: {bank.shape} != {expected_shape}")
    return bank, metadata
