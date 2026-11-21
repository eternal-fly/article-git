"""Exact final-Z/ZZ suffix banks for event-wise reset-channel DTC-QRC."""

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
    G,
    LAYERS,
    OBSERVABLES,
    PROTOCOL_VERSION,
    QRC_BACKEND,
    T_VALUES,
    W,
    qrc_bank_path,
    readout_dim,
    token_cache_path,
)


if str(QRC_BACKEND) not in sys.path:
    sys.path.insert(0, str(QRC_BACKEND))

from gpu_qrc_statevector import (  # noqa: E402
    DTCQRCConfig,
    adapt_token_dim,
    apply_reservoir_layers,
    apply_token_injection,
    build_dtc_diagonal_phase,
    build_observable_signs,
    compute_z_zz_from_state,
    generate_dtc_params,
    real_dtype_from_complex,
    rx_gate,
    set_seed,
    torch_complex_dtype,
)


def adaptive_batch_size(q: int) -> int:
    if q >= 20:
        return 2
    if q >= 16:
        return 8
    if q >= 13:
        return 16
    if q >= 9:
        return 32
    return 64


def refresh_suffix_weights(sequence_length: int, probability: float) -> np.ndarray:
    """Weights for suffix starts 0..T-1 followed by the reset anchor."""
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
    """Return exact reset-channel final Z/ZZ expectations."""
    source = np.asarray(suffixes, dtype=np.float64)
    if source.ndim != 3 or source.shape[0] < 1:
        raise ValueError(source.shape)
    weights = refresh_suffix_weights(source.shape[0], probability)
    mixed = np.tensordot(weights[:-1], source, axes=(0, 0))
    # |0...0> has +1 expectation for every Z and ZZ observable.
    mixed += weights[-1]
    return mixed.astype(np.float32)


def parameter_sha256(parameters: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(parameters):
        value = np.ascontiguousarray(parameters[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _atomic_save_bank(path: Path, suffixes: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        suffixes=np.asarray(suffixes, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        protocol_version=np.asarray(PROTOCOL_VERSION),
        channel_protocol=np.asarray(CHANNEL_PROTOCOL),
        dataset=np.asarray(metadata["dataset"]),
        token_variant=np.asarray(metadata["token_variant"]),
        token_sha256=np.asarray(metadata["token_sha256"]),
        dataset_fingerprint=np.asarray(metadata["dataset_fingerprint"]),
        q=np.int16(metadata["q"]),
        t=np.int16(metadata["t"]),
        seed=np.int32(metadata["seed"]),
        g=np.float32(metadata["g"]),
        W=np.float32(metadata["W"]),
        layers=np.int16(metadata["layers"]),
        observables=np.asarray(OBSERVABLES),
        parameter_sha256=np.asarray(metadata["parameter_sha256"]),
    )
    partial.replace(path)


def load_qrc_bank(
    dataset: str,
    variant: str,
    q: int,
    t: int,
    seed: int,
    *,
    n_samples: int | None = None,
    token_sha256: str | None = None,
) -> np.ndarray:
    path = qrc_bank_path(dataset, variant, q, t, seed)
    with np.load(path, allow_pickle=False) as archive:
        suffixes = archive["suffixes"].astype(np.float32, copy=False)
        identity = {
            "protocol": str(archive["protocol_version"].item()),
            "channel": str(archive["channel_protocol"].item()),
            "dataset": str(archive["dataset"].item()),
            "variant": str(archive["token_variant"].item()),
            "q": int(archive["q"]),
            "t": int(archive["t"]),
            "seed": int(archive["seed"]),
            "g": float(archive["g"]),
            "W": float(archive["W"]),
            "layers": int(archive["layers"]),
            "observables": str(archive["observables"].item()),
            "token_sha256": str(archive["token_sha256"].item()),
        }
    expected = {
        "protocol": PROTOCOL_VERSION,
        "channel": CHANNEL_PROTOCOL,
        "dataset": dataset.upper(),
        "variant": variant,
        "q": int(q),
        "t": int(t),
        "seed": int(seed),
        "g": float(np.float32(G)),
        "W": float(np.float32(W)),
        "layers": LAYERS,
        "observables": OBSERVABLES,
        "token_sha256": token_sha256 or identity["token_sha256"],
    }
    if identity != expected:
        raise ValueError(f"stale QRC bank {path}: {identity} != {expected}")
    expected_shape = (int(t), n_samples or suffixes.shape[1], readout_dim(q))
    if suffixes.shape != expected_shape or not np.isfinite(suffixes).all():
        raise ValueError(f"invalid QRC bank {path}: {suffixes.shape} != {expected_shape}")
    return suffixes


@torch.inference_mode()
def compute_all_t_suffix_banks(
    dataset: str,
    variant: str,
    q: int,
    seed: int,
    *,
    device: str,
    batch_size: int | None = None,
    force: bool = False,
    max_samples: int | None = None,
) -> list[dict]:
    """Compute all requested T banks in one triangular state-evolution pass."""
    dataset = dataset.upper()
    cache_path = token_cache_path(dataset, variant)
    with np.load(cache_path, allow_pickle=False) as archive:
        source = archive["tokens"].astype(np.float32, copy=False)
        token_sha = str(archive["tokens_sha256"].item())
    metadata = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if max_samples is not None:
        source = source[: int(max_samples)]
    source = np.ascontiguousarray(source[:, : max(T_VALUES), : 3 * int(q)], dtype=np.float32)
    pending = []
    for t in T_VALUES:
        path = qrc_bank_path(dataset, variant, q, t, seed)
        if not force and path.exists():
            try:
                load_qrc_bank(
                    dataset,
                    variant,
                    q,
                    t,
                    seed,
                    n_samples=len(source),
                    token_sha256=token_sha,
                )
                continue
            except Exception:
                pass
        pending.append(int(t))
    if not pending:
        return [{"dataset": dataset, "variant": variant, "q": q, "seed": seed, "status": "cached"}]

    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"QRC extraction requires visible CUDA, got {device}")
    torch.cuda.set_device(torch_device)
    q = int(q)
    cfg = DTCQRCConfig(
        num_qubits=q,
        features_per_qubit=3,
        g=G,
        W_scale=W,
        layers_per_injection=LAYERS,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    set_seed(int(seed))
    params = generate_dtc_params(q, seed=int(seed), W_scale=W)
    param_hash = parameter_sha256(params)
    J = torch.from_numpy(params["J"])
    h = torch.from_numpy(params["h"])
    complex_dtype = torch_complex_dtype(cfg.complex_dtype)
    real_dtype = real_dtype_from_complex(complex_dtype)
    rx = rx_gate(cfg.g * math.pi, device=torch_device, complex_dtype=complex_dtype)
    phase = build_dtc_diagonal_phase(q, J, h, torch_device, complex_dtype)
    signs = build_observable_signs(q, device=torch_device, dtype=real_dtype)
    output = {
        t: np.empty((t, len(source), cfg.output_dim), dtype=np.float32) for t in pending
    }
    values = torch.from_numpy(source)
    batch_size = int(batch_size or adaptive_batch_size(q))
    started = time.time()
    max_target = max(pending)
    for suffix_start in range(max_target):
        targets = [t for t in pending if suffix_start < t]
        if not targets:
            continue
        for batch_start in range(0, len(source), batch_size):
            batch_end = min(batch_start + batch_size, len(source))
            x = values[batch_start:batch_end].to(
                device=torch_device, dtype=real_dtype, non_blocking=True
            )
            x = adapt_token_dim(x, cfg.input_dim, cfg.token_dim_policy)
            x = torch.tanh(x)
            state = torch.zeros(
                (len(x), cfg.state_dim), device=torch_device, dtype=complex_dtype
            )
            state[:, 0] = 1.0 + 0.0j
            target_set = set(targets)
            for event in range(suffix_start, max_target):
                state = apply_token_injection(state, x[:, event, :], cfg, complex_dtype)
                state = apply_reservoir_layers(state, cfg, rx, phase)
                end = event + 1
                if end in target_set:
                    features = compute_z_zz_from_state(state, signs)
                    output[end][suffix_start, batch_start:batch_end] = (
                        features.detach().cpu().numpy()
                    )
                    del features
            del state, x
        torch.cuda.empty_cache()
        print(
            f"[{device}] {dataset}/{variant}/Q{q}/seed{seed}: "
            f"suffix-start {suffix_start + 1}/{max_target}",
            flush=True,
        )

    elapsed = time.time() - started
    rows = []
    for t in pending:
        bank = output[t]
        # p=0 must be the uninterrupted full-prefix trajectory.
        if not np.array_equal(mix_suffix_bank(bank, 0.0), bank[0]):
            raise AssertionError("p=0 suffix identity failed")
        bank_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "channel_protocol": CHANNEL_PROTOCOL,
            "dataset": dataset,
            "token_variant": variant,
            "token_schema_version": metadata["schema_version"],
            "token_sha256": token_sha,
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "q": q,
            "t": t,
            "seed": int(seed),
            "g": G,
            "W": W,
            "layers": LAYERS,
            "observables": OBSERVABLES,
            "parameter_sha256": param_hash,
            "device": str(torch_device),
            "device_name": torch.cuda.get_device_name(torch_device),
            "batch_size": batch_size,
            "joint_all_t_elapsed_seconds": elapsed,
            "n_samples": len(source),
        }
        _atomic_save_bank(qrc_bank_path(dataset, variant, q, t, seed), bank, bank_metadata)
        rows.append({**bank_metadata, "status": "computed"})
    return rows

