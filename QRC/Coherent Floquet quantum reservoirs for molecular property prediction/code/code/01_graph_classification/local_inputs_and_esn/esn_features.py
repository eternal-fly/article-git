"""Matched and wide dense ESN features on the full canonical Q-T grid."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from protocol import (
    ESN_INPUT_SCALE,
    ESN_LEAK_RATE,
    ESN_SPECTRAL_RADIUS,
    LAYERS,
    PROTOCOL_VERSION,
    T_VALUES,
    esn_feature_path,
    esn_parameter_path,
    token_cache_path,
)


ESN_PROTOCOL = "dense_gaussian_esn_final_state_width_scan_v1"


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_or_load_recurrent(width: int, seed: int, *, force: bool = False) -> tuple[np.ndarray, dict]:
    """Exact original ESN recurrent draw and exact eigenvalue normalization."""
    width = int(width)
    seed = int(seed)
    path = esn_parameter_path(width, seed)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as archive:
            recurrent = archive["recurrent"].astype(np.float32, copy=False)
            metadata = json.loads(str(archive["metadata_json"].item()))
        if (
            metadata.get("protocol_version") == PROTOCOL_VERSION
            and metadata.get("esn_protocol") == ESN_PROTOCOL
            and metadata.get("recurrent_sha256") == array_sha256(recurrent)
            and recurrent.shape == (width, width)
        ):
            return recurrent, metadata
    rng = np.random.default_rng(seed)
    recurrent = rng.normal(size=(width, width)).astype(np.float32)
    state_after_recurrent = rng.bit_generator.state
    started = time.time()
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent.astype(np.float64)))))
    recurrent *= np.float32(ESN_SPECTRAL_RADIUS / max(radius, 1e-8))
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "esn_protocol": ESN_PROTOCOL,
        "width": width,
        "seed": seed,
        "spectral_radius_target": ESN_SPECTRAL_RADIUS,
        "spectral_radius_before_scaling": radius,
        "normalization": "exact numpy float64 eigenvalues, matching manuscript standard ESN",
        "rng_state_after_recurrent": state_after_recurrent,
        "recurrent_sha256": array_sha256(recurrent),
        "eigendecomposition_seconds": time.time() - started,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez(
        partial,
        recurrent=recurrent,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    partial.replace(path)
    return recurrent, metadata


def _input_parameters(metadata: dict, width: int, input_dim: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    rng.bit_generator.state = metadata["rng_state_after_recurrent"]
    input_weights = (
        rng.normal(size=(int(width), int(input_dim))).astype(np.float32)
        / math.sqrt(max(1, int(input_dim)))
    )
    bias = rng.normal(scale=0.10, size=int(width)).astype(np.float32)
    return input_weights, bias


def load_esn_features(
    dataset: str,
    variant: str,
    q: int,
    t: int,
    width: int,
    seed: int,
    *,
    n_samples: int | None = None,
    token_sha256: str | None = None,
) -> np.ndarray:
    path = esn_feature_path(dataset, variant, q, t, width, seed)
    with np.load(path, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32, copy=False)
        metadata = json.loads(str(archive["metadata_json"].item()))
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "esn_protocol": ESN_PROTOCOL,
        "dataset": dataset.upper(),
        "variant": variant,
        "q": int(q),
        "t": int(t),
        "width": int(width),
        "seed": int(seed),
        "token_sha256": token_sha256 or metadata.get("token_sha256"),
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"stale ESN cache {path}: {actual} != {expected}")
    expected_shape = (n_samples or features.shape[0], int(width))
    if features.shape != expected_shape or not np.isfinite(features).all():
        raise ValueError(f"invalid ESN features {path}: {features.shape} != {expected_shape}")
    return features


@torch.inference_mode()
def compute_esn_all_t(
    dataset: str,
    variant: str,
    q: int,
    width: int,
    seed: int,
    *,
    device: str,
    force: bool = False,
    max_samples: int | None = None,
) -> list[dict]:
    dataset = dataset.upper()
    cache_path = token_cache_path(dataset, variant)
    with np.load(cache_path, allow_pickle=False) as archive:
        values = archive["tokens"].astype(np.float32, copy=False)
        token_sha = str(archive["tokens_sha256"].item())
    if max_samples is not None:
        values = values[: int(max_samples)]
    values = np.ascontiguousarray(values[:, : max(T_VALUES), : 3 * int(q)], dtype=np.float32)
    pending = []
    for t in T_VALUES:
        path = esn_feature_path(dataset, variant, q, t, width, seed)
        if not force and path.exists():
            try:
                load_esn_features(
                    dataset,
                    variant,
                    q,
                    t,
                    width,
                    seed,
                    n_samples=len(values),
                    token_sha256=token_sha,
                )
                continue
            except Exception:
                pass
        pending.append(int(t))
    if not pending:
        return [{"dataset": dataset, "variant": variant, "q": q, "width": width, "seed": seed, "status": "cached"}]

    recurrent, recurrent_metadata = build_or_load_recurrent(width, seed)
    input_weights, bias = _input_parameters(recurrent_metadata, width, 3 * int(q))
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"ESN CUDA device unavailable: {device}")
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    x = torch.tanh(torch.from_numpy(values).to(torch_device))
    recurrent_t = torch.from_numpy(recurrent).to(torch_device)
    input_t = torch.from_numpy(input_weights).to(torch_device)
    bias_t = torch.from_numpy(bias).to(torch_device)
    state = torch.zeros((len(values), int(width)), dtype=torch.float32, device=torch_device)
    snapshots: dict[int, np.ndarray] = {}
    target_set = set(pending)
    started = time.time()
    for event in range(max(pending)):
        injection = ESN_INPUT_SCALE * (x[:, event] @ input_t.T)
        for _ in range(LAYERS):
            proposal = torch.tanh(state @ recurrent_t.T + injection + bias_t)
            state = (1.0 - ESN_LEAK_RATE) * state + ESN_LEAK_RATE * proposal
        t = event + 1
        if t in target_set:
            snapshots[t] = state.detach().cpu().numpy().astype(np.float32, copy=True)
        print(
            f"[{device}] ESN {dataset}/{variant}/Q{q}/width{width}/seed{seed}: event {t}/{max(pending)}",
            flush=True,
        )
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.time() - started
    rows = []
    parameter_hash = hashlib.sha256(
        (recurrent_metadata["recurrent_sha256"] + array_sha256(input_weights) + array_sha256(bias)).encode("ascii")
    ).hexdigest()
    for t in pending:
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "esn_protocol": ESN_PROTOCOL,
            "dataset": dataset,
            "variant": variant,
            "q": int(q),
            "t": int(t),
            "width": int(width),
            "seed": int(seed),
            "token_sha256": token_sha,
            "layers_per_event": LAYERS,
            "spectral_radius": ESN_SPECTRAL_RADIUS,
            "input_scale": ESN_INPUT_SCALE,
            "leak_rate": ESN_LEAK_RATE,
            "readout": "final_ESN_state_direct",
            "parameter_sha256": parameter_hash,
            "device": str(torch_device),
            "all_t_elapsed_seconds": elapsed,
            "n_samples": len(values),
        }
        destination = esn_feature_path(dataset, variant, q, t, width, seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".partial.npz")
        np.savez_compressed(
            partial,
            features=snapshots[t],
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        partial.replace(destination)
        rows.append({**metadata, "status": "computed"})
    return rows

