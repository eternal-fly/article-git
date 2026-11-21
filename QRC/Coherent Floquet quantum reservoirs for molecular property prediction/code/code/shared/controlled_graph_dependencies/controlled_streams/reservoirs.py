"""Reservoir feature maps for controlled graph-event experiments.

QRC uses the exact GPU statevector implementation from FINAL/2_linear_readout.
The scalar CRC reproduces that module's recurrence.  ESN is a stronger, simple
classical nonlinear control with an orthogonal recurrent matrix and a state width
equal to the QRC Z+ZZ readout width.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


FINAL_ROOT = Path(__file__).resolve().parents[2]
LINEAR_ROOT = FINAL_ROOT / "2_linear_readout"
if str(LINEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(LINEAR_ROOT))

from core.gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu  # noqa: E402
from core.crc3_baseline import (  # noqa: E402
    CRC3_Z_AXIS,
    CRC3_Z_ZZ_METHOD,
    CRC3_Z_ZZ_PROTOCOL,
    compute_crc3_z_zz_features,
)
from core.esn_baseline import (  # noqa: E402
    ESN_STANDARD_METHOD,
    ESN_STANDARD_PROTOCOL,
    compute_esn_standard_features,
)


def readout_dim(q: int) -> int:
    q = int(q)
    return q + q * (q - 1) // 2


def resolve_device(value: str | torch.device | None) -> torch.device:
    raw = "auto" if value is None else str(value).lower()
    if raw in {"", "auto", "cuda", "gpu"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


@torch.no_grad()
def qrc_features(
    tokens: np.ndarray,
    q: int,
    seed: int,
    device: str | torch.device,
    *,
    g: float = 0.84,
    W: float = 0.60,
    layers: int = 3,
    batch_size: int | None = None,
    progress: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute FINAL-compatible Z+all-ZZ QRC features."""

    q = int(q)
    cfg = DTCQRCConfig(
        num_qubits=q,
        features_per_qubit=3,
        g=float(g),
        W_scale=float(W),
        layers_per_injection=int(layers),
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    if batch_size is None:
        if q <= 8:
            batch_size = 256
        elif q <= 12:
            batch_size = 96
        elif q <= 16:
            batch_size = 24
        else:
            batch_size = 4
    features, params = compute_qrc_features_gpu(
        torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.float32)),
        cfg,
        seed=int(seed),
        batch_size=int(batch_size),
        device=resolve_device(device),
        progress=bool(progress),
    )
    metadata = {
        "type": "FINAL_DTC_QRC_statevector",
        "q": q,
        "g": float(g),
        "W": float(W),
        "layers": int(layers),
        "readout": "Z+all_ZZ",
        "preprocessing": "tanh_before_U3",
        "device": str(resolve_device(device)),
        "J": params["J"].tolist(),
        "h": params["h"].tolist(),
    }
    return features.numpy().astype(np.float32), metadata


@torch.no_grad()
def crc_features(
    tokens: np.ndarray,
    q: int,
    seed: int,
    device: str | torch.device = "auto",
    *,
    layers: int = 3,
    state_scale: float = 0.85,
    inject_scale: float = 0.75,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact scalar-Q CRC recurrence used in FINAL/2_linear_readout."""

    q = int(q)
    compute_device = resolve_device(device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    encoder_cpu = torch.randn((q, 3), generator=generator) / math.sqrt(3.0)
    enc_bias_cpu = torch.randn(q, generator=generator) / math.sqrt(max(1, q))
    W_state_cpu = torch.randn((q, q), generator=generator) / math.sqrt(max(1, q))
    W_inj_cpu = torch.randn((q, q), generator=generator) / math.sqrt(max(1, q))
    bias_cpu = torch.randn(q, generator=generator) / math.sqrt(max(1, q))
    encoder = encoder_cpu.to(compute_device)
    enc_bias = enc_bias_cpu.to(compute_device)
    W_state = W_state_cpu.to(compute_device)
    W_inj = W_inj_cpu.to(compute_device)
    bias = bias_cpu.to(compute_device)
    x = torch.tanh(torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.float32)).to(compute_device))
    x = x.reshape(len(tokens), -1, q, 3)
    injections = torch.sin(torch.einsum("ntqf,qf->ntq", x, encoder) + enc_bias)
    state = torch.zeros((len(tokens), q), dtype=torch.float32, device=compute_device)
    for time_index in range(injections.shape[1]):
        inj = injections[:, time_index]
        state = torch.tanh(float(state_scale) * state + float(inject_scale) * inj)
        for _ in range(int(layers)):
            state = torch.tanh(
                float(state_scale) * (state @ W_state.T)
                + float(inject_scale) * (inj @ W_inj.T)
                + bias
            )
    columns = [state]
    if q >= 2:
        columns.append(
            torch.stack([state[:, i] * state[:, j] for i in range(q) for j in range(i + 1, q)], dim=1)
        )
    if compute_device.type == "cuda":
        torch.cuda.synchronize(compute_device)
    features = torch.cat(columns, dim=1).cpu().numpy().astype(np.float32)
    return features, {
        "type": "FINAL_scalar_Q_CRC",
        "q": q,
        "layers": int(layers),
        "state_scale": float(state_scale),
        "inject_scale": float(inject_scale),
        "readout": "state+all_pair_products",
        "preprocessing": "tanh_then_sine_random_encoder",
        "device": str(compute_device),
    }


def crc3_z_zz_matched_features(
    tokens: np.ndarray,
    q: int,
    seed: int,
    device: str | torch.device = "auto",
    *,
    layers: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Three-axis CRC with a direct, no-PCA z/zz readout.

    The shared implementation uses the requested accelerator when available.
    Its internal state has shape ``[N,Q,3]`` and only the fixed z coordinate plus pairwise
    z-products are exposed, so the output width exactly equals QRC Z+ZZ.
    """

    q = int(q)
    features, params = compute_crc3_z_zz_features(
        np.ascontiguousarray(tokens, dtype=np.float32),
        q=q,
        layers=int(layers),
        seed=int(seed),
        device=device,
    )
    metadata: dict[str, Any] = {
        "type": CRC3_Z_ZZ_METHOD,
        "protocol": CRC3_Z_ZZ_PROTOCOL,
        "q": q,
        "internal_state_dim": 3 * q,
        "layers": int(layers),
        "readout": "fixed_z+all_z_z_products",
        "readout_axis": int(CRC3_Z_AXIS),
        "preprocessing": "tanh; three coordinates retained throughout recurrence",
        "transform": "none",
        "pca": False,
        "requested_device": str(device),
    }
    metadata.update(params)
    return features, metadata


@torch.no_grad()
def esn_features(
    tokens: np.ndarray,
    q: int,
    seed: int,
    device: str | torch.device = "auto",
    *,
    layers: int = 3,
    spectral_radius: float = 0.90,
    input_scale: float = 0.75,
    leak_rate: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dimension-aligned orthogonal echo-state network control."""

    q = int(q)
    state_dim = readout_dim(q)
    input_dim = 3 * q
    rng = np.random.default_rng(int(seed))
    gaussian = rng.standard_normal((state_dim, state_dim), dtype=np.float32)
    orthogonal, _ = np.linalg.qr(gaussian)
    recurrent = (float(spectral_radius) * orthogonal.astype(np.float32))
    W_in = (rng.standard_normal((state_dim, input_dim), dtype=np.float32) / math.sqrt(input_dim))
    bias = (rng.standard_normal(state_dim, dtype=np.float32) / math.sqrt(state_dim))
    compute_device = resolve_device(device)
    recurrent_t = torch.from_numpy(recurrent).to(compute_device)
    W_in_t = torch.from_numpy(W_in).to(compute_device)
    bias_t = torch.from_numpy(bias).to(compute_device)
    x = torch.tanh(torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.float32)).to(compute_device))
    state = torch.zeros((len(tokens), state_dim), dtype=torch.float32, device=compute_device)
    alpha = float(leak_rate)
    for time_index in range(x.shape[1]):
        injection = float(input_scale) * (x[:, time_index] @ W_in_t.T)
        for _ in range(int(layers)):
            proposal = torch.tanh(state @ recurrent_t.T + injection + bias_t)
            state = (1.0 - alpha) * state + alpha * proposal
    if compute_device.type == "cuda":
        torch.cuda.synchronize(compute_device)
    return state.cpu().numpy().astype(np.float32), {
        "type": "orthogonal_ESN",
        "state_dim": state_dim,
        "layers": int(layers),
        "spectral_radius": float(spectral_radius),
        "input_scale": float(input_scale),
        "leak_rate": float(leak_rate),
        "device": str(compute_device),
    }


def esn_standard_features(
    tokens: np.ndarray,
    q: int,
    seed: int,
    device: str | torch.device = "auto",
    *,
    layers: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    features, params = compute_esn_standard_features(
        tokens, q=q, layers=layers, seed=seed, device=device
    )
    return features, {
        "type": ESN_STANDARD_METHOD,
        "protocol": ESN_STANDARD_PROTOCOL,
        "q": int(q),
        "layers": int(layers),
        "readout": "final_ESN_state_direct",
        "transform": "none",
        "pca": False,
        **params,
    }
