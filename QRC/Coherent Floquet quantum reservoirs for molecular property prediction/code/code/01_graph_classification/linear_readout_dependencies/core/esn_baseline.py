"""Shared standard echo-state-network baseline for FINAL experiments."""

from __future__ import annotations

import math

import numpy as np
import torch


ESN_STANDARD_METHOD = "esn_standard"
ESN_STANDARD_PROTOCOL = "esn_standard_dim_matched_v1"


def esn_standard_dim(q: int) -> int:
    q = int(q)
    return q + q * (q - 1) // 2


def _resolve_device(device: str | torch.device | None) -> torch.device:
    requested = "auto" if device is None else str(device).lower()
    if requested in {"", "auto", "cuda", "gpu"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


@torch.no_grad()
def compute_esn_standard_features(
    tokens: np.ndarray,
    q: int,
    layers: int,
    seed: int,
    device: str | torch.device | None = "cpu",
    spectral_radius: float = 0.90,
    input_scale: float = 0.75,
    leak_rate: float = 0.65,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return a textbook fixed ESN state with QRC-matched output width.

    The reservoir has ``D=Q(Q+1)/2`` tanh nodes and exposes that state directly.
    Only the final affine classifier is trained.  The recurrent matrix is a
    seeded Gaussian matrix normalized to the requested spectral radius; no PCA,
    labels, or train/validation/test split enters this feature map.
    """

    q = int(q)
    layers = int(layers)
    values = np.ascontiguousarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 * q:
        raise ValueError(f"Standard ESN expects tokens [N,T,{3*q}], got {values.shape}")
    if layers < 1:
        raise ValueError("layers must be positive")
    if not (0.0 < float(spectral_radius) < 1.0 + 1e-12):
        raise ValueError("spectral_radius must lie in (0,1]")
    if not (0.0 < float(leak_rate) <= 1.0):
        raise ValueError("leak_rate must lie in (0,1]")

    state_dim = esn_standard_dim(q)
    input_dim = 3 * q
    rng = np.random.default_rng(int(seed))
    recurrent = rng.normal(size=(state_dim, state_dim)).astype(np.float32)
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent.astype(np.float64)))))
    recurrent *= np.float32(float(spectral_radius) / max(radius, 1e-8))
    input_weights = (
        rng.normal(size=(state_dim, input_dim)).astype(np.float32) / math.sqrt(max(1, input_dim))
    )
    bias = rng.normal(scale=0.10, size=state_dim).astype(np.float32)

    compute_device = _resolve_device(device)
    x = torch.tanh(torch.from_numpy(values).to(compute_device))
    recurrent_t = torch.from_numpy(recurrent).to(compute_device)
    input_weights_t = torch.from_numpy(input_weights).to(compute_device)
    bias_t = torch.from_numpy(bias).to(compute_device)
    state = torch.zeros((len(values), state_dim), dtype=torch.float32, device=compute_device)
    alpha = float(leak_rate)
    for time_index in range(x.shape[1]):
        injection = float(input_scale) * (x[:, time_index] @ input_weights_t.T)
        for _ in range(layers):
            proposal = torch.tanh(state @ recurrent_t.T + injection + bias_t)
            state = (1.0 - alpha) * state + alpha * proposal
    if compute_device.type == "cuda":
        torch.cuda.synchronize(compute_device)
    features = state.cpu().numpy().astype(np.float32, copy=False)
    if features.shape != (len(values), state_dim) or not np.isfinite(features).all():
        raise RuntimeError(f"Invalid standard ESN output shape/values: {features.shape}")
    return features, {
        "recurrent": recurrent,
        "input_weights": input_weights,
        "bias": bias,
        "spectral_radius": np.asarray(float(spectral_radius), dtype=np.float32),
        "input_scale": np.asarray(float(input_scale), dtype=np.float32),
        "leak_rate": np.asarray(float(leak_rate), dtype=np.float32),
        "compute_device": np.asarray(str(compute_device)),
    }
