"""Three-axis classical reservoir baselines shared across FINAL modules.

``crc3_z_zz_matched`` preserves all three coordinates assigned to every
QRC qubit during the recurrent dynamics, but exposes only the classical
z coordinate and all z-z pair products.  It therefore has exactly the same
readout width and one-/two-body form as the standard QRC Z/ZZ measurement,
without PCA or any label-dependent feature selection.
"""

from __future__ import annotations

import math

import numpy as np
import torch


CRC3_Z_ZZ_METHOD = "crc3_z_zz_matched"
CRC3_Z_ZZ_PROTOCOL = "crc3_z_zz_matched_v1"
CRC3_Z_AXIS = 2


def crc3_z_zz_dim(q: int) -> int:
    q = int(q)
    return q + q * (q - 1) // 2


def crc3_full_dim(q: int) -> int:
    q = int(q)
    return 3 * q + 9 * q * (q - 1) // 2


def _orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(int(n), int(n))))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return (q * signs).astype(np.float32)


def compute_crc3_state(
    tokens: np.ndarray,
    q: int,
    layers: int,
    seed: int,
    device: str | torch.device | None = "cpu",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the final ``[sample, Q, xyz]`` state of the fixed CRC3 map.

    The equations intentionally match the CRC3 strong control introduced in
    FINAL/8.  Input and reservoir matrices vary with the paired reservoir
    seed, while no label or split information enters feature construction.
    """

    q = int(q)
    layers = int(layers)
    values = np.ascontiguousarray(tokens, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3 * q:
        raise ValueError(f"CRC3 expects tokens [N,T,{3*q}], got {values.shape}")
    if layers < 1:
        raise ValueError("layers must be positive")

    rng = np.random.default_rng(int(seed))
    qubit_mix = _orthogonal(q, rng)
    axis_mix = _orthogonal(3, rng)
    input_mix = _orthogonal(3, rng)
    bias = rng.normal(scale=1 / math.sqrt(max(1, q)), size=(q, 3)).astype(np.float32)
    requested = "cpu" if device is None else str(device)
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    compute_device = torch.device(requested)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        compute_device = torch.device("cpu")

    with torch.no_grad():
        x = torch.tanh(torch.from_numpy(values).to(compute_device)).reshape(len(values), -1, q, 3)
        qubit_mix_t = torch.from_numpy(qubit_mix).to(compute_device)
        axis_mix_t = torch.from_numpy(axis_mix).to(compute_device)
        input_mix_t = torch.from_numpy(input_mix).to(compute_device)
        bias_t = torch.from_numpy(bias).to(compute_device)
        state_t = torch.zeros((len(values), q, 3), dtype=torch.float32, device=compute_device)
        for time_index in range(x.shape[1]):
            injection = x[:, time_index] @ input_mix_t
            state_t = torch.tanh(0.55 * state_t + 0.75 * injection)
            for _ in range(layers):
                recurrent = torch.einsum("ij,nja,ab->nib", qubit_mix_t, state_t, axis_mix_t)
                state_t = torch.tanh(0.85 * recurrent + 0.65 * injection + bias_t)
        if compute_device.type == "cuda":
            torch.cuda.synchronize(compute_device)
        state = state_t.cpu().numpy().astype(np.float32, copy=False)
    params = {
        "qubit_mix": qubit_mix,
        "axis_mix": axis_mix,
        "input_mix": input_mix,
        "bias": bias,
        "input_state_scale": np.asarray(0.55, dtype=np.float32),
        "input_inject_scale": np.asarray(0.75, dtype=np.float32),
        "recurrent_scale": np.asarray(0.85, dtype=np.float32),
        "recurrent_inject_scale": np.asarray(0.65, dtype=np.float32),
        "compute_device": np.asarray(str(compute_device)),
    }
    return state.astype(np.float32, copy=False), params


def z_zz_readout(state: np.ndarray, axis: int = CRC3_Z_AXIS) -> np.ndarray:
    """Direct ``z_i`` plus ``z_i*z_j`` readout; never applies PCA."""

    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 3 or state.shape[2] != 3:
        raise ValueError(f"CRC3 state must have shape [N,Q,3], got {state.shape}")
    q = int(state.shape[1])
    z = state[:, :, int(axis)]
    columns = [z]
    if q >= 2:
        columns.append(
            np.stack([z[:, i] * z[:, j] for i in range(q) for j in range(i + 1, q)], axis=1)
        )
    output = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
    expected = crc3_z_zz_dim(q)
    if output.shape != (len(state), expected):
        raise RuntimeError(f"Unexpected CRC3 z/zz shape {output.shape}; expected {(len(state), expected)}")
    return output


def full_xyz_pair_readout(state: np.ndarray) -> np.ndarray:
    """Original CRC3 full xyz plus all 3x3 inter-node outer products."""

    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 3 or state.shape[2] != 3:
        raise ValueError(f"CRC3 state must have shape [N,Q,3], got {state.shape}")
    q = int(state.shape[1])
    columns = [state.reshape(len(state), 3 * q)]
    for i in range(q):
        for j in range(i + 1, q):
            columns.append(
                (state[:, i, :, None] * state[:, j, None, :]).reshape(len(state), 9)
            )
    output = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
    expected = crc3_full_dim(q)
    if output.shape != (len(state), expected):
        raise RuntimeError(f"Unexpected CRC3 full shape {output.shape}; expected {(len(state), expected)}")
    return output


def compute_crc3_z_zz_features(
    tokens: np.ndarray,
    q: int,
    layers: int,
    seed: int,
    device: str | torch.device | None = "cpu",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    state, params = compute_crc3_state(tokens, q=q, layers=layers, seed=seed, device=device)
    return z_zz_readout(state), params
