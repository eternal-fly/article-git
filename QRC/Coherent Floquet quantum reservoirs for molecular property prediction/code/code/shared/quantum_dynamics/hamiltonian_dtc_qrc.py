"""Batched statevector DTC-QRC with single-qubit Hamiltonian input modulation.

The DTC reservoir matches BFS-QRC/FINAL/2_linear_readout/core/gpu_qrc_statevector.py:
fixed J_i ~ U(-0.9, 0.9), h_i ~ U(-W, W), all-qubit RX(g*pi), nearest-neighbor
ZZ plus onsite Z phase, repeated L times after every event.  Only the old tanh+U3
input gate is replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np
import torch


@dataclass(frozen=True)
class DTCQRCConfig:
    num_qubits: int = 20
    features_per_qubit: int = 3
    g: float = 0.84
    W_scale: float = 0.60
    layers_per_injection: int = 3
    complex_dtype: str = "complex64"

    @property
    def input_dim(self) -> int:
        return self.num_qubits * self.features_per_qubit

    @property
    def state_dim(self) -> int:
        return 1 << self.num_qubits

    @property
    def output_dim(self) -> int:
        return self.num_qubits + self.num_qubits * (self.num_qubits - 1) // 2


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def generate_dtc_params(num_qubits: int, seed: int, W_scale: float) -> dict[str, np.ndarray]:
    """Unchanged BFS-QRC J/h generation."""
    rng = np.random.default_rng(int(seed))
    J = rng.uniform(-0.9, 0.9, size=int(num_qubits) - 1).astype(np.float32)
    h = (rng.uniform(-1.0, 1.0, size=int(num_qubits)) * float(W_scale)).astype(np.float32)
    return {"J": J, "h": h, "seed": np.asarray([int(seed)], dtype=np.int64)}


def _complex_dtype(name: str) -> torch.dtype:
    if name == "complex64":
        return torch.complex64
    if name == "complex128":
        return torch.complex128
    raise ValueError(name)


def _real_dtype(complex_dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if complex_dtype == torch.complex64 else torch.float64


def _basis_z_signs(num_qubits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    dim = 1 << int(num_qubits)
    index = torch.arange(dim, device=device, dtype=torch.long)
    shifts = torch.arange(int(num_qubits) - 1, -1, -1, device=device, dtype=torch.long)
    bits = ((index[:, None] >> shifts[None, :]) & 1).to(dtype)
    return 1.0 - 2.0 * bits


def build_observable_signs(num_qubits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    z = _basis_z_signs(num_qubits, device, dtype)
    rows = [z[:, i] for i in range(num_qubits)]
    rows.extend(z[:, i] * z[:, j] for i in range(num_qubits) for j in range(i + 1, num_qubits))
    return torch.stack(rows, dim=0).contiguous()


def build_dtc_diagonal_phase(
    cfg: DTCQRCConfig,
    J: torch.Tensor,
    h: torch.Tensor,
    device: torch.device,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    real_dtype = _real_dtype(complex_dtype)
    z = _basis_z_signs(cfg.num_qubits, device, real_dtype)
    J = J.to(device=device, dtype=real_dtype)
    h = h.to(device=device, dtype=real_dtype)
    angle = torch.zeros(z.shape[0], device=device, dtype=real_dtype)
    angle += torch.sum((z[:, :-1] * z[:, 1:]) * J[None, :], dim=1)
    angle += math.pi * torch.sum(z * h[None, :], dim=1)
    return torch.polar(torch.ones_like(angle), -0.5 * angle).to(complex_dtype)


def rx_gate(angle: float, device: torch.device, complex_dtype: torch.dtype) -> torch.Tensor:
    half = torch.tensor(float(angle) / 2.0, device=device, dtype=_real_dtype(complex_dtype))
    c, s = torch.cos(half), torch.sin(half)
    gate = torch.zeros((2, 2), device=device, dtype=complex_dtype)
    gate[0, 0] = gate[1, 1] = c
    gate[0, 1] = gate[1, 0] = -1j * s
    return gate


def hamiltonian_modulation_gates(components: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """Return exp[-i/2 (theta_x X + theta_y Y + theta_z Z)] for a batch.

    ``components`` has shape [B, 3].  This is one Hamiltonian exponential, not an
    Euler-angle U3 sequence.  The three input features directly modulate X/Y/Z.
    """
    values = components.to(dtype=_real_dtype(complex_dtype))
    x, y, z = values[:, 0], values[:, 1], values[:, 2]
    radius = torch.linalg.vector_norm(values, dim=1)
    cosine = torch.cos(radius / 2.0)
    sine_over_radius = torch.where(
        radius > 1e-8,
        torch.sin(radius / 2.0) / radius,
        torch.full_like(radius, 0.5),
    )
    factor = (-1j * sine_over_radius).to(complex_dtype)
    gate = torch.empty((len(values), 2, 2), device=values.device, dtype=complex_dtype)
    gate[:, 0, 0] = cosine + factor * z
    gate[:, 1, 1] = cosine - factor * z
    gate[:, 0, 1] = factor * (x - 1j * y)
    gate[:, 1, 0] = factor * (x + 1j * y)
    return gate


def apply_1q_gate(state: torch.Tensor, gate: torch.Tensor, wire: int, num_qubits: int) -> torch.Tensor:
    batch = int(state.shape[0])
    tensor = state.reshape(batch, *([2] * int(num_qubits)))
    axis = 1 + int(wire)
    permutation = [0, axis] + [i for i in range(1, int(num_qubits) + 1) if i != axis]
    inverse = [0] * (int(num_qubits) + 1)
    for index, original in enumerate(permutation):
        inverse[original] = index
    temporary = tensor.permute(permutation).reshape(batch, 2, -1)
    if gate.ndim == 2:
        output = torch.einsum("ij,bjk->bik", gate, temporary)
    else:
        output = torch.einsum("bij,bjk->bik", gate, temporary)
    return output.reshape(batch, 2, *([2] * (int(num_qubits) - 1))).permute(inverse).reshape(batch, -1).contiguous()


def apply_hamiltonian_injection(
    state: torch.Tensor,
    token: torch.Tensor,
    cfg: DTCQRCConfig,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    if token.shape[-1] != cfg.input_dim:
        raise ValueError(f"Expected token dimension {cfg.input_dim}, got {token.shape[-1]}.")
    grouped = token.reshape(len(token), cfg.num_qubits, cfg.features_per_qubit)
    if cfg.features_per_qubit != 3:
        raise ValueError("Hamiltonian modulation requires exactly three features per qubit.")
    for wire in range(cfg.num_qubits):
        gate = hamiltonian_modulation_gates(grouped[:, wire], complex_dtype)
        state = apply_1q_gate(state, gate, wire, cfg.num_qubits)
    return state


def apply_reservoir_layers(
    state: torch.Tensor,
    cfg: DTCQRCConfig,
    rx: torch.Tensor,
    dtc_phase: torch.Tensor,
) -> torch.Tensor:
    for _ in range(cfg.layers_per_injection):
        for wire in range(cfg.num_qubits):
            state = apply_1q_gate(state, rx, wire, cfg.num_qubits)
        state = state * dtc_phase[None, :]
    return state


@torch.inference_mode()
def evolve_batch(
    tokens: torch.Tensor,
    cfg: DTCQRCConfig,
    J: torch.Tensor,
    h: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    complex_dtype = _complex_dtype(cfg.complex_dtype)
    tokens = tokens.to(device=device, dtype=_real_dtype(complex_dtype), non_blocking=True)
    state = torch.zeros((len(tokens), cfg.state_dim), device=device, dtype=complex_dtype)
    state[:, 0] = 1.0
    rx = rx_gate(cfg.g * math.pi, device, complex_dtype)
    phase = build_dtc_diagonal_phase(cfg, J, h, device, complex_dtype)
    for event in range(tokens.shape[1]):
        state = apply_hamiltonian_injection(state, tokens[:, event], cfg, complex_dtype)
        state = apply_reservoir_layers(state, cfg, rx, phase)
    return state


@torch.inference_mode()
def compute_qrc_features(
    tokens: np.ndarray | torch.Tensor,
    cfg: DTCQRCConfig,
    *,
    seed: int,
    batch_size: int,
    device: str,
    progress_every: int = 20,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    set_seed(seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable in this process; cannot use {device}.")
    params = generate_dtc_params(cfg.num_qubits, seed, cfg.W_scale)
    J = torch.from_numpy(params["J"])
    h = torch.from_numpy(params["h"])
    source = torch.as_tensor(tokens, dtype=torch.float32, device="cpu")
    signs = build_observable_signs(cfg.num_qubits, torch_device, _real_dtype(_complex_dtype(cfg.complex_dtype)))
    output = np.empty((len(source), cfg.output_dim), dtype=np.float32)
    total_batches = math.ceil(len(source) / int(batch_size))
    for batch_index, start in enumerate(range(0, len(source), int(batch_size)), start=1):
        end = min(start + int(batch_size), len(source))
        state = evolve_batch(source[start:end], cfg, J, h, torch_device)
        probabilities = torch.abs(state).square().real
        output[start:end] = (probabilities @ signs.T).float().cpu().numpy()
        del state, probabilities
        if batch_index == 1 or batch_index == total_batches or batch_index % int(progress_every) == 0:
            print(f"[QRC seed={seed} {device}] {end}/{len(source)}", flush=True)
    return output, params


def injection_audit(angle_components: np.ndarray) -> dict[str, float]:
    """Numerically audit unitary accuracy and noncommutation with X/Z DTC factors."""
    values = torch.as_tensor(np.asarray(angle_components, dtype=np.float64).reshape(-1, 3), dtype=torch.float64)
    gates = hamiltonian_modulation_gates(values, torch.complex128)
    identity = torch.eye(2, dtype=torch.complex128)[None]
    unitarity = torch.max(torch.abs(gates.conj().transpose(1, 2) @ gates - identity)).item()
    X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    H = 0.5 * (values[:, 0, None, None] * X + values[:, 1, None, None] * Y + values[:, 2, None, None] * Z)
    comm_x = torch.linalg.matrix_norm(H @ X - X @ H, dim=(-2, -1))
    comm_z = torch.linalg.matrix_norm(H @ Z - Z @ H, dim=(-2, -1))
    common_commutator = torch.maximum(comm_x, comm_z)
    nonzero = torch.linalg.vector_norm(values, dim=1) > 1e-12
    minimum = float(torch.min(common_commutator[nonzero]).item()) if torch.any(nonzero) else 0.0
    return {
        "max_unitarity_error": float(unitarity),
        "minimum_max_commutator_norm_over_nonzero_samples": minimum,
        "mean_commutator_with_X": float(torch.mean(comm_x).item()),
        "mean_commutator_with_Z": float(torch.mean(comm_z).item()),
    }
