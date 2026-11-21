"""
GPU statevector implementation of the DTC-QRC reservoir used in the uploaded notebook.

Circuit preserved from the original notebook:
  For each token t:
    1) U3(theta, phi, lambda) injection on every qubit.
    2) Repeat L DTC reservoir layers:
       - RX(g*pi) on all qubits.
       - nearest-neighbor ZZ/Ising coupling: CNOT - RZ(J_i) - CNOT.
       - onsite disorder: RZ(h_i*pi).
    3) Read out all Z_i and Z_i Z_j expectations.

The implementation does not use PennyLane. It evolves a batched complex statevector directly
with PyTorch on CUDA, which is usually much faster for many samples than CPU QNode evaluation.
"""

from __future__ import annotations

Q24_SAFE_SCRIPT_VERSION = "2026-06-15-q24-chunked-observables-no-dense-signs"

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import math
import os
import random

import numpy as np
import torch


TokenDimPolicy = Literal["strict", "truncate_pad"]
ComplexDTypeName = Literal["complex64", "complex128"]


@dataclass(frozen=True)
class DTCQRCConfig:
    num_qubits: int = 16
    features_per_qubit: int = 3
    g: float = 0.80
    W_scale: float = 0.80
    layers_per_injection: int = 3
    use_tanh_before_encoding: bool = True
    token_dim_policy: TokenDimPolicy = "strict"
    complex_dtype: ComplexDTypeName = "complex64"

    @property
    def input_dim(self) -> int:
        return int(self.num_qubits) * int(self.features_per_qubit)

    @property
    def state_dim(self) -> int:
        return 1 << int(self.num_qubits)

    @property
    def output_dim(self) -> int:
        q = int(self.num_qubits)
        return q + (q * (q - 1)) // 2


def set_seed(seed: int) -> None:
    """Match the original notebook's seed setting as closely as possible."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_complex_dtype(name: ComplexDTypeName) -> torch.dtype:
    if name == "complex64":
        return torch.complex64
    if name == "complex128":
        return torch.complex128
    raise ValueError(f"Unsupported complex dtype: {name}")


def real_dtype_from_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    raise ValueError(f"Expected a complex dtype, got {dtype}")


def generate_dtc_params(
    num_qubits: int,
    seed: int = 0,
    W_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Generate fixed random DTC-QRC reservoir parameters.

    This mirrors the uploaded notebook. After `set_seed(seed)`, the original class used
    `torch.initial_seed() % 2**32` as the NumPy RNG seed. Since `torch.manual_seed(seed)`
    makes `torch.initial_seed() == seed`, the effective parameter seed is the run seed.

    J_i ~ Uniform(-0.9, 0.9) for nearest-neighbor ZZ couplings.
    h_i ~ Uniform(-W, W) for onsite disorder fields.
    """
    rng = np.random.default_rng(int(seed))
    q = int(num_qubits)
    J = rng.uniform(-0.9, 0.9, size=q - 1).astype(np.float32)
    h = (rng.uniform(-1.0, 1.0, size=q) * float(W_scale)).astype(np.float32)
    packed = np.concatenate([J, h]).astype(np.float32)
    return {"J": J, "h": h, "packed": packed, "seed": np.array([int(seed)], dtype=np.int64)}


def make_z_zz_feature_names(num_qubits: int) -> List[str]:
    q = int(num_qubits)
    names = [f"Z_{i}" for i in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            names.append(f"ZZ_{i}_{j}")
    return names


def adapt_token_dim(tokens: torch.Tensor, input_dim: int, policy: TokenDimPolicy) -> torch.Tensor:
    """Ensure the last token dimension matches num_qubits * features_per_qubit."""
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [N, T, token_dim], got {tuple(tokens.shape)}")
    current = int(tokens.shape[-1])
    target = int(input_dim)
    if current == target:
        return tokens
    if policy == "strict":
        raise ValueError(
            f"Token dimension mismatch: data token_dim={current}, required={target}. "
            f"Use --token-dim-policy truncate_pad only for controlled ablation experiments."
        )
    if policy != "truncate_pad":
        raise ValueError(f"Unsupported token_dim_policy: {policy}")
    if current > target:
        return tokens[..., :target]
    pad_shape = list(tokens.shape)
    pad_shape[-1] = target - current
    pad = torch.zeros(*pad_shape, dtype=tokens.dtype, device=tokens.device)
    return torch.cat([tokens, pad], dim=-1)


def _basis_z_signs(num_qubits: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return signs[d, q] = +1/-1 for Z_q on computational basis state d.

    Wire 0 is treated as the most significant tensor axis. This is consistent internally
    for gate application and readout naming.
    """
    q = int(num_qubits)
    dim = 1 << q
    idx = torch.arange(dim, device=device, dtype=torch.long)
    shifts = torch.arange(q - 1, -1, -1, device=device, dtype=torch.long)
    bits = ((idx[:, None] >> shifts[None, :]) & 1).to(dtype)
    return 1.0 - 2.0 * bits


def build_observable_signs(
    num_qubits: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return [output_dim, 2**num_qubits] signs for all Z and pairwise ZZ observables."""
    z = _basis_z_signs(num_qubits, device=device, dtype=dtype)  # [D, q]
    rows = [z[:, i] for i in range(num_qubits)]
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            rows.append(z[:, i] * z[:, j])
    return torch.stack(rows, dim=0).contiguous()


def build_dtc_diagonal_phase(
    num_qubits: int,
    J: torch.Tensor,
    h: torch.Tensor,
    device: torch.device,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Build the diagonal phase for all ZZ couplings plus onsite RZ disorder.

    CNOT-RZ(J_i)-CNOT contributes exp[-i * J_i * Z_i Z_{i+1} / 2].
    RZ(h_i*pi) contributes exp[-i * h_i*pi * Z_i / 2].
    """
    real_dtype = real_dtype_from_complex(complex_dtype)
    z = _basis_z_signs(num_qubits, device=device, dtype=real_dtype)  # [D, q]
    J = J.to(device=device, dtype=real_dtype)
    h = h.to(device=device, dtype=real_dtype)

    phase_angle = torch.zeros(z.shape[0], device=device, dtype=real_dtype)
    if num_qubits >= 2:
        phase_angle = phase_angle + torch.sum((z[:, :-1] * z[:, 1:]) * J[None, :], dim=1)
    phase_angle = phase_angle + math.pi * torch.sum(z * h[None, :], dim=1)

    # exp(-i * phase_angle / 2)
    return torch.polar(torch.ones_like(phase_angle), -0.5 * phase_angle).to(complex_dtype)


def rx_gate(angle: float, device: torch.device, complex_dtype: torch.dtype) -> torch.Tensor:
    """PennyLane-compatible RX(angle)."""
    real_dtype = real_dtype_from_complex(complex_dtype)
    half = torch.tensor(float(angle) / 2.0, device=device, dtype=real_dtype)
    c = torch.cos(half)
    s = torch.sin(half)
    gate = torch.zeros((2, 2), device=device, dtype=complex_dtype)
    gate[0, 0] = c
    gate[1, 1] = c
    gate[0, 1] = -1j * s
    gate[1, 0] = -1j * s
    return gate


def u3_gates(theta: torch.Tensor, phi: torch.Tensor, lam: torch.Tensor, complex_dtype: torch.dtype) -> torch.Tensor:
    """Return batched PennyLane-compatible U3(theta, phi, lambda), shape [B, 2, 2]."""
    theta = theta.to(dtype=real_dtype_from_complex(complex_dtype))
    phi = phi.to(dtype=theta.dtype)
    lam = lam.to(dtype=theta.dtype)

    ct = torch.cos(theta / 2.0)
    st = torch.sin(theta / 2.0)
    exp_phi = torch.polar(torch.ones_like(phi), phi)
    exp_lam = torch.polar(torch.ones_like(lam), lam)
    exp_phi_lam = torch.polar(torch.ones_like(phi), phi + lam)

    B = int(theta.shape[0])
    gate = torch.empty((B, 2, 2), device=theta.device, dtype=complex_dtype)
    gate[:, 0, 0] = ct
    gate[:, 0, 1] = -exp_lam * st
    gate[:, 1, 0] = exp_phi * st
    gate[:, 1, 1] = exp_phi_lam * ct
    return gate


def apply_1q_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    wire: int,
    num_qubits: int,
) -> torch.Tensor:
    """Apply one-qubit gate to a batched statevector.

    state: [B, 2**q]
    gate:  [2, 2] for same gate across batch, or [B, 2, 2] for sample-specific gates.
    """
    B = int(state.shape[0])
    q = int(num_qubits)
    wire = int(wire)
    if not (0 <= wire < q):
        raise ValueError(f"wire must be in [0, {q}), got {wire}")

    tensor = state.reshape(B, *([2] * q))
    gate_axis = 1 + wire
    perm = [0, gate_axis] + [ax for ax in range(1, q + 1) if ax != gate_axis]
    inv_perm = [0] * (q + 1)
    for i, ax in enumerate(perm):
        inv_perm[ax] = i

    tmp = tensor.permute(perm).reshape(B, 2, -1)
    if gate.ndim == 2:
        out = torch.einsum("ij,bjk->bik", gate, tmp)
    elif gate.ndim == 3:
        if int(gate.shape[0]) != B:
            raise ValueError(f"Batched gate has batch {gate.shape[0]}, but state batch is {B}")
        out = torch.einsum("bij,bjk->bik", gate, tmp)
    else:
        raise ValueError(f"gate must have shape [2,2] or [B,2,2], got {tuple(gate.shape)}")

    out = out.reshape(B, 2, *([2] * (q - 1))).permute(inv_perm).reshape(B, 1 << q)
    return out.contiguous()


def apply_token_injection(
    state: torch.Tensor,
    token_t: torch.Tensor,
    cfg: DTCQRCConfig,
    complex_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply per-qubit U3 injection for one token step."""
    B = int(state.shape[0])
    x = token_t.reshape(B, cfg.num_qubits, cfg.features_per_qubit)
    zero = torch.zeros(B, device=token_t.device, dtype=token_t.dtype)
    for wire in range(cfg.num_qubits):
        theta = x[:, wire, 0]
        phi = x[:, wire, 1] if cfg.features_per_qubit >= 2 else zero
        lam = x[:, wire, 2] if cfg.features_per_qubit >= 3 else zero
        gate = u3_gates(theta, phi, lam, complex_dtype)
        state = apply_1q_gate(state, gate, wire, cfg.num_qubits)
    return state


def apply_reservoir_layers(
    state: torch.Tensor,
    cfg: DTCQRCConfig,
    rx: torch.Tensor,
    dtc_phase: torch.Tensor,
) -> torch.Tensor:
    """Apply cfg.layers_per_injection fixed DTC reservoir layers."""
    for _ in range(cfg.layers_per_injection):
        for wire in range(cfg.num_qubits):
            state = apply_1q_gate(state, rx, wire, cfg.num_qubits)
        state = state * dtc_phase[None, :]
    return state


@torch.no_grad()
def evolve_batch_final_state(
    tokens_batch: torch.Tensor,
    cfg: DTCQRCConfig,
    J: torch.Tensor,
    h: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Evolve a batch of token sequences and return final statevectors [B, 2**q]."""
    complex_dtype = torch_complex_dtype(cfg.complex_dtype)
    real_dtype = real_dtype_from_complex(complex_dtype)

    tokens_batch = tokens_batch.to(device=device, dtype=real_dtype, non_blocking=True)
    tokens_batch = adapt_token_dim(tokens_batch, cfg.input_dim, cfg.token_dim_policy)
    if cfg.use_tanh_before_encoding:
        tokens_batch = torch.tanh(tokens_batch)

    B, T, _ = tokens_batch.shape
    state = torch.zeros((B, cfg.state_dim), device=device, dtype=complex_dtype)
    state[:, 0] = 1.0 + 0.0j

    rx = rx_gate(cfg.g * math.pi, device=device, complex_dtype=complex_dtype)
    dtc_phase = build_dtc_diagonal_phase(cfg.num_qubits, J, h, device, complex_dtype)

    for t in range(int(T)):
        state = apply_token_injection(state, tokens_batch[:, t, :], cfg, complex_dtype)
        state = apply_reservoir_layers(state, cfg, rx, dtc_phase)
    return state


@torch.no_grad()
def compute_z_zz_from_state(
    state: torch.Tensor,
    obs_signs: torch.Tensor,
) -> torch.Tensor:
    """Compute all Z and ZZ expectation values from final statevectors.

    This dense-sign implementation is fast for moderate q, but the sign matrix
    has shape [q + q(q-1)/2, 2**q]. For q=24 that is roughly 300 x 16.8M,
    i.e. about 20 GB in float32. Use compute_z_zz_from_state_chunked for q=24.
    """
    probs = torch.abs(state).square().real
    return torch.matmul(probs, obs_signs.t()).float()


@torch.no_grad()
def compute_z_zz_from_state_chunked(
    state: torch.Tensor,
    num_qubits: int,
    chunk_size: int = 262_144,
) -> torch.Tensor:
    """Memory-efficient Z/ZZ expectation values for large-q statevectors.

    It streams over computational-basis index chunks and accumulates
    E[Z_i] and E[Z_i Z_j] without materializing the full observable sign matrix.
    This is the intended path for q=24 on 24 GB GPUs.
    """
    if state.ndim != 2:
        raise ValueError(f"state must have shape [B, 2**q], got {tuple(state.shape)}")
    B, D = int(state.shape[0]), int(state.shape[1])
    q = int(num_qubits)
    if D != (1 << q):
        raise ValueError(f"state width {D} does not match 2**num_qubits={1 << q}")

    device = state.device
    probs = torch.abs(state).square().real
    real_dtype = probs.dtype

    z_acc = torch.zeros((B, q), device=device, dtype=real_dtype)
    zz_acc = torch.zeros((B, q, q), device=device, dtype=real_dtype)
    shifts = torch.arange(q - 1, -1, -1, device=device, dtype=torch.long)

    for start in range(0, D, int(chunk_size)):
        end = min(start + int(chunk_size), D)
        idx = torch.arange(start, end, device=device, dtype=torch.long)
        bits = ((idx[:, None] >> shifts[None, :]) & 1).to(real_dtype)
        signs = 1.0 - 2.0 * bits  # [C, q], +1 for |0>, -1 for |1>
        p = probs[:, start:end]     # [B, C]

        z_acc += p @ signs
        weighted = p[:, :, None] * signs[None, :, :]
        zz_acc += torch.einsum("bcq,cr->bqr", weighted, signs)

    rows = [z_acc[:, i] for i in range(q)]
    for i in range(q):
        for j in range(i + 1, q):
            rows.append(zz_acc[:, i, j])
    return torch.stack(rows, dim=1).float()


@torch.no_grad()
def compute_qrc_features_gpu(
    tokens: torch.Tensor,
    cfg: DTCQRCConfig,
    seed: int,
    batch_size: int,
    device: str | torch.device = "cuda:0",
    progress: bool = True,
) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    """Compute [N, q + C(q,2)] Z/ZZ features for one seed on one device."""
    set_seed(seed)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")

    params = generate_dtc_params(cfg.num_qubits, seed=int(seed), W_scale=cfg.W_scale)
    J = torch.tensor(params["J"], dtype=torch.float32)
    h = torch.tensor(params["h"], dtype=torch.float32)
    # For q=24, a dense [output_dim, 2**q] sign matrix is too close to the
    # 24 GB limit of an RTX 4090. Use the dense path only for smaller q.
    obs_signs = None
    if cfg.num_qubits <= 20:
        obs_signs = build_observable_signs(
            cfg.num_qubits,
            device=device,
            dtype=real_dtype_from_complex(torch_complex_dtype(cfg.complex_dtype)),
        )

    tokens = tokens.detach().cpu().float()
    N = int(tokens.shape[0])
    out = torch.empty((N, cfg.output_dim), dtype=torch.float32, device="cpu")

    for start in range(0, N, int(batch_size)):
        end = min(start + int(batch_size), N)
        state = evolve_batch_final_state(tokens[start:end], cfg, J, h, device=device)
        if obs_signs is None:
            feats = compute_z_zz_from_state_chunked(state, cfg.num_qubits).detach().cpu()
        else:
            feats = compute_z_zz_from_state(state, obs_signs).detach().cpu()
        out[start:end] = feats
        del state, feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if progress:
            print(f"[seed={seed} | {device}] encoded {end}/{N}", flush=True)
    return out, params


@torch.no_grad()
def build_reservoir_unitary_matrix(
    cfg: DTCQRCConfig,
    seed: int,
    device: str | torch.device = "cuda:0",
    basis_batch_size: int = 256,
) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
    """Materialize the fixed DTC reservoir unitary for L layers, excluding token U3 injection.

    This is intended only for small q. A full matrix has shape [2**q, 2**q].
    For q=16, complex64 requires roughly 34.36 GB, so materialization is not practical
    on a 24 GB RTX 4090. Use implicit statevector evolution for q=16.

    Returns U where columns are evolved computational basis vectors: |psi_out> = U |psi_in>.
    """
    set_seed(seed)
    device = torch.device(device)
    complex_dtype = torch_complex_dtype(cfg.complex_dtype)
    D = cfg.state_dim

    params = generate_dtc_params(cfg.num_qubits, seed=int(seed), W_scale=cfg.W_scale)
    J = torch.tensor(params["J"], dtype=torch.float32)
    h = torch.tensor(params["h"], dtype=torch.float32)
    rx = rx_gate(cfg.g * math.pi, device=device, complex_dtype=complex_dtype)
    dtc_phase = build_dtc_diagonal_phase(cfg.num_qubits, J, h, device, complex_dtype)

    U = torch.empty((D, D), dtype=complex_dtype, device="cpu")
    for start in range(0, D, int(basis_batch_size)):
        end = min(start + int(basis_batch_size), D)
        B = end - start
        state = torch.zeros((B, D), dtype=complex_dtype, device=device)
        cols = torch.arange(start, end, device=device)
        state[torch.arange(B, device=device), cols] = 1.0 + 0.0j
        state = apply_reservoir_layers(state, cfg, rx, dtc_phase)
        # Rows are output statevectors for input basis columns; transpose to U[:, start:end].
        U[:, start:end] = state.t().detach().cpu()
        del state
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[seed={seed} | {device}] unitary columns {end}/{D}", flush=True)
    return U, params


def feature_file_payload(
    *,
    seed: int,
    features: torch.Tensor,
    params: Dict[str, np.ndarray],
    y: Optional[torch.Tensor],
    smiles: Optional[Sequence[str]],
    split_indices: Optional[object],
    cfg: DTCQRCConfig,
    source_processed_path: str,
    extra_config: Optional[dict] = None,
) -> dict:
    """Build a torch-saveable feature package compatible with the split notebook style."""
    payload = {
        "method": "qrc_gpu_statevector",
        "seed": int(seed),
        "features": features.detach().cpu().float(),
        "feature_names": make_z_zz_feature_names(cfg.num_qubits),
        "params": {
            "J": torch.tensor(params["J"], dtype=torch.float32),
            "h": torch.tensor(params["h"], dtype=torch.float32),
            "packed": torch.tensor(params["packed"], dtype=torch.float32),
            "seed": int(seed),
        },
        "qrc_gpu_config": cfg.__dict__.copy(),
        "source_processed_path": os.path.abspath(source_processed_path),
    }
    if y is not None:
        payload["y"] = y.detach().cpu().float() if isinstance(y, torch.Tensor) else y
    if smiles is not None:
        payload["smiles"] = list(smiles)
    if split_indices is not None:
        payload["split_indices"] = split_indices
    if extra_config is not None:
        payload["config"] = extra_config
    return payload
