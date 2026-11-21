"""Grouped local gates for the unchanged event-dephasing statevector protocol.

All scientific gate matrices come from hamiltonian_dtc_qrc.py. Triton only
applies their tensor products without permuting/copying the whole state for
each individual wire. Each kernel owns all amplitudes of its local block.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", str(Path(__file__).resolve().parent / "dephasing_full_k/.triton_cache"))

from common import *
import torch
import triton
import triton.language as tl
from hamiltonian_dtc_qrc import (
    DTCQRCConfig, generate_dtc_params, build_dtc_diagonal_phase,
    build_observable_signs, rx_gate, hamiltonian_modulation_gates,
    apply_1q_gate, apply_hamiltonian_injection, _basis_z_signs,
)


@triton.jit
def _group_gate(Src, Dst, Gates, Phase, Noise,
                GROUPS_TOTAL: tl.constexpr, DIM: tl.constexpr, Q: tl.constexpr,
                EVENTS: tl.constexpr, EVENT, REPEATS: tl.constexpr,
                WIRE: tl.constexpr, GROUP: tl.constexpr, INPUT: tl.constexpr,
                PHASE: tl.constexpr, DEPHASE: tl.constexpr,
                COS: tl.constexpr, SIN: tl.constexpr, BLOCK: tl.constexpr):
    group_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = group_idx < GROUPS_TOTAL
    row = group_idx // (DIM >> GROUP)
    reduced = group_idx % (DIM >> GROUP)
    shift: tl.constexpr = Q - WIRE - GROUP
    base = ((reduced >> shift) << (shift + GROUP)) | (reduced & ((1 << shift) - 1))
    local = tl.arange(0, 1 << GROUP)
    amplitude = base[:, None] + (local[None, :] << shift)
    address = row[:, None] * DIM + amplitude
    real = tl.load(Src + address * 2, valid[:, None], 0)
    imag = tl.load(Src + address * 2 + 1, valid[:, None], 0)
    for k in tl.static_range(GROUP):
        bit = 1 << (GROUP - 1 - k)
        peer = tl.broadcast_to((local ^ bit)[None, :], (BLOCK, 1 << GROUP))
        pr = tl.gather(real, peer, 1)
        pi = tl.gather(imag, peer, 1)
        if INPUT:
            gate = (((row // REPEATS) * EVENTS + EVENT) * Q + WIRE + k) * 8
            g00r = tl.load(Gates + gate, valid, 0)[:, None]
            g00i = tl.load(Gates + gate + 1, valid, 0)[:, None]
            g01r = tl.load(Gates + gate + 2, valid, 0)[:, None]
            g01i = tl.load(Gates + gate + 3, valid, 0)[:, None]
            g10r = tl.load(Gates + gate + 4, valid, 0)[:, None]
            g10i = tl.load(Gates + gate + 5, valid, 0)[:, None]
            g11r = tl.load(Gates + gate + 6, valid, 0)[:, None]
            g11i = tl.load(Gates + gate + 7, valid, 0)[:, None]
            high = (local[None, :] & bit) != 0
            dr = tl.where(high, g11r, g00r)
            di = tl.where(high, g11i, g00i)
            cr = tl.where(high, g10r, g01r)
            ci = tl.where(high, g10i, g01i)
            rr = dr * real - di * imag + cr * pr - ci * pi
            ii = dr * imag + di * real + cr * pi + ci * pr
        else:
            rr = COS * real + SIN * pi
            ii = COS * imag - SIN * pr
        real, imag = rr, ii
    if PHASE:
        vr = tl.load(Phase + amplitude * 2)
        vi = tl.load(Phase + amplitude * 2 + 1)
        rr = real * vr - imag * vi
        imag = real * vi + imag * vr
        real = rr
    if DEPHASE:
        flips = tl.load(Noise + row * EVENTS + EVENT, valid, 0)[:, None]
        parity = amplitude & flips
        parity = parity ^ (parity >> 16)
        parity = parity ^ (parity >> 8)
        parity = parity ^ (parity >> 4)
        parity = parity ^ (parity >> 2)
        parity = parity ^ (parity >> 1)
        sign = 1.0 - 2.0 * (parity & 1).to(tl.float32)
        real, imag = real * sign, imag * sign
    tl.store(Dst + address * 2, real, valid[:, None])
    tl.store(Dst + address * 2 + 1, imag, valid[:, None])


class StatevectorBackend:
    def __init__(self, q, seed, device="cuda:0", group_size=3):
        self.q, self.seed, self.group_size = q, seed, group_size
        self.device = torch.device(device)
        self.cfg = DTCQRCConfig(num_qubits=q, g=.84, W_scale=.5, layers_per_injection=3)
        self.params = generate_dtc_params(q, seed, .5)
        self.phase = build_dtc_diagonal_phase(self.cfg, torch.from_numpy(self.params["J"]),
                       torch.from_numpy(self.params["h"]), self.device, torch.complex64)
        self.rx = rx_gate(.84 * math.pi, self.device, torch.complex64)
        self.cos = float(self.rx[0, 0].real)
        self.sin = -float(self.rx[0, 1].imag)
        self.signs = build_observable_signs(q, self.device, torch.float32)
        self.basis = _basis_z_signs(q, self.device, torch.float32)

    def gates(self, tokens):
        values = torch.as_tensor(tokens, dtype=torch.float32, device=self.device)
        raw = hamiltonian_modulation_gates(values.reshape(-1, 3), torch.complex64)
        # Input injection followed by the first all-wire RX drive.
        return (self.rx @ raw).reshape(*values.shape[:2], self.q, 2, 2).contiguous()

    @torch.inference_mode()
    def run(self, tokens, masks, suffix=0, tm3=False, reference=False):
        values = torch.as_tensor(tokens, dtype=torch.float32, device=self.device)
        n, t, _ = values.shape
        k = masks.shape[1]
        if masks.shape != (n, k, t):
            raise ValueError(masks.shape)
        noise = torch.as_tensor(masks.reshape(n*k, t), dtype=torch.int32, device=self.device)
        state = torch.zeros((n*k, 1 << self.q), dtype=torch.complex64, device=self.device)
        state[:, 0] = 1.
        if reference:
            source = values.repeat_interleave(k, 0)
            for event in range(suffix, t):
                state = apply_hamiltonian_injection(state, source[:, event], self.cfg, torch.complex64)
                for _ in range(3):
                    for wire in range(self.q):
                        state = apply_1q_gate(state, self.rx, wire, self.q)
                    state = state * self.phase
                for wire in range(self.q):
                    flip = ((noise[:, event] >> (self.q-1-wire)) & 1).bool()
                    state = torch.where(flip[:, None], state * self.basis[:, wire], state)
            output = [state.abs().square() @ self.signs.T]
            if tm3:
                for _ in range(2):
                    for wire in range(self.q):
                        state = apply_1q_gate(state, self.rx, wire, self.q)
                    state = state * self.phase
                    output.append(state.abs().square() @ self.signs.T)
        else:
            gates = self.gates(tokens)
            scratch = torch.empty_like(state)
            def cycle(event, injection=False, dephase=False):
                nonlocal state, scratch
                for wire in range(0, self.q, self.group_size):
                    group = min(self.group_size, self.q-wire)
                    last = wire + group == self.q
                    blocks = n*k*((1 << self.q) >> group)
                    _group_gate[(triton.cdiv(blocks, 128),)](
                        torch.view_as_real(state), torch.view_as_real(scratch),
                        torch.view_as_real(gates), torch.view_as_real(self.phase), noise,
                        blocks, 1 << self.q, self.q, t, event, k, wire, group,
                        injection, last, last and dephase,
                        self.cos, self.sin, 128, num_warps=4, enable_fp_fusion=False)
                    state, scratch = scratch, state
            for event in range(suffix, t):
                cycle(event, injection=True)
                cycle(event)
                cycle(event, dephase=True)
            output = [state.abs().square() @ self.signs.T]
            if tm3:
                for _ in range(2):
                    cycle(0)
                    output.append(state.abs().square() @ self.signs.T)
        result = torch.cat(output, 1).reshape(n, k, -1).cpu().numpy()
        norm_error = float(torch.max(torch.abs(state.abs().square().sum(1) - 1)).cpu())
        return result, norm_error


def draw_masks(rng, n, k, t, q, strength, suffix=0):
    """Row-major stream: chunking rows does not change any noise realization."""
    uniforms = rng.random((n, k, t-suffix, q), dtype=np.float32)
    powers = np.left_shift(np.int32(1), np.arange(q-1, -1, -1, dtype=np.int32))
    masks = np.zeros((n, k, t), dtype=np.int32)
    masks[:, :, suffix:] = ((uniforms < strength/2) * powers).sum(-1, dtype=np.int32)
    return masks
