"""Independent-gate, density-channel, and original full-Q anchor checks."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *
from backend import suffixes, mix, compile_kernel
import torch
from hamiltonian_dtc_qrc import (DTCQRCConfig, generate_dtc_params, build_dtc_diagonal_phase,
    rx_gate, build_observable_signs, apply_hamiltonian_injection, apply_1q_gate,
    hamiltonian_modulation_gates)

torch.set_num_threads(1)


def torch_bank(tokens, seed, g):
    n, t, d = tokens.shape
    q = d//3
    cfg = DTCQRCConfig(num_qubits=q, g=g, W_scale=W, layers_per_injection=3)
    device = torch.device("cpu")
    params = generate_dtc_params(q, seed, W)
    drive = rx_gate(g*np.pi, device, torch.complex64)
    phase = build_dtc_diagonal_phase(cfg, torch.from_numpy(params["J"]), torch.from_numpy(params["h"]), device, torch.complex64)
    signs = build_observable_signs(q, device, torch.float32)
    values = torch.from_numpy(tokens * np.float32(ANGLE))
    def cycle(state):
        for wire in range(q):
            state = apply_1q_gate(state, drive, wire, q)
        return state * phase[None, :]
    result = []
    with torch.inference_mode():
        for suffix in range(t):
            state = torch.zeros((n, 1<<q), dtype=torch.complex64)
            state[:, 0] = 1
            for event in range(suffix, t):
                state = apply_hamiltonian_injection(state, values[:, event], cfg, torch.complex64)
                for _ in range(3):
                    state = cycle(state)
            snapshots = []
            for k in range(3):
                if k:
                    state = cycle(state)
                probabilities = state.real.square()+state.imag.square()
                snapshots.append((probabilities @ signs.T).numpy())
            result.append(np.concatenate(snapshots, axis=1))
    return np.stack(result)


def tensor(gates):
    result = np.asarray([[1.]], dtype=np.complex128)
    for gate in gates:
        result = np.kron(result, gate)
    return result


def density_reference(tokens, seed, g, p):
    _, t, d = tokens.shape
    q = d//3
    cfg = DTCQRCConfig(num_qubits=q, g=g, W_scale=W)
    device = torch.device("cpu")
    params = generate_dtc_params(q, seed, W)
    phase = build_dtc_diagonal_phase(cfg, torch.from_numpy(params["J"]), torch.from_numpy(params["h"]), device, torch.complex64).numpy()
    rx = rx_gate(g*np.pi, device, torch.complex64).numpy()
    cycle = phase[:, None] * tensor([rx]*q)
    cycles = np.linalg.matrix_power(cycle, 3)
    gates = hamiltonian_modulation_gates(torch.from_numpy(tokens.reshape(-1, 3)*np.float32(ANGLE)), torch.complex64).numpy().reshape(t, q, 2, 2)
    zero = np.zeros((1<<q, 1<<q), dtype=np.complex128)
    zero[0, 0] = 1
    rho = zero.copy()
    for event in range(t):
        if event:
            rho = (1-p)*rho+p*zero
        operator = cycles @ tensor(gates[event])
        rho = operator @ rho @ operator.conj().T
    signs = build_observable_signs(q, device, torch.float32).numpy()
    results = []
    for k in range(3):
        if k:
            rho = cycle @ rho @ cycle.conj().T
        results.append(signs @ np.diag(rho).real)
    return np.concatenate(results)


def main():
    compile_kernel()
    audit = validate_inputs()
    records = []
    rng = np.random.default_rng(20260908)
    small = rng.normal(size=(3, 4, 9)).astype(np.float32)
    small[..., 1::3] = 0
    for g in (0., .34, .84, .98):
        bank, metadata = suffixes(small, seed=10, g=g, threads=3)
        ref = torch_bank(small, 10, g)
        err = float(np.max(np.abs(bank-ref)))
        assert err < 1e-5, (g, err)
        records.append(dict(check="independent_Torch_gates", g=g, error=err))
        for p in (0., .1, .65, 1.):
            mixed = np.tensordot(reset_suffix_weights(4, p), bank[:, 0], axes=(0, 0))
            density = density_reference(small[:1], 10, g, p)
            err = float(np.max(np.abs(mixed-density)))
            assert err < 1e-5, (g, p, err)
            records.append(dict(check="direct_density_between_event_reset", g=g, p=p, error=err))
    data = dataset()
    idx = np.array([0, 1424, 1896])
    for seed in SEEDS:
        bank, meta = suffixes(data["tokens"][idx], seed=seed, g=.84, threads=3)
        with np.load(HOP/"features"/f"qrc_post_tm3_suffixes_seed{seed:03d}.npz", allow_pickle=False) as z:
            original = z["suffixes"][:, idx]
        assert original.shape == bank.shape == (20, 3, 360)
        err = float(np.max(np.abs(bank-original)))
        assert err < 3e-5, (seed, err)
        a, b = mix(bank), mix(original)
        mixed_error = max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
        assert mixed_error < 3e-5
        records.append(dict(check="Q15_frozen_hopping_bank", error=err,
                            mixed_feature_error=mixed_error, **meta))
        print(f"[verified Q15 seed={seed}] max error={err:.3g}", flush=True)
    jwrite(HERE/"backend_verification.json", dict(passed=True, records=records, input_audit=audit,
            maximum_feature_error=max(r["error"] for r in records), kernel_sha256=sha(HERE/"cpu_kernel.so")))
    print("PASS: original gates, reset positions, final/TM3 ordering, six frozen Q15 anchors", flush=True)


if __name__ == "__main__":
    main()
