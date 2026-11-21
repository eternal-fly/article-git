from __future__ import annotations

import argparse
import math
import time

from common import *
import torch
from hamiltonian_dtc_qrc import (DTCQRCConfig, generate_dtc_params, build_observable_signs,
    build_dtc_diagonal_phase, rx_gate, apply_1q_gate, apply_hamiltonian_injection, _basis_z_signs)


def representatives(task):
    if task == "bace":
        path = FINAL / "16_dissipative_protocol_migration/cache/tokens/BACE_canonical_general_Q24_T30.npz"
        with np.load(path) as z:
            tokens = z["tokens"][:, :30, :36]
        split = json.loads((FINAL / "16_dissipative_protocol_migration/cache/splits/BACE/seed000.json").read_text())
        indices = np.concatenate([np.sort(split[key])[np.linspace(0, len(split[key]) - 1, 8, dtype=int)]
                     for key in ("train_indices", "val_indices", "test_indices")])
        q, scale = 12, 1.
    else:
        data = hopping_data()
        tokens = data["tokens"]
        indices = []
        for split in range(3):
            ids = np.unique(data["trajectory_id"][data["split"] == split])
            for trajectory in ids[np.linspace(0, len(ids) - 1, 8, dtype=int)]:
                rows = np.flatnonzero(data["trajectory_id"] == trajectory)
                indices.append(rows[len(rows) // 2])
        indices = np.asarray(indices)
        q, scale = 15, .5
    selected = tokens[indices]
    write_npz(HERE / f"dephasing/{task}/representatives.npz", indices=indices, tokens=selected)
    return selected, indices, q, scale


@torch.inference_mode()
def simulate_suffix(tokens, q, seed, strength, suffix, device, mc=128, batch_size=256):
    n, t, _ = tokens.shape
    cfg = DTCQRCConfig(num_qubits=q, g=.84, W_scale=.5, layers_per_injection=3)
    params = generate_dtc_params(q, seed, .5)
    dev = torch.device(device)
    phase = build_dtc_diagonal_phase(cfg, torch.from_numpy(params["J"]), torch.from_numpy(params["h"]), dev, torch.complex64)
    rx = rx_gate(.84 * math.pi, dev, torch.complex64)
    signs = build_observable_signs(q, dev, torch.float32)
    basis = _basis_z_signs(q, dev, torch.float32)
    # Generate the fixed maximum ensemble, independent of computational batching.
    # Each suffix has an independent stream; lambda conditions share uniforms.
    rng = np.random.default_rng(stable_seed("pauli-nested-v1", q, seed, suffix))
    uniforms = rng.random((n, 128, t - suffix, q), dtype=np.float32)[:, :mc]
    flips = (uniforms < strength / 2).reshape(n * mc, t - suffix, q)
    source = np.repeat(tokens, mc, axis=0)
    result = np.empty((n * mc, cfg.output_dim), dtype=np.float32)
    for start in range(0, len(source), batch_size):
        values = torch.from_numpy(source[start:start + batch_size]).to(dev)
        noise = torch.from_numpy(flips[start:start + batch_size]).to(dev)
        state = torch.zeros((len(values), 2**q), dtype=torch.complex64, device=dev)
        state[:, 0] = 1.
        for event in range(suffix, t):
            state = apply_hamiltonian_injection(state, values[:, event], cfg, torch.complex64)
            for _ in range(3):
                for wire in range(q):
                    state = apply_1q_gate(state, rx, wire, q)
                state = state * phase
            if strength > 0:
                for wire in range(q):
                    state = torch.where(noise[:, event - suffix, wire, None], state * basis[:, wire], state)
        result[start:start + len(values)] = (state.abs().square() @ signs.T).cpu().numpy()
    return result.reshape(n, mc, cfg.output_dim).transpose(1, 0, 2)


def suffix_weights(t, p, task):
    if task == "bace":
        return np.asarray([(1-p)**t] + [p * (1-p)**(t-s) for s in range(1, t)]), p
    return np.asarray([(1-p)**(t-1)] + [p * (1-p)**(t-1-s) for s in range(1, t)]), 0.


def run_case(task, seed, strength, device):
    folder = HERE / f"dephasing/{task}/lambda{strength:.1f}/seed{seed:03d}"
    if (folder / "convergence.csv").exists():
        print(f"[cached dephasing] {task} seed={seed} lambda={strength}", flush=True)
        return
    tokens, indices, q, scale = representatives(task)
    t = tokens.shape[1]
    if task == "bace":
        path = MIG / f"cache/qrc_suffix_banks/BACE/canonical_general/Q12/g0.840_W0.500/T30_seed{seed:03d}.npz"
    else:
        path = HOP / f"features/qrc_post_tm3_suffixes_seed{seed:03d}.npz"
    with np.load(path) as z:
        coherent_bank = z["suffixes"][:, indices, :q*(q+1)//2]
    scaled = tokens * np.float32(scale)
    check = simulate_suffix(scaled, q, seed, 0., 0, device, mc=1)[0]
    parity_error = float(np.max(np.abs(check - coherent_bank[0])))
    if parity_error > 3e-5:
        raise RuntimeError(f"coherent parity failure {task} {parity_error}")
    outputs = np.zeros((2, 128, len(tokens), q*(q+1)//2), dtype=np.float64)
    weights = [suffix_weights(t, p, task) for p in (0., .1)]
    for i, (_, anchor) in enumerate(weights):
        outputs[i] += anchor
    start_time = time.monotonic()
    for suffix in range(t):
        trial_features = simulate_suffix(scaled, q, seed, strength, suffix, device)
        for i, (w, _) in enumerate(weights):
            if w[suffix]:
                outputs[i] += w[suffix] * trial_features
        if suffix % 5 == 0:
            print(f"[dephase {task} q{q} lambda{strength} seed{seed}] suffix {suffix+1}/{t}", flush=True)
    rows = []
    for i, p in enumerate((0., .1)):
        w, anchor = weights[i]
        coherent = np.tensordot(w, coherent_bank, axes=(0, 0)) + anchor
        ref = outputs[i].mean(0)
        physical_rms = float(np.sqrt(np.mean((ref - coherent)**2)))
        split64_rms = float(np.sqrt(np.mean((outputs[i, :64].mean(0) - outputs[i, 64:].mean(0))**2)))
        for k in (8, 32, 128):
            subset = outputs[i, :k]
            mean = subset.mean(0)
            se = subset.std(0, ddof=1) / np.sqrt(k)
            se_rms = float(np.sqrt(np.mean(se**2)))
            row = dict(task=task, q=q, t=t, seed=seed, p=p, dephasing=strength, trajectories=k,
                       n_representatives=len(tokens), feature_se_rms=se_rms,
                       nested_difference_to_128_rms=float(np.sqrt(np.mean((mean - ref)**2))),
                       coherent_difference_rms=physical_rms, se_to_physical_rms=se_rms/max(physical_rms, 1e-12),
                       independent_64_difference_rms=split64_rms, coherent_parity_error=parity_error)
            rows.append(row)
            write_npz(folder / f"K{k:03d}/p{p:.1f}.npz", features=mean, standard_error=se,
                      original_indices=indices, coherent_features=coherent)
    write_npz(folder / "nested_ensemble_128.npz", features_by_p_and_trial=outputs.astype(np.float32), original_indices=indices)
    write_json(folder / "metadata.json", dict(task=task, q=q, t=t, g=.84, W=.5, L=3, seed=seed,
               lambda_value=strength, p_values=[0., .1], K_values=[8, 32, 128], token_hash=array_hash(tokens),
               random_scheme="pauli-nested-v1; suffix-independent; lambda-paired; common prefix in K; batch-independent",
               original_8_trajectory_cache_reused=False, feature_diagnostic_only=True,
               elapsed_seconds=time.monotonic()-start_time))
    write_csv(folder / "convergence.csv", rows)
    print(f"[dephase complete] {task} lambda={strength} seed={seed} {time.monotonic()-start_time:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=["bace", "hopping", "all"], default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for task in (("bace", "hopping") if args.task == "all" else (args.task,)):
        for strength in (.5, 1.):
            for seed in SEEDS:
                run_case(task, seed, strength, args.device)
    files = sorted((HERE / "dephasing").glob("*/lambda*/seed*/convergence.csv"))
    write_csv(HERE / "dephasing/convergence_all_seeds.csv", pd.concat([pd.read_csv(f) for f in files]))
