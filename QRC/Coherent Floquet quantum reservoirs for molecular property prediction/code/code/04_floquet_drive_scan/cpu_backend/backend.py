"""CPU backend with the original hopping gates and reset timing."""
from __future__ import annotations
import ctypes
import subprocess
import time
from common import *
import torch
from hamiltonian_dtc_qrc import (DTCQRCConfig, generate_dtc_params,
    build_dtc_diagonal_phase, rx_gate, hamiltonian_modulation_gates)

torch.set_num_threads(1)


def compile_kernel():
    source, output = HERE / "cpu_kernel.cpp", HERE / "cpu_kernel.so"
    if not output.exists() or output.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC",
                        "-std=c++17", str(source), "-o", str(output)], check=True)
    return output


def suffixes(tokens, *, seed, g, threads=24):
    started = time.monotonic()
    values = np.ascontiguousarray(tokens, dtype=np.float32) * np.float32(ANGLE)
    n, t, d = values.shape
    q = d // 3
    cfg = DTCQRCConfig(num_qubits=q, g=float(g), W_scale=W, layers_per_injection=LAYERS)
    params = generate_dtc_params(q, seed, W)
    device = torch.device("cpu")
    phase = build_dtc_diagonal_phase(cfg, torch.from_numpy(params["J"]), torch.from_numpy(params["h"]), device, torch.complex64)
    drive = rx_gate(float(g)*np.pi, device, torch.complex64)
    raw = hamiltonian_modulation_gates(torch.from_numpy(values.reshape(-1, 3)), torch.complex64)
    fused = (drive @ raw).reshape(n, t, q, 2, 2).contiguous()
    gate_data = np.ascontiguousarray(torch.view_as_real(fused).numpy())
    phase_data = np.ascontiguousarray(torch.view_as_real(phase).numpy())
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib = ctypes.CDLL(str(compile_kernel()))
    lib.hopping_suffixes.argtypes = [fp, fp, ctypes.c_float, ctypes.c_float,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, dp, dp]
    lib.hopping_suffixes.restype = ctypes.c_int
    out = np.empty((t, n, 3*q*(q+1)//2), dtype=np.float64)
    errors = np.empty(n, dtype=np.float64)
    status = lib.hopping_suffixes(gate_data, phase_data, float(drive[0, 0].real),
                                 -float(drive[0, 1].imag), n, t, q, LAYERS, threads, out, errors)
    assert status == 0 and np.isfinite(out).all()
    assert errors.max() < 5e-4 and np.max(np.abs(out)) < 1.002
    return out.astype(np.float32), dict(maximum_norm_error=float(errors.max()),
            elapsed_seconds=time.monotonic()-started, seed=seed, g=g,
            params={k: params[k].tolist() for k in ("J", "h")})


def mix(bank):
    return {pkey(p): np.asarray(np.tensordot(reset_suffix_weights(len(bank), p), bank, axes=(0, 0))[:, :width],
                               dtype=np.float32) for p, width in widths().items()}


def features(g, seed, threads=24, chunk_size=96):
    folder = case_dir(g, seed)
    output = folder / "features.npz"
    if (folder / "features_complete.json").exists():
        assert output.exists()
        return output
    data = dataset()
    tokens = data["tokens"]
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    if np.isclose(g, .84):
        source = HOP / "features" / f"qrc_post_tm3_suffixes_seed{seed:03d}.npz"
        with np.load(source, allow_pickle=False) as z:
            bank = z["suffixes"]
        assert bank.shape == (20, 2376, 360)
        arrays = mix(bank)
        origin = "frozen-g0.84-hopping-feature-bank"
    else:
        origin = "exact-CPU-full-suffix-mixture"
        arrays = {pkey(p): np.empty((len(tokens), width), dtype=np.float32) for p, width in widths().items()}
        for start in range(0, len(tokens), chunk_size):
            end = min(start+chunk_size, len(tokens))
            chunk = folder / "chunks" / f"rows{start:04d}_{end:04d}.npz"
            metadata = chunk.with_suffix(".json")
            if chunk.exists() and metadata.exists():
                with np.load(chunk, allow_pickle=False) as z:
                    mixed = {k: z[k] for k in z.files}
                record = json.loads(metadata.read_text())
            else:
                bank, record = suffixes(tokens[start:end], seed=seed, g=g, threads=threads)
                mixed = mix(bank)
                nwrite(chunk, **mixed)
                record.update(start=start, end=end)
                jwrite(metadata, record)
            for key, values in mixed.items():
                arrays[key][start:end] = values
            records.append(record)
            jwrite(folder / "feature_progress.json", dict(g=g, seed=seed, rows_done=end, rows_total=len(tokens),
                    elapsed_seconds=time.monotonic()-started))
            print(f"[features g={g:.2f} seed={seed}] {end}/{len(tokens)}", flush=True)
    assert all(np.isfinite(a).all() for a in arrays.values())
    nwrite(output, **arrays)
    jwrite(folder / "features_complete.json", dict(version=VERSION, g=g, seed=seed, source=origin,
            p_widths=widths(), rows=len(tokens), elapsed_seconds=time.monotonic()-started,
            maximum_norm_error=max((r["maximum_norm_error"] for r in records), default=0.),
            bytes=output.stat().st_size, sha256=sha(output)))
    # Only redundant, verified chunk checkpoints of this exact new case are removed.
    # The complete mixed features are now durably saved and checked above.
    for path in sorted((folder / "chunks").glob("rows*.npz")):
        path.unlink()
    for path in sorted((folder / "chunks").glob("rows*.json")):
        path.unlink()
    return output
