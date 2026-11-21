"""Paper-aligned post-input TM3 readout with Z+ZZ observables.

Unlike Task 7, all three reservoir layers belonging to the last real input are
completed before temporal sampling begins.  The samples are O(t0), O(t1), and
O(t2), where t0 is the original Task-6 terminal state and t1/t2 follow one/two
additional input-independent reservoir cycles.  No reset occurs in this
readout-only window.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK_DIR.parent
WORKSPACE = TRANSITION_DIR.parent
TASK7_DIR = TRANSITION_DIR / "task7_q12_xz_tm3_decoders"
TASK6_DIR = TRANSITION_DIR / "task6_q12_xz_dissipation_scan"
TASK5_DIR = TRANSITION_DIR / "task5_xonly_dissipative"
TASK1_DIR = TRANSITION_DIR / "task1"
HAMILTONIAN_DIR = WORKSPACE / "spatrace_dtc_qrc" / "01_algorithm_validation"
QRC_CORE_DIR = WORKSPACE / "BFS-QRC" / "FINAL" / "2_linear_readout" / "core"
for directory in (
    TASK7_DIR, TASK6_DIR, TASK5_DIR, TASK1_DIR, HAMILTONIAN_DIR, QRC_CORE_DIR,
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from task1_core import _masked_errors  # noqa: E402
from task5_core import mix_suffix_bank  # noqa: E402
from task7_core import (  # noqa: E402
    Task7Config,
    _energy_mae,
    _esn_parameters,
    _fit_linear_at_selection,
    _fit_svr_at_selection,
    _rho_mae,
    _select_linear_parameters,
    _write_rows,
    load_config as load_task7_config,
)


@dataclass(frozen=True)
class Task8Config:
    task7: Task7Config
    source_internal_feature_dir: Path
    source_task7_output_dir: Path
    temporal_nodes: int
    sample_cycles: tuple[int, ...]
    observable_family: str

    @property
    def task6(self):
        return self.task7.task6

    @property
    def output_dim(self) -> int:
        return self.task7.output_dim

    @property
    def temporal_dim(self) -> int:
        return self.temporal_nodes * self.output_dim


def load_config(path: str | Path | None = None) -> Task8Config:
    path = TASK_DIR / "config.json" if path is None else Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    task7 = load_task7_config((TASK_DIR / payload["task7_config"]).resolve())
    cfg = Task8Config(
        task7=task7,
        source_internal_feature_dir=(TASK_DIR / payload["source_internal_feature_dir"]).resolve(),
        source_task7_output_dir=(TASK_DIR / payload["source_task7_output_dir"]).resolve(),
        temporal_nodes=int(payload["temporal_nodes"]),
        sample_cycles=tuple(map(int, payload["sample_times_in_readout_cycles"])),
        observable_family=str(payload["observable_family"]),
    )
    if cfg.temporal_nodes != 3 or cfg.sample_cycles != (0, 1, 2):
        raise ValueError("post-input TM3 must sample at readout cycles t0,t1,t2")
    if cfg.observable_family != "Z+ZZ" or cfg.output_dim != 78:
        raise ValueError("Task 8 locks Q12 Z+ZZ, D=78 per temporal node")
    task6 = cfg.task6
    if (task6.q, task6.g, task6.W, task6.layers, task6.injection_axis) != (12, 0.84, 0.5, 3, "XZ"):
        raise ValueError("Task 8 must inherit Q12/g0.84/W0.5/L3/XZ")
    return cfg


def _observe_z2(state, signs):
    import torch

    probabilities = torch.abs(state).square().real
    return probabilities @ signs.T


def _qrc_cycle(state, cfg, rx, phase):
    from hamiltonian_dtc_qrc import apply_1q_gate

    for wire in range(cfg.num_qubits):
        state = apply_1q_gate(state, rx, wire, cfg.num_qubits)
    return state * phase[None, :]


def _compute_qrc_post_tm3(
    tokens: np.ndarray,
    *,
    q: int,
    g: float,
    W: float,
    layers: int,
    angle_scale: float,
    injection_scale: float,
    seed: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from hamiltonian_dtc_qrc import (
        DTCQRCConfig,
        _complex_dtype,
        _real_dtype,
        apply_hamiltonian_injection,
        build_dtc_diagonal_phase,
        build_observable_signs,
        generate_dtc_params,
        rx_gate,
        set_seed,
    )

    cfg = DTCQRCConfig(
        num_qubits=q, features_per_qubit=3, g=g, W_scale=W,
        layers_per_injection=layers, complex_dtype="complex64",
    )
    if layers != 3:
        raise ValueError("Task 8 locks L=3")
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"formal QRC extraction requires CUDA, got {device}")
    set_seed(seed)
    params = generate_dtc_params(q, seed, W)
    J, h = torch.from_numpy(params["J"]), torch.from_numpy(params["h"])
    complex_dtype = _complex_dtype(cfg.complex_dtype)
    real_dtype = _real_dtype(complex_dtype)
    rx = rx_gate(cfg.g * math.pi, torch_device, complex_dtype)
    phase = build_dtc_diagonal_phase(cfg, J, h, torch_device, complex_dtype)
    signs = build_observable_signs(q, torch_device, real_dtype)
    source = torch.as_tensor(
        np.ascontiguousarray(tokens, dtype=np.float32)
        * np.float32(angle_scale * injection_scale),
        dtype=torch.float32,
        device="cpu",
    )
    output = np.empty((len(source), 3 * cfg.output_dim), dtype=np.float32)
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(source), batch_size), start=1):
            end = min(start + batch_size, len(source))
            values = source[start:end].to(torch_device, dtype=real_dtype, non_blocking=True)
            state = torch.zeros((len(values), cfg.state_dim), device=torch_device, dtype=complex_dtype)
            state[:, 0] = 1.0
            for event in range(values.shape[1]):
                state = apply_hamiltonian_injection(state, values[:, event], cfg, complex_dtype)
                for _ in range(layers):
                    state = _qrc_cycle(state, cfg, rx, phase)

            # FID-like discrete readout: measure terminal state, then evolve with
            # no new input and measure twice more.  No reset is applied here.
            snapshots = [_observe_z2(state, signs)]
            for _ in range(2):
                state = _qrc_cycle(state, cfg, rx, phase)
                snapshots.append(_observe_z2(state, signs))
            temporal = torch.stack(snapshots, dim=1).reshape(len(values), -1)
            output[start:end] = temporal.float().cpu().numpy()
            if batch_number == 1 or end == len(source):
                print(f"[QRC-post-TM3 seed={seed} {device}] {end}/{len(source)}", flush=True)
    if output.shape != (len(source), 3 * cfg.output_dim) or not np.isfinite(output).all():
        raise RuntimeError(f"invalid QRC post-TM3 output {output.shape}")
    return output, {"J": params["J"].tolist(), "h": params["h"].tolist()}


def _compute_esn_post_tm3(
    tokens: np.ndarray,
    *,
    q: int,
    layers: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if layers != 3:
        raise ValueError("Task 8 locks three internal updates per input")
    values = np.ascontiguousarray(tokens, dtype=np.float32)
    recurrent, input_weights, bias = _esn_parameters(q, seed)
    compute_device = torch.device(device)
    if compute_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"formal ESN extraction requires CUDA, got {device}")
    x = torch.tanh(torch.from_numpy(values).to(compute_device))
    recurrent_t = torch.from_numpy(recurrent).to(compute_device)
    input_weights_t = torch.from_numpy(input_weights).to(compute_device)
    bias_t = torch.from_numpy(bias).to(compute_device)
    state = torch.zeros((len(values), len(bias)), dtype=torch.float32, device=compute_device)
    with torch.inference_mode():
        for time_index in range(x.shape[1]):
            injection = 0.75 * (x[:, time_index] @ input_weights_t.T)
            for _ in range(layers):
                proposal = torch.tanh(state @ recurrent_t.T + injection + bias_t)
                state = 0.35 * state + 0.65 * proposal

        snapshots = [state.clone()]
        for _ in range(2):
            # Matched input-independent readout dynamics: the external injection
            # is exactly zero, while recurrent dynamics and fixed bias continue.
            proposal = torch.tanh(state @ recurrent_t.T + bias_t)
            state = 0.35 * state + 0.65 * proposal
            snapshots.append(state.clone())
    torch.cuda.synchronize(compute_device)
    output = torch.stack(snapshots, dim=1).reshape(len(values), -1).cpu().numpy().astype(np.float32)
    if output.shape != (len(values), 3 * len(bias)) or not np.isfinite(output).all():
        raise RuntimeError(f"invalid ESN post-TM3 output {output.shape}")
    return output, {
        "protocol": "esn_standard_dim_matched_v1_post_input_free_readout",
        "spectral_radius": 0.90, "input_scale": 0.75, "leak_rate": 0.65,
    }


def _feature_worker(
    system: str,
    seed: int,
    gpu_id: int,
    indices: np.ndarray,
    dataset_path: str,
    output_path: str,
    payload: dict[str, Any],
    error_queue,
) -> None:
    try:
        import torch

        torch.cuda.set_device(gpu_id)
        with np.load(dataset_path, allow_pickle=False) as data:
            tokens = np.ascontiguousarray(data["tokens"][indices], dtype=np.float32)
        banks, parameter_summary = [], {}
        started = time.perf_counter()
        for suffix_start in range(tokens.shape[1]):
            if system == "qrc":
                feature, parameter_summary = _compute_qrc_post_tm3(
                    tokens[:, suffix_start:], q=payload["q"], g=payload["g"], W=payload["W"],
                    layers=payload["layers"], angle_scale=payload["angle_scale"],
                    injection_scale=payload["injection_scale"], seed=seed,
                    batch_size=payload["batch_size"], device=f"cuda:{gpu_id}",
                )
            elif system == "esn":
                feature, parameter_summary = _compute_esn_post_tm3(
                    tokens[:, suffix_start:], q=payload["q"], layers=payload["layers"],
                    seed=seed, device=f"cuda:{gpu_id}",
                )
            else:
                raise ValueError(system)
            banks.append(feature)
        np.savez_compressed(
            output_path,
            indices=indices,
            suffixes=np.stack(banks).astype(np.float32),
            system=np.asarray(system), seed=np.int64(seed), gpu_id=np.int64(gpu_id),
            elapsed_seconds=np.float64(time.perf_counter() - started),
            parameter_summary=np.asarray(json.dumps(parameter_summary)),
        )
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


def compute_post_tm3_suffix_bank_dual_gpu(
    system: str,
    seed: int,
    cfg: Task8Config,
    feature_dir: Path,
    force: bool = False,
) -> Path:
    task6 = cfg.task6
    feature_dir.mkdir(parents=True, exist_ok=True)
    combined = feature_dir / f"{system}_post_tm3_suffixes_seed{seed:03d}.npz"
    if combined.exists() and not force:
        return combined
    indices = np.array_split(np.arange(task6.total_count, dtype=np.int64), len(task6.gpu_ids))
    parts = [
        feature_dir / f"{system}_post_tm3_suffixes_seed{seed:03d}_gpu{gpu}_part.npz"
        for gpu in task6.gpu_ids
    ]
    context, processes = mp.get_context("spawn"), []
    errors = context.Queue()
    payload = {
        "q": task6.q, "g": task6.g, "W": task6.W, "layers": task6.layers,
        "angle_scale": task6.angle_scale, "injection_scale": task6.injection_scale,
        "batch_size": 256,
    }
    for gpu, index, part in zip(task6.gpu_ids, indices, parts):
        process = context.Process(
            target=_feature_worker,
            args=(system, seed, gpu, index, str(task6.processed_dataset), str(part), payload, errors),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    messages = []
    while not errors.empty():
        messages.append(errors.get())
    if messages or any(process.exitcode != 0 for process in processes):
        raise RuntimeError("dual-GPU post-TM3 extraction failed\n" + "\n".join(messages))
    suffixes = np.empty((task6.source_images, task6.total_count, cfg.temporal_dim), dtype=np.float32)
    metadata = []
    for part in parts:
        with np.load(part, allow_pickle=False) as package:
            index, values = package["indices"], package["suffixes"]
            expected = (task6.source_images, len(index), cfg.temporal_dim)
            if values.shape != expected:
                raise RuntimeError(f"bad post-TM3 suffix shape {values.shape} != {expected}")
            suffixes[:, index] = values
            metadata.append({
                "gpu_id": int(package["gpu_id"]), "num_samples": len(index),
                "elapsed_seconds": float(package["elapsed_seconds"]),
                "parameter_summary": json.loads(str(package["parameter_summary"].item())),
            })
    final_path = cfg.task7.source_feature_dir / f"{system}_x_suffixes_seed{seed:03d}.npz"
    with np.load(final_path, allow_pickle=False) as source:
        final_bank = source["suffixes"].astype(np.float32)
    terminal_error = float(np.max(np.abs(suffixes[..., : cfg.output_dim] - final_bank)))
    if terminal_error > 2e-5:
        raise RuntimeError(f"post-TM3 t0 does not reproduce Task-6 final: {terminal_error}")
    np.savez_compressed(
        combined,
        suffixes=suffixes,
        system=np.asarray(system), seed=np.int64(seed),
        temporal_nodes=np.int16(cfg.temporal_nodes),
        sample_cycles=np.asarray(cfg.sample_cycles, dtype=np.int16),
        observables_per_node=np.int16(cfg.output_dim),
        temporal_feature_dim=np.int16(cfg.temporal_dim),
        observable_family=np.asarray(cfg.observable_family),
        t0_final_max_abs_error=np.float64(terminal_error),
        input_during_readout=np.asarray("none"),
        reset_during_readout=np.asarray("none"),
        g=np.float32(task6.g), W=np.float32(task6.W), layers=np.int16(task6.layers),
        injection_axis=np.asarray(task6.injection_axis),
        reset_boundaries=np.arange(task6.source_images, dtype=np.int16),
        gpu_parts=np.asarray(json.dumps(metadata)),
    )
    return combined


def _post_bank(system: str, seed: int, feature_dir: Path) -> np.ndarray:
    path = feature_dir / f"{system}_post_tm3_suffixes_seed{seed:03d}.npz"
    with np.load(path, allow_pickle=False) as package:
        return package["suffixes"].astype(np.float32)


def _new_metric_row(
    representation: str,
    decoder: str,
    seed: int,
    bank: np.ndarray,
    selection: dict[str, dict[str, float]],
    decoder_meta: dict[str, dict[str, float]],
    rho_prediction: np.ndarray,
    energy_prediction: np.ndarray,
    target_rho: np.ndarray,
    target_energy: np.ndarray,
    valid: np.ndarray,
    changed: np.ndarray,
    test: np.ndarray,
    epsilon: float,
) -> dict[str, Any]:
    all_mae, all_rmse = _masked_errors(target_rho[test], rho_prediction, valid[test])
    changed_mae, changed_rmse = _masked_errors(
        target_rho[test], rho_prediction, valid[test] & changed[test]
    )
    energy_error = energy_prediction - target_energy[test]
    rho_meta, energy_meta = decoder_meta["rho"], decoder_meta["energy"]
    return {
        "representation": representation,
        "system": representation.split("_")[0],
        "readout": "post_tm3",
        "decoder": decoder,
        "seed": seed,
        "feature_dim": bank.shape[-1],
        "rho_selected_p": selection["rho"]["p"],
        "energy_selected_p": selection["energy"]["p"],
        "rho_val_mae": rho_meta["val_mae"],
        "energy_val_mae": energy_meta["val_mae"],
        "rho_alpha": rho_meta.get("alpha", ""),
        "energy_alpha": energy_meta.get("alpha", ""),
        "rho_svr_C": rho_meta.get("C", ""),
        "rho_svr_gamma_scale": rho_meta.get("gamma_scale", ""),
        "rho_svr_gamma": rho_meta.get("gamma", ""),
        "energy_svr_C": energy_meta.get("C", ""),
        "energy_svr_gamma_scale": energy_meta.get("gamma_scale", ""),
        "energy_svr_gamma": energy_meta.get("gamma", ""),
        "svr_epsilon": epsilon if decoder == "svr" else "",
        "rho_mae_all_pairs": all_mae,
        "rho_rmse_all_pairs": all_rmse,
        "rho_mae_changed_pairs": changed_mae,
        "rho_rmse_changed_pairs": changed_rmse,
        "energy_mae_eV": float(np.mean(np.abs(energy_error))),
        "energy_rmse_eV": float(np.sqrt(np.mean(energy_error ** 2))),
    }


def evaluate(cfg: Task8Config, feature_dir: Path, output_dir: Path) -> Path:
    task6 = cfg.task6
    with np.load(task6.processed_dataset, allow_pickle=False) as data:
        splits = data["splits"]
        target_rho = data["target_delta_rho"].astype(np.float64)
        target_energy = data["target_delta_energy"].astype(np.float64)
        valid, changed = data["valid_pair_mask"], data["changed_pair_mask"]
    test, val = splits == "test", splits == "val"
    new_rows: list[dict[str, Any]] = []
    new_selections: list[dict[str, Any]] = []
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for system in ("qrc", "esn"):
        representation = f"{system}_post_tm3"
        for seed in task6.reservoir_seeds:
            bank = _post_bank(system, seed, feature_dir)
            print(f"[evaluate] {representation} seed={seed}: selecting p/alpha", flush=True)
            selection = _select_linear_parameters(
                bank, target_rho, target_energy, splits, valid,
                task6.p_grid, task6.alpha_grid,
            )
            new_selections.append({
                "representation": representation, "seed": seed,
                "rho_selected_p": selection["rho"]["p"],
                "rho_alpha": selection["rho"]["alpha"],
                "rho_linear_val_mae": selection["rho"]["val_mae"],
                "energy_selected_p": selection["energy"]["p"],
                "energy_alpha": selection["energy"]["alpha"],
                "energy_linear_val_mae": selection["energy"]["val_mae"],
            })
            features = {
                name: mix_suffix_bank(bank, selection[name]["p"])
                for name in ("rho", "energy")
            }
            predictions: dict[str, dict[str, np.ndarray]] = {"linear": {}, "svr": {}}
            meta: dict[str, dict[str, dict[str, float]]] = {
                "linear": {
                    "rho": {"val_mae": selection["rho"]["val_mae"], "alpha": selection["rho"]["alpha"]},
                    "energy": {"val_mae": selection["energy"]["val_mae"], "alpha": selection["energy"]["alpha"]},
                },
                "svr": {},
            }
            predictions["linear"]["rho"] = _fit_linear_at_selection(
                features["rho"], target_rho, splits, selection["rho"]["alpha"]
            )
            predictions["linear"]["energy"] = _fit_linear_at_selection(
                features["energy"], target_energy, splits, selection["energy"]["alpha"]
            )
            print(f"[evaluate] {representation} seed={seed}: tuning SVR", flush=True)
            predictions["svr"]["rho"], meta["svr"]["rho"] = _fit_svr_at_selection(
                features["rho"], target_rho, splits, _rho_mae, valid[val], cfg.task7, seed
            )
            predictions["svr"]["energy"], meta["svr"]["energy"] = _fit_svr_at_selection(
                features["energy"], target_energy, splits, _energy_mae, None, cfg.task7, seed
            )
            for decoder in ("linear", "svr"):
                rho_prediction = predictions[decoder]["rho"]
                energy_prediction = predictions[decoder]["energy"]
                new_rows.append(_new_metric_row(
                    representation, decoder, seed, bank, selection, meta[decoder],
                    rho_prediction, energy_prediction, target_rho, target_energy,
                    valid, changed, test, cfg.task7.svr_epsilon,
                ))
                np.savez_compressed(
                    prediction_dir / f"{representation}_{decoder}_seed{seed:03d}.npz",
                    target_rho=target_rho[test], rho_prediction=rho_prediction,
                    target_energy=target_energy[test], energy_prediction=energy_prediction,
                    valid_mask=valid[test], changed_mask=changed[test],
                    representation=np.asarray(representation), decoder=np.asarray(decoder),
                    seed=np.int64(seed), rho_p=np.float64(selection["rho"]["p"]),
                    energy_p=np.float64(selection["energy"]["p"]),
                )

    old = pd.read_csv(cfg.source_task7_output_dir / "metrics_by_seed.csv")
    old["representation"] = old["representation"].replace({
        "qrc_tm3": "qrc_internal_tm3", "esn_tm3": "esn_internal_tm3",
    })
    old.loc[old.readout == "tm3", "readout"] = "internal_tm3"
    frame = pd.concat((old, pd.DataFrame(new_rows)), ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "metrics_by_seed.csv", index=False)

    old_selection = pd.read_csv(cfg.source_task7_output_dir / "linear_representation_selection.csv")
    old_selection["representation"] = old_selection["representation"].replace({
        "qrc_tm3": "qrc_internal_tm3", "esn_tm3": "esn_internal_tm3",
    })
    selection_frame = pd.concat((old_selection, pd.DataFrame(new_selections)), ignore_index=True)
    selection_frame.to_csv(output_dir / "linear_representation_selection.csv", index=False)

    metrics = ("rho_mae_all_pairs", "rho_mae_changed_pairs", "energy_mae_eV")
    summary = frame.groupby(
        ["representation", "system", "readout", "decoder"], as_index=False
    ).agg(
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_sample_std": (metric, "std") for metric in metrics},
        num_seeds=("seed", "count"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_key = {
        (row.representation, row.decoder, int(row.seed)): row
        for row in frame.itertuples(index=False)
    }
    final_rows, internal_rows, decoder_rows, system_rows = [], [], [], []
    for system in ("qrc", "esn"):
        for decoder in ("linear", "svr"):
            for seed in task6.reservoir_seeds:
                final = by_key[(f"{system}_final", decoder, seed)]
                internal = by_key[(f"{system}_internal_tm3", decoder, seed)]
                post = by_key[(f"{system}_post_tm3", decoder, seed)]
                final_row = {"system": system, "decoder": decoder, "seed": seed}
                internal_row = {"system": system, "decoder": decoder, "seed": seed}
                for metric in metrics:
                    final_row[f"post_tm3_minus_final_{metric}"] = getattr(post, metric) - getattr(final, metric)
                    internal_row[f"post_tm3_minus_internal_tm3_{metric}"] = getattr(post, metric) - getattr(internal, metric)
                final_rows.append(final_row)
                internal_rows.append(internal_row)
    representations = (
        "qrc_final", "qrc_internal_tm3", "qrc_post_tm3",
        "esn_final", "esn_internal_tm3", "esn_post_tm3",
    )
    for representation in representations:
        for seed in task6.reservoir_seeds:
            linear = by_key[(representation, "linear", seed)]
            svr = by_key[(representation, "svr", seed)]
            row = {"representation": representation, "seed": seed}
            for metric in metrics:
                row[f"svr_minus_linear_{metric}"] = getattr(svr, metric) - getattr(linear, metric)
            decoder_rows.append(row)
    for readout in ("final", "internal_tm3", "post_tm3"):
        for decoder in ("linear", "svr"):
            for seed in task6.reservoir_seeds:
                qrc = by_key[(f"qrc_{readout}", decoder, seed)]
                esn = by_key[(f"esn_{readout}", decoder, seed)]
                row = {"readout": readout, "decoder": decoder, "seed": seed}
                for metric in metrics:
                    row[f"qrc_minus_esn_{metric}"] = getattr(qrc, metric) - getattr(esn, metric)
                system_rows.append(row)
    _write_rows(output_dir / "post_tm3_minus_final_by_seed.csv", final_rows)
    _write_rows(output_dir / "post_tm3_minus_internal_tm3_by_seed.csv", internal_rows)
    _write_rows(output_dir / "svr_minus_linear_by_seed.csv", decoder_rows)
    _write_rows(output_dir / "qrc_minus_esn_by_seed.csv", system_rows)
    report = {
        "status": "completed",
        "protocol": "transition1x-task8-q12-xz-post-input-tm3-z2-v1",
        "dataset": str(task6.processed_dataset),
        "qrc": {"Q": task6.q, "g": task6.g, "W": task6.W, "layers_per_input": task6.layers},
        "injection": "24 values as 12 simultaneous X/Z pairs; once per real time step",
        "observable_family": cfg.observable_family,
        "post_input_tm": {
            "sample_cycles": list(cfg.sample_cycles),
            "definition": "t0 after all 3 layers of final input; t1/t2 after 1/2 input-independent cycles",
            "evolution_between_samples": "one full DTC Floquet cycle: global RX followed by diagonal Z/ZZ phase",
            "input_during_readout": None,
            "reset_during_readout": None,
            "feature_dim": cfg.temporal_dim,
        },
        "p_selection": "validation-selected by standardized Ridge per representation/seed/target; frozen across decoders",
        "reservoir_seeds": list(task6.reservoir_seeds),
        "summary": summary.to_dict(orient="records"),
    }
    path = output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
