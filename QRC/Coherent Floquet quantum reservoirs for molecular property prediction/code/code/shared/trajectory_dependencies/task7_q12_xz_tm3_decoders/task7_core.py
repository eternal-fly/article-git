"""Matched TM3 readout and Linear-vs-SVR study for Transition1x Task 6.

TM3 observes the reservoir after each of the three DTC/ESN internal updates of
the final real time step.  The input is injected once per real time step; it is
not repeated and no micro-time-step serialization is introduced.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


TASK_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK_DIR.parent
WORKSPACE = TRANSITION_DIR.parent
TASK6_DIR = TRANSITION_DIR / "task6_q12_xz_dissipation_scan"
TASK5_DIR = TRANSITION_DIR / "task5_xonly_dissipative"
TASK1_DIR = TRANSITION_DIR / "task1"
HAMILTONIAN_DIR = WORKSPACE / "spatrace_dtc_qrc" / "01_algorithm_validation"
QRC_CORE_DIR = WORKSPACE / "BFS-QRC" / "FINAL" / "2_linear_readout" / "core"
for directory in (TASK6_DIR, TASK5_DIR, TASK1_DIR, HAMILTONIAN_DIR, QRC_CORE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from task1_core import _masked_errors  # noqa: E402
from task5_core import mix_suffix_bank  # noqa: E402
from task6_core import Task6Config, load_config as load_task6_config  # noqa: E402


@dataclass(frozen=True)
class Task7Config:
    task6: Task6Config
    source_feature_dir: Path
    temporal_nodes: int
    svr_C: tuple[float, ...]
    svr_gamma_scales: tuple[float, ...]
    svr_epsilon: float

    @property
    def output_dim(self) -> int:
        return self.task6.output_dim

    @property
    def temporal_dim(self) -> int:
        return self.temporal_nodes * self.output_dim


def load_config(path: str | Path | None = None) -> Task7Config:
    path = TASK_DIR / "config.json" if path is None else Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    task6_path = (TASK_DIR / payload["task6_config"]).resolve()
    task6 = load_task6_config(task6_path)
    readouts = payload["readouts"]
    cfg = Task7Config(
        task6=task6,
        source_feature_dir=(TASK_DIR / payload["source_feature_dir"]).resolve(),
        temporal_nodes=int(payload["temporal_nodes"]),
        svr_C=tuple(map(float, readouts["svr_C"])),
        svr_gamma_scales=tuple(map(float, readouts["svr_gamma_scales"])),
        svr_epsilon=float(readouts["svr_epsilon"]),
    )
    if cfg.temporal_nodes != task6.layers or cfg.temporal_nodes != 3:
        raise ValueError("TM3 requires one snapshot after each of the three internal layers")
    if task6.q != 12 or task6.injection_axis != "XZ" or task6.W != 0.5:
        raise ValueError("Task 7 must inherit Task 6 Q12/XZ/W=0.5 exactly")
    return cfg


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _compute_qrc_tm3(
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
        apply_1q_gate,
        apply_hamiltonian_injection,
        build_dtc_diagonal_phase,
        build_observable_signs,
        generate_dtc_params,
        rx_gate,
        set_seed,
    )

    cfg = DTCQRCConfig(
        num_qubits=int(q), features_per_qubit=3, g=float(g), W_scale=float(W),
        layers_per_injection=int(layers), complex_dtype="complex64",
    )
    if layers != 3:
        raise ValueError("this experiment locks three temporal nodes")
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"formal QRC extraction requires CUDA, got {device}")
    set_seed(int(seed))
    params = generate_dtc_params(q, seed, W)
    J, h = torch.from_numpy(params["J"]), torch.from_numpy(params["h"])
    complex_dtype = _complex_dtype(cfg.complex_dtype)
    real_dtype = _real_dtype(complex_dtype)
    rx = rx_gate(cfg.g * math.pi, torch_device, complex_dtype)
    phase = build_dtc_diagonal_phase(cfg, J, h, torch_device, complex_dtype)
    signs = build_observable_signs(cfg.num_qubits, torch_device, real_dtype)
    source = torch.as_tensor(
        np.ascontiguousarray(tokens, dtype=np.float32)
        * np.float32(float(angle_scale) * float(injection_scale)),
        dtype=torch.float32,
        device="cpu",
    )
    output = np.empty((len(source), layers * cfg.output_dim), dtype=np.float32)
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(source), int(batch_size)), start=1):
            end = min(start + int(batch_size), len(source))
            values = source[start:end].to(torch_device, dtype=real_dtype, non_blocking=True)
            state = torch.zeros((len(values), cfg.state_dim), device=torch_device, dtype=complex_dtype)
            state[:, 0] = 1.0
            snapshots = []
            for event in range(values.shape[1]):
                state = apply_hamiltonian_injection(state, values[:, event], cfg, complex_dtype)
                for _ in range(layers):
                    for wire in range(cfg.num_qubits):
                        state = apply_1q_gate(state, rx, wire, cfg.num_qubits)
                    state = state * phase[None, :]
                    if event == values.shape[1] - 1:
                        probabilities = torch.abs(state).square().real
                        snapshots.append(probabilities @ signs.T)
            temporal = torch.stack(snapshots, dim=1).reshape(len(values), -1)
            output[start:end] = temporal.float().cpu().numpy()
            if batch_number == 1 or end == len(source):
                print(f"[QRC-TM3 seed={seed} {device}] {end}/{len(source)}", flush=True)
    if output.shape[1] != 3 * cfg.output_dim or not np.isfinite(output).all():
        raise RuntimeError(f"invalid QRC TM3 output {output.shape}")
    return output, {"J": params["J"].tolist(), "h": params["h"].tolist()}


def _esn_parameters(q: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_dim = q * (q + 1) // 2
    input_dim = 3 * q
    rng = np.random.default_rng(int(seed))
    recurrent = rng.normal(size=(state_dim, state_dim)).astype(np.float32)
    radius = float(np.max(np.abs(np.linalg.eigvals(recurrent.astype(np.float64)))))
    recurrent *= np.float32(0.90 / max(radius, 1e-8))
    input_weights = rng.normal(size=(state_dim, input_dim)).astype(np.float32) / math.sqrt(input_dim)
    bias = rng.normal(scale=0.10, size=state_dim).astype(np.float32)
    return recurrent, input_weights, bias


def _compute_esn_tm3(
    tokens: np.ndarray,
    *,
    q: int,
    layers: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if layers != 3:
        raise ValueError("this experiment locks three temporal nodes")
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
    snapshots = []
    with torch.inference_mode():
        for time_index in range(x.shape[1]):
            injection = 0.75 * (x[:, time_index] @ input_weights_t.T)
            for _ in range(layers):
                proposal = torch.tanh(state @ recurrent_t.T + injection + bias_t)
                state = 0.35 * state + 0.65 * proposal
                if time_index == x.shape[1] - 1:
                    snapshots.append(state.clone())
    torch.cuda.synchronize(compute_device)
    output = torch.stack(snapshots, dim=1).reshape(len(values), -1).cpu().numpy().astype(np.float32)
    if output.shape != (len(values), 3 * len(bias)) or not np.isfinite(output).all():
        raise RuntimeError(f"invalid ESN TM3 output {output.shape}")
    return output, {
        "protocol": "esn_standard_dim_matched_v1", "spectral_radius": 0.90,
        "input_scale": 0.75, "leak_rate": 0.65,
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
        for start in range(tokens.shape[1]):
            if system == "qrc":
                feature, parameter_summary = _compute_qrc_tm3(
                    tokens[:, start:], q=payload["q"], g=payload["g"], W=payload["W"],
                    layers=payload["layers"], angle_scale=payload["angle_scale"],
                    injection_scale=payload["injection_scale"], seed=seed,
                    batch_size=payload["batch_size"], device=f"cuda:{gpu_id}",
                )
            elif system == "esn":
                feature, parameter_summary = _compute_esn_tm3(
                    tokens[:, start:], q=payload["q"], layers=payload["layers"],
                    seed=seed, device=f"cuda:{gpu_id}",
                )
            else:
                raise ValueError(system)
            banks.append(feature)
        np.savez_compressed(
            output_path,
            indices=indices,
            suffixes=np.stack(banks).astype(np.float32),
            system=np.asarray(system),
            seed=np.int64(seed),
            gpu_id=np.int64(gpu_id),
            elapsed_seconds=np.float64(time.perf_counter() - started),
            parameter_summary=np.asarray(json.dumps(parameter_summary)),
        )
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


def compute_tm3_suffix_bank_dual_gpu(
    system: str,
    seed: int,
    cfg: Task7Config,
    feature_dir: Path,
    force: bool = False,
) -> Path:
    task6 = cfg.task6
    feature_dir.mkdir(parents=True, exist_ok=True)
    combined = feature_dir / f"{system}_tm3_suffixes_seed{seed:03d}.npz"
    if combined.exists() and not force:
        return combined
    indices = np.array_split(np.arange(task6.total_count, dtype=np.int64), len(task6.gpu_ids))
    parts = [
        feature_dir / f"{system}_tm3_suffixes_seed{seed:03d}_gpu{gpu}_part.npz"
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
        raise RuntimeError("dual-GPU TM3 extraction failed\n" + "\n".join(messages))
    suffixes = np.empty(
        (task6.source_images, task6.total_count, cfg.temporal_dim), dtype=np.float32
    )
    metadata = []
    for part in parts:
        with np.load(part, allow_pickle=False) as package:
            index, values = package["indices"], package["suffixes"]
            expected = (task6.source_images, len(index), cfg.temporal_dim)
            if values.shape != expected:
                raise RuntimeError(f"bad TM3 suffix shape {values.shape} != {expected}")
            suffixes[:, index] = values
            metadata.append({
                "gpu_id": int(package["gpu_id"]), "num_samples": len(index),
                "elapsed_seconds": float(package["elapsed_seconds"]),
                "parameter_summary": json.loads(str(package["parameter_summary"].item())),
            })
    source_path = cfg.source_feature_dir / f"{system}_x_suffixes_seed{seed:03d}.npz"
    with np.load(source_path, allow_pickle=False) as source:
        final_bank = source["suffixes"].astype(np.float32)
    final_block_max_abs_error = float(np.max(np.abs(suffixes[..., -cfg.output_dim:] - final_bank)))
    if final_block_max_abs_error > 2e-5:
        raise RuntimeError(
            f"TM3 final block does not reproduce Task 6 final features: {final_block_max_abs_error}"
        )
    np.savez_compressed(
        combined,
        suffixes=suffixes,
        system=np.asarray(system),
        seed=np.int64(seed),
        temporal_nodes=np.int16(cfg.temporal_nodes),
        observables_per_node=np.int16(cfg.output_dim),
        temporal_feature_dim=np.int16(cfg.temporal_dim),
        final_block_max_abs_error=np.float64(final_block_max_abs_error),
        g=np.float32(task6.g), W=np.float32(task6.W), layers=np.int16(task6.layers),
        injection_axis=np.asarray(task6.injection_axis),
        reset_boundaries=np.arange(task6.source_images, dtype=np.int16),
        gpu_parts=np.asarray(json.dumps(metadata)),
    )
    return combined


def _representation_bank(
    representation: str, seed: int, cfg: Task7Config, feature_dir: Path
) -> np.ndarray:
    system, kind = representation.split("_")
    if kind == "final":
        path = cfg.source_feature_dir / f"{system}_x_suffixes_seed{seed:03d}.npz"
    elif kind == "tm3":
        path = feature_dir / f"{system}_tm3_suffixes_seed{seed:03d}.npz"
    else:
        raise ValueError(representation)
    with np.load(path, allow_pickle=False) as package:
        return package["suffixes"].astype(np.float32)


def _rho_mae(y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - y)[mask]))


def _energy_mae(y: np.ndarray, prediction: np.ndarray, _mask: np.ndarray | None = None) -> float:
    return float(np.mean(np.abs(prediction - y)))


def _select_linear_parameters(
    bank: np.ndarray,
    target_rho: np.ndarray,
    target_energy: np.ndarray,
    splits: np.ndarray,
    valid: np.ndarray,
    p_grid: tuple[float, ...],
    alpha_grid: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    train, val = splits == "train", splits == "val"
    combined_target = np.column_stack((target_rho, target_energy))
    best: dict[str, tuple[tuple[float, float, float], float, float]] = {}
    for p in p_grid:
        features = mix_suffix_bank(bank, p)
        scaler = StandardScaler().fit(features[train])
        x_train, x_val = scaler.transform(features[train]), scaler.transform(features[val])
        for alpha in alpha_grid:
            prediction = Ridge(alpha=alpha).fit(x_train, combined_target[train]).predict(x_val)
            scores = {
                "rho": _rho_mae(target_rho[val], prediction[:, : target_rho.shape[1]], valid[val]),
                "energy": _energy_mae(target_energy[val], prediction[:, -1]),
            }
            for target_name, score in scores.items():
                key = (score, float(p), float(alpha))
                if target_name not in best or key < best[target_name][0]:
                    best[target_name] = (key, float(p), float(alpha))
    return {
        name: {"val_mae": values[0][0], "p": values[1], "alpha": values[2]}
        for name, values in best.items()
    }


def _base_gamma(x_train_scaled: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    count = min(512, len(x_train_scaled))
    selected = rng.choice(len(x_train_scaled), size=count, replace=False)
    distances = pairwise_distances(x_train_scaled[selected], metric="sqeuclidean")
    nonzero = distances[np.triu_indices_from(distances, k=1)]
    nonzero = nonzero[nonzero > 1e-12]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    return 1.0 / max(median, 1e-12)


def _fit_one_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
) -> np.ndarray:
    return SVR(kernel="rbf", C=C, gamma=gamma, epsilon=epsilon).fit(
        x_train, y_train
    ).predict(x_predict)


def _svr_predict_multi(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    C: float,
    gamma: float,
    epsilon: float,
) -> np.ndarray:
    if y_train.ndim == 1:
        return _fit_one_svr(x_train, y_train, x_predict, C, gamma, epsilon)
    columns = Parallel(n_jobs=min(y_train.shape[1], 8), prefer="threads")(
        delayed(_fit_one_svr)(x_train, y_train[:, column], x_predict, C, gamma, epsilon)
        for column in range(y_train.shape[1])
    )
    return np.column_stack(columns)


def _fit_linear_at_selection(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    alpha: float,
) -> np.ndarray:
    train_val, test = splits != "test", splits == "test"
    scaler = StandardScaler().fit(features[train_val])
    model = Ridge(alpha=alpha).fit(scaler.transform(features[train_val]), target[train_val])
    return np.asarray(model.predict(scaler.transform(features[test])))


def _fit_svr_at_selection(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float],
    validation_mask: np.ndarray | None,
    cfg: Task7Config,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    train, val, test = splits == "train", splits == "val", splits == "test"
    scaler = StandardScaler().fit(features[train])
    x_train, x_val = scaler.transform(features[train]), scaler.transform(features[val])
    base_gamma = _base_gamma(x_train, seed)
    best = None
    for C in cfg.svr_C:
        for gamma_scale in cfg.svr_gamma_scales:
            gamma = float(gamma_scale) * base_gamma
            prediction = _svr_predict_multi(
                x_train, target[train], x_val, C, gamma, cfg.svr_epsilon
            )
            score = metric(target[val], prediction, validation_mask)
            key = (float(score), float(C), float(gamma_scale))
            if best is None or key < best[0]:
                best = (key, float(C), float(gamma_scale))
    if best is None:
        raise RuntimeError("SVR validation grid produced no model")
    _, selected_C, selected_gamma_scale = best
    train_val = train | val
    scaler = StandardScaler().fit(features[train_val])
    x_train_val, x_test = scaler.transform(features[train_val]), scaler.transform(features[test])
    final_base_gamma = _base_gamma(x_train_val, seed)
    final_gamma = selected_gamma_scale * final_base_gamma
    prediction = _svr_predict_multi(
        x_train_val, target[train_val], x_test,
        selected_C, final_gamma, cfg.svr_epsilon,
    )
    return np.asarray(prediction), {
        "val_mae": float(best[0][0]), "C": selected_C,
        "gamma_scale": selected_gamma_scale, "gamma": float(final_gamma),
        "epsilon": cfg.svr_epsilon,
    }


def evaluate(cfg: Task7Config, feature_dir: Path, output_dir: Path) -> Path:
    task6 = cfg.task6
    with np.load(task6.processed_dataset, allow_pickle=False) as data:
        splits = data["splits"]
        target_rho = data["target_delta_rho"].astype(np.float64)
        target_energy = data["target_delta_energy"].astype(np.float64)
        valid, changed = data["valid_pair_mask"], data["changed_pair_mask"]
    test = splits == "test"
    representations = ("qrc_final", "qrc_tm3", "esn_final", "esn_tm3")
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for representation in representations:
        for seed in task6.reservoir_seeds:
            print(f"[evaluate] {representation} seed={seed}: selecting p/alpha", flush=True)
            bank = _representation_bank(representation, seed, cfg, feature_dir)
            selection = _select_linear_parameters(
                bank, target_rho, target_energy, splits, valid,
                task6.p_grid, task6.alpha_grid,
            )
            selection_rows.append({
                "representation": representation, "seed": seed,
                "rho_selected_p": selection["rho"]["p"],
                "rho_alpha": selection["rho"]["alpha"],
                "rho_linear_val_mae": selection["rho"]["val_mae"],
                "energy_selected_p": selection["energy"]["p"],
                "energy_alpha": selection["energy"]["alpha"],
                "energy_linear_val_mae": selection["energy"]["val_mae"],
            })
            features = {
                target_name: mix_suffix_bank(bank, selection[target_name]["p"])
                for target_name in ("rho", "energy")
            }
            predictions: dict[str, dict[str, np.ndarray]] = {"linear": {}, "svr": {}}
            decoder_meta: dict[str, dict[str, dict[str, float]]] = {
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
            predictions["svr"]["rho"], decoder_meta["svr"]["rho"] = _fit_svr_at_selection(
                features["rho"], target_rho, splits, _rho_mae, valid[splits == "val"], cfg, seed
            )
            predictions["svr"]["energy"], decoder_meta["svr"]["energy"] = _fit_svr_at_selection(
                features["energy"], target_energy, splits, _energy_mae, None, cfg, seed
            )
            for decoder in ("linear", "svr"):
                rho_prediction = predictions[decoder]["rho"]
                energy_prediction = predictions[decoder]["energy"]
                all_mae, all_rmse = _masked_errors(target_rho[test], rho_prediction, valid[test])
                changed_mae, changed_rmse = _masked_errors(
                    target_rho[test], rho_prediction, valid[test] & changed[test]
                )
                energy_error = energy_prediction - target_energy[test]
                rho_meta, energy_meta = decoder_meta[decoder]["rho"], decoder_meta[decoder]["energy"]
                rows.append({
                    "representation": representation,
                    "system": representation.split("_")[0],
                    "readout": representation.split("_")[1],
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
                    "svr_epsilon": cfg.svr_epsilon if decoder == "svr" else "",
                    "rho_mae_all_pairs": all_mae,
                    "rho_rmse_all_pairs": all_rmse,
                    "rho_mae_changed_pairs": changed_mae,
                    "rho_rmse_changed_pairs": changed_rmse,
                    "energy_mae_eV": float(np.mean(np.abs(energy_error))),
                    "energy_rmse_eV": float(np.sqrt(np.mean(energy_error ** 2))),
                })
                np.savez_compressed(
                    prediction_dir / f"{representation}_{decoder}_seed{seed:03d}.npz",
                    target_rho=target_rho[test], rho_prediction=rho_prediction,
                    target_energy=target_energy[test], energy_prediction=energy_prediction,
                    valid_mask=valid[test], changed_mask=changed[test],
                    representation=np.asarray(representation), decoder=np.asarray(decoder),
                    seed=np.int64(seed), rho_p=np.float64(selection["rho"]["p"]),
                    energy_p=np.float64(selection["energy"]["p"]),
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "linear_representation_selection.csv", selection_rows)
    _write_rows(output_dir / "metrics_by_seed.csv", rows)
    frame = pd.DataFrame(rows)
    metrics = ("rho_mae_all_pairs", "rho_mae_changed_pairs", "energy_mae_eV")
    summary = frame.groupby(["representation", "system", "readout", "decoder"], as_index=False).agg(
        **{
            f"{metric}_mean": (metric, "mean") for metric in metrics
        },
        **{
            f"{metric}_sample_std": (metric, "std") for metric in metrics
        },
        num_seeds=("seed", "count"),
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_key = {
        (row["representation"], row["decoder"], int(row["seed"])): row for row in rows
    }
    tm_rows, decoder_rows = [], []
    for system in ("qrc", "esn"):
        for decoder in ("linear", "svr"):
            for seed in task6.reservoir_seeds:
                final = by_key[(f"{system}_final", decoder, seed)]
                tm3 = by_key[(f"{system}_tm3", decoder, seed)]
                row = {"system": system, "decoder": decoder, "seed": seed}
                for metric in metrics:
                    row[f"tm3_minus_final_{metric}"] = tm3[metric] - final[metric]
                tm_rows.append(row)
    for representation in representations:
        for seed in task6.reservoir_seeds:
            linear = by_key[(representation, "linear", seed)]
            svr = by_key[(representation, "svr", seed)]
            row = {"representation": representation, "seed": seed}
            for metric in metrics:
                row[f"svr_minus_linear_{metric}"] = svr[metric] - linear[metric]
            decoder_rows.append(row)
    _write_rows(output_dir / "tm3_minus_final_by_seed.csv", tm_rows)
    _write_rows(output_dir / "svr_minus_linear_by_seed.csv", decoder_rows)
    report = {
        "status": "completed",
        "protocol": "transition1x-task7-q12-xz-tm3-linear-svr-v1",
        "dataset": str(task6.processed_dataset),
        "qrc": {"Q": task6.q, "g": task6.g, "W": task6.W, "layers": task6.layers},
        "injection": "24 values injected simultaneously as 12 (X,Z) pairs once per real time step",
        "temporal_readout": "Z+ZZ after each of the 3 internal layers of the final real time step",
        "representations": {"final": task6.output_dim, "tm3": cfg.temporal_dim},
        "p_selection": "per representation/seed/target, validation-selected by standardized Ridge; frozen across decoders",
        "svr_grid": {"C": list(cfg.svr_C), "gamma_scale": list(cfg.svr_gamma_scales), "epsilon": cfg.svr_epsilon},
        "reservoir_seeds": list(task6.reservoir_seeds),
        "summary": summary.to_dict(orient="records"),
    }
    path = output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
