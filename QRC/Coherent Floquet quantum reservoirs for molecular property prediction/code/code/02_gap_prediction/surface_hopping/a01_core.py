"""SHNITSEL A01 real-time surface-hopping benchmark for Q15 DTC-QRC."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


TASK_DIR = Path(__file__).resolve().parent
WORKSPACE = TASK_DIR.parents[1]
TASK8_DIR = TASK_DIR.parent / "task8_q12_xz_post_tm3_z2"
TASK7_DIR = TASK_DIR.parent / "task7_q12_xz_tm3_decoders"
HAMILTONIAN_DIR = WORKSPACE / "spatrace_dtc_qrc" / "01_algorithm_validation"
for directory in (TASK8_DIR, TASK7_DIR, HAMILTONIAN_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


PAIR_INDEX = np.asarray(list(combinations(range(6), 2)), dtype=np.int16)
TARGET_NAMES = ("gap_1fs", "gap_5fs", "gap_10fs", "min_gap_next_5fs")
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class Config:
    payload: dict[str, Any]
    raw_path: Path
    archive_path: Path
    history_steps: int
    horizons: tuple[int, ...]
    min_gap_steps: int
    samples_per_trajectory: int
    split_seed: int
    q: int
    clip: float
    angle_scale: float
    g: float
    W: float
    layers: int
    p_grid: tuple[float, ...]
    seeds: tuple[int, ...]
    batch_size: int

    @property
    def output_dim(self) -> int:
        return self.q * (self.q + 1) // 2

    @property
    def tm3_dim(self) -> int:
        return 3 * self.output_dim

    @property
    def processed_path(self) -> Path:
        return TASK_DIR / "data" / "processed" / "a01_windows.npz"


def load_config() -> Config:
    payload = json.loads((TASK_DIR / "config.json").read_text(encoding="utf-8"))
    data, qrc = payload["dataset"], payload["qrc"]
    cfg = Config(
        payload=payload,
        raw_path=(TASK_DIR / data["dynamic_file"]).resolve(),
        archive_path=(TASK_DIR / data["archive"]).resolve(),
        history_steps=int(data["history_steps"]),
        horizons=tuple(map(int, data["forecast_horizons_steps"])),
        min_gap_steps=int(data["minimum_gap_window_steps"]),
        samples_per_trajectory=int(data["samples_per_trajectory"]),
        split_seed=int(data["split_seed"]),
        q=int(qrc["num_qubits"]), clip=float(qrc["z_clip"]),
        angle_scale=float(qrc["injection_angle_scale"]),
        g=float(qrc["g"]), W=float(qrc["W"]),
        layers=int(qrc["layers_per_input"]),
        p_grid=tuple(map(float, qrc["p_grid"])),
        seeds=tuple(map(int, qrc["reservoir_seeds"])),
        batch_size=int(qrc["batch_size"]),
    )
    if (cfg.q, cfg.g, cfg.W, cfg.layers) != (15, 0.84, 0.5, 3):
        raise ValueError("protocol locks Q15, g=0.84, W=0.5, L=3")
    if len(PAIR_INDEX) != cfg.q or (cfg.output_dim, cfg.tm3_dim) != (120, 360):
        raise ValueError("ethene must map 15 pairs to Q15 and Z+ZZ=120D/TM3=360D")
    if cfg.history_steps != 20 or cfg.horizons != (2, 10, 20) or cfg.min_gap_steps != 10:
        raise ValueError("protocol locks 10-fs history and 1/5/10-fs targets")
    tiny_p = {0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08}
    uniform_p = {round(0.05 * index, 2) for index in range(1, 21)}
    expected_p_grid = tuple(sorted(tiny_p | uniform_p))
    if len(cfg.p_grid) != len(expected_p_grid) or not np.allclose(
        cfg.p_grid, expected_p_grid
    ):
        raise ValueError(
            "protocol locks the retained tiny-p cells plus the uniform 0.05 grid"
        )
    return cfg


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_data(cfg: Config) -> dict[str, Any]:
    if not cfg.raw_path.exists():
        raise FileNotFoundError(cfg.raw_path)
    expected = cfg.payload["dataset"]["dynamic_sha256"]
    actual = _digest(cfg.raw_path, "sha256")
    if actual != expected:
        raise RuntimeError(f"SHNITSEL A01 hash mismatch: {actual}")
    with h5py.File(cfg.raw_path, "r") as source:
        shapes = {key: list(source[key].shape) for key in ("atXYZ", "energy", "trajid", "time", "astate")}
        atoms = [value.decode() for value in source["atNames"][...]]
        dt = float(source.attrs["delta_t"])
        energy_units = source["energy"].attrs["units"]
        if isinstance(energy_units, bytes):
            energy_units = energy_units.decode()
    if shapes != {
        "atXYZ": [6, 3, 62031], "energy": [3, 62031], "trajid": [62031],
        "time": [62031], "astate": [62031],
    } or atoms != ["C", "C", "H", "H", "H", "H"] or dt != 0.5 or energy_units != "eV":
        raise RuntimeError(f"unexpected A01 contract: {shapes}, {atoms}, {dt}, {energy_units}")
    return {"verified": True, "sha256": actual, "shapes": shapes, "atoms": atoms,
            "delta_t_fs": dt, "energy_units": energy_units}


def _trajectory_splits(trajid: np.ndarray, astate: np.ndarray, seed: int) -> dict[int, int]:
    unique = np.unique(trajid)
    hop_bins = []
    for trajectory in unique:
        states = astate[trajid == trajectory]
        hops = int(np.count_nonzero(states[1:] != states[:-1]))
        hop_bins.append(0 if hops == 0 else 1 if hops == 1 else 2)
    hop_bins = np.asarray(hop_bins)
    outer = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
    train_idx, remainder_idx = next(outer.split(unique, hop_bins))
    inner = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
    val_rel, test_rel = next(inner.split(unique[remainder_idx], hop_bins[remainder_idx]))
    mapping: dict[int, int] = {}
    for index in train_idx:
        mapping[int(unique[index])] = 0
    for index in remainder_idx[val_rel]:
        mapping[int(unique[index])] = 1
    for index in remainder_idx[test_rel]:
        mapping[int(unique[index])] = 2
    if len(mapping) != len(unique):
        raise AssertionError("trajectory split is incomplete")
    return mapping


def _pair_distances(xyz: np.ndarray) -> np.ndarray:
    delta = xyz[:, PAIR_INDEX[:, 0]] - xyz[:, PAIR_INDEX[:, 1]]
    return np.linalg.norm(delta, axis=-1)


def build_dataset(cfg: Config, force: bool = False) -> Path:
    if cfg.processed_path.exists() and not force:
        return cfg.processed_path
    raw_audit = verify_raw_data(cfg)
    with h5py.File(cfg.raw_path, "r") as source:
        xyz = np.moveaxis(source["atXYZ"][...], -1, 0)
        energy = source["energy"][...].T
        trajid = source["trajid"][...]
        time_fs = source["time"][...]
        astate = source["astate"][...]
    split_map = _trajectory_splits(trajid, astate, cfg.split_seed)
    token_windows, targets, trajectory_ids, endpoints, splits = [], [], [], [], []
    current_gaps, hop_targets = [], []
    max_future = max(max(cfg.horizons), cfg.min_gap_steps)
    for trajectory in np.unique(trajid):
        frame_indices = np.flatnonzero(trajid == trajectory)
        if not np.allclose(np.diff(time_fs[frame_indices]), 0.5):
            raise RuntimeError(f"non-contiguous trajectory {trajectory}")
        valid = np.linspace(
            cfg.history_steps,
            len(frame_indices) - max_future - 1,
            cfg.samples_per_trajectory,
            dtype=np.int64,
        )
        if len(np.unique(valid)) != cfg.samples_per_trajectory:
            raise RuntimeError(f"trajectory {trajectory} is too short")
        coordinates = xyz[frame_indices]
        distances = _pair_distances(coordinates)
        gaps = energy[frame_indices, 1] - energy[frame_indices, 0]
        states = astate[frame_indices]
        for endpoint in valid:
            distance_history = distances[endpoint - cfg.history_steps:endpoint + 1]
            radial_velocity = np.diff(distance_history, axis=0) / 0.5
            pair_values = np.stack((distance_history[1:], radial_velocity), axis=-1)
            future = [gaps[endpoint + horizon] for horizon in cfg.horizons]
            future.append(np.min(gaps[endpoint + 1:endpoint + cfg.min_gap_steps + 1]))
            token_windows.append(pair_values)
            targets.append(future)
            trajectory_ids.append(int(trajectory))
            endpoints.append(int(endpoint))
            splits.append(split_map[int(trajectory)])
            current_gaps.append(gaps[endpoint])
            hop_targets.append(float(np.any(states[endpoint + 1:endpoint + cfg.min_gap_steps + 1] != states[endpoint])))
    pair_windows = np.asarray(token_windows, dtype=np.float32)
    split = np.asarray(splits, dtype=np.int8)
    train_values = pair_windows[split == 0]
    mean = train_values.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = train_values.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    normalized = np.clip((pair_windows - mean) / std, -cfg.clip, cfg.clip)
    tokens = np.zeros((len(normalized), cfg.history_steps, cfg.q, 3), dtype=np.float32)
    tokens[..., 0] = normalized[..., 0]
    tokens[..., 2] = normalized[..., 1]
    cfg.processed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.processed_path,
        tokens=tokens.reshape(len(tokens), cfg.history_steps, 3 * cfg.q),
        pair_windows=pair_windows,
        targets=np.asarray(targets, dtype=np.float32),
        target_names=np.asarray(TARGET_NAMES),
        current_gap=np.asarray(current_gaps, dtype=np.float32),
        hop_within_5fs=np.asarray(hop_targets, dtype=np.float32),
        trajectory_id=np.asarray(trajectory_ids, dtype=np.int32),
        endpoint=np.asarray(endpoints, dtype=np.int16), split=split,
        split_names=np.asarray(SPLIT_NAMES), pair_index=PAIR_INDEX,
        normalization_mean=mean, normalization_std=std,
        normalization_clip=np.float32(cfg.clip),
        raw_audit=np.asarray(json.dumps(raw_audit)),
        source_generated=np.asarray(False),
    )
    return cfg.processed_path


def reset_suffix_weights(length: int, p: float) -> np.ndarray:
    if length < 1 or not 0.0 <= p <= 1.0:
        raise ValueError("invalid reset arguments")
    survival = 1.0 - p
    weights = np.empty(length, dtype=np.float64)
    weights[0] = survival ** (length - 1)
    for start in range(1, length):
        weights[start] = p * survival ** (length - 1 - start)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise AssertionError(weights)
    return weights


def mix_suffix_bank(bank: np.ndarray, p: float, readout: str, output_dim: int) -> np.ndarray:
    mixed = np.tensordot(reset_suffix_weights(len(bank), p), bank, axes=(0, 0))
    if readout == "final":
        return np.asarray(mixed[:, :output_dim], dtype=np.float32)
    if readout == "tm3":
        return np.asarray(mixed, dtype=np.float32)
    raise ValueError(readout)


def feature_path(system: str, seed: int) -> Path:
    return TASK_DIR / "features" / f"{system}_post_tm3_suffixes_seed{seed:03d}.npz"


def _compute_features(tokens: np.ndarray, system: str, seed: int, cfg: Config, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    from task8_core import _compute_esn_post_tm3, _compute_qrc_post_tm3
    banks, metadata = [], {}
    for suffix in range(cfg.history_steps):
        if system == "qrc":
            values, metadata = _compute_qrc_post_tm3(
                tokens[:, suffix:], q=cfg.q, g=cfg.g, W=cfg.W, layers=cfg.layers,
                angle_scale=cfg.angle_scale, injection_scale=1.0, seed=seed,
                batch_size=cfg.batch_size, device=device,
            )
        elif system == "esn":
            values, metadata = _compute_esn_post_tm3(
                tokens[:, suffix:], q=cfg.q, layers=cfg.layers, seed=seed, device=device,
            )
        else:
            raise ValueError(system)
        banks.append(values)
        print(f"[{system} seed={seed} {device}] suffix {suffix + 1}/{cfg.history_steps}", flush=True)
    return np.stack(banks).astype(np.float32), metadata


def compute_feature_bank(cfg: Config, system: str, seed: int, device: str, force: bool = False) -> Path:
    path = feature_path(system, seed)
    if path.exists() and not force:
        return path
    if system not in ("qrc", "esn") or seed not in cfg.seeds:
        raise ValueError("invalid feature request")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("formal Q15/ESN extraction requires a visible CUDA GPU")
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32)
    started = time.perf_counter()
    bank, metadata = _compute_features(tokens, system, seed, cfg, device)
    expected = (cfg.history_steps, len(tokens), cfg.tm3_dim)
    if bank.shape != expected or not np.isfinite(bank).all():
        raise RuntimeError(f"bad {system} feature bank {bank.shape} != {expected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, suffixes=bank, system=np.asarray(system), seed=np.int64(seed),
        elapsed_seconds=np.float64(time.perf_counter() - started),
        parameter_summary=np.asarray(json.dumps(metadata)),
        readout=np.asarray("post-input t0,t1,t2; no input/reset in readout window"),
        observable_family=np.asarray("Z+ZZ" if system == "qrc" else "dimension-matched ESN state"),
        real_inputs_per_step=np.int16(30), repeated_inputs_per_step=np.int16(0),
    )
    return path


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae_ev": float(mean_absolute_error(target, prediction)),
        "rmse_ev": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
    }


def _scaled_arrays(x_train: np.ndarray, x_validation: np.ndarray, x_test: np.ndarray):
    scaler = StandardScaler().fit(x_train)
    return scaler, scaler.transform(x_train), scaler.transform(x_validation), scaler.transform(x_test)


def _best_ridge(x_train, y_train, x_validation, y_validation, alpha_grid):
    scaler = StandardScaler().fit(x_train)
    xt, xv = scaler.transform(x_train), scaler.transform(x_validation)
    best = None
    for alpha in alpha_grid:
        prediction = Ridge(alpha=alpha).fit(xt, y_train).predict(xv)
        candidate = (float(mean_absolute_error(y_validation, prediction)), float(alpha))
        if best is None or candidate < best:
            best = candidate
    return {"validation_mae_ev": best[0], "alpha": best[1]}


def _best_svr(x_train, y_train, x_validation, y_validation, candidates, cache_size):
    scaler = StandardScaler().fit(x_train)
    xt, xv = scaler.transform(x_train), scaler.transform(x_validation)
    dimension = xt.shape[1]
    best = None
    for C, gamma_scale, epsilon in candidates:
        model = SVR(C=C, gamma=gamma_scale / dimension, epsilon=epsilon, cache_size=cache_size)
        prediction = model.fit(xt, y_train).predict(xv)
        candidate = (float(mean_absolute_error(y_validation, prediction)), C, gamma_scale, epsilon)
        if best is None or candidate < best:
            best = candidate
    return {"validation_mae_ev": best[0], "C": best[1], "gamma_scale": best[2],
            "gamma": best[2] / dimension, "epsilon": best[3]}


def _refit_ridge(x_train, y_train, x_validation, y_validation, x_test, alpha):
    x = np.concatenate((x_train, x_validation)); y = np.concatenate((y_train, y_validation))
    scaler = StandardScaler().fit(x)
    return Ridge(alpha=alpha).fit(scaler.transform(x), y).predict(scaler.transform(x_test))


def _refit_svr(x_train, y_train, x_validation, y_validation, x_test, params, cache_size):
    x = np.concatenate((x_train, x_validation)); y = np.concatenate((y_train, y_validation))
    scaler = StandardScaler().fit(x)
    model = SVR(C=params["C"], gamma=params["gamma"], epsilon=params["epsilon"], cache_size=cache_size)
    return model.fit(scaler.transform(x), y).predict(scaler.transform(x_test))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _evaluate_reservoir_seed(system: str, seed: int) -> dict[str, list[dict[str, Any]]]:
    cfg = load_config()
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        targets = package["targets"].astype(np.float64); split = package["split"]
        trajectory_id = package["trajectory_id"]
    train, validation, test = split == 0, split == 1, split == 2
    decoder = cfg.payload["decoders"]
    alpha_grid = tuple(map(float, decoder["ridge_alpha_grid"]))
    reference = decoder["svr_p_scan_reference"]
    reference_candidates = [(float(reference["C"]), float(reference["gamma_scale"]), float(reference["epsilon"]))]
    full_candidates = [(float(C), float(gamma), float(epsilon))
                       for C in decoder["svr_C_grid"] for gamma in decoder["svr_gamma_scale_grid"]
                       for epsilon in decoder["svr_epsilon_grid"]]
    cache_size = float(decoder["svr_cache_size_mb"])
    scan_rows, selected_rows, prediction_rows = [], [], []
    prediction_dir = TASK_DIR / "outputs" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with np.load(feature_path(system, seed), allow_pickle=False) as package:
        bank = package["suffixes"].astype(np.float32)
    for readout in ("final", "tm3"):
        feature_by_p = {p: mix_suffix_bank(bank, p, readout, cfg.output_dim) for p in cfg.p_grid}
        for target_index, target_name in enumerate(TARGET_NAMES):
            y = targets[:, target_index]
            linear_by_p, svr_by_p = {}, {}
            for p, features in feature_by_p.items():
                linear = _best_ridge(features[train], y[train], features[validation], y[validation], alpha_grid)
                reference_svr = _best_svr(features[train], y[train], features[validation], y[validation], reference_candidates, cache_size)
                linear_by_p[p], svr_by_p[p] = linear, reference_svr
                scan_rows.extend((
                    {"system": system, "seed": seed, "readout": readout, "target": target_name,
                     "decoder": "linear", "p": p, **linear},
                    {"system": system, "seed": seed, "readout": readout, "target": target_name,
                     "decoder": "svr", "p": p, "svr_stage": "fixed-reference-p-scan", **reference_svr},
                ))
            for decoder_name, selected_p in (
                ("linear", min(cfg.p_grid, key=lambda p: (linear_by_p[p]["validation_mae_ev"], p))),
                ("svr", min(cfg.p_grid, key=lambda p: (svr_by_p[p]["validation_mae_ev"], p))),
            ):
                features = feature_by_p[selected_p]
                if decoder_name == "linear":
                    params = linear_by_p[selected_p]
                    prediction = _refit_ridge(features[train], y[train], features[validation], y[validation],
                                              features[test], params["alpha"])
                else:
                    params = _best_svr(features[train], y[train], features[validation], y[validation],
                                       full_candidates, cache_size)
                    prediction = _refit_svr(features[train], y[train], features[validation], y[validation],
                                            features[test], params, cache_size)
                row = {"system": system, "seed": seed, "readout": readout, "feature_dim": features.shape[1],
                       "target": target_name, "decoder": decoder_name, "selected_p": selected_p,
                       "validation_mae_ev": params["validation_mae_ev"], **regression_metrics(y[test], prediction)}
                for key in ("alpha", "C", "gamma_scale", "gamma", "epsilon"):
                    row[key] = params.get(key, "")
                selected_rows.append(row)
                np.savez_compressed(
                    prediction_dir / f"{system}_seed{seed:03d}_{readout}_{decoder_name}_{target_name}.npz",
                    target=y[test].astype(np.float32), prediction=prediction.astype(np.float32),
                    trajectory_id=trajectory_id[test], selected_p=np.float64(selected_p),
                )
                print(f"[{system} seed={seed}] {target_name} {readout}/{decoder_name} "
                      f"p={selected_p:g} test={row['mae_ev']:.5f} eV", flush=True)
    return {"scan": scan_rows, "selected": selected_rows}


def evaluate_reservoirs(cfg: Config) -> Path:
    jobs = [(system, seed) for system in ("qrc", "esn") for seed in cfg.seeds]
    missing = [feature_path(*job) for job in jobs if not feature_path(*job).exists()]
    if missing:
        raise FileNotFoundError("missing banks:\n" + "\n".join(map(str, missing)))
    scan_rows, selected_rows = [], []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {executor.submit(_evaluate_reservoir_seed, *job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            scan_rows.extend(result["scan"]); selected_rows.extend(result["selected"])
    scan_rows.sort(key=lambda row: (row["system"], row["seed"], row["readout"],
                                    row["target"], row["decoder"], row["p"]))
    selected_rows.sort(key=lambda row: (row["system"], row["seed"], row["readout"],
                                        row["target"], row["decoder"]))
    _write_rows(TASK_DIR / "outputs" / "p_scan_validation.csv", scan_rows)
    _write_rows(TASK_DIR / "outputs" / "reservoir_selected_test.csv", selected_rows)
    frame = pd.DataFrame(selected_rows)
    frame.groupby(["system", "readout", "decoder", "target"], as_index=False).agg(
        mae_ev_mean=("mae_ev", "mean"), mae_ev_sample_std=("mae_ev", "std"),
        rmse_ev_mean=("rmse_ev", "mean"), r2_mean=("r2", "mean"), num_seeds=("seed", "size"),
    ).to_csv(TASK_DIR / "outputs" / "reservoir_summary.csv", index=False)
    best_rows = []
    for (system, seed, target), group in frame.groupby(["system", "seed", "target"]):
        chosen = group.sort_values(["validation_mae_ev", "feature_dim", "decoder"]).iloc[0].to_dict()
        chosen["selection_scope"] = "readout and decoder selected only by validation MAE"
        best_rows.append(chosen)
    _write_rows(TASK_DIR / "outputs" / "reservoir_best_validation_selected.csv", best_rows)
    return TASK_DIR / "outputs" / "reservoir_best_validation_selected.csv"


def _classical_candidates(name: str):
    if name == "extra_trees":
        for max_features in (0.5, 1.0):
            for min_samples_leaf in (1, 3):
                yield {"max_features": max_features, "min_samples_leaf": min_samples_leaf}
    elif name == "random_forest":
        for max_features in (0.5, 1.0):
            for min_samples_leaf in (1, 3):
                yield {"max_features": max_features, "min_samples_leaf": min_samples_leaf}
    elif name == "hist_gradient_boosting":
        for max_leaf_nodes in (15, 31):
            for l2_regularization in (0.0, 1.0):
                yield {"max_leaf_nodes": max_leaf_nodes, "l2_regularization": l2_regularization}
    elif name == "knn":
        for n_neighbors in (5, 15, 30):
            yield {"n_neighbors": n_neighbors, "weights": "distance"}
    else:
        raise ValueError(name)


def _classical_model(name: str, params: dict[str, Any]):
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=400, random_state=0, n_jobs=-1, **params)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1, **params)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0, **params)
    if name == "knn":
        return KNeighborsRegressor(n_jobs=-1, **params)
    raise ValueError(name)


def evaluate_classical(cfg: Config) -> Path:
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32).reshape(-1, cfg.history_steps, cfg.q, 3)
        targets = package["targets"].astype(np.float64); split = package["split"]
        current_gap = package["current_gap"].astype(np.float64)
        trajectory_id = package["trajectory_id"]
    x_history = tokens[..., (0, 2)].reshape(len(tokens), -1)
    x_last = tokens[:, -1][:, :, (0, 2)].reshape(len(tokens), -1)
    train, validation, test = split == 0, split == 1, split == 2
    decoder = cfg.payload["decoders"]
    alpha_grid = tuple(map(float, decoder["ridge_alpha_grid"]))
    svr_candidates = [(float(C), float(gamma), float(epsilon))
                      for C in decoder["svr_C_grid"] for gamma in decoder["svr_gamma_scale_grid"]
                      for epsilon in decoder["svr_epsilon_grid"]]
    cache_size = float(decoder["svr_cache_size_mb"])
    rows = []
    prediction_dir = TASK_DIR / "outputs" / "predictions"
    for target_index, target_name in enumerate(TARGET_NAMES):
        y = targets[:, target_index]
        persistence = current_gap[test]
        rows.append({"model": "current_gap_persistence", "target": target_name,
                     "validation_mae_ev": float(mean_absolute_error(y[validation], current_gap[validation])),
                     "input_scope": "target-informed current gap", **regression_metrics(y[test], persistence)})
        np.savez_compressed(prediction_dir / f"classical_persistence_{target_name}.npz",
                            target=y[test].astype(np.float32), prediction=persistence.astype(np.float32),
                            trajectory_id=trajectory_id[test])
        for scope, features in (("last_frame", x_last), ("history_10fs", x_history)):
            ridge = _best_ridge(features[train], y[train], features[validation], y[validation], alpha_grid)
            prediction = _refit_ridge(features[train], y[train], features[validation], y[validation],
                                      features[test], ridge["alpha"])
            rows.append({"model": f"ridge_{scope}", "target": target_name,
                         "validation_mae_ev": ridge["validation_mae_ev"], "alpha": ridge["alpha"],
                         "input_scope": scope, **regression_metrics(y[test], prediction)})
            svr = _best_svr(features[train], y[train], features[validation], y[validation],
                            svr_candidates, cache_size)
            prediction = _refit_svr(features[train], y[train], features[validation], y[validation],
                                    features[test], svr, cache_size)
            rows.append({"model": f"svr_{scope}", "target": target_name,
                         "validation_mae_ev": svr["validation_mae_ev"], "C": svr["C"],
                         "gamma_scale": svr["gamma_scale"], "epsilon": svr["epsilon"],
                         "input_scope": scope, **regression_metrics(y[test], prediction)})
        scaler, xt, xv, xs = _scaled_arrays(x_history[train], x_history[validation], x_history[test])
        for model_name in ("extra_trees", "random_forest", "hist_gradient_boosting", "knn"):
            best = None
            for params in _classical_candidates(model_name):
                model = _classical_model(model_name, params).fit(xt, y[train])
                score = float(mean_absolute_error(y[validation], model.predict(xv)))
                candidate = (score, json.dumps(params, sort_keys=True), params)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            x_refit = np.concatenate((x_history[train], x_history[validation]))
            y_refit = np.concatenate((y[train], y[validation]))
            scaler = StandardScaler().fit(x_refit)
            model = _classical_model(model_name, best[2]).fit(scaler.transform(x_refit), y_refit)
            prediction = model.predict(scaler.transform(x_history[test]))
            rows.append({"model": model_name, "target": target_name,
                         "validation_mae_ev": best[0], "parameters": best[1],
                         "input_scope": "history_10fs", **regression_metrics(y[test], prediction)})
            np.savez_compressed(prediction_dir / f"classical_{model_name}_{target_name}.npz",
                                target=y[test].astype(np.float32), prediction=np.asarray(prediction, dtype=np.float32),
                                trajectory_id=trajectory_id[test])
            print(f"[classical] {target_name} {model_name} test={rows[-1]['mae_ev']:.5f} eV", flush=True)
    _write_rows(TASK_DIR / "outputs" / "classical_test.csv", rows)
    return TASK_DIR / "outputs" / "classical_test.csv"


def _trajectory_mae(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        error = np.abs(package["prediction"].astype(np.float64) - package["target"].astype(np.float64))
        trajectory = package["trajectory_id"].astype(np.int64)
    unique = np.unique(trajectory)
    return unique, np.asarray([error[trajectory == item].mean() for item in unique])


def _reservoir_prediction_path(row: pd.Series) -> Path:
    return TASK_DIR / "outputs" / "predictions" / (
        f"{row.system}_seed{int(row.seed):03d}_{row.readout}_{row.decoder}_{row.target}.npz"
    )


def _classical_prediction_path(model: str, target: str) -> Path:
    label = "persistence" if model == "current_gap_persistence" else model
    return TASK_DIR / "outputs" / "predictions" / f"classical_{label}_{target}.npz"


def _paired_trajectory_comparisons(
    reservoir: pd.DataFrame, classical: pd.DataFrame, *, bootstrap_samples: int = 10000,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260826)
    rows = []
    for target in TARGET_NAMES:
        errors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for system in ("qrc", "esn"):
            selected = reservoir[(reservoir.system == system) & (reservoir.target == target)]
            per_seed = []
            reference_ids = None
            for _, selected_row in selected.sort_values("seed").iterrows():
                ids, values = _trajectory_mae(_reservoir_prediction_path(selected_row))
                if reference_ids is None:
                    reference_ids = ids
                elif not np.array_equal(ids, reference_ids):
                    raise RuntimeError("reservoir predictions use different test trajectories")
                per_seed.append(values)
            errors[system] = (reference_ids, np.mean(per_seed, axis=0))
        selected_classic = classical[classical.target == target].sort_values(
            ["validation_mae_ev", "model"]
        ).iloc[0]
        classic_path = _classical_prediction_path(str(selected_classic.model), target)
        if not classic_path.exists():
            raise FileNotFoundError(f"validation-selected classical prediction missing: {classic_path}")
        errors["classical"] = _trajectory_mae(classic_path)
        for baseline in ("esn", "classical"):
            ids, qrc_error = errors["qrc"]
            other_ids, baseline_error = errors[baseline]
            if not np.array_equal(ids, other_ids):
                raise RuntimeError("paired comparison trajectory mismatch")
            difference = qrc_error - baseline_error
            indices = rng.integers(0, len(ids), size=(bootstrap_samples, len(ids)))
            bootstrap = difference[indices].mean(axis=1)
            rows.append({
                "target": target, "comparison": f"qrc_minus_{baseline}",
                "baseline_model": (str(selected_classic.model) if baseline == "classical" else "validation-selected ESN"),
                "mean_mae_difference_ev": float(difference.mean()),
                "bootstrap_95ci_low_ev": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95ci_high_ev": float(np.quantile(bootstrap, 0.975)),
                "qrc_better_trajectory_fraction": float(np.mean(difference < 0)),
                "num_test_trajectories": len(ids), "bootstrap_samples": bootstrap_samples,
            })
    return rows


def build_report(cfg: Config) -> Path:
    reservoir = pd.read_csv(TASK_DIR / "outputs" / "reservoir_best_validation_selected.csv")
    classical = pd.read_csv(TASK_DIR / "outputs" / "classical_test.csv")
    reservoir_summary = reservoir.groupby(["system", "target"], as_index=False).agg(
        mae_ev_mean=("mae_ev", "mean"), mae_ev_sample_std=("mae_ev", "std"),
        rmse_ev_mean=("rmse_ev", "mean"), r2_mean=("r2", "mean"), num_repeats=("seed", "size"),
    )
    reservoir_summary.to_csv(TASK_DIR / "outputs" / "best_reservoir_summary.csv", index=False)
    primary = "min_gap_next_5fs"
    qrc_primary = reservoir_summary[(reservoir_summary.system == "qrc") & (reservoir_summary.target == primary)].iloc[0]
    esn_primary = reservoir_summary[(reservoir_summary.system == "esn") & (reservoir_summary.target == primary)].iloc[0]
    classic_primary = classical[classical.target == primary].sort_values(["validation_mae_ev", "model"]).iloc[0]
    paired_rows = _paired_trajectory_comparisons(reservoir, classical)
    _write_rows(TASK_DIR / "outputs" / "paired_trajectory_comparisons.csv", paired_rows)
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        split = package["split"]; trajectory = package["trajectory_id"]
        hop_rate = float(package["hop_within_5fs"].mean())
        split_stats = {SPLIT_NAMES[i]: {"trajectories": int(len(np.unique(trajectory[split == i]))),
                                        "samples": int(np.count_nonzero(split == i))} for i in range(3)}
    report = {
        "status": "completed",
        "protocol": "shnitsel-a01-q15-xz-pair-distance-velocity-exact-last-reset-final-vs-post-tm3-v1",
        "dataset": {"source": cfg.payload["dataset"]["source"], "raw_file": str(cfg.raw_path),
                    "split_statistics": split_stats, "hop_within_5fs_rate_at_sampled_endpoints": hop_rate,
                    "trajectory_level_split": True, "test_used_for_selection": False},
        "qrc": {"q": cfg.q, "g": cfg.g, "W": cfg.W, "layers": cfg.layers,
                "injection": "15 fixed atom pairs; X distance and Z radial velocity; 30 simultaneous values/frame",
                "micro_time_steps": False, "observables": "Z+ZZ", "final_dim": cfg.output_dim,
                "tm3_dim": cfg.tm3_dim, "p_grid": list(cfg.p_grid), "seeds": list(cfg.seeds)},
        "targets": list(TARGET_NAMES),
        "primary_comparison_mae_ev": {
            "qrc_mean": float(qrc_primary.mae_ev_mean), "qrc_sample_std": float(qrc_primary.mae_ev_sample_std),
            "esn_mean": float(esn_primary.mae_ev_mean), "esn_sample_std": float(esn_primary.mae_ev_sample_std),
            "validation_selected_classical_model": str(classic_primary.model),
            "classical_test": float(classic_primary.mae_ev),
        },
        "reservoir_summary": reservoir_summary.to_dict(orient="records"),
        "classical_results": classical.to_dict(orient="records"),
        "paired_trajectory_comparisons": paired_rows,
    }
    path = TASK_DIR / "outputs" / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    paired = pd.DataFrame(paired_rows)
    qrc_primary_mae = float(qrc_primary.mae_ev_mean)
    esn_primary_mae = float(esn_primary.mae_ev_mean)
    classic_primary_mae = float(classic_primary.mae_ev)
    markdown = f"""# 已弃用的选择式 SHNITSEL A01 报告

此兼容入口会用验证集选择物理轴，只保留作历史复现。正式六种子固定轴结果
由 `evaluate_strict_axes.py`、`run_uniform_p_scan.py` 和
`figures/strict_six_seed/` 给出。

## 协议

- 官方 v2 A01 动态文件：297 条完整乙烯轨迹、62,031 个真实 0.5-fs 帧；SHA-256 已锁定。
- 乙烯 6 原子产生 15 个固定原子对。每帧同时注入 30 个数：X 为距离，Z 为径向速度；没有微时间步和重复注入。
- Q=15、g=0.84、W=0.5、L=3；观测 Z+ZZ；final=120D，输入后 TM3=360D。
- 精确扫描 {len(cfg.p_grid)} 个 p；{len(cfg.seeds)} 个 reservoir seed；旧流程会由验证轨迹选择物理轴。
- 轨迹级分层切分：178/59/60 条 train/validation/test，对应 1424/472/480 个窗口。测试轨迹不参与归一化或选择。

## 主要结果

主目标“未来 5 fs 内最小 S1-S0 能隙”的测试 MAE：

| 方法 | MAE (eV) |
|---|---:|
| 最优 QRC（验证选择，3 seed） | **{qrc_primary_mae:.3f} ± {float(qrc_primary.mae_ev_sample_std):.3f}** |
| 匹配 ESN（验证选择，3 seed） | {esn_primary_mae:.3f} ± {float(esn_primary.mae_ev_sample_std):.3f} |
| 验证选择经典模型（{classic_primary.model}） | {classic_primary_mae:.3f} |

QRC 相对 ESN 降低 MAE {(esn_primary_mae-qrc_primary_mae)/esn_primary_mae*100:.1f}%，相对验证选择的经典模型降低 {(classic_primary_mae-qrc_primary_mae)/classic_primary_mae*100:.1f}%。

四个目标的验证选择后结果：

| 目标 | QRC MAE | ESN MAE | 验证选择经典模型及 MAE |
|---|---:|---:|---:|
"""
    for target in TARGET_NAMES:
        q = reservoir_summary[(reservoir_summary.system == "qrc") & (reservoir_summary.target == target)].iloc[0]
        e = reservoir_summary[(reservoir_summary.system == "esn") & (reservoir_summary.target == target)].iloc[0]
        c = classical[classical.target == target].sort_values(["validation_mae_ev", "model"]).iloc[0]
        markdown += f"| {target} | **{q.mae_ev_mean:.3f}** | {e.mae_ev_mean:.3f} | {c.model}: {c.mae_ev:.3f} |\n"
    markdown += """

1-fs 的 current-gap persistence 使用了 QRC/ESN 没有输入的当前电子能隙，因而是 target-informed 上界式基线，不能作为输入匹配的公平反例。10-fs 目标上，验证选择的 Extra Trees 与 QRC 接近且略优；QRC 的优势集中在 1-fs、5-fs 和未来最小能隙三个目标。

## p、TM3 与解码器

- 四个目标在三个 QRC seed 上最终都由 SVR 胜出。
- 该旧流程的验证选择结果不进入正式证据链；固定轴结果不以验证集选择 p。
- final 并非总胜：验证集在 10-fs 的两个 seed、1-fs 的一个 seed、最小能隙的一个 seed 选择 TM3；其余选择 final。因此 TM3 有条件地提供增益，但不是统一最优。

## 轨迹级配对检验

`paired_trajectory_comparisons.csv` 对 60 条测试轨迹先求 MAE，再在 QRC 三 seed 上平均，并进行 10,000 次轨迹 bootstrap。负的 QRC−baseline 表示 QRC 更好；该检验不改变任何模型选择。
"""
    for _, row in paired.iterrows():
        if row.target == primary:
            markdown += (f"- {row.comparison}: {row.mean_mae_difference_ev:+.3f} eV，"
                         f"95% CI [{row.bootstrap_95ci_low_ev:+.3f}, {row.bootstrap_95ci_high_ev:+.3f}]，"
                         f"QRC 在 {row.qrc_better_trajectory_fraction*100:.1f}% 的测试轨迹上更好。\n")
    markdown += """

## 结论边界

这是同一输入、同一轨迹切分下的预测基准，不说明 QRC 普遍优于所有经典学习器。当前证据支持：15Q 距离/径向速度编码在短中期非绝热能隙预测上显著优于匹配 ESN，并在主目标上优于所测输入匹配经典算法；10-fs 远期预测则没有形成经典优势。
"""
    (TASK_DIR / "RESULTS.md").write_text(markdown, encoding="utf-8")
    return path
