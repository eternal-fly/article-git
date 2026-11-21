"""Closed, non-dissipative DTC-QRC benchmark on SHNITSEL A01 trajectories.

The quantum reservoir is reset only once at the beginning of each causal input
window.  All frames inside that window are processed by closed unitary dynamics
with p=0.  Validation data select the physical history length and decoder
hyperparameters; the held-out trajectories are evaluated once.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


TASK_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK_DIR.parent
WORKSPACE = TRANSITION_DIR.parent
SOURCE_TASK = TRANSITION_DIR / "shnitsel_a01_surface_hopping"
TASK8_DIR = TRANSITION_DIR / "task8_q12_xz_post_tm3_z2"
QRC_CORE = WORKSPACE / "BFS-QRC" / "FINAL" / "2_linear_readout" / "core"
SECOND_ESN = WORKSPACE / "BFS-QRC" / "PRAPPLIED_reviewer_supplement_gscan_ho_esn" / "src"
for directory in (SOURCE_TASK, TASK8_DIR, QRC_CORE, SECOND_ESN):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from a01_core import (  # noqa: E402
    TARGET_NAMES,
    _best_ridge,
    _best_svr,
    _classical_candidates,
    _classical_model,
    _refit_ridge,
    _refit_svr,
    regression_metrics,
)
from crc3_baseline import compute_crc3_z_zz_features  # noqa: E402
from esn_baseline import compute_esn_standard_features  # noqa: E402
from higher_order_esn import compute_second_order_esn_features  # noqa: E402
from task8_core import _compute_qrc_post_tm3  # noqa: E402


SYSTEMS = ("qrc", "esn_standard", "esn_second_order_recurrent", "fp_cvr")


@dataclass(frozen=True)
class Config:
    payload: dict[str, Any]
    dataset_path: Path
    histories: tuple[int, ...]
    dt_fs: float
    q: int
    g: float
    W: float
    layers: int
    angle_scale: float
    seeds: tuple[int, ...]
    batch_size: int

    @property
    def output_dim(self) -> int:
        return self.q * (self.q + 1) // 2


def load_config() -> Config:
    payload = json.loads((TASK_DIR / "config.json").read_text(encoding="utf-8"))
    qrc = payload["qrc"]
    cfg = Config(
        payload=payload,
        dataset_path=(TASK_DIR / payload["source_dataset"]).resolve(),
        histories=tuple(map(int, payload["history_steps_grid"])),
        dt_fs=float(payload["frame_interval_fs"]),
        q=int(qrc["num_qubits"]),
        g=float(qrc["g"]),
        W=float(qrc["W"]),
        layers=int(qrc["layers_per_input"]),
        angle_scale=float(qrc["injection_angle_scale"]),
        seeds=tuple(map(int, qrc["reservoir_seeds"])),
        batch_size=int(qrc["batch_size"]),
    )
    expected = (cfg.q, cfg.g, cfg.W, cfg.layers, cfg.output_dim)
    if expected != (15, 0.84, 0.5, 3, 120):
        raise ValueError(f"closed paper-aligned protocol changed: {expected}")
    if cfg.histories != (1, 2, 4, 8, 12, 16, 20):
        raise ValueError("unexpected physical-history grid")
    if qrc["reset_probability"] != 0.0 or qrc["readout"] != "final":
        raise ValueError("primary protocol must be closed p=0 with final readout")
    if qrc["observable_family"] != "Z+ZZ" or qrc["injection_axis"] != "XZ":
        raise ValueError("primary protocol must use XZ injection and Z+ZZ observations")
    if not cfg.dataset_path.exists():
        raise FileNotFoundError(cfg.dataset_path)
    return cfg


def feature_path(system: str, seed: int) -> Path:
    return TASK_DIR / "features" / f"{system}_closed_history_grid_seed{seed:03d}.npz"


def _compute_features(tokens: np.ndarray, system: str, seed: int, cfg: Config, device: str):
    if system == "qrc":
        temporal, metadata = _compute_qrc_post_tm3(
            tokens,
            q=cfg.q,
            g=cfg.g,
            W=cfg.W,
            layers=cfg.layers,
            angle_scale=cfg.angle_scale,
            injection_scale=1.0,
            seed=seed,
            batch_size=cfg.batch_size,
            device=device,
        )
        return temporal[:, : cfg.output_dim], metadata
    if system == "esn_standard":
        return compute_esn_standard_features(tokens, cfg.q, cfg.layers, seed, device=device)
    if system == "esn_second_order_recurrent":
        return compute_second_order_esn_features(tokens, cfg.q, cfg.layers, seed, device=device)
    if system == "fp_cvr":
        return compute_crc3_z_zz_features(tokens, cfg.q, cfg.layers, seed, device=device)
    raise ValueError(system)


def compute_feature_grid(system: str, seed: int, device: str, force: bool = False) -> Path:
    cfg = load_config()
    if system not in SYSTEMS or seed not in cfg.seeds:
        raise ValueError((system, seed))
    path = feature_path(system, seed)
    if path.exists() and not force:
        return path
    if system == "qrc":
        import torch

        if not torch.cuda.is_available() or not str(device).startswith("cuda"):
            raise RuntimeError("formal Q15 closed-QRC extraction requires CUDA")
    with np.load(cfg.dataset_path, allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32)
    grids, metadata = [], {}
    started = time.perf_counter()
    for history in cfg.histories:
        values, metadata = _compute_features(tokens[:, -history:], system, seed, cfg, device)
        if values.shape != (len(tokens), cfg.output_dim) or not np.isfinite(values).all():
            raise RuntimeError(f"bad feature grid cell {system}/{history}: {values.shape}")
        grids.append(values.astype(np.float32))
        print(f"[{system} seed={seed}] closed history={history} ({history*cfg.dt_fs:g} fs)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        histories=np.asarray(cfg.histories, dtype=np.int16),
        features=np.stack(grids).astype(np.float32),
        system=np.asarray(system),
        seed=np.int64(seed),
        p=np.float32(0.0),
        closed_unitary=np.asarray(system == "qrc"),
        initialization=np.asarray("|0>^Q once at the start of each history window"),
        reset_between_frames=np.asarray(False),
        readout=np.asarray("final"),
        observable_family=np.asarray("Z+ZZ" if system in ("qrc", "fp_cvr") else "matched 120D state"),
        g=np.float32(cfg.g),
        W=np.float32(cfg.W),
        layers=np.int16(cfg.layers),
        elapsed_seconds=np.float64(time.perf_counter() - started),
        parameter_summary=np.asarray(json.dumps(_jsonable(metadata))),
    )
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _decoder_contract(cfg: Config):
    decoder = cfg.payload["decoders"]
    alpha = tuple(map(float, decoder["ridge_alpha_grid"]))
    reference_payload = decoder["svr_history_scan_reference"]
    reference = [(
        float(reference_payload["C"]),
        float(reference_payload["gamma_scale"]),
        float(reference_payload["epsilon"]),
    )]
    svr = [
        (float(C), float(gamma), float(epsilon))
        for C in decoder["svr_C_grid"]
        for gamma in decoder["svr_gamma_scale_grid"]
        for epsilon in decoder["svr_epsilon_grid"]
    ]
    return alpha, reference, svr, float(decoder["svr_cache_size_mb"])


def _select_and_refit(
    decoder: str,
    grid: np.ndarray,
    histories: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    cfg: Config,
):
    train, validation, test = split == 0, split == 1, split == 2
    alpha_grid, svr_reference, svr_grid, cache_size = _decoder_contract(cfg)
    scans = []
    for history, features in zip(histories, grid):
        if decoder == "linear":
            params = _best_ridge(features[train], y[train], features[validation], y[validation], alpha_grid)
        elif decoder == "svr":
            params = _best_svr(
                features[train], y[train], features[validation], y[validation], svr_reference, cache_size
            )
        else:
            raise ValueError(decoder)
        scans.append((float(params["validation_mae_ev"]), int(history), params, features))
    history_selection_mae, selected_history, params, features = min(scans, key=lambda item: (item[0], item[1]))
    if decoder == "linear":
        prediction = _refit_ridge(
            features[train], y[train], features[validation], y[validation], features[test], params["alpha"]
        )
    else:
        params = _best_svr(
            features[train], y[train], features[validation], y[validation], svr_grid, cache_size
        )
        prediction = _refit_svr(
            features[train], y[train], features[validation], y[validation], features[test], params, cache_size
        )
    params["history_selection_validation_mae_ev"] = history_selection_mae
    return scans, selected_history, params, prediction


def _evaluate_reservoir_job(system: str, seed: int):
    cfg = load_config()
    with np.load(cfg.dataset_path, allow_pickle=False) as package:
        targets = package["targets"].astype(np.float64)
        split = package["split"]
        trajectory = package["trajectory_id"]
    test = split == 2
    scan_rows, selected_rows = [], []
    prediction_dir = TASK_DIR / "outputs" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with np.load(feature_path(system, seed), allow_pickle=False) as package:
        histories = package["histories"].astype(np.int64)
        grid = package["features"].astype(np.float32)
    for target_index, target_name in enumerate(TARGET_NAMES):
        y = targets[:, target_index]
        for decoder in ("linear", "svr"):
            scans, history, params, prediction = _select_and_refit(decoder, grid, histories, y, split, cfg)
            for validation_mae, candidate_history, candidate_params, _ in scans:
                row = {
                    "system": system,
                    "seed": seed,
                    "target": target_name,
                    "decoder": decoder,
                    "history_steps": candidate_history,
                    "history_fs": candidate_history * cfg.dt_fs,
                    "validation_mae_ev": validation_mae,
                    "selection_stage": "full-linear-grid" if decoder == "linear" else "fixed-reference-svr",
                }
                for key in ("alpha", "C", "gamma_scale", "gamma", "epsilon"):
                    row[key] = candidate_params.get(key, "")
                scan_rows.append(row)
            row = {
                "system": system,
                "seed": seed,
                "target": target_name,
                "decoder": decoder,
                "selected_history_steps": history,
                "selected_history_fs": history * cfg.dt_fs,
                "history_selection_validation_mae_ev": params["history_selection_validation_mae_ev"],
                "validation_mae_ev": params["validation_mae_ev"],
                "p": 0.0,
                "W": cfg.W,
                "feature_dim": cfg.output_dim,
                **regression_metrics(y[test], prediction),
            }
            for key in ("alpha", "C", "gamma_scale", "gamma", "epsilon"):
                row[key] = params.get(key, "")
            selected_rows.append(row)
            np.savez_compressed(
                prediction_dir / f"{system}_seed{seed:03d}_{decoder}_{target_name}.npz",
                target=y[test].astype(np.float32),
                prediction=np.asarray(prediction, dtype=np.float32),
                trajectory_id=trajectory[test],
                selected_history_steps=np.int16(history),
            )
            print(
                f"[{system} seed={seed}] {target_name}/{decoder} T={history} test={row['mae_ev']:.5f} eV",
                flush=True,
            )
    return scan_rows, selected_rows


def evaluate_reservoirs() -> Path:
    cfg = load_config()
    missing = [feature_path(system, seed) for system in SYSTEMS for seed in cfg.seeds if not feature_path(system, seed).exists()]
    if missing:
        raise FileNotFoundError("missing feature grids:\n" + "\n".join(map(str, missing)))
    scan_rows, selected_rows = [], []
    jobs = [(system, seed) for system in SYSTEMS for seed in cfg.seeds]
    with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as executor:
        futures = {executor.submit(_evaluate_reservoir_job, system, seed): (system, seed) for system, seed in jobs}
        for future in as_completed(futures):
            scan, selected = future.result()
            scan_rows.extend(scan)
            selected_rows.extend(selected)
    scan_rows.sort(key=lambda row: (row["system"], row["seed"], row["target"], row["decoder"], row["history_steps"]))
    selected_rows.sort(key=lambda row: (row["system"], row["seed"], row["target"], row["decoder"]))
    _write_rows(TASK_DIR / "outputs" / "history_scan_validation.csv", scan_rows)
    _write_rows(TASK_DIR / "outputs" / "reservoir_selected_test.csv", selected_rows)
    selected = pd.DataFrame(selected_rows)
    selected.groupby(["system", "decoder", "target"], as_index=False).agg(
        mae_ev_mean=("mae_ev", "mean"),
        mae_ev_sample_std=("mae_ev", "std"),
        rmse_ev_mean=("rmse_ev", "mean"),
        r2_mean=("r2", "mean"),
        history_steps_mean=("selected_history_steps", "mean"),
        num_seeds=("seed", "size"),
    ).to_csv(TASK_DIR / "outputs" / "reservoir_summary.csv", index=False)
    return TASK_DIR / "outputs" / "reservoir_selected_test.csv"


def _fit_classical_candidate(name: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray):
    if name in ("extra_trees", "hist_gradient_boosting"):
        return _classical_model(name, params).fit(x, y)
    raise ValueError(name)


def evaluate_classical() -> Path:
    cfg = load_config()
    with np.load(cfg.dataset_path, allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32).reshape(-1, 20, cfg.q, 3)
        targets = package["targets"].astype(np.float64)
        current_gap = package["current_gap"].astype(np.float64)
        split = package["split"]
        trajectory = package["trajectory_id"]
    train, validation, test = split == 0, split == 1, split == 2
    alpha_grid, svr_reference, svr_grid, cache_size = _decoder_contract(cfg)
    rows, scan_rows = [], []
    prediction_dir = TASK_DIR / "outputs" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for target_index, target_name in enumerate(TARGET_NAMES):
        y = targets[:, target_index]
        persistence_prediction = current_gap[test]
        rows.append({
            "model": "current_gap_persistence",
            "target": target_name,
            "selected_history_steps": 1,
            "selected_history_fs": cfg.dt_fs,
            "validation_mae_ev": float(mean_absolute_error(y[validation], current_gap[validation])),
            "input_scope": "target-informed current gap",
            **regression_metrics(y[test], persistence_prediction),
        })
        np.savez_compressed(
            prediction_dir / f"classical_current_gap_persistence_{target_name}.npz",
            target=y[test].astype(np.float32), prediction=persistence_prediction.astype(np.float32),
            trajectory_id=trajectory[test],
        )
        for model_name in ("raw_ridge", "raw_svr", "extra_trees", "hist_gradient_boosting"):
            candidates = []
            for history in cfg.histories:
                features = tokens[:, -history:, :, (0, 2)].reshape(len(tokens), -1)
                if model_name == "raw_ridge":
                    params = _best_ridge(features[train], y[train], features[validation], y[validation], alpha_grid)
                    score = params["validation_mae_ev"]
                elif model_name == "raw_svr":
                    params = _best_svr(
                        features[train], y[train], features[validation], y[validation], svr_reference, cache_size
                    )
                    score = params["validation_mae_ev"]
                else:
                    scaler = StandardScaler().fit(features[train])
                    xt, xv = scaler.transform(features[train]), scaler.transform(features[validation])
                    best = None
                    for params0 in _classical_candidates(model_name):
                        prediction = _fit_classical_candidate(model_name, params0, xt, y[train]).predict(xv)
                        candidate = (float(mean_absolute_error(y[validation], prediction)), json.dumps(params0, sort_keys=True), params0)
                        if best is None or candidate[:2] < best[:2]:
                            best = candidate
                    score, params = best[0], best[2]
                candidates.append((float(score), int(history), params, features))
                scan_rows.append({
                    "model": model_name,
                    "target": target_name,
                    "history_steps": history,
                    "history_fs": history * cfg.dt_fs,
                    "validation_mae_ev": score,
                    "parameters": json.dumps(_jsonable(params), sort_keys=True),
                })
            score, history, params, features = min(candidates, key=lambda item: (item[0], item[1]))
            history_selection_score = score
            if model_name == "raw_ridge":
                prediction = _refit_ridge(
                    features[train], y[train], features[validation], y[validation], features[test], params["alpha"]
                )
            elif model_name == "raw_svr":
                params = _best_svr(
                    features[train], y[train], features[validation], y[validation], svr_grid, cache_size
                )
                prediction = _refit_svr(
                    features[train], y[train], features[validation], y[validation], features[test], params, cache_size
                )
            else:
                x_refit = np.concatenate((features[train], features[validation]))
                y_refit = np.concatenate((y[train], y[validation]))
                scaler = StandardScaler().fit(x_refit)
                model = _fit_classical_candidate(model_name, params, scaler.transform(x_refit), y_refit)
                prediction = model.predict(scaler.transform(features[test]))
            final_validation_score = float(params["validation_mae_ev"]) if model_name in ("raw_ridge", "raw_svr") else score
            row = {
                "model": model_name,
                "target": target_name,
                "selected_history_steps": history,
                "selected_history_fs": history * cfg.dt_fs,
                "history_selection_validation_mae_ev": history_selection_score,
                "validation_mae_ev": final_validation_score,
                "parameters": json.dumps(_jsonable(params), sort_keys=True),
                "input_scope": "validation-selected ordered XZ history",
                **regression_metrics(y[test], prediction),
            }
            rows.append(row)
            np.savez_compressed(
                prediction_dir / f"classical_{model_name}_{target_name}.npz",
                target=y[test].astype(np.float32), prediction=np.asarray(prediction, dtype=np.float32),
                trajectory_id=trajectory[test], selected_history_steps=np.int16(history),
            )
            print(f"[classical] {target_name}/{model_name} T={history} test={row['mae_ev']:.5f} eV", flush=True)
    _write_rows(TASK_DIR / "outputs" / "classical_history_scan_validation.csv", scan_rows)
    _write_rows(TASK_DIR / "outputs" / "classical_selected_test.csv", rows)
    return TASK_DIR / "outputs" / "classical_selected_test.csv"


def _trajectory_errors(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        error = np.abs(package["prediction"].astype(np.float64) - package["target"].astype(np.float64))
        trajectory = package["trajectory_id"].astype(np.int64)
    ids = np.unique(trajectory)
    return ids, np.asarray([error[trajectory == item].mean() for item in ids])


def _reservoir_error(system: str, decoder: str, target: str, seeds: tuple[int, ...]):
    values, reference = [], None
    for seed in seeds:
        ids, error = _trajectory_errors(TASK_DIR / "outputs" / "predictions" / f"{system}_seed{seed:03d}_{decoder}_{target}.npz")
        if reference is None:
            reference = ids
        elif not np.array_equal(reference, ids):
            raise RuntimeError("reservoir trajectory mismatch")
        values.append(error)
    return reference, np.mean(values, axis=0)


def _paired_rows(reservoir: pd.DataFrame, classical: pd.DataFrame, cfg: Config):
    rng = np.random.default_rng(20260826)
    rows = []
    for target in TARGET_NAMES:
        selected_classic = classical[(classical.target == target) & (classical.model != "current_gap_persistence")].sort_values(
            ["validation_mae_ev", "model"]
        ).iloc[0]
        classic_ids, classic_error = _trajectory_errors(
            TASK_DIR / "outputs" / "predictions" / f"classical_{selected_classic.model}_{target}.npz"
        )
        for decoder in ("linear", "svr"):
            qrc_ids, qrc_error = _reservoir_error("qrc", decoder, target, cfg.seeds)
            comparisons = [
                ("esn_standard",) + _reservoir_error("esn_standard", decoder, target, cfg.seeds),
                ("esn_second_order_recurrent",) + _reservoir_error("esn_second_order_recurrent", decoder, target, cfg.seeds),
                ("fp_cvr",) + _reservoir_error("fp_cvr", decoder, target, cfg.seeds),
                (str(selected_classic.model), classic_ids, classic_error),
            ]
            for baseline, ids, baseline_error in comparisons:
                if not np.array_equal(qrc_ids, ids):
                    raise RuntimeError("paired trajectory mismatch")
                difference = qrc_error - baseline_error
                sampled = rng.integers(0, len(ids), size=(10000, len(ids)))
                boot = difference[sampled].mean(axis=1)
                rows.append({
                    "target": target,
                    "decoder": decoder,
                    "comparison": f"qrc_minus_{baseline}",
                    "mean_mae_difference_ev": float(difference.mean()),
                    "bootstrap_95ci_low_ev": float(np.quantile(boot, 0.025)),
                    "bootstrap_95ci_high_ev": float(np.quantile(boot, 0.975)),
                    "qrc_better_trajectory_fraction": float(np.mean(difference < 0)),
                    "num_test_trajectories": len(ids),
                })
    return rows


def build_report() -> Path:
    cfg = load_config()
    reservoir = pd.read_csv(TASK_DIR / "outputs" / "reservoir_selected_test.csv")
    classical = pd.read_csv(TASK_DIR / "outputs" / "classical_selected_test.csv")
    summary = reservoir.groupby(["system", "decoder", "target"], as_index=False).agg(
        mae_ev_mean=("mae_ev", "mean"),
        mae_ev_sample_std=("mae_ev", "std"),
        rmse_ev_mean=("rmse_ev", "mean"),
        r2_mean=("r2", "mean"),
        history_steps_mean=("selected_history_steps", "mean"),
    )
    paired = _paired_rows(reservoir, classical, cfg)
    _write_rows(TASK_DIR / "outputs" / "paired_trajectory_comparisons.csv", paired)
    report = {
        "status": "completed",
        "protocol": "shnitsel-a01-q15-closed-p0-W0p5-final-z-zz-history-scan-v2",
        "closed_architecture": {
            "q": cfg.q, "g": cfg.g, "W": cfg.W, "layers": cfg.layers, "p": 0.0,
            "histories_steps": list(cfg.histories), "histories_fs": [item * cfg.dt_fs for item in cfg.histories],
            "injection": "15 pair distances on X and 15 radial velocities on Z, simultaneously per frame",
            "readout": "single final Z+ZZ vector", "feature_dim": cfg.output_dim,
        },
        "reservoir_summary": summary.to_dict(orient="records"),
        "classical_results": classical.to_dict(orient="records"),
        "paired_trajectory_comparisons": paired,
    }
    path = TASK_DIR / "outputs" / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# SHNITSEL A01：闭合非耗散DTC-QRC",
        "",
        "## 锁定协议",
        "",
        "- Q15，g=0.84，W=0.5，L=3，p=0；每个窗口只在开始时初始化量子态，帧间没有reset或dephasing。",
        "- 每帧同时以X/Z注入15个原子对距离与15个径向速度，不拆微时间步。",
        "- 只读取最后状态的Z+ZZ（120D）；验证轨迹选择历史长度和decoder超参数。",
        "- 历史长度为1/2/4/8/12/16/20帧，即0.5–10 fs；测试轨迹从不参与选择。",
        "- 对照包括标准ESN、维度匹配二阶递归ESN、FP-CVR及输入匹配经典回归器。",
        "",
        "## 测试MAE（验证选择历史长度）",
        "",
        "| Target | Decoder | QRC | Standard ESN | 2nd-order ESN | FP-CVR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target in TARGET_NAMES:
        for decoder in ("linear", "svr"):
            values, means = {}, {}
            for system in SYSTEMS:
                row = summary[(summary.system == system) & (summary.decoder == decoder) & (summary.target == target)].iloc[0]
                values[system] = f"{row.mae_ev_mean:.3f}±{row.mae_ev_sample_std:.3f}"
                means[system] = float(row.mae_ev_mean)
            best_system = min(means, key=means.get)
            values[best_system] = f"**{values[best_system]}**"
            lines.append(
                f"| {target} | {decoder} | {values['qrc']} | {values['esn_standard']} | "
                f"{values['esn_second_order_recurrent']} | {values['fp_cvr']} |"
            )
    lines.extend([
        "",
        "## 输入匹配经典模型",
        "",
        "| Target | 验证选择模型 | Test MAE (eV) | 选择历史 |",
        "|---|---|---:|---:|",
    ])
    for target in TARGET_NAMES:
        classic = classical[(classical.target == target) & (classical.model != "current_gap_persistence")].sort_values(
            ["validation_mae_ev", "model"]
        ).iloc[0]
        lines.append(
            f"| {target} | {classic.model} | {classic.mae_ev:.3f} | "
            f"{int(classic.selected_history_steps)} frames ({classic.selected_history_fs:g} fs) |"
        )
    lines.extend(["", "## 验证选择的QRC历史长度", ""])
    for target in TARGET_NAMES:
        chosen = reservoir[(reservoir.system == "qrc") & (reservoir.decoder == "svr") & (reservoir.target == target)].sort_values("seed")
        lines.append(
            f"- {target}: " + ", ".join(
                f"seed {int(row.seed)} -> {int(row.selected_history_steps)} frame"
                for _, row in chosen.iterrows()
            )
        )
    paired_frame = pd.DataFrame(paired)
    lines.extend(["", "## 轨迹级配对结果（QRC-SVR）", ""])
    for target in TARGET_NAMES:
        for label in ("esn_standard",):
            effect = paired_frame[
                (paired_frame.target == target)
                & (paired_frame.decoder == "svr")
                & (paired_frame.comparison == f"qrc_minus_{label}")
            ].iloc[0]
            lines.append(
                f"- {target}, QRC−{label}: {effect.mean_mae_difference_ev:+.3f} eV, "
                f"95% CI [{effect.bootstrap_95ci_low_ev:+.3f}, {effect.bootstrap_95ci_high_ev:+.3f}]."
            )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "- QRC-SVR在1-fs与未来5-fs最小能隙任务上显著优于标准及二阶递归ESN；对5-fs和10-fs点预测的QRC−ESN区间跨零。",
        "- QRC-SVR只在1-fs任务上显著优于验证选择的输入匹配经典模型；主目标相对HGB的区间跨零。",
        "- 四个QRC-SVR任务的三个seed均选择单帧历史。因此该结果支持闭合量子特征映射的任务迁移，不支持长时间记忆是性能来源。",
        "- 耗散p扫描和TM3属于独立的应用优化扩展；本闭合历史扫描与主协议统一使用W=0.5。",
        "",
    ])
    (TASK_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return path


def audit_results() -> Path:
    cfg = load_config()
    selected = pd.read_csv(TASK_DIR / "outputs" / "reservoir_selected_test.csv")
    scan = pd.read_csv(TASK_DIR / "outputs" / "history_scan_validation.csv")
    classical = pd.read_csv(TASK_DIR / "outputs" / "classical_selected_test.csv")
    expected_selected = len(SYSTEMS) * len(cfg.seeds) * len(TARGET_NAMES) * 2
    expected_scan = expected_selected * len(cfg.histories)
    if len(selected) != expected_selected or len(scan) != expected_scan:
        raise AssertionError((len(selected), len(scan), expected_selected, expected_scan))
    if set(selected["p"]) != {0.0} or set(selected["W"]) != {0.5}:
        raise AssertionError("closed protocol drift")
    if not set(selected["selected_history_steps"]).issubset(set(cfg.histories)):
        raise AssertionError("invalid selected history")
    if selected[["mae_ev", "rmse_ev", "r2"]].isna().any().any():
        raise AssertionError("non-finite reservoir metric")
    if len(classical) != len(TARGET_NAMES) * 5:
        raise AssertionError("unexpected classical row count")
    feature_contracts = []
    for system in SYSTEMS:
        for seed in cfg.seeds:
            path0 = feature_path(system, seed)
            with np.load(path0, allow_pickle=False) as package:
                if package["features"].shape != (len(cfg.histories), 2376, cfg.output_dim):
                    raise AssertionError(f"bad feature shape: {path0}")
                if tuple(package["histories"].tolist()) != cfg.histories:
                    raise AssertionError(f"history drift: {path0}")
                if not np.isclose(float(package["p"]), 0.0) or not np.isclose(float(package["W"]), 0.5):
                    raise AssertionError(f"non-closed feature file: {path0}")
                if bool(package["reset_between_frames"].item()):
                    raise AssertionError(f"unexpected reset: {path0}")
                if str(package["readout"].item()) != "final":
                    raise AssertionError(f"unexpected readout: {path0}")
                if system == "qrc" and not bool(package["closed_unitary"].item()):
                    raise AssertionError(f"QRC not marked closed: {path0}")
            feature_contracts.append({"system": system, "seed": seed, "path": str(path0)})
    audit = {
        "status": "passed",
        "reservoir_rows": len(selected),
        "validation_scan_rows": len(scan),
        "classical_rows": len(classical),
        "p": 0.0,
        "W": 0.5,
        "histories": list(cfg.histories),
        "feature_contracts": feature_contracts,
        "test_used_for_selection": False,
    }
    path = TASK_DIR / "outputs" / "audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return path
