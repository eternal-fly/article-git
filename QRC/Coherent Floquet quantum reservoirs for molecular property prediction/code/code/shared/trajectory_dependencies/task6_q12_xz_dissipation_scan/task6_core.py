"""Q12 simultaneous X/Z injection and exact dissipation scan on Transition1x."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TASK_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK_DIR.parent
TASK5_DIR = TRANSITION_DIR / "task5_xonly_dissipative"
TASK1_DIR = TRANSITION_DIR / "task1"
for directory in (TASK5_DIR, TASK1_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from task1_core import _masked_errors  # noqa: E402
from task5_core import (  # noqa: E402
    compute_suffix_bank_dual_gpu,
    mix_suffix_bank,
)


@dataclass(frozen=True)
class Task6Config:
    source_processed_dataset: Path
    processed_dataset: Path
    train_count: int
    val_count: int
    test_count: int
    q: int
    source_images: int
    fields_per_image: int
    total_injections: int
    target_image_index: int
    g: float
    W: float
    layers: int
    angle_scale: float
    injection_scale: float
    injection_axis: str
    p_grid: tuple[float, ...]
    alpha_grid: tuple[float, ...]
    reservoir_seeds: tuple[int, ...]
    gpu_ids: tuple[int, ...]

    @property
    def total_count(self) -> int:
        return self.train_count + self.val_count + self.test_count

    @property
    def output_dim(self) -> int:
        return self.q * (self.q + 1) // 2


def load_config(path: str | Path | None = None) -> Task6Config:
    path = TASK_DIR / "config.json" if path is None else Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts, qrc = payload["sample_counts"], payload["qrc"]
    cfg = Task6Config(
        source_processed_dataset=(TASK_DIR / payload["source_processed_dataset"]).resolve(),
        processed_dataset=(TASK_DIR / payload["processed_dataset"]).resolve(),
        train_count=int(counts["train"]), val_count=int(counts["val"]),
        test_count=int(counts["test"]), q=int(payload["num_qubits"]),
        source_images=int(payload["source_time_steps"]), fields_per_image=1,
        total_injections=int(payload["total_injections"]),
        target_image_index=int(payload["target_image_index"]),
        g=float(qrc["g"]), W=float(qrc["W"]),
        layers=int(qrc["layers_per_injection"]),
        angle_scale=float(qrc["angle_scale"]),
        injection_scale=float(qrc["injection_scale"]),
        injection_axis=str(qrc["injection_axis"]),
        p_grid=tuple(map(float, payload["dissipation"]["p_grid"])),
        alpha_grid=tuple(map(float, payload["ridge_alpha_grid"])),
        reservoir_seeds=tuple(map(int, payload["reservoir_seeds"])),
        gpu_ids=tuple(map(int, payload["gpus_per_method"])),
    )
    if (cfg.q, cfg.source_images, cfg.total_injections, cfg.injection_axis) != (12, 6, 6, "XZ"):
        raise ValueError("Task 6 locks Q12, T6, and simultaneous XZ injection")
    if (cfg.g, cfg.W, cfg.layers) != (0.84, 0.5, 3):
        raise ValueError("Task 6 locks g=0.84, W=0.5, and L=3")
    if cfg.p_grid[0] != 0.0 or cfg.p_grid[-1] != 1.0:
        raise ValueError("Task 6 p scan must include both closed and last-step limits")
    return cfg


def q12_xz_tokens(source_tokens: np.ndarray) -> np.ndarray:
    """Map each ordered pair of the 24 source values to X/Z on one qubit."""
    source = np.asarray(source_tokens, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 24:
        raise ValueError(f"expected [N,T,24], got {source.shape}")
    paired = source.reshape(len(source), source.shape[1], 12, 2)
    output = np.zeros((len(source), source.shape[1], 12, 3), dtype=np.float32)
    output[..., 0] = paired[..., 0]
    output[..., 2] = paired[..., 1]
    if np.count_nonzero(output[..., 1]):
        raise AssertionError("Y injection must be exactly zero")
    return output.reshape(len(source), source.shape[1], 36)


def recover_source_values(xz_tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(xz_tokens, dtype=np.float32).reshape(*xz_tokens.shape[:-1], 12, 3)
    return np.stack((values[..., 0], values[..., 2]), axis=-1).reshape(*xz_tokens.shape[:-1], 24)


def build_dataset(cfg: Task6Config, force: bool = False) -> Path:
    if cfg.processed_dataset.exists() and not force:
        return cfg.processed_dataset
    with np.load(cfg.source_processed_dataset, allow_pickle=False) as source:
        source_tokens = source["tokens"].astype(np.float32)
        expected = (cfg.total_count, cfg.source_images, 24)
        if source_tokens.shape != expected:
            raise RuntimeError(f"source shape {source_tokens.shape} != {expected}")
        replaced = {"tokens", "dataset_version", "source_processed_dataset", "injection_axis"}
        arrays = {key: source[key] for key in source.files if key not in replaced}
    tokens = q12_xz_tokens(source_tokens)
    np.testing.assert_array_equal(recover_source_values(tokens), source_tokens)
    cfg.processed_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.processed_dataset, tokens=tokens,
        dataset_version=np.asarray("transition1x-task6-q12-simultaneous-xz-v1"),
        source_processed_dataset=np.asarray(str(cfg.source_processed_dataset)),
        injection_axis=np.asarray("XZ"), source_token_dim=np.int16(24),
        **arrays,
    )
    return cfg.processed_dataset


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _fit_fixed_p(
    bank: np.ndarray, target: np.ndarray, splits: np.ndarray, p: float,
    alpha_grid: tuple[float, ...], metric,
) -> tuple[np.ndarray, float, float]:
    features = mix_suffix_bank(bank, p)
    train, val, test = splits == "train", splits == "val", splits == "test"
    scaler = StandardScaler().fit(features[train])
    x_train, x_val = scaler.transform(features[train]), scaler.transform(features[val])
    best = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha).fit(x_train, target[train])
        score = float(metric(target[val], model.predict(x_val), val))
        key = (score, alpha)
        if best is None or key < best:
            best = key
    assert best is not None
    score, alpha = best
    train_val = train | val
    scaler = StandardScaler().fit(features[train_val])
    model = Ridge(alpha=alpha).fit(scaler.transform(features[train_val]), target[train_val])
    prediction = model.predict(scaler.transform(features[test]))
    return np.asarray(prediction), float(alpha), float(score)


def evaluate_p_scan(cfg: Task6Config, feature_dir: Path, output_dir: Path) -> Path:
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        splits = data["splits"]
        target_rho = data["target_delta_rho"].astype(np.float64)
        target_energy = data["target_delta_energy"].astype(np.float64)
        valid, changed = data["valid_pair_mask"], data["changed_pair_mask"]
    test = splits == "test"
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int, float], tuple[np.ndarray, np.ndarray]] = {}
    for system in ("qrc", "esn"):
        for seed in cfg.reservoir_seeds:
            with np.load(feature_dir / f"{system}_x_suffixes_seed{seed:03d}.npz", allow_pickle=False) as package:
                bank = package["suffixes"].astype(np.float32)
            for p in cfg.p_grid:
                def rho_metric(y, prediction, mask):
                    return np.mean(np.abs(prediction - y)[valid[mask]])
                rho_prediction, rho_alpha, rho_val = _fit_fixed_p(
                    bank, target_rho, splits, p, cfg.alpha_grid, rho_metric,
                )
                def energy_metric(y, prediction, _mask):
                    return np.mean(np.abs(prediction - y))
                energy_prediction, energy_alpha, energy_val = _fit_fixed_p(
                    bank, target_energy, splits, p, cfg.alpha_grid, energy_metric,
                )
                all_mae, all_rmse = _masked_errors(target_rho[test], rho_prediction, valid[test])
                changed_mae, changed_rmse = _masked_errors(
                    target_rho[test], rho_prediction, valid[test] & changed[test]
                )
                energy_error = energy_prediction - target_energy[test]
                rows.append({
                    "system": system, "seed": seed, "p": p, "feature_dim": cfg.output_dim,
                    "rho_alpha": rho_alpha, "rho_val_mae": rho_val,
                    "energy_alpha": energy_alpha, "energy_val_mae": energy_val,
                    "rho_mae_all_pairs": all_mae, "rho_rmse_all_pairs": all_rmse,
                    "rho_mae_changed_pairs": changed_mae, "rho_rmse_changed_pairs": changed_rmse,
                    "energy_mae_eV": float(np.mean(np.abs(energy_error))),
                    "energy_rmse_eV": float(np.sqrt(np.mean(energy_error ** 2))),
                })
                predictions[(system, seed, p)] = (rho_prediction, energy_prediction)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "p_scan_by_seed.csv", rows)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["system", "p"], as_index=False).agg(
        rho_val_mae_mean=("rho_val_mae", "mean"),
        energy_val_mae_mean=("energy_val_mae", "mean"),
        rho_mae_all_pairs_mean=("rho_mae_all_pairs", "mean"),
        rho_mae_all_pairs_std=("rho_mae_all_pairs", "std"),
        rho_mae_changed_pairs_mean=("rho_mae_changed_pairs", "mean"),
        rho_mae_changed_pairs_std=("rho_mae_changed_pairs", "std"),
        energy_mae_eV_mean=("energy_mae_eV", "mean"),
        energy_mae_eV_std=("energy_mae_eV", "std"),
    )
    summary.to_csv(output_dir / "p_scan_summary.csv", index=False)

    selected_rows = []
    selected_predictions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for system in ("qrc", "esn"):
        for seed in cfg.reservoir_seeds:
            candidates = frame[(frame.system == system) & (frame.seed == seed)]
            rho_row = candidates.sort_values(["rho_val_mae", "p", "rho_alpha"]).iloc[0]
            energy_row = candidates.sort_values(["energy_val_mae", "p", "energy_alpha"]).iloc[0]
            selected_rows.append({
                "system": system, "seed": seed,
                "rho_selected_p": rho_row.p, "rho_alpha": rho_row.rho_alpha,
                "rho_val_mae": rho_row.rho_val_mae,
                "rho_mae_all_pairs": rho_row.rho_mae_all_pairs,
                "rho_mae_changed_pairs": rho_row.rho_mae_changed_pairs,
                "energy_selected_p": energy_row.p, "energy_alpha": energy_row.energy_alpha,
                "energy_val_mae": energy_row.energy_val_mae,
                "energy_mae_eV": energy_row.energy_mae_eV,
            })
            rho_prediction = predictions[(system, seed, float(rho_row.p))][0]
            energy_prediction = predictions[(system, seed, float(energy_row.p))][1]
            selected_predictions[(system, seed)] = (rho_prediction, energy_prediction)
    _write_rows(output_dir / "validation_selected_by_seed.csv", selected_rows)
    selected_frame = pd.DataFrame(selected_rows)
    selected_summary = selected_frame.groupby("system", as_index=False).agg(
        rho_mae_all_pairs_mean=("rho_mae_all_pairs", "mean"),
        rho_mae_all_pairs_std=("rho_mae_all_pairs", "std"),
        rho_mae_changed_pairs_mean=("rho_mae_changed_pairs", "mean"),
        rho_mae_changed_pairs_std=("rho_mae_changed_pairs", "std"),
        energy_mae_eV_mean=("energy_mae_eV", "mean"),
        energy_mae_eV_std=("energy_mae_eV", "std"),
    )
    selected_summary.to_csv(output_dir / "validation_selected_summary.csv", index=False)

    report = {
        "status": "completed", "protocol": "transition1x-task6-q12-xz-p-scan-v1",
        "dataset": str(cfg.processed_dataset), "q": cfg.q, "g": cfg.g, "W": cfg.W,
        "layers": cfg.layers, "time_steps": cfg.total_injections,
        "source_values_per_step": 24,
        "mapping": "(d[2j],d[2j+1]) -> (X_j,Z_j); Y_j=0; j=0..11",
        "p_grid": list(cfg.p_grid), "p_scan_rows": len(rows),
        "reservoir_seeds": list(cfg.reservoir_seeds),
        "gpu_contract": {"qrc": list(cfg.gpu_ids), "esn": list(cfg.gpu_ids)},
        "validation_selected_summary": selected_summary.to_dict(orient="records"),
    }
    path = output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

