"""X-only exact-reset DTC-QRC and matched ESN for Transition1x."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


TASK_DIR = Path(__file__).resolve().parent
TRANSITION_DIR = TASK_DIR.parent
WORKSPACE = TRANSITION_DIR.parent
TASK1_DIR = TRANSITION_DIR / "task1"
HAMILTONIAN_DIR = WORKSPACE / "spatrace_dtc_qrc" / "01_algorithm_validation"
QRC_CORE_DIR = WORKSPACE / "BFS-QRC" / "FINAL" / "2_linear_readout" / "core"
for directory in (TASK1_DIR, HAMILTONIAN_DIR, QRC_CORE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from task1_core import _masked_errors  # noqa: E402


METHODS = (
    "qrc_x_dissipative",
    "qrc_x_closed",
    "esn_x_dissipative",
    "esn_x_closed",
)


@dataclass(frozen=True)
class Task5Config:
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


def load_config(path: str | Path | None = None) -> Task5Config:
    config_path = TASK_DIR / "config.json" if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    counts, qrc = payload["sample_counts"], payload["qrc"]
    cfg = Task5Config(
        source_processed_dataset=(TASK_DIR / payload["source_processed_dataset"]).resolve(),
        processed_dataset=(TASK_DIR / payload["processed_dataset"]).resolve(),
        train_count=int(counts["train"]),
        val_count=int(counts["val"]),
        test_count=int(counts["test"]),
        q=int(payload["num_qubits"]),
        source_images=int(payload["source_images"]),
        fields_per_image=int(payload["fields_per_image"]),
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
    if (cfg.q, cfg.source_images, cfg.fields_per_image, cfg.total_injections) != (8, 6, 3, 18):
        raise ValueError("Task 5 locks Q=8, six images, three fields, and 18 injections")
    if cfg.g != 0.84 or cfg.W != 0.5 or cfg.layers != 3:
        raise ValueError("Task 5 locks g=0.84, W=0.5, and L=3")
    if not cfg.p_grid or cfg.p_grid[0] != 0.0 or any(not 0 <= p < 1 for p in cfg.p_grid):
        raise ValueError("p grid must start at zero and remain in [0,1)")
    return cfg


def x_only_field_injections(source_tokens: np.ndarray, q: int) -> np.ndarray:
    """Serialize identity/geometry/motion as separate X-only injections."""
    source = np.asarray(source_tokens, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] != 3 * int(q):
        raise ValueError(f"expected [N,T,{3 * int(q)}], got {source.shape}")
    structured = source.reshape(len(source), source.shape[1], int(q), 3)
    output = np.zeros((len(source), 3 * source.shape[1], int(q), 3), dtype=np.float32)
    for field in range(3):
        output[:, field::3, :, 0] = structured[:, :, :, field]
    if np.any(output[..., 1:] != 0):
        raise AssertionError("Y and Z injection channels must be exactly zero")
    return output.reshape(len(source), 3 * source.shape[1], 3 * int(q))


def build_dataset(cfg: Task5Config, force: bool = False) -> Path:
    if cfg.processed_dataset.exists() and not force:
        return cfg.processed_dataset
    with np.load(cfg.source_processed_dataset, allow_pickle=False) as source:
        tokens = source["tokens"].astype(np.float32)
        expected = (cfg.total_count, cfg.source_images, 3 * cfg.q)
        if tokens.shape != expected:
            raise RuntimeError(f"source token shape {tokens.shape} != {expected}")
        replaced = {
            "tokens", "dataset_version", "source_processed_dataset",
            "injection_axis", "image_start_indices",
        }
        arrays = {key: source[key] for key in source.files if key not in replaced}
    expanded = x_only_field_injections(tokens, cfg.q)
    if expanded.shape != (cfg.total_count, cfg.total_injections, 3 * cfg.q):
        raise AssertionError(expanded.shape)
    cfg.processed_dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.processed_dataset,
        tokens=expanded,
        dataset_version=np.asarray("transition1x-task5-xonly-field-serialization-v1"),
        source_processed_dataset=np.asarray(str(cfg.source_processed_dataset)),
        injection_axis=np.asarray("X"),
        image_start_indices=np.arange(0, cfg.total_injections, cfg.fields_per_image, dtype=np.int16),
        **arrays,
    )
    return cfg.processed_dataset


def reset_suffix_weights(num_images: int, probability: float) -> np.ndarray:
    """Last-reset weights with resets after images 1..T-1 and none after T."""
    t, p = int(num_images), float(probability)
    if t < 1 or not 0 <= p <= 1:
        raise ValueError("invalid reset arguments")
    survival = 1.0 - p
    weights = np.empty(t, dtype=np.float64)
    weights[0] = survival ** (t - 1)
    for start in range(1, t):
        weights[start] = p * survival ** (t - 1 - start)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise AssertionError(weights)
    return weights


def mix_suffix_bank(suffixes: np.ndarray, probability: float) -> np.ndarray:
    suffixes = np.asarray(suffixes, dtype=np.float64)
    if suffixes.ndim != 3:
        raise ValueError("suffix bank must be [image_start,N,D]")
    return np.tensordot(reset_suffix_weights(len(suffixes), probability), suffixes, axes=(0, 0)).astype(np.float32)


def _feature_worker(
    system: str, seed: int, gpu_id: int, indices: np.ndarray,
    dataset_path: str, output_path: str, cfg_payload: dict[str, Any], error_queue,
) -> None:
    try:
        import torch
        torch.cuda.set_device(gpu_id)
        with np.load(dataset_path, allow_pickle=False) as data:
            tokens = np.ascontiguousarray(data["tokens"][indices], dtype=np.float32)
        q, fields = int(cfg_payload["q"]), int(cfg_payload["fields_per_image"])
        image_starts = range(0, tokens.shape[1], fields)
        banks, parameter_summary = [], {}
        started = time.perf_counter()
        if system == "qrc":
            from hamiltonian_dtc_qrc import DTCQRCConfig, compute_qrc_features
            qrc_cfg = DTCQRCConfig(
                num_qubits=q, features_per_qubit=3, g=float(cfg_payload["g"]),
                W_scale=float(cfg_payload["W"]), layers_per_injection=int(cfg_payload["layers"]),
                complex_dtype="complex64",
            )
            scaled = tokens * np.float32(cfg_payload["angle_scale"] * cfg_payload["injection_scale"])
            for image_start in image_starts:
                feature, params = compute_qrc_features(
                    scaled[:, image_start:, :], qrc_cfg, seed=seed,
                    batch_size=int(cfg_payload["batch_size"]), device=f"cuda:{gpu_id}", progress_every=1_000_000,
                )
                banks.append(feature)
            parameter_summary = {"J": params["J"].tolist(), "h": params["h"].tolist()}
        elif system == "esn":
            from esn_baseline import ESN_STANDARD_PROTOCOL, compute_esn_standard_features
            for image_start in image_starts:
                feature, params = compute_esn_standard_features(
                    tokens[:, image_start:, :], q=q, layers=int(cfg_payload["layers"]),
                    seed=seed, device=f"cuda:{gpu_id}",
                )
                banks.append(feature)
            parameter_summary = {
                "protocol": ESN_STANDARD_PROTOCOL,
                "spectral_radius": float(params["spectral_radius"]),
                "input_scale": float(params["input_scale"]),
                "leak_rate": float(params["leak_rate"]),
            }
        else:
            raise ValueError(system)
        bank = np.stack(banks).astype(np.float32)
        np.savez_compressed(
            output_path, indices=indices, suffixes=bank, system=np.asarray(system),
            seed=np.int64(seed), gpu_id=np.int64(gpu_id),
            elapsed_seconds=np.float64(time.perf_counter() - started),
            parameter_summary=np.asarray(json.dumps(parameter_summary)),
        )
    except Exception:
        error_queue.put(traceback.format_exc())
        raise


def compute_suffix_bank_dual_gpu(
    system: str, seed: int, cfg: Task5Config, feature_dir: Path, force: bool = False,
) -> Path:
    feature_dir.mkdir(parents=True, exist_ok=True)
    combined = feature_dir / f"{system}_x_suffixes_seed{seed:03d}.npz"
    if combined.exists() and not force:
        return combined
    indices = np.array_split(np.arange(cfg.total_count, dtype=np.int64), len(cfg.gpu_ids))
    parts = [feature_dir / f"{system}_x_suffixes_seed{seed:03d}_gpu{gpu}_part.npz" for gpu in cfg.gpu_ids]
    context, processes = mp.get_context("spawn"), []
    errors = context.Queue()
    payload = {
        "q": cfg.q, "fields_per_image": cfg.fields_per_image, "g": cfg.g, "W": cfg.W,
        "layers": cfg.layers, "angle_scale": cfg.angle_scale,
        "injection_scale": cfg.injection_scale, "batch_size": 256,
    }
    for gpu, index, part in zip(cfg.gpu_ids, indices, parts):
        process = context.Process(target=_feature_worker, args=(
            system, seed, gpu, index, str(cfg.processed_dataset), str(part), payload, errors,
        ))
        process.start(); processes.append(process)
    for process in processes:
        process.join()
    messages = []
    while not errors.empty():
        messages.append(errors.get())
    if messages or any(process.exitcode != 0 for process in processes):
        raise RuntimeError("dual-GPU extraction failed\n" + "\n".join(messages))
    suffixes = np.empty((cfg.source_images, cfg.total_count, cfg.output_dim), dtype=np.float32)
    metadata = []
    for part in parts:
        with np.load(part, allow_pickle=False) as package:
            index, values = package["indices"], package["suffixes"]
            if values.shape != (cfg.source_images, len(index), cfg.output_dim):
                raise RuntimeError(f"bad suffix shape {values.shape}")
            suffixes[:, index] = values
            metadata.append({
                "gpu_id": int(package["gpu_id"]), "num_samples": len(index),
                "elapsed_seconds": float(package["elapsed_seconds"]),
                "parameter_summary": json.loads(str(package["parameter_summary"].item())),
            })
    np.savez_compressed(
        combined, suffixes=suffixes, system=np.asarray(system), seed=np.int64(seed),
        g=np.float32(cfg.g), W=np.float32(cfg.W), layers=np.int16(cfg.layers),
        injection_axis=np.asarray(cfg.injection_axis), reset_boundaries=np.arange(cfg.source_images, dtype=np.int16),
        gpu_parts=np.asarray(json.dumps(metadata)),
    )
    return combined


def _validation_selected_prediction(
    bank: np.ndarray, target: np.ndarray, splits: np.ndarray, p_grid: tuple[float, ...],
    alpha_grid: tuple[float, ...], metric, closed: bool,
) -> tuple[np.ndarray, float, float, float]:
    train, val, test = splits == "train", splits == "val", splits == "test"
    candidates = (0.0,) if closed else p_grid
    best = None
    for p in candidates:
        features = mix_suffix_bank(bank, p)
        scaler = StandardScaler().fit(features[train])
        x_train, x_val = scaler.transform(features[train]), scaler.transform(features[val])
        for alpha in alpha_grid:
            model = Ridge(alpha=alpha).fit(x_train, target[train])
            score = float(metric(target[val], model.predict(x_val), val))
            key = (score, p, alpha)
            if best is None or key < best[0]:
                best = (key, p, alpha)
    assert best is not None
    _, selected_p, selected_alpha = best
    features = mix_suffix_bank(bank, selected_p)
    train_val = train | val
    scaler = StandardScaler().fit(features[train_val])
    model = Ridge(alpha=selected_alpha).fit(scaler.transform(features[train_val]), target[train_val])
    prediction = model.predict(scaler.transform(features[test]))
    return np.asarray(prediction), float(selected_p), float(selected_alpha), float(best[0][0])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _bootstrap_comparisons(
    prediction_dir: Path,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    output_dir: Path,
    repeats: int = 20_000,
) -> list[dict[str, Any]]:
    ensembles: dict[str, dict[str, np.ndarray]] = {}
    reference = None
    for method in methods:
        packages = []
        for seed in seeds:
            with np.load(prediction_dir / f"{method}_seed{seed:03d}.npz", allow_pickle=False) as package:
                packages.append({key: package[key] for key in package.files})
        current = packages[0]
        if reference is None:
            reference = current
        for key in ("reaction_ids", "formulas", "target_rho", "target_energy", "valid_mask", "changed_mask"):
            if not np.array_equal(current[key], reference[key]):
                raise RuntimeError(f"prediction audit identity mismatch: {method}/{key}")
        ensembles[method] = {
            "rho_prediction": np.mean([p["rho_prediction"] for p in packages], axis=0),
            "energy_prediction": np.mean([p["energy_prediction"] for p in packages], axis=0),
        }
    assert reference is not None
    valid = reference["valid_mask"].astype(bool)
    changed = valid & reference["changed_mask"].astype(bool)
    losses: dict[str, dict[str, np.ndarray]] = {}
    for method, prediction in ensembles.items():
        rho_error = np.abs(prediction["rho_prediction"] - reference["target_rho"])
        losses[method] = {
            "rho_mae_all_pairs": np.sum(rho_error * valid, axis=1) / np.sum(valid, axis=1),
            "rho_mae_changed_pairs": np.sum(rho_error * changed, axis=1) / np.sum(changed, axis=1),
            "energy_mae_eV": np.abs(prediction["energy_prediction"] - reference["target_energy"]),
        }
    ensemble_rows = [
        {"method": method, "metric": metric, "ensemble_test_mae": float(values.mean())}
        for method, method_losses in losses.items()
        for metric, values in method_losses.items()
    ]
    _write_rows(output_dir / "ensemble_metrics.csv", ensemble_rows)
    comparisons = (
        ("qrc_x_dissipative", "esn_x_dissipative"),
        ("qrc_x_dissipative", "qrc_x_closed"),
        ("esn_x_dissipative", "esn_x_closed"),
    )
    formulas = reference["formulas"].astype(str)
    unique_formulas = np.unique(formulas)
    formula_indices = [np.flatnonzero(formulas == formula) for formula in unique_formulas]
    rng = np.random.default_rng(20260826)
    rows: list[dict[str, Any]] = []
    for left, right in comparisons:
        for metric in ("rho_mae_all_pairs", "rho_mae_changed_pairs", "energy_mae_eV"):
            difference = losses[left][metric] - losses[right][metric]
            reaction_draws = rng.integers(0, len(difference), size=(repeats, len(difference)))
            reaction_values = difference[reaction_draws].mean(axis=1)
            formula_values = np.empty(repeats, dtype=np.float64)
            for repeat in range(repeats):
                chosen = rng.integers(0, len(unique_formulas), size=len(unique_formulas))
                sampled = np.concatenate([formula_indices[index] for index in chosen])
                formula_values[repeat] = difference[sampled].mean()
            for unit, values in (("reaction", reaction_values), ("formula_cluster", formula_values)):
                rows.append({
                    "left_method": left, "right_method": right, "metric": metric,
                    "difference_definition": "left MAE minus right MAE; negative favours left",
                    "bootstrap_unit": unit, "n_test_reactions": len(difference),
                    "n_test_formulas": len(unique_formulas), "bootstrap_repeats": repeats,
                    "mean_difference": float(difference.mean()),
                    "ci_2p5": float(np.quantile(values, 0.025)),
                    "ci_97p5": float(np.quantile(values, 0.975)),
                    "probability_left_better": float(np.mean(values < 0)),
                })
    _write_rows(output_dir / "paired_bootstrap.csv", rows)
    return rows


def evaluate(cfg: Task5Config, feature_dir: Path, output_dir: Path) -> Path:
    with np.load(cfg.processed_dataset, allow_pickle=False) as data:
        splits, reaction_ids, formulas = data["splits"], data["reaction_ids"], data["formulas"]
        target_rho = data["target_delta_rho"].astype(np.float64)
        target_energy = data["target_delta_energy"].astype(np.float64)
        valid, changed = data["valid_pair_mask"], data["changed_pair_mask"]
    test = splits == "test"
    rows, prediction_rows = [], []
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for method in METHODS:
        system = method.split("_")[0]
        closed = method.endswith("closed")
        for seed in cfg.reservoir_seeds:
            with np.load(feature_dir / f"{system}_x_suffixes_seed{seed:03d}.npz", allow_pickle=False) as package:
                bank = package["suffixes"].astype(np.float32)
            def rho_metric(y, prediction, mask):
                return np.mean(np.abs(prediction - y)[valid[mask]])
            rho_pred, rho_p, rho_alpha, rho_val = _validation_selected_prediction(
                bank, target_rho, splits, cfg.p_grid, cfg.alpha_grid, rho_metric, closed,
            )
            def energy_metric(y, prediction, _mask):
                return np.mean(np.abs(prediction - y))
            energy_pred, energy_p, energy_alpha, energy_val = _validation_selected_prediction(
                bank, target_energy, splits, cfg.p_grid, cfg.alpha_grid, energy_metric, closed,
            )
            all_mae, all_rmse = _masked_errors(target_rho[test], rho_pred, valid[test])
            changed_mae, changed_rmse = _masked_errors(target_rho[test], rho_pred, valid[test] & changed[test])
            energy_error = energy_pred - target_energy[test]
            rows.append({
                "method": method, "seed": seed, "feature_dim": cfg.output_dim,
                "rho_selected_p": rho_p, "rho_alpha": rho_alpha, "rho_val_mae": rho_val,
                "energy_selected_p": energy_p, "energy_alpha": energy_alpha, "energy_val_mae": energy_val,
                "rho_mae_all_pairs": all_mae, "rho_rmse_all_pairs": all_rmse,
                "rho_mae_changed_pairs": changed_mae, "rho_rmse_changed_pairs": changed_rmse,
                "energy_mae_eV": float(np.mean(np.abs(energy_error))),
                "energy_rmse_eV": float(np.sqrt(np.mean(energy_error ** 2))),
            })
            np.savez_compressed(
                prediction_dir / f"{method}_seed{seed:03d}.npz",
                reaction_ids=reaction_ids[test], formulas=formulas[test],
                target_rho=target_rho[test], rho_prediction=rho_pred,
                target_energy=target_energy[test], energy_prediction=energy_pred,
                valid_mask=valid[test], changed_mask=changed[test],
                method=np.asarray(method), seed=np.int64(seed),
            )
            for local, sample in enumerate(np.flatnonzero(test)):
                prediction_rows.append({
                    "reaction_id": str(reaction_ids[sample]), "method": method, "seed": seed,
                    "target_delta_energy_eV": float(target_energy[sample]),
                    "predicted_delta_energy_eV": float(energy_pred[local]),
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "metrics_by_seed.csv", rows)
    _write_rows(output_dir / "energy_test_predictions.csv", prediction_rows)
    metrics = ("rho_mae_all_pairs", "rho_mae_changed_pairs", "energy_mae_eV")
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected])
            summary.append({"method": method, "metric": metric, "mean": values.mean(), "sample_std": values.std(ddof=1), "num_seeds": len(values)})
    _write_rows(output_dir / "summary.csv", summary)
    by_key = {(row["method"], row["seed"]): row for row in rows}
    pairs = []
    for seed in cfg.reservoir_seeds:
        row = {"seed": seed}
        for metric in metrics:
            row[f"qrc_open_minus_esn_open_{metric}"] = by_key[("qrc_x_dissipative", seed)][metric] - by_key[("esn_x_dissipative", seed)][metric]
            row[f"qrc_open_minus_closed_{metric}"] = by_key[("qrc_x_dissipative", seed)][metric] - by_key[("qrc_x_closed", seed)][metric]
            row[f"esn_open_minus_closed_{metric}"] = by_key[("esn_x_dissipative", seed)][metric] - by_key[("esn_x_closed", seed)][metric]
        pairs.append(row)
    _write_rows(output_dir / "paired_differences.csv", pairs)
    bootstrap_rows = _bootstrap_comparisons(
        prediction_dir, METHODS, cfg.reservoir_seeds, output_dir
    )
    report = {
        "status": "completed", "protocol": "transition1x-task5-xonly-exact-reset-W0p5-v2",
        "dataset": str(cfg.processed_dataset), "split_counts": {name: int(np.sum(splits == name)) for name in ("train", "val", "test")},
        "q": cfg.q, "g": cfg.g, "W": cfg.W, "layers": cfg.layers,
        "source_images": cfg.source_images, "total_x_injections": cfg.total_injections,
        "injection": "identity, geometry, motion serialized as three [value,0,0] injections per image",
        "dissipation": "exact reset-to-|0> between images; validation-selected p; no post-final reset",
        "p_grid": list(cfg.p_grid), "reservoir_seeds": list(cfg.reservoir_seeds),
        "gpu_contract": {"qrc": list(cfg.gpu_ids), "esn": list(cfg.gpu_ids), "strategy": "sample-axis split for every system and seed"},
        "summary": summary, "paired_bootstrap": bootstrap_rows,
    }
    path = output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
