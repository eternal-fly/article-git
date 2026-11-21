"""End-to-end feature extraction, nested readout analysis, and baselines."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from data_pipeline import apply_linear_angle_normalizer, fit_linear_angle_normalizer
from hamiltonian_dtc_qrc import DTCQRCConfig, compute_qrc_features, injection_audit


SPLITS = ("train", "validation", "test")
METRICS = ("macro_nrmse", "global_rmse", "macro_r2", "macro_pearson", "mae")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_with_linear_angles(
    dataset_path: str | Path,
    *,
    angle_scale: float,
    output_dir: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    normalizer = fit_linear_angle_normalizer(data["train_tokens"], float(angle_scale))
    for split in SPLITS:
        data[f"{split}_angles"] = apply_linear_angle_normalizer(data[f"{split}_tokens"], normalizer)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "linear_angle_normalizer.npz",
        slope=normalizer["slope"],
        intercept=normalizer["intercept"],
        angle_scale=np.asarray([normalizer["angle_scale"]], dtype=np.float32),
    )
    # An affine-map identity check.  No clipping is applied to validation/test values.
    a = data["train_tokens"][0, 0]
    b = data["train_tokens"][-1, -1]
    weight = 0.37
    left = apply_linear_angle_normalizer(weight * a + (1.0 - weight) * b, normalizer)
    right = weight * apply_linear_angle_normalizer(a, normalizer) + (1.0 - weight) * apply_linear_angle_normalizer(b, normalizer)
    audit = {
        "preprocessing": "featurewise train-only affine min-max map to [-angle_scale, angle_scale]",
        "tanh": False,
        "clipping": False,
        "angle_scale": float(angle_scale),
        "affine_identity_max_error": float(np.max(np.abs(left - right))),
        "validation_outside_training_angle_range_fraction": float(
            np.mean(np.abs(data["validation_angles"]) > float(angle_scale) + 1e-6)
        ),
        "test_outside_training_angle_range_fraction": float(
            np.mean(np.abs(data["test_angles"]) > float(angle_scale) + 1e-6)
        ),
    }
    (output_dir / "linear_normalization_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data, audit


def run_injection_audit(data: dict[str, np.ndarray], output_dir: str | Path) -> dict[str, float]:
    sample = data["train_angles"][: min(16, len(data["train_angles"]))].reshape(-1, 3)
    audit = injection_audit(sample)
    audit["interpretation"] = (
        "Any nonzero single-qubit Hamiltonian cannot commute with both the DTC X-pulse "
        "and its Z/ZZ sector. The affine preprocessing is linear, but the exponential "
        "encoding and Born readout are still nonlinear; L=0 is therefore retained as a control."
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hamiltonian_injection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def _qrc_config(config: dict, layers: int | None = None) -> DTCQRCConfig:
    return DTCQRCConfig(
        num_qubits=int(config["num_qubits"]),
        features_per_qubit=int(config["features_per_qubit"]),
        g=float(config["g"]),
        W_scale=float(config["W"]),
        layers_per_injection=int(config["L"] if layers is None else layers),
        complex_dtype=str(config["complex_dtype"]),
    )


def _feature_cache_path(cache_dir: str | Path, method: str, seed: int) -> Path:
    return Path(cache_dir) / f"{method}_reservoir_seed{int(seed):04d}.npz"


def _compute_seed_group(
    device: str,
    seeds: list[int],
    angles_path: str,
    split_sizes: dict[str, int],
    qrc_config: dict,
    cache_dir: str,
    dataset_hash: str,
    batch_size: int,
    force: bool,
) -> list[str]:
    with np.load(angles_path, allow_pickle=False) as archive:
        combined = archive["angles"]
    cfg = _qrc_config(qrc_config)
    outputs: list[str] = []
    offsets = np.cumsum([0] + [split_sizes[name] for name in SPLITS])
    for seed in seeds:
        destination = _feature_cache_path(cache_dir, "dtc_qrc_l3", seed)
        if destination.exists() and not force:
            outputs.append(str(destination))
            print(f"[QRC seed={seed}] reuse {destination.name}", flush=True)
            continue
        features, params = compute_qrc_features(
            combined,
            cfg,
            seed=int(seed),
            batch_size=int(batch_size),
            device=device,
        )
        payload: dict[str, Any] = {
            "J": params["J"],
            "h": params["h"],
            "reservoir_seed": np.asarray([seed], dtype=np.int64),
            "dataset_sha256": np.asarray(dataset_hash),
            "config_json": np.asarray(json.dumps(qrc_config, sort_keys=True)),
        }
        for index, split in enumerate(SPLITS):
            payload[split] = features[offsets[index] : offsets[index + 1]]
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".partial.npz")
        np.savez_compressed(partial, **payload)
        partial.replace(destination)
        outputs.append(str(destination))
        del features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def extract_qrc_features(
    data: dict[str, np.ndarray],
    *,
    dataset_path: str | Path,
    qrc_config: dict,
    reservoir_seeds: list[int],
    devices: list[str],
    batch_size: int,
    cache_dir: str | Path,
    force: bool,
) -> list[Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined = np.concatenate([data[f"{split}_angles"] for split in SPLITS], axis=0)
    angles_path = cache_dir / "fixed_linear_angles_all_splits.npz"
    np.savez_compressed(angles_path, angles=combined)
    split_sizes = {split: int(len(data[f"{split}_angles"])) for split in SPLITS}
    dataset_hash = _sha256(dataset_path)
    assignments = [[] for _ in devices]
    for index, seed in enumerate(reservoir_seeds):
        assignments[index % len(devices)].append(int(seed))
    jobs = [(devices[i], assignments[i]) for i in range(len(devices)) if assignments[i]]
    if len(jobs) == 1:
        paths = _compute_seed_group(
            jobs[0][0], jobs[0][1], str(angles_path), split_sizes, qrc_config,
            str(cache_dir), dataset_hash, int(batch_size), bool(force)
        )
    else:
        context = mp.get_context("spawn")
        paths = []
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as executor:
            futures = [
                executor.submit(
                    _compute_seed_group,
                    device,
                    seeds,
                    str(angles_path),
                    split_sizes,
                    qrc_config,
                    str(cache_dir),
                    dataset_hash,
                    int(batch_size),
                    bool(force),
                )
                for device, seeds in jobs
            ]
            for future in as_completed(futures):
                paths.extend(future.result())
    by_seed = {int(Path(path).stem.rsplit("seed", 1)[1]): Path(path) for path in paths}
    return [by_seed[int(seed)] for seed in reservoir_seeds]


def extract_injection_only_features(
    data: dict[str, np.ndarray],
    *,
    qrc_config: dict,
    device: str,
    batch_size: int,
    cache_dir: str | Path,
    force: bool,
) -> Path:
    destination = _feature_cache_path(cache_dir, "injection_only_l0", 0)
    if destination.exists() and not force:
        return destination
    combined = np.concatenate([data[f"{split}_angles"] for split in SPLITS], axis=0)
    cfg = _qrc_config(qrc_config, layers=0)
    features, _ = compute_qrc_features(combined, cfg, seed=0, batch_size=batch_size, device=device)
    sizes = [len(data[f"{split}_angles"]) for split in SPLITS]
    offsets = np.cumsum([0] + sizes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        **{split: features[offsets[index] : offsets[index + 1]] for index, split in enumerate(SPLITS)},
    )
    return destination


def esn_features(
    tokens: np.ndarray,
    *,
    hidden_size: int,
    seed: int,
    spectral_radius: float,
    input_scale: float,
    leak_rate: float,
) -> np.ndarray:
    """Standard fixed tanh ESN; final hidden dimension matches the 210 QRC observables."""
    rng = np.random.default_rng(int(seed))
    input_dim = tokens.shape[-1]
    W_in = rng.normal(0.0, float(input_scale) / math.sqrt(input_dim), size=(hidden_size, input_dim))
    W = rng.normal(0.0, 1.0 / math.sqrt(hidden_size), size=(hidden_size, hidden_size))
    radius = float(np.max(np.abs(np.linalg.eigvals(W))))
    W *= float(spectral_radius) / max(radius, 1e-12)
    state = np.zeros((len(tokens), hidden_size), dtype=np.float64)
    for event in range(tokens.shape[1]):
        proposal = np.tanh(tokens[:, event] @ W_in.T + state @ W.T)
        state = (1.0 - float(leak_rate)) * state + float(leak_rate) * proposal
    return state.astype(np.float32)


def _set_training_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray, train_std: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - y_true
    rmse_gene = np.sqrt(np.mean(error**2, axis=0))
    denominator = np.asarray(train_std, dtype=np.float64).copy()
    denominator[denominator < 1e-8] = 1.0
    nrmse_gene = rmse_gene / denominator
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    ss_total = np.sum(centered**2, axis=0)
    ss_residual = np.sum(error**2, axis=0)
    valid_r2 = ss_total > 1e-12
    r2 = np.full(y_true.shape[1], np.nan)
    r2[valid_r2] = 1.0 - ss_residual[valid_r2] / ss_total[valid_r2]
    pearson = []
    for gene in range(y_true.shape[1]):
        if np.std(y_true[:, gene]) > 1e-12 and np.std(prediction[:, gene]) > 1e-12:
            pearson.append(float(np.corrcoef(y_true[:, gene], prediction[:, gene])[0, 1]))
    return {
        "macro_nrmse": float(np.mean(nrmse_gene)),
        "global_rmse": float(np.sqrt(np.mean(error**2))),
        "macro_r2": float(np.nanmean(r2)),
        "macro_pearson": float(np.mean(pearson)) if pearson else float("nan"),
        "mae": float(np.mean(np.abs(error))),
    }


def train_affine_readout(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    seed: int,
    config: dict,
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """Fit exactly one multi-output affine layer, with validation-only epoch selection."""
    _set_training_seed(seed)
    requested = str(config["device"])
    device = torch.device(requested if "cuda" not in requested or torch.cuda.is_available() else "cpu")
    x_mean = features["train"].mean(axis=0, keepdims=True)
    x_std = features["train"].std(axis=0, keepdims=True)
    x_std[x_std < 1e-7] = 1.0
    y_mean = targets["train"].mean(axis=0, keepdims=True)
    y_std = targets["train"].std(axis=0, keepdims=True)
    y_std[y_std < 1e-7] = 1.0
    x = {
        split: torch.as_tensor((features[split] - x_mean) / x_std, dtype=torch.float32, device=device)
        for split in SPLITS
    }
    y = {
        split: torch.as_tensor((targets[split] - y_mean) / y_std, dtype=torch.float32, device=device)
        for split in SPLITS
    }
    model = torch.nn.Linear(x["train"].shape[1], y["train"].shape[1], bias=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    loss_fn = torch.nn.MSELoss()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train()
        order = torch.randperm(len(x["train"]), generator=generator, device=device)
        for start in range(0, len(order), int(config["batch_size"])):
            batch = order[start : start + int(config["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x["train"][batch]), y["train"][batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x["validation"]), y["validation"]).item())
        if validation_loss < best_loss - float(config["minimum_delta"]):
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config["patience"]):
                break
    if best_state is None:
        raise RuntimeError("Readout did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction_z = model(x["test"]).cpu().numpy()
    prediction = prediction_z * y_std + y_mean
    metrics = regression_metrics(targets["test"], prediction, targets["train"].std(axis=0))
    details = {"best_epoch": int(best_epoch), "best_validation_mse_z": float(best_loss), "device": str(device)}
    return metrics, prediction.astype(np.float32), details


def _load_feature_file(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {split: archive[split] for split in SPLITS}


def _feature_dict_from_all(all_features: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    sizes = [len(data[f"{split}_targets"]) for split in SPLITS]
    offsets = np.cumsum([0] + sizes)
    return {split: all_features[offsets[index] : offsets[index + 1]] for index, split in enumerate(SPLITS)}


def run_nested_analysis(
    data: dict[str, np.ndarray],
    *,
    qrc_feature_files: list[str | Path],
    injection_only_file: str | Path | None,
    reservoir_seeds: list[int],
    training_seeds: list[int],
    readout_config: dict,
    esn_config: dict,
    angle_scale: float,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {split: data[f"{split}_targets"].astype(np.float32) for split in SPLITS}
    rows: list[dict[str, Any]] = []

    def evaluate(method: str, features: dict[str, np.ndarray], reservoir_seed: int, training_seed: int) -> None:
        metrics, _prediction, details = train_affine_readout(
            features, targets, seed=int(training_seed), config=readout_config
        )
        rows.append(
            {
                "method": method,
                "reservoir_seed": int(reservoir_seed),
                "training_seed": int(training_seed),
                "feature_dim": int(features["train"].shape[1]),
                **details,
                **metrics,
            }
        )

    for reservoir_seed, feature_file in zip(reservoir_seeds, qrc_feature_files, strict=True):
        features = _load_feature_file(feature_file)
        for training_seed in training_seeds:
            evaluate("DTC_QRC_HAMILTONIAN_L3", features, reservoir_seed, training_seed)

    unit_tokens = {split: data[f"{split}_angles"] / float(angle_scale) for split in SPLITS}
    for reservoir_seed in reservoir_seeds:
        esn_all = esn_features(
            np.concatenate([unit_tokens[split] for split in SPLITS], axis=0),
            hidden_size=int(esn_config["hidden_size"]),
            seed=int(reservoir_seed),
            spectral_radius=float(esn_config["spectral_radius"]),
            input_scale=float(esn_config["input_scale"]),
            leak_rate=float(esn_config["leak_rate"]),
        )
        features = _feature_dict_from_all(esn_all, data)
        for training_seed in training_seeds:
            evaluate("ESN_TANH_MATCHED_210", features, reservoir_seed, training_seed)

    raw_features = {split: unit_tokens[split].reshape(len(unit_tokens[split]), -1) for split in SPLITS}
    for training_seed in training_seeds:
        evaluate("RAW_3STEP_AFFINE", raw_features, -1, training_seed)

    if injection_only_file is not None:
        injection_features = _load_feature_file(injection_only_file)
        for training_seed in training_seeds:
            evaluate("HAMILTONIAN_INJECTION_ONLY_L0", injection_features, -1, training_seed)

    train_std = targets["train"].std(axis=0)
    persistence = data["test_tokens"][:, -1, -targets["test"].shape[1] :]
    mean_prediction = np.repeat(targets["train"].mean(axis=0, keepdims=True), len(targets["test"]), axis=0)
    for method, prediction in (("PERSISTENCE_LAST_TG", persistence), ("TRAIN_MEAN", mean_prediction)):
        rows.append(
            {
                "method": method,
                "reservoir_seed": -1,
                "training_seed": -1,
                "feature_dim": int(prediction.shape[1]),
                "best_epoch": 0,
                "best_validation_mse_z": float("nan"),
                "device": "analytic",
                **regression_metrics(targets["test"], prediction, train_std),
            }
        )

    metrics_frame = pd.DataFrame(rows).sort_values(["method", "reservoir_seed", "training_seed"]).reset_index(drop=True)
    metrics_frame.to_csv(output_dir / "nested_metrics.csv", index=False)
    summary_rows = []
    for method, group in metrics_frame.groupby("method", sort=False):
        reservoir_group = group[group["reservoir_seed"] >= 0].groupby("reservoir_seed")
        row: dict[str, Any] = {
            "method": method,
            "n_runs": int(len(group)),
            "n_reservoirs": int(group.loc[group["reservoir_seed"] >= 0, "reservoir_seed"].nunique()),
            "n_training_seeds": int(group.loc[group["training_seed"] >= 0, "training_seed"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd_all_runs"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
            if len(reservoir_group):
                reservoir_means = reservoir_group[metric].mean()
                row[f"{metric}_sd_between_reservoirs"] = (
                    float(reservoir_means.std(ddof=1)) if len(reservoir_means) > 1 else 0.0
                )
                within = reservoir_group[metric].std(ddof=1).dropna()
                row[f"{metric}_mean_within_reservoir_sd"] = float(within.mean()) if len(within) else 0.0
            else:
                row[f"{metric}_sd_between_reservoirs"] = float("nan")
                row[f"{metric}_mean_within_reservoir_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("macro_nrmse_mean").reset_index(drop=True)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    qrc = metrics_frame[metrics_frame["method"] == "DTC_QRC_HAMILTONIAN_L3"]
    esn = metrics_frame[metrics_frame["method"] == "ESN_TANH_MATCHED_210"]
    paired = qrc.merge(esn, on=["reservoir_seed", "training_seed"], suffixes=("_qrc", "_esn"))
    for metric in METRICS:
        paired[f"delta_qrc_minus_esn_{metric}"] = paired[f"{metric}_qrc"] - paired[f"{metric}_esn"]
    paired.to_csv(output_dir / "paired_qrc_vs_esn.csv", index=False)
    paired_summary = pd.DataFrame(
        [
            {
                "n_pairs": int(len(paired)),
                "qrc_nrmse_win_fraction": float(np.mean(paired["macro_nrmse_qrc"] < paired["macro_nrmse_esn"])),
                "mean_delta_qrc_minus_esn_macro_nrmse": float(paired["delta_qrc_minus_esn_macro_nrmse"].mean()),
                "median_delta_qrc_minus_esn_macro_nrmse": float(paired["delta_qrc_minus_esn_macro_nrmse"].median()),
            }
        ]
    )
    paired_summary.to_csv(output_dir / "paired_qrc_vs_esn_summary.csv", index=False)
    return metrics_frame, summary, paired_summary


def plot_summary(summary: pd.DataFrame, output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("macro_nrmse_mean", ascending=True)
    error = ordered["macro_nrmse_sd_between_reservoirs"].fillna(ordered["macro_nrmse_sd_all_runs"])
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    positions = np.arange(len(ordered))
    axis.barh(positions, ordered["macro_nrmse_mean"], xerr=error, color="#3B6FB6", alpha=0.88, capsize=3)
    axis.set_yticks(positions, ordered["method"])
    axis.set_xlabel("Macro NRMSE (lower is better)")
    axis.set_title("Fixed SpaTRACE 3-event → next-TG validation")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path

