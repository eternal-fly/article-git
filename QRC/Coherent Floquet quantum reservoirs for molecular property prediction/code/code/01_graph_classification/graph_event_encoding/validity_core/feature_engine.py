"""Aligned QRC/CRC/raw feature extraction and frozen-probe evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


MODULE_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = MODULE_ROOT.parent
BASELINE_ROOT = FINAL_ROOT / "2_linear_readout"
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from core.gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu  # noqa: E402
from core.crc3_baseline import (  # noqa: E402
    compute_crc3_z_zz_features,
    crc3_z_zz_dim,
)
from core.linear_readout import (  # noqa: E402
    SELECTION_RULE,
    ExperimentConfig,
    compute_crc_features,
    scaffold_split_indices,
    train_linear_probe,
)
from .token_variants import build_order_tensor


def json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_cache_extended(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32, copy=False)
        labels = package["labels"].astype(np.int64, copy=False)
        smiles = package["smiles"].astype(str)
        if "event_counts" in package.files:
            counts = package["event_counts"].astype(np.int16, copy=False)
        else:
            counts = np.count_nonzero(np.any(np.abs(tokens) > 1e-12, axis=2), axis=1).astype(np.int16)
    return tokens, labels, smiles, counts


def adaptive_batch_size(q: int) -> int:
    if q >= 22:
        return 1
    if q >= 20:
        return 2
    if q >= 18:
        return 4
    if q >= 16:
        return 8
    if q >= 13:
        return 16
    if q >= 9:
        return 32
    return 64


def output_dim(q: int) -> int:
    return int(q + q * (q - 1) // 2)


def _qrc_tensor_worker(job: dict) -> list[str]:
    """One GPU owns a disjoint set of reservoir seeds."""
    with np.load(job["tensor_path"], allow_pickle=False) as package:
        token_tensor = package["tokens"].astype(np.float32)
        variant_names = package["variant_names"].astype(str)
    if token_tensor.ndim != 4:
        raise ValueError(f"Expected [variant,N,T,3Q], got {token_tensor.shape}")
    v, n, _, _ = token_tensor.shape
    flat = torch.from_numpy(np.ascontiguousarray(token_tensor.reshape(v * n, *token_tensor.shape[2:])))
    q = int(job["q"])
    cfg = DTCQRCConfig(
        num_qubits=q,
        features_per_qubit=3,
        g=float(job["g"]),
        W_scale=float(job["W"]),
        layers_per_injection=int(job["layers"]),
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for raw_seed in job["seeds"]:
        seed = int(raw_seed)
        output = output_dir / f"qrc_seed{seed:03d}.npz"
        if output.exists() and not job["force"]:
            try:
                with np.load(output, allow_pickle=False) as old:
                    if old["features"].shape == (v, n, cfg.output_dim) and np.array_equal(
                        old["variant_names"].astype(str), variant_names
                    ):
                        completed.append(str(output))
                        continue
            except Exception:
                pass
        features, params = compute_qrc_features_gpu(
            flat,
            cfg=cfg,
            seed=seed,
            batch_size=int(job["batch_size"]),
            device=str(job["device"]),
            progress=True,
        )
        np.savez_compressed(
            output,
            features=features.numpy().reshape(v, n, cfg.output_dim).astype(np.float32),
            variant_names=variant_names,
            J=params["J"],
            h=params["h"],
            seed=np.int32(seed),
            q=np.int16(q),
            t=np.int16(job["t"]),
            g=np.float32(job["g"]),
            W=np.float32(job["W"]),
            layers=np.int16(job["layers"]),
        )
        completed.append(str(output))
    return completed


def _qrc_padding_worker(job: dict) -> list[str]:
    """Extract typed-pad, zero-pad, and true variable-length masked features."""
    with np.load(job["tensor_path"], allow_pickle=False) as package:
        typed = package["typed_tokens"].astype(np.float32)
        zero = package["zero_tokens"].astype(np.float32)
        counts = package["event_counts"].astype(np.int16)
    q, t = int(job["q"]), int(job["t"])
    cfg = DTCQRCConfig(
        num_qubits=q,
        features_per_qubit=3,
        g=float(job["g"]),
        W_scale=float(job["W"]),
        layers_per_injection=int(job["layers"]),
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    names = np.asarray(["typed_pad", "zero_identity_pad", "length_masked"])
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for raw_seed in job["seeds"]:
        seed = int(raw_seed)
        output = output_dir / f"qrc_seed{seed:03d}.npz"
        if output.exists() and not job["force"]:
            try:
                with np.load(output, allow_pickle=False) as old:
                    if old["features"].shape == (3, len(typed), cfg.output_dim):
                        completed.append(str(output))
                        continue
            except Exception:
                pass
        joined = torch.from_numpy(np.concatenate([typed, zero], axis=0))
        first, params = compute_qrc_features_gpu(
            joined,
            cfg=cfg,
            seed=seed,
            batch_size=int(job["batch_size"]),
            device=str(job["device"]),
            progress=True,
        )
        first = first.numpy().reshape(2, len(typed), cfg.output_dim)
        masked = np.zeros((len(typed), cfg.output_dim), dtype=np.float32)
        clipped = np.clip(counts.astype(int), 1, t)
        for length in sorted(np.unique(clipped)):
            indices = np.flatnonzero(clipped == length)
            group, _ = compute_qrc_features_gpu(
                torch.from_numpy(np.ascontiguousarray(typed[indices, :length])),
                cfg=cfg,
                seed=seed,
                batch_size=int(job["batch_size"]),
                device=str(job["device"]),
                progress=False,
            )
            masked[indices] = group.numpy()
        features = np.concatenate([first, masked[None, ...]], axis=0).astype(np.float32)
        np.savez_compressed(
            output,
            features=features,
            variant_names=names,
            J=params["J"],
            h=params["h"],
            seed=np.int32(seed),
            q=np.int16(q),
            t=np.int16(t),
            g=np.float32(job["g"]),
            W=np.float32(job["W"]),
            layers=np.int16(job["layers"]),
            masking=np.asarray("grouped exact variable-length evolution"),
        )
        completed.append(str(output))
    return completed


def _qrc_order_worker(job: dict) -> list[str]:
    with np.load(job["base_path"], allow_pickle=False) as package:
        tokens = package["tokens"].astype(np.float32)
        counts = package["event_counts"].astype(np.int16)
    q, t = int(job["q"]), int(job["t"])
    cfg = DTCQRCConfig(
        num_qubits=q,
        features_per_qubit=3,
        g=float(job["g"]),
        W_scale=float(job["W"]),
        layers_per_injection=int(job["layers"]),
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for raw_seed, raw_permutation_seed in zip(job["seeds"], job["permutation_seeds"]):
        seed = int(raw_seed)
        permutation_seed = int(raw_permutation_seed)
        output = output_dir / f"qrc_seed{seed:03d}.npz"
        ordered, names = build_order_tensor(
            tokens,
            counts,
            permutation_seed=permutation_seed,
            q=q,
            t=t,
            block_size=int(job["block_size"]),
        )
        if output.exists() and not job["force"]:
            try:
                with np.load(output, allow_pickle=False) as old:
                    if (
                        old["features"].shape == (len(names), len(tokens), cfg.output_dim)
                        and int(old["permutation_seed"]) == permutation_seed
                    ):
                        completed.append(str(output))
                        continue
            except Exception:
                pass
        flat = torch.from_numpy(
            np.ascontiguousarray(ordered.reshape(len(names) * len(tokens), *ordered.shape[2:]))
        )
        features, params = compute_qrc_features_gpu(
            flat,
            cfg=cfg,
            seed=seed,
            batch_size=int(job["batch_size"]),
            device=str(job["device"]),
            progress=True,
        )
        np.savez_compressed(
            output,
            features=features.numpy().reshape(len(names), len(tokens), cfg.output_dim).astype(np.float32),
            variant_names=np.asarray(names),
            J=params["J"],
            h=params["h"],
            seed=np.int32(seed),
            permutation_seed=np.int64(permutation_seed),
            q=np.int16(q),
            t=np.int16(t),
            g=np.float32(job["g"]),
            W=np.float32(job["W"]),
            layers=np.int16(job["layers"]),
            block_size=np.int16(job["block_size"]),
        )
        completed.append(str(output))
    return completed


def extract_qrc_dual_gpu(
    tensor_path: Path,
    output_dir: Path,
    *,
    q: int,
    t: int,
    seeds: Sequence[int],
    devices: Sequence[str],
    g: float,
    W: float,
    layers: int,
    force: bool,
    padding_mode: bool = False,
    batch_size: int | None = None,
) -> None:
    if not devices:
        raise ValueError("At least one QRC device is required")
    assignments = [list(seeds[index :: len(devices)]) for index in range(len(devices))]
    jobs = [
        {
            "device": device,
            "seeds": assigned,
            "tensor_path": str(tensor_path),
            "output_dir": str(output_dir),
            "q": q,
            "t": t,
            "g": g,
            "W": W,
            "layers": layers,
            "batch_size": batch_size or adaptive_batch_size(q),
            "force": force,
        }
        for device, assigned in zip(devices, assignments)
        if assigned
    ]
    worker = _qrc_padding_worker if padding_mode else _qrc_tensor_worker
    if len(jobs) == 1:
        worker(jobs[0])
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


def extract_qrc_order_dual_gpu(
    base_path: Path,
    output_dir: Path,
    *,
    q: int,
    t: int,
    seeds: Sequence[int],
    permutation_seeds: Sequence[int],
    devices: Sequence[str],
    g: float,
    W: float,
    layers: int,
    block_size: int,
    force: bool,
    batch_size: int | None = None,
) -> None:
    if len(seeds) != len(permutation_seeds):
        raise ValueError("Each reservoir/split seed must have one separate permutation seed")
    if not devices:
        raise ValueError("At least one QRC device is required")
    jobs = []
    for device_index, device in enumerate(devices):
        indices = list(range(device_index, len(seeds), len(devices)))
        if not indices:
            continue
        jobs.append(
            {
                "device": device,
                "seeds": [int(seeds[index]) for index in indices],
                "permutation_seeds": [int(permutation_seeds[index]) for index in indices],
                "base_path": str(base_path),
                "output_dir": str(output_dir),
                "q": q,
                "t": t,
                "g": g,
                "W": W,
                "layers": layers,
                "block_size": block_size,
                "batch_size": batch_size or adaptive_batch_size(q),
                "force": force,
            }
        )
    if len(jobs) == 1:
        _qrc_order_worker(jobs[0])
        return
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(jobs), mp_context=context) as pool:
        futures = [pool.submit(_qrc_order_worker, job) for job in jobs]
        for future in as_completed(futures):
            future.result()


def compute_crc_tensor(
    token_tensor: np.ndarray,
    *,
    q: int,
    layers: int,
    seed: int,
    device: str,
    state_scale: float,
    inject_scale: float,
) -> np.ndarray:
    v, n = token_tensor.shape[:2]
    flat = np.ascontiguousarray(token_tensor.reshape(v * n, *token_tensor.shape[2:]), dtype=np.float32)
    features, _ = compute_crc_features(
        flat,
        q=q,
        layers=layers,
        seed=seed,
        state_scale=state_scale,
        inject_scale=inject_scale,
        device=device,
    )
    return features.reshape(v, n, -1).astype(np.float32)


def compute_crc3_z_zz_tensor(
    token_tensor: np.ndarray,
    *,
    q: int,
    layers: int,
    seed: int,
    device: str = "auto",
) -> np.ndarray:
    """Map every token variant through one paired Q-by-3 CRC3 reservoir."""
    v, n = token_tensor.shape[:2]
    flat = np.ascontiguousarray(
        token_tensor.reshape(v * n, *token_tensor.shape[2:]), dtype=np.float32
    )
    features, _ = compute_crc3_z_zz_features(
        flat, q=q, layers=layers, seed=seed, device=device
    )
    return features.reshape(v, n, crc3_z_zz_dim(q)).astype(np.float32)


def compute_crc_padding(
    typed: np.ndarray,
    zero: np.ndarray,
    counts: np.ndarray,
    *,
    q: int,
    t: int,
    layers: int,
    seed: int,
    device: str,
    state_scale: float,
    inject_scale: float,
) -> np.ndarray:
    pair = compute_crc_tensor(
        np.stack([typed, zero]),
        q=q,
        layers=layers,
        seed=seed,
        device=device,
        state_scale=state_scale,
        inject_scale=inject_scale,
    )
    masked = np.zeros((len(typed), output_dim(q)), dtype=np.float32)
    clipped = np.clip(counts.astype(int), 1, t)
    for length in sorted(np.unique(clipped)):
        indices = np.flatnonzero(clipped == length)
        features, _ = compute_crc_features(
            np.ascontiguousarray(typed[indices, :length]),
            q=q,
            layers=layers,
            seed=seed,
            state_scale=state_scale,
            inject_scale=inject_scale,
            device=device,
        )
        masked[indices] = features
    return np.concatenate([pair, masked[None, ...]], axis=0)


def compute_crc3_z_zz_padding(
    typed: np.ndarray,
    zero: np.ndarray,
    counts: np.ndarray,
    *,
    q: int,
    t: int,
    layers: int,
    seed: int,
    device: str = "auto",
) -> np.ndarray:
    pair = compute_crc3_z_zz_tensor(
        np.stack([typed, zero]), q=q, layers=layers, seed=seed, device=device
    )
    masked = np.zeros((len(typed), crc3_z_zz_dim(q)), dtype=np.float32)
    clipped = np.clip(counts.astype(int), 1, t)
    for length in sorted(np.unique(clipped)):
        indices = np.flatnonzero(clipped == length)
        features, _ = compute_crc3_z_zz_features(
            np.ascontiguousarray(typed[indices, :length]),
            q=q,
            layers=layers,
            seed=seed,
            device=device,
        )
        masked[indices] = features
    return np.concatenate([pair, masked[None, ...]], axis=0)


def append_ensemble_variants(
    features: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    names = list(names)
    arrays = [features]
    appended = []
    for prefix, ensemble_name in (
        ("atom_relabel_r", "atom_relabel_ensemble"),
        ("random_smiles_r", "random_smiles_ensemble"),
        ("root_r", "multi_root_ensemble"),
    ):
        indices = [index for index, name in enumerate(names) if name.startswith(prefix)]
        if indices:
            arrays.append(features[indices].mean(axis=0, keepdims=True))
            appended.append(ensemble_name)
    traversal_indices = [
        index
        for index, name in enumerate(names)
        if name.startswith(("atom_relabel_r", "random_smiles_r", "root_r"))
    ]
    if traversal_indices:
        arrays.append(features[traversal_indices].mean(axis=0, keepdims=True))
        appended.append("traversal_ensemble")
    return np.concatenate(arrays, axis=0), names + appended


def _safe_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    numerator = np.sum(x * y, axis=1)
    denominator = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - x.mean(axis=0, keepdims=True)
    y = y.astype(np.float64) - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    return float(np.sum(cross * cross) / np.sqrt(max(np.sum(xx * xx) * np.sum(yy * yy), 1e-24)))


def representation_robustness_rows(
    features: np.ndarray,
    names: Sequence[str],
    *,
    method: str,
    seed: int,
    reference: str,
) -> list[dict]:
    names = list(names)
    ref = features[names.index(reference)].reshape(len(features[0]), -1)
    rows = []
    for index, name in enumerate(names):
        current = features[index].reshape(len(ref), -1)
        delta = current - ref
        relative = np.linalg.norm(delta, axis=1) / np.maximum(np.linalg.norm(ref, axis=1), 1e-12)
        cosine = _safe_cosine(current, ref)
        exact = np.all(current == ref, axis=1)
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "reference": reference,
                "variant": name,
                "feature_dim": int(current.shape[1]),
                "exact_match_fraction": float(exact.mean()),
                "cosine_mean": float(cosine.mean()),
                "cosine_std": float(cosine.std(ddof=1)),
                "relative_l2_mean": float(relative.mean()),
                "relative_l2_std": float(relative.std(ddof=1)),
                "linear_cka": linear_cka(ref, current),
            }
        )
    return rows


def split_payload(
    smiles: Sequence[str],
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict:
    train, val, test = scaffold_split_indices(smiles, train_ratio, val_ratio, seed)
    payload = {
        "seed": int(seed),
        "split_type": "stable_scaffold_seeded_tie_shuffle",
        "train_indices": train.tolist(),
        "val_indices": val.tolist(),
        "test_indices": test.tolist(),
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "class_counts": {
            name: {str(c): int((labels[idx] == c).sum()) for c in (0, 1)}
            for name, idx in (("train", train), ("val", val), ("test", test))
        },
    }
    for name, idx in (("validation", val), ("test", test)):
        if len(np.unique(labels[idx])) < 2:
            raise RuntimeError(f"Seed {seed} produced one-class {name} split")
    return payload


def metric_bundle(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else float("nan"),
        "acc": float(accuracy_score(y, predictions)),
    }


def evaluate_frozen_head(
    features_by_method: Mapping[str, np.ndarray],
    variant_names: Sequence[str],
    labels: np.ndarray,
    smiles: np.ndarray,
    *,
    seed: int,
    output_dir: Path,
    linear_cfg: ExperimentConfig,
    q: int,
    t: int,
    train_variant: str,
    experiment: str,
    permutation_seed: int | None = None,
) -> pd.DataFrame:
    """Train once on ``train_variant`` and evaluate every variant without refitting."""
    variant_names = list(variant_names)
    train_index = variant_names.index(train_variant)
    split = split_payload(
        smiles,
        labels,
        seed=seed,
        train_ratio=linear_cfg.train_ratio,
        val_ratio=linear_cfg.val_ratio,
    )
    json_dump(output_dir / "splits" / f"seed{seed:03d}.json", split)
    test_idx = np.asarray(split["test_indices"], dtype=np.int64)
    rows = []
    prediction_frames = []
    prediction_dir = output_dir / "frozen_probe" / f"seed{seed:03d}"
    combined_metrics_path = prediction_dir / "cross_variant_metrics.csv"
    combined_predictions_path = prediction_dir / "cross_variant_predictions.csv"
    existing_metrics = (
        pd.read_csv(combined_metrics_path)
        if combined_metrics_path.exists() and not linear_cfg.force
        else pd.DataFrame()
    )
    existing_predictions = (
        pd.read_csv(combined_predictions_path)
        if combined_predictions_path.exists() and not linear_cfg.force
        else pd.DataFrame()
    )
    for method, feature_tensor in features_by_method.items():
        method_dir = output_dir / "frozen_probe" / f"seed{seed:03d}" / method
        checkpoint_path = method_dir / "checkpoints" / f"{method}_seed{seed:03d}_best.pt"
        cached_rows = (
            existing_metrics[existing_metrics.method == method]
            if "method" in existing_metrics
            else pd.DataFrame()
        )
        cached_predictions = (
            existing_predictions[existing_predictions.method == method]
            if "method" in existing_predictions
            else pd.DataFrame()
        )
        expected_variants = set(variant_names)
        cache_ok = (
            not linear_cfg.force
            and checkpoint_path.exists()
            and (method_dir / "training_history.csv").exists()
            and len(cached_rows) == len(variant_names)
            and set(cached_rows.test_variant.astype(str)) == expected_variants
            and set(cached_predictions.test_variant.astype(str)) == expected_variants
        )
        if cache_ok:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                cache_ok = int(checkpoint["input_dim"]) == int(feature_tensor.shape[-1])
            except Exception:
                cache_ok = False
        if cache_ok:
            rows.extend(cached_rows.to_dict(orient="records"))
            prediction_frames.append(cached_predictions)
            continue
        row, history, _ = train_linear_probe(
            feature_tensor[train_index],
            labels,
            split,
            method,
            seed,
            method_dir,
            linear_cfg,
            q,
            t,
        )
        history.to_csv(method_dir / "training_history.csv", index=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weight = checkpoint["state_dict"]["weight"].numpy().reshape(-1)
        bias = float(checkpoint["state_dict"]["bias"].numpy().reshape(-1)[0])
        for variant_index, variant in enumerate(variant_names):
            current = feature_tensor[variant_index]
            logits = current[test_idx] @ weight + bias
            metrics = metric_bundle(labels[test_idx], logits)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "seed": int(seed),
                    "permutation_seed": permutation_seed,
                    "q": int(q),
                    "t": int(t),
                    "train_variant": train_variant,
                    "test_variant": variant,
                    "best_epoch": int(row["best_epoch"]),
                    "selection_rule": SELECTION_RULE,
                    "val_auc_original": float(row["val_auc"]),
                    "val_loss_original": float(row["val_loss"]),
                    "val_acc_original": float(row["val_acc"]),
                    "test_auc": metrics["auc"],
                    "test_loss": metrics["loss"],
                    "test_acc": metrics["acc"],
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "experiment": experiment,
                        "method": method,
                        "seed": int(seed),
                        "permutation_seed": permutation_seed,
                        "train_variant": train_variant,
                        "test_variant": variant,
                        "sample_index": test_idx,
                        "y_true": labels[test_idx],
                        "probability": probabilities,
                        "prediction": (probabilities >= 0.5).astype(int),
                    }
                )
            )
    metrics = pd.DataFrame(rows)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(prediction_dir / "cross_variant_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        prediction_dir / "cross_variant_predictions.csv", index=False
    )
    return metrics


def summarize_frozen_metrics(table: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    group_columns = ["experiment", "method", "train_variant", "test_variant"]
    for keys, subset in table.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(subset.seed.nunique())
        for metric in ("test_auc", "test_loss", "test_acc"):
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_std"] = float(subset[metric].std(ddof=1))
        rows.append(row)
    summary = pd.DataFrame(rows)
    original = table[table.test_variant == table.train_variant].set_index(["method", "seed"])
    effects = []
    for _, row in table.iterrows():
        key = (row.method, int(row.seed))
        if key not in original.index:
            continue
        ref = original.loc[key]
        record = row.to_dict()
        record["delta_test_auc_vs_original"] = float(row.test_auc - ref.test_auc)
        record["delta_test_acc_vs_original"] = float(row.test_acc - ref.test_acc)
        effects.append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "frozen_cross_variant_metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "frozen_cross_variant_summary.csv", index=False)
    pd.DataFrame(effects).to_csv(output_dir / "frozen_cross_variant_effects.csv", index=False)
    return summary


def evaluate_ensemble_augmented_head(
    features_by_method: Mapping[str, np.ndarray],
    variant_names: Sequence[str],
    labels: np.ndarray,
    smiles: np.ndarray,
    *,
    seed: int,
    output_dir: Path,
    linear_cfg: ExperimentConfig,
    q: int,
    t: int,
) -> pd.DataFrame:
    """Traversal augmentation on train; ensemble aggregation on val/test.

    Each individual atom-relabel/random-SMILES/root realization contributes a
    labelled training representation. Validation and test each use only the
    post-map mean ``traversal_ensemble`` representation, so model selection does
    not favor one arbitrary serialization.
    """
    names = list(variant_names)
    member_indices = [
        index
        for index, name in enumerate(names)
        if name.startswith(("atom_relabel_r", "random_smiles_r", "root_r"))
    ]
    if "traversal_ensemble" not in names or not member_indices:
        raise ValueError("Traversal ensemble and its members are required")
    ensemble_index = names.index("traversal_ensemble")
    base_split = split_payload(
        smiles,
        labels,
        seed=seed,
        train_ratio=linear_cfg.train_ratio,
        val_ratio=linear_cfg.val_ratio,
    )
    n = len(labels)
    train_base = np.asarray(base_split["train_indices"], dtype=int)
    val_base = np.asarray(base_split["val_indices"], dtype=int)
    test_base = np.asarray(base_split["test_indices"], dtype=int)
    rows = []
    combined_path = output_dir / "ensemble_augmented_probe" / f"seed{seed:03d}_metrics.csv"
    existing = (
        pd.read_csv(combined_path)
        if combined_path.exists() and not linear_cfg.force
        else pd.DataFrame()
    )
    for method, tensor in features_by_method.items():
        member_arrays = [tensor[index] for index in member_indices]
        assembled = np.concatenate(member_arrays + [tensor[ensemble_index]], axis=0)
        assembled_labels = np.tile(labels, len(member_arrays) + 1)
        ensemble_offset = len(member_arrays) * n
        split = {
            "seed": int(seed),
            "split_type": "scaffold split; traversal-augmented train; traversal-ensemble validation/test",
            "train_indices": np.concatenate(
                [member * n + train_base for member in range(len(member_arrays))]
            ).tolist(),
            "val_indices": (ensemble_offset + val_base).tolist(),
            "test_indices": (ensemble_offset + test_base).tolist(),
        }
        method_name = f"{method}_traversal_aug"
        setting_dir = output_dir / "ensemble_augmented_probe" / f"seed{seed:03d}" / method
        cached = (
            existing[existing.base_method == method]
            if "base_method" in existing
            else pd.DataFrame()
        )
        checkpoint_path = setting_dir / "checkpoints" / f"{method_name}_seed{seed:03d}_best.pt"
        cache_ok = (
            not linear_cfg.force
            and len(cached) == 1
            and checkpoint_path.exists()
            and (setting_dir / "training_history.csv").exists()
            and (setting_dir / "test_predictions.csv").exists()
        )
        if cache_ok:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                cache_ok = int(checkpoint["input_dim"]) == int(assembled.shape[1])
            except Exception:
                cache_ok = False
        if cache_ok:
            rows.extend(cached.to_dict(orient="records"))
            continue
        row, history, predictions = train_linear_probe(
            assembled,
            assembled_labels,
            split,
            method_name,
            seed,
            setting_dir,
            linear_cfg,
            q,
            t,
        )
        history.to_csv(setting_dir / "training_history.csv", index=False)
        predictions["source_sample_index"] = predictions.sample_index.astype(int) - ensemble_offset
        predictions.to_csv(setting_dir / "test_predictions.csv", index=False)
        rows.append(
            {
                **row,
                "base_method": method,
                "training_protocol": "individual traversal augmentation",
                "validation_test_protocol": "post-map traversal ensemble",
                "n_train_traversals": len(member_arrays),
            }
        )
    table = pd.DataFrame(rows)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(combined_path, index=False)
    return table
