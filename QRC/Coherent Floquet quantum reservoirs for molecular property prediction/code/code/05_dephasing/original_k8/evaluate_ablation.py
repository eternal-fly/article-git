"""RBF-SVR evaluation for the frozen mechanism feature banks."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, pairwise_distances, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mechanism_features import connected_zz_diagnostics
from protocol import (
    BACE_Q,
    BACE_SEEDS,
    BACE_SVR_C_VALUES,
    BACE_SVR_EPSILON,
    BACE_SVR_GAMMA_SCALES,
    BACE_T,
    DECODER,
    HOPPING_Q,
    HOPPING_ROOT,
    HOPPING_SEEDS,
    HOPPING_T,
    MAIN_ROOT,
    P_VALUES,
    PROTOCOL_VERSION,
    ROOT,
    VARIANTS,
    bace_split_path,
    bace_token_path,
    coherent_bace_path,
    coherent_hopping_path,
    feature_path,
)


if str(HOPPING_ROOT) not in sys.path:
    sys.path.insert(0, str(HOPPING_ROOT))

from a01_core import (  # noqa: E402
    TARGET_NAMES,
    _best_svr,
    _refit_svr,
    load_config,
    regression_metrics,
)


RESULTS = ROOT / "results"


def _score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    raw = np.asarray(score, dtype=np.float64)
    bounded = np.clip(raw, 0.0, 1.0)
    return {
        "auc": float(roc_auc_score(y, raw)),
        "acc": float(accuracy_score(y, bounded >= 0.5)),
        "brier": float(np.mean((bounded - y) ** 2)),
        "rmse": float(np.sqrt(np.mean((bounded - y) ** 2))),
    }


def _base_gamma(x_train: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    count = min(512, len(x_train))
    selected = rng.choice(len(x_train), size=count, replace=False)
    distances = pairwise_distances(x_train[selected], metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    nonzero = upper[upper > 1e-12]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    return 1.0 / max(median, 1e-12)


def fit_bace_svr(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
) -> tuple[dict, dict]:
    train, validation, test = (
        np.asarray(split[key], dtype=np.int64)
        for key in ("train_indices", "val_indices", "test_indices")
    )
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_validation = scaler.transform(features[validation])
    x_test = scaler.transform(features[test])
    base_gamma = _base_gamma(x_train, seed)
    screen, best_key, best_model, best_row = [], None, None, None
    for c_value in BACE_SVR_C_VALUES:
        for gamma_scale in BACE_SVR_GAMMA_SCALES:
            gamma = float(gamma_scale * base_gamma)
            model = SVR(
                kernel="rbf",
                C=float(c_value),
                gamma=gamma,
                epsilon=BACE_SVR_EPSILON,
            ).fit(x_train, labels[train])
            metrics = _score_metrics(labels[validation], model.predict(x_validation))
            row = {
                "C": float(c_value),
                "gamma_scale": float(gamma_scale),
                "gamma": gamma,
                "epsilon": BACE_SVR_EPSILON,
                **{f"val_{name}": value for name, value in metrics.items()},
            }
            screen.append(row)
            key = (metrics["auc"], -metrics["brier"], -float(c_value), -float(gamma_scale))
            if best_key is None or key > best_key:
                best_key, best_model, best_row = key, copy.deepcopy(model), row
    if best_model is None or best_row is None:
        raise RuntimeError("no BACE SVR candidate")
    test_scores = best_model.predict(x_test)
    result = {
        **best_row,
        **{f"test_{name}": value for name, value in _score_metrics(labels[test], test_scores).items()},
    }
    return result, {
        "screen": pd.DataFrame(screen),
        "scaler": scaler,
        "model": best_model,
        "test_indices": test,
        "test_scores": test_scores,
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.csv")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def _load_bank(task: str, variant: str, seed: int) -> np.ndarray:
    if task == "bace":
        path = coherent_bace_path(seed) if variant == "coherent" else feature_path(task, variant, seed)
    elif task == "hopping":
        path = coherent_hopping_path(seed) if variant == "coherent" else feature_path(task, variant, seed)
    else:
        raise ValueError(task)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as package:
        bank = package["suffixes"].astype(np.float32)
    return bank


def mix_bace_bank(bank: np.ndarray, p: float) -> np.ndarray:
    source = np.asarray(bank, dtype=np.float64)
    length = len(source)
    survival = 1.0 - float(p)
    weights = [survival**length]
    weights.extend(float(p) * survival ** (length - start) for start in range(1, length))
    weights.append(float(p))
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise AssertionError(weights.sum())
    mixed = np.tensordot(weights[:-1], source, axes=(0, 0))
    mixed += weights[-1]  # the final reset anchor |0...0>
    return mixed.astype(np.float32)


def mix_hopping_bank(bank: np.ndarray, p: float, readout: str) -> np.ndarray:
    source = np.asarray(bank, dtype=np.float64)
    length = len(source)
    survival = 1.0 - float(p)
    weights = np.empty(length, dtype=np.float64)
    weights[0] = survival ** (length - 1)
    for start in range(1, length):
        weights[start] = float(p) * survival ** (length - 1 - start)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise AssertionError(weights.sum())
    mixed = np.tensordot(weights, source, axes=(0, 0)).astype(np.float32)
    width = HOPPING_Q * (HOPPING_Q + 1) // 2
    if readout == "final":
        return mixed[:, :width]
    if readout == "tm3":
        return mixed
    raise ValueError(readout)


def evaluate_bace(force: bool = False) -> tuple[Path, Path]:
    metrics_path = RESULTS / "bace_metrics_by_seed.csv"
    predictions_path = RESULTS / "bace_test_predictions.csv"
    if metrics_path.exists() and predictions_path.exists() and not force:
        return metrics_path, predictions_path
    with np.load(bace_token_path(), allow_pickle=False) as package:
        labels = package["labels"].astype(np.int64)
    metric_rows, prediction_rows = [], []
    for variant in VARIANTS:
        for seed in BACE_SEEDS:
            split = json.loads(bace_split_path(seed).read_text(encoding="utf-8"))
            bank = _load_bank("bace", variant.name, seed)
            if bank.shape != (BACE_T, len(labels), BACE_Q * (BACE_Q + 1) // 2):
                raise ValueError((variant.name, seed, bank.shape))
            for p in P_VALUES:
                features = mix_bace_bank(bank, p)
                metrics, artifacts = fit_bace_svr(features, labels, split, seed=seed)
                connected_rms, connected_max = connected_zz_diagnostics(features, BACE_Q)
                metric_rows.append(
                    {
                        "task": "BACE",
                        "variant": variant.name,
                        "j_scale": variant.j_scale,
                        "dephasing_strength": variant.dephasing_strength,
                        "dephasing_trajectories": variant.trajectories,
                        "seed": seed,
                        "q": BACE_Q,
                        "t": BACE_T,
                        "p": p,
                        "decoder": DECODER,
                        "readout": "final",
                        "connected_zz_rms": connected_rms,
                        "connected_zz_max_abs": connected_max,
                        **metrics,
                        "protocol_version": PROTOCOL_VERSION,
                    }
                )
                for sample_index, score in zip(artifacts["test_indices"], artifacts["test_scores"]):
                    prediction_rows.append(
                        {
                            "variant": variant.name,
                            "seed": seed,
                            "p": p,
                            "sample_index": int(sample_index),
                            "label": int(labels[sample_index]),
                            "score": float(score),
                        }
                    )
            print(f"[BACE evaluation] {variant.name} seed={seed}", flush=True)
    _atomic_csv(pd.DataFrame(metric_rows), metrics_path)
    _atomic_csv(pd.DataFrame(prediction_rows), predictions_path)
    return metrics_path, predictions_path


def evaluate_hopping(force: bool = False) -> tuple[Path, Path]:
    metrics_path = RESULTS / "hopping_metrics_by_seed.csv"
    predictions_path = RESULTS / "hopping_predictions_all_fixed_axes.csv"
    cfg = load_config()
    with np.load(cfg.processed_path, allow_pickle=False) as package:
        targets = package["targets"].astype(np.float64)
        split = package["split"].astype(np.int8)
        trajectory_id = package["trajectory_id"].astype(np.int32)
        endpoint = package["endpoint"].astype(np.int16)
    train, validation, test = split == 0, split == 1, split == 2
    test_indices = np.flatnonzero(test)
    decoder_cfg = cfg.payload["decoders"]
    candidates = [
        (float(C), float(gamma), float(epsilon))
        for C in decoder_cfg["svr_C_grid"]
        for gamma in decoder_cfg["svr_gamma_scale_grid"]
        for epsilon in decoder_cfg["svr_epsilon_grid"]
    ]
    cache_size = float(decoder_cfg["svr_cache_size_mb"])
    metric_keys = ["variant", "seed", "p", "readout", "target"]
    expected_metrics_per_seed = len(VARIANTS) * len(P_VALUES) * 2 * len(TARGET_NAMES)
    expected_predictions_per_seed = expected_metrics_per_seed * int(np.count_nonzero(test))
    if not force and metrics_path.exists() and predictions_path.exists():
        prior_metrics = pd.read_csv(metrics_path)
        prior_predictions = pd.read_csv(predictions_path)
    else:
        prior_metrics = pd.DataFrame()
        prior_predictions = pd.DataFrame()
    completed_seeds: set[int] = set()
    if not prior_metrics.empty and not prior_predictions.empty:
        for seed in HOPPING_SEEDS:
            current_metrics = prior_metrics[prior_metrics.seed == seed]
            current_predictions = prior_predictions[prior_predictions.seed == seed]
            if (
                len(current_metrics) == expected_metrics_per_seed
                and len(current_metrics[metric_keys].drop_duplicates()) == expected_metrics_per_seed
                and len(current_predictions) == expected_predictions_per_seed
                and set(current_metrics.protocol_version) == {PROTOCOL_VERSION}
            ):
                completed_seeds.add(int(seed))
        prior_metrics = prior_metrics[prior_metrics.seed.isin(completed_seeds)].copy()
        prior_predictions = prior_predictions[prior_predictions.seed.isin(completed_seeds)].copy()
    if completed_seeds == set(HOPPING_SEEDS):
        return metrics_path, predictions_path

    for seed in HOPPING_SEEDS:
        if seed in completed_seeds:
            print(f"[hopping evaluation] seed={seed} cached", flush=True)
            continue
        seed_metric_rows: list[dict] = []
        seed_prediction_rows: list[dict] = []
        for variant in VARIANTS:
            bank = _load_bank("hopping", variant.name, seed)
            expected_width = 3 * HOPPING_Q * (HOPPING_Q + 1) // 2
            if bank.shape != (HOPPING_T, len(targets), expected_width):
                raise ValueError((variant.name, seed, bank.shape))
            for p in P_VALUES:
                for readout in ("final", "tm3"):
                    features = mix_hopping_bank(bank, p, readout)
                    connected_rms, connected_max = connected_zz_diagnostics(features, HOPPING_Q)
                    for target_index, target_name in enumerate(TARGET_NAMES):
                        y = targets[:, target_index]
                        selected = _best_svr(
                            features[train], y[train],
                            features[validation], y[validation],
                            candidates, cache_size,
                        )
                        prediction = _refit_svr(
                            features[train], y[train],
                            features[validation], y[validation],
                            features[test], selected, cache_size,
                        )
                        row = {
                            "task": "SHNITSEL_A01",
                            "target": target_name,
                            "variant": variant.name,
                            "j_scale": variant.j_scale,
                            "dephasing_strength": variant.dephasing_strength,
                            "dephasing_trajectories": variant.trajectories,
                            "seed": seed,
                            "q": HOPPING_Q,
                            "t": HOPPING_T,
                            "p": p,
                            "decoder": DECODER,
                            "readout": readout,
                            "feature_dim": features.shape[1],
                            "connected_zz_rms": connected_rms,
                            "connected_zz_max_abs": connected_max,
                            **selected,
                            **regression_metrics(y[test], prediction),
                            "protocol_version": PROTOCOL_VERSION,
                        }
                        seed_metric_rows.append(row)
                        for local_index, sample_index in enumerate(test_indices):
                            seed_prediction_rows.append(
                            {
                                "target": target_name,
                                "variant": variant.name,
                                "seed": seed,
                                "p": p,
                                "decoder": DECODER,
                                "readout": readout,
                                "sample_index": int(sample_index),
                                "trajectory_id": int(trajectory_id[sample_index]),
                                "endpoint": int(endpoint[sample_index]),
                                "target_ev": float(targets[sample_index, target_index]),
                                "prediction_ev": float(prediction[local_index]),
                                "absolute_error_ev": float(
                                    abs(prediction[local_index] - targets[sample_index, target_index])
                                ),
                            }
                        )
            print(f"[hopping evaluation] {variant.name} seed={seed}", flush=True)
        prior_metrics = pd.concat(
            [prior_metrics, pd.DataFrame(seed_metric_rows)], ignore_index=True
        ).sort_values(metric_keys).reset_index(drop=True)
        prior_predictions = pd.concat(
            [prior_predictions, pd.DataFrame(seed_prediction_rows)], ignore_index=True
        ).sort_values(metric_keys + ["sample_index"]).reset_index(drop=True)
        _atomic_csv(prior_metrics, metrics_path)
        _atomic_csv(prior_predictions, predictions_path)
        print(f"[hopping evaluation] seed={seed} checkpointed", flush=True)
    return metrics_path, predictions_path


def evaluate_all(force: bool = False) -> dict[str, list[str]]:
    return {
        "bace": [str(path) for path in evaluate_bace(force)],
        "hopping": [str(path) for path in evaluate_hopping(force)],
    }
