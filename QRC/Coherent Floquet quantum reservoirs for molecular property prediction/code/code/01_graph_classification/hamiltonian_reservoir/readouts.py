"""Paired linear and RBF-SVR decoders for frozen reservoir features."""

from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, pairwise_distances, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from protocol import (
    LINEAR_BATCH_SIZE,
    LINEAR_EPOCHS,
    LINEAR_LEARNING_RATE,
    SVR_C_VALUES,
    SVR_EPSILON,
    SVR_GAMMA_SCALES,
)


def split_indices(split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.asarray(split[key], dtype=np.int64)
        for key in ("train_indices", "val_indices", "test_indices")
    )


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return {
        "auc": float(roc_auc_score(y, p)),
        "loss": float(log_loss(y, p, labels=[0, 1])),
        "acc": float(accuracy_score(y, p >= 0.5)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    raw = np.asarray(score, dtype=np.float64)
    bounded = np.clip(raw, 0.0, 1.0)
    return {
        "auc": float(roc_auc_score(y, raw)),
        "loss": float(log_loss(y, bounded, labels=[0, 1])),
        "acc": float(accuracy_score(y, bounded >= 0.5)),
        "brier": float(np.mean((bounded - y) ** 2)),
        "rmse": float(np.sqrt(np.mean((bounded - y) ** 2))),
    }


def _rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.arange(1, len(score) + 1, dtype=np.float64)
    cuts = np.concatenate(
        ([0], np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1, [len(score)])
    )
    for start, stop in zip(cuts[:-1], cuts[1:]):
        if stop - start > 1:
            ranks[start:stop] = 0.5 * (start + 1 + stop)
    positive = y[order] == 1
    n_positive = int(positive.sum())
    n_negative = len(y) - n_positive
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def probability_metric_bank(
    y_true: np.ndarray, probability: np.ndarray
) -> list[dict[str, float]]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if p.ndim != 2 or p.shape[1] != len(y):
        raise ValueError((p.shape, len(y)))
    targets = y[None]
    losses = -np.mean(targets * np.log(p) + (1 - targets) * np.log1p(-p), axis=1)
    accuracies = np.mean((p >= 0.5) == targets, axis=1)
    briers = np.mean((p - targets) ** 2, axis=1)
    return [
        {
            "auc": _rank_auc(y, p[index]),
            "loss": float(losses[index]),
            "acc": float(accuracies[index]),
            "brier": float(briers[index]),
        }
        for index in range(len(p))
    ]


def fit_linear_bank(
    feature_bank: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
    device: str,
    row_values: tuple[float, ...] | None = None,
    epochs: int = LINEAR_EPOCHS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Train independent affine heads in one vectorized Adam computation."""
    values = np.asarray(feature_bank, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(values.shape)
    labels = np.asarray(labels, dtype=np.int64)
    if values.shape[1] != len(labels):
        raise ValueError((values.shape, labels.shape))
    count, _, width = values.shape
    identifiers = row_values or tuple(float(index) for index in range(count))
    if len(identifiers) != count:
        raise ValueError((len(identifiers), count))

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"unavailable decoder device {device}")
    x = torch.from_numpy(values).to(compute_device)
    y = torch.from_numpy(labels.astype(np.float32)).to(compute_device)
    train_np, val_np, test_np = split_indices(split)
    train = torch.from_numpy(train_np).to(compute_device)
    val = torch.from_numpy(val_np).to(compute_device)
    test = torch.from_numpy(test_np).to(compute_device)

    reference = torch.nn.Linear(width, 1, bias=True).to(compute_device)
    weight = torch.nn.Parameter(reference.weight.detach().repeat(count, 1))
    bias = torch.nn.Parameter(reference.bias.detach().repeat(count))
    del reference
    optimizer = torch.optim.Adam(
        (weight, bias), lr=LINEAR_LEARNING_RATE, weight_decay=0.0
    )
    generator = torch.Generator(device=compute_device).manual_seed(int(seed))
    best_keys = [None] * count
    best_epochs = [-1] * count
    best_validation = [None] * count
    best_weights = [None] * count
    best_biases = [None] * count

    for epoch in range(1, int(epochs) + 1):
        order = train[
            torch.randperm(len(train), generator=generator, device=compute_device)
        ]
        for start in range(0, len(order), LINEAR_BATCH_SIZE):
            batch = order[start : start + LINEAR_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = torch.einsum("pbd,pd->pb", x[:, batch], weight) + bias[:, None]
            targets = y[batch][None].expand(count, -1)
            losses = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none"
            ).mean(dim=1)
            losses.sum().backward()
            optimizer.step()
        with torch.no_grad():
            val_probability = torch.sigmoid(
                torch.einsum("pbd,pd->pb", x[:, val], weight) + bias[:, None]
            ).cpu().numpy()
        metrics_bank = probability_metric_bank(labels[val_np], val_probability)
        for index, metrics in enumerate(metrics_bank):
            key = (metrics["auc"], -metrics["loss"], metrics["acc"])
            if best_keys[index] is None or key > best_keys[index]:
                best_keys[index] = key
                best_epochs[index] = epoch
                best_validation[index] = metrics
                best_weights[index] = weight[index].detach().cpu().clone()
                best_biases[index] = bias[index].detach().cpu().clone()

    test_probabilities = []
    rows = []
    for index, identifier in enumerate(identifiers):
        selected_weight = best_weights[index].to(compute_device)
        selected_bias = best_biases[index].to(compute_device)
        with torch.no_grad():
            probability = torch.sigmoid(
                x[index, test] @ selected_weight + selected_bias
            ).cpu().numpy()
        test_probabilities.append(probability)
        test_metrics = probability_metrics(labels[test_np], probability)
        rows.append(
            {
                "row_value": float(identifier),
                "best_epoch": int(best_epochs[index]),
                **{
                    f"val_{name}": value
                    for name, value in best_validation[index].items()
                },
                **{f"test_{name}": value for name, value in test_metrics.items()},
            }
        )
    artifacts = {
        "weights": np.stack([value.numpy() for value in best_weights]),
        "biases": np.asarray([value.item() for value in best_biases], dtype=np.float32),
        "test_indices": test_np,
        "test_probabilities": np.stack(test_probabilities).astype(np.float32),
    }
    return pd.DataFrame(rows), artifacts


def _base_gamma(x_train: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    count = min(512, len(x_train))
    selected = rng.choice(len(x_train), size=count, replace=False)
    distances = pairwise_distances(x_train[selected], metric="sqeuclidean")
    upper = distances[np.triu_indices_from(distances, k=1)]
    nonzero = upper[upper > 1e-12]
    median = float(np.median(nonzero)) if len(nonzero) else 1.0
    return 1.0 / max(median, 1e-12)


def fit_svr(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
) -> tuple[dict, dict]:
    train, val, test = split_indices(split)
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_val = scaler.transform(features[val])
    x_test = scaler.transform(features[test])
    base_gamma = _base_gamma(x_train, seed)
    screen = []
    best_key = None
    best_model = None
    best_row = None
    for c_value in SVR_C_VALUES:
        for gamma_scale in SVR_GAMMA_SCALES:
            gamma = float(gamma_scale * base_gamma)
            model = SVR(
                kernel="rbf", C=float(c_value), gamma=gamma, epsilon=SVR_EPSILON
            )
            model.fit(x_train, labels[train])
            metrics = score_metrics(labels[val], model.predict(x_val))
            row = {
                "C": float(c_value),
                "gamma_scale": float(gamma_scale),
                "gamma": gamma,
                "epsilon": SVR_EPSILON,
                **{f"val_{name}": value for name, value in metrics.items()},
            }
            screen.append(row)
            # Decoder hyperparameters are the only quantities selected here.
            # Follow the registered hierarchy: validation AUC, then log loss,
            # then accuracy.  The final two entries make exact metric ties
            # deterministic without consulting test data.
            key = (
                metrics["auc"], -metrics["loss"], metrics["acc"],
                -float(c_value), -float(gamma_scale),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_model = copy.deepcopy(model)
                best_row = row
    if best_model is None or best_row is None:
        raise RuntimeError("no valid SVR candidate")
    test_score = best_model.predict(x_test)
    test_metrics = score_metrics(labels[test], test_score)
    result = {
        **best_row,
        **{f"test_{name}": value for name, value in test_metrics.items()},
    }
    artifacts = {
        "screen": pd.DataFrame(screen),
        "scaler": scaler,
        "model": best_model,
        "test_indices": test,
        "test_scores": test_score,
    }
    return result, artifacts
