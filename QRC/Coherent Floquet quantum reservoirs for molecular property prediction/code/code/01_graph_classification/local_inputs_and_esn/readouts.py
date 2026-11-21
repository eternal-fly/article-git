"""Validation-only p screening and manuscript affine readout."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from protocol import (
    LINEAR_BATCH_SIZE,
    LINEAR_EPOCHS,
    LINEAR_LR,
    P_VALUES,
    PROTOCOL_VERSION,
    RESULT_ROOT,
    SELECTION_RULE,
    split_path,
    token_cache_path,
)
from qrc_features import load_qrc_bank, mix_suffix_bank


P_SELECTION_PROTOCOL = "same_unstandardized_affine_500epoch_validation_scan_v1"


def _indices(split: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.asarray(split[f"{name}_indices"], dtype=np.int64)
        for name in ("train", "val", "test")
    )


def metric_bundle(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    prediction = (p >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, p)),
        "loss": float(log_loss(y, p, labels=[0, 1])),
        "acc": float(accuracy_score(y, prediction)),
        "brier": float(np.mean((p - y) ** 2)),
    }


def _binary_auc_from_ranks(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary ROC-AUC via the same average-rank definition as sklearn.

    This tiny implementation avoids constructing thousands of sklearn metric
    objects inside the 500-epoch, ten-p validation loop.  Equal scores receive
    their average rank, so the result also covers the (rare) tied-score case.
    """
    y = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.arange(1, len(score) + 1, dtype=np.float64)
    boundaries = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(sorted_score[1:] != sorted_score[:-1]) + 1,
            np.array([len(score)], dtype=np.int64),
        )
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start > 1:
            ranks[start:stop] = 0.5 * ((start + 1) + stop)
    positive = y[order] == 1
    n_positive = int(positive.sum())
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC-AUC requires both classes")
    u_statistic = float(ranks[positive].sum()) - n_positive * (n_positive + 1) / 2.0
    return u_statistic / float(n_positive * n_negative)


def metric_bundles(
    y_true: np.ndarray, probabilities: np.ndarray
) -> list[dict[str, float]]:
    """Evaluate a bank of probability rows with one vectorized CPU transfer."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if p.ndim != 2 or p.shape[1] != len(y):
        raise ValueError(f"expected probability bank (P,{len(y)}), got {p.shape}")
    targets = y[None, :]
    losses = -np.mean(targets * np.log(p) + (1 - targets) * np.log1p(-p), axis=1)
    accuracies = np.mean((p >= 0.5) == targets, axis=1)
    briers = np.mean((p - targets) ** 2, axis=1)
    return [
        {
            "auc": float(_binary_auc_from_ranks(y, p[index])),
            "loss": float(losses[index]),
            "acc": float(accuracies[index]),
            "brier": float(briers[index]),
        }
        for index in range(p.shape[0])
    ]


def screen_p(
    suffixes: np.ndarray,
    labels: np.ndarray,
    split: dict,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    """Select p with a fixed validation-only standardized logistic probe."""
    train, val, _ = _indices(split)
    rows = []
    for probability in P_VALUES:
        features = mix_suffix_bank(suffixes, probability).astype(np.float64)
        probe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=1.0,
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        probe.fit(features[train], labels[train])
        validation = metric_bundle(labels[val], probe.predict_proba(features[val])[:, 1])
        rows.append({"p": float(probability), **{f"val_{k}": v for k, v in validation.items()}})
    table = pd.DataFrame(rows)
    ranked = table.sort_values(
        ["val_auc", "val_loss", "val_acc", "p"],
        ascending=[False, True, False, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0].p), table.sort_values("p").reset_index(drop=True)


def _torch_probabilities(model: torch.nn.Linear, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(x).squeeze(1)).detach().cpu().numpy()


def fit_linear(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
    device: str,
    artifact_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Exact manuscript head: unstandardized Linear+BCE+Adam for 500 epochs."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        compute_device = torch.device("cpu")
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(compute_device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(compute_device)
    train_np, val_np, test_np = _indices(split)
    train = torch.from_numpy(train_np).to(compute_device)
    val = torch.from_numpy(val_np).to(compute_device)
    test = torch.from_numpy(test_np).to(compute_device)
    model = torch.nn.Linear(x.shape[1], 1, bias=True).to(compute_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LINEAR_LR, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator(device=compute_device).manual_seed(int(seed))
    best_key = None
    best_state = None
    best_epoch = -1
    best_validation = None
    history = []
    for epoch in range(1, LINEAR_EPOCHS + 1):
        model.train()
        order = train[torch.randperm(len(train), generator=generator, device=compute_device)]
        losses = []
        for start in range(0, len(order), LINEAR_BATCH_SIZE):
            batch = order[start : start + LINEAR_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch]).squeeze(1), y[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        validation = metric_bundle(labels[val_np], _torch_probabilities(model, x[val]))
        key = (validation["auc"], -validation["loss"], validation["acc"])
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_validation = validation
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **{f"val_{key}": value for key, value in validation.items()},
            }
        )
    if best_state is None or best_validation is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(compute_device).eval()
    test_probability = _torch_probabilities(model, x[test])
    test_metrics = metric_bundle(labels[test_np], test_probability)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "model": "torch.nn.Linear",
            "state_dict": best_state,
            "input_dim": int(x.shape[1]),
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "selection_rule": SELECTION_RULE,
            "standardized": False,
            "weight_decay": 0.0,
        },
        artifact_dir / "best.pt",
    )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(artifact_dir / "history.csv", index=False)
    predictions = pd.DataFrame(
        {
            "sample_index": test_np,
            "y_true": labels[test_np],
            "probability": test_probability,
            "prediction": (test_probability >= 0.5).astype(np.int64),
        }
    )
    predictions.to_csv(artifact_dir / "test_predictions.csv", index=False)
    metrics = {
        "protocol_version": PROTOCOL_VERSION,
        "feature_dim": int(x.shape[1]),
        "seed": int(seed),
        "epochs": LINEAR_EPOCHS,
        "best_epoch": int(best_epoch),
        "selection_rule": SELECTION_RULE,
        "standardized": False,
        "weight_decay": 0.0,
        **{f"val_{key}": value for key, value in best_validation.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics, history_frame, predictions


def fit_linear_p_scan(
    suffixes: np.ndarray,
    labels: np.ndarray,
    split: dict,
    *,
    seed: int,
    device: str,
    artifact_dir: Path,
) -> tuple[float, pd.DataFrame, dict[float, dict], dict[float, pd.DataFrame]]:
    """Train ten independent manuscript heads in one vectorized Adam graph.

    Every p row starts from the exact same ``torch.nn.Linear`` initialization
    and sees the exact same seeded minibatch order that a separate run would
    receive.  Summing the per-row BCE losses leaves every row's gradient
    independent, so Adam's elementwise states reproduce ten independent
    optimizers without a tenfold Python loop.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"p scan requested unavailable CUDA device {device}")
    feature_bank = np.stack(
        [mix_suffix_bank(suffixes, probability) for probability in P_VALUES]
    ).astype(np.float32)
    x = torch.from_numpy(feature_bank).to(compute_device)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(compute_device)
    train_np, val_np, test_np = _indices(split)
    train = torch.from_numpy(train_np).to(compute_device)
    val = torch.from_numpy(val_np).to(compute_device)
    test = torch.from_numpy(test_np).to(compute_device)
    reference = torch.nn.Linear(x.shape[2], 1, bias=True).to(compute_device)
    weight = torch.nn.Parameter(reference.weight.detach().repeat(len(P_VALUES), 1))
    bias = torch.nn.Parameter(reference.bias.detach().repeat(len(P_VALUES)))
    del reference
    optimizer = torch.optim.Adam((weight, bias), lr=LINEAR_LR, weight_decay=0.0)
    generator = torch.Generator(device=compute_device).manual_seed(int(seed))
    best_keys: list[tuple[float, float, float] | None] = [None] * len(P_VALUES)
    best_epochs = [-1] * len(P_VALUES)
    best_validation: list[dict | None] = [None] * len(P_VALUES)
    best_weights: list[torch.Tensor | None] = [None] * len(P_VALUES)
    best_biases: list[torch.Tensor | None] = [None] * len(P_VALUES)
    history_rows = []
    for epoch in range(1, LINEAR_EPOCHS + 1):
        order = train[torch.randperm(len(train), generator=generator, device=compute_device)]
        epoch_losses: list[torch.Tensor] = []
        for start in range(0, len(order), LINEAR_BATCH_SIZE):
            batch = order[start : start + LINEAR_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = torch.einsum("pbd,pd->pb", x[:, batch, :], weight) + bias[:, None]
            targets = y[batch][None, :].expand(len(P_VALUES), -1)
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(dim=1)
            losses.sum().backward()
            optimizer.step()
            epoch_losses.append(losses.detach().cpu())
        with torch.no_grad():
            val_logits = torch.einsum("pbd,pd->pb", x[:, val, :], weight) + bias[:, None]
            val_probability = torch.sigmoid(val_logits).cpu().numpy()
        mean_train_loss = torch.stack(epoch_losses).mean(dim=0).numpy()
        validation_bank = metric_bundles(labels[val_np], val_probability)
        for index, probability in enumerate(P_VALUES):
            validation = validation_bank[index]
            key = (validation["auc"], -validation["loss"], validation["acc"])
            if best_keys[index] is None or key > best_keys[index]:
                best_keys[index] = key
                best_epochs[index] = epoch
                best_validation[index] = validation
                best_weights[index] = weight[index].detach().cpu().clone()
                best_biases[index] = bias[index].detach().cpu().clone()
            history_rows.append(
                {
                    "p": float(probability),
                    "epoch": epoch,
                    "train_loss": float(mean_train_loss[index]),
                    **{f"val_{name}": value for name, value in validation.items()},
                }
            )
    if any(value is None for value in best_validation + best_weights + best_biases):
        raise RuntimeError("incomplete affine p scan")
    screen_rows = [
        {
            "p": float(probability),
            "best_epoch": int(best_epochs[index]),
            **{
                f"val_{name}": value
                for name, value in best_validation[index].items()  # type: ignore[union-attr]
            },
        }
        for index, probability in enumerate(P_VALUES)
    ]
    screen = pd.DataFrame(screen_rows)
    selected = screen.sort_values(
        ["val_auc", "val_loss", "val_acc", "p"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).iloc[0]
    selected_p = float(selected.p)
    selected_indices = {0, P_VALUES.index(selected_p)}
    metrics_by_p: dict[float, dict] = {}
    predictions_by_p: dict[float, pd.DataFrame] = {}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(history_rows)
    history.to_csv(artifact_dir / "p_scan_history.csv", index=False)
    screen.to_csv(artifact_dir / "p_validation_screen.csv", index=False)
    for index in selected_indices:
        probability = float(P_VALUES[index])
        best_weight = best_weights[index].to(compute_device)  # type: ignore[union-attr]
        best_bias = best_biases[index].to(compute_device)  # type: ignore[union-attr]
        with torch.no_grad():
            test_logits = torch.einsum("bd,d->b", x[index, test, :], best_weight) + best_bias
            test_probability = torch.sigmoid(test_logits).cpu().numpy()
        test_metrics = metric_bundle(labels[test_np], test_probability)
        metrics = {
            "protocol_version": PROTOCOL_VERSION,
            "p_selection_protocol": P_SELECTION_PROTOCOL,
            "feature_dim": int(x.shape[2]),
            "seed": int(seed),
            "epochs": LINEAR_EPOCHS,
            "best_epoch": int(best_epochs[index]),
            "selection_rule": SELECTION_RULE,
            "standardized": False,
            "weight_decay": 0.0,
            **{
                f"val_{name}": value
                for name, value in best_validation[index].items()  # type: ignore[union-attr]
            },
            **{f"test_{name}": value for name, value in test_metrics.items()},
        }
        predictions = pd.DataFrame(
            {
                "sample_index": test_np,
                "y_true": labels[test_np],
                "probability": test_probability,
                "prediction": (test_probability >= 0.5).astype(np.int64),
            }
        )
        torch.save(
            {
                "protocol_version": PROTOCOL_VERSION,
                "p_selection_protocol": P_SELECTION_PROTOCOL,
                "model": "torch.nn.Linear",
                "weight": best_weights[index],
                "bias": best_biases[index],
                "p": probability,
                "seed": int(seed),
                "best_epoch": int(best_epochs[index]),
            },
            artifact_dir / (f"p{probability:.3f}".replace(".", "p") + ".pt"),
        )
        predictions.to_csv(
            artifact_dir / (f"p{probability:.3f}".replace(".", "p") + "_test_predictions.csv"),
            index=False,
        )
        metrics_by_p[probability] = metrics
        predictions_by_p[probability] = predictions
    return selected_p, screen, metrics_by_p, predictions_by_p


def qrc_result_dir(dataset: str, variant: str, q: int, t: int, seed: int) -> Path:
    return (
        RESULT_ROOT
        / "qrc"
        / dataset.upper()
        / variant
        / f"Q{int(q):02d}"
        / f"T{int(t):02d}"
        / f"seed{int(seed):03d}"
    )


def evaluate_qrc_setting(
    dataset: str,
    variant: str,
    q: int,
    t: int,
    seed: int,
    *,
    device: str = "cpu",
    force: bool = False,
) -> list[dict]:
    destination = qrc_result_dir(dataset, variant, q, t, seed)
    metrics_path = destination / "metrics.json"
    if metrics_path.exists() and not force:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            payload.get("protocol_version") == PROTOCOL_VERSION
            and payload.get("p_selection_protocol") == P_SELECTION_PROTOCOL
        ):
            return payload["rows"]
    with np.load(token_cache_path(dataset, variant), allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    split = json.loads(split_path(dataset, seed).read_text(encoding="utf-8"))
    suffixes = load_qrc_bank(
        dataset,
        variant,
        q,
        t,
        seed,
        n_samples=len(labels),
        token_sha256=token_sha,
    )
    destination.mkdir(parents=True, exist_ok=True)
    selected_p, screen, metrics_by_p, _ = fit_linear_p_scan(
        suffixes,
        labels,
        split,
        seed=seed,
        device=device,
        artifact_dir=destination,
    )
    rows = []
    for system, probability in (("QRC_CLOSED", 0.0), ("QRC_OPEN", selected_p)):
        metrics = metrics_by_p[probability]
        rows.append(
            {
                **metrics,
                "dataset": dataset.upper(),
                "token_variant": variant,
                "system": system,
                "q": int(q),
                "t": int(t),
                "p": float(probability),
                "p_selection": (
                    "fixed p=0"
                    if system == "QRC_CLOSED"
                    else "same affine 500-epoch head; validation max AUC > min loss > max ACC > smaller p"
                ),
            }
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "p_selection_protocol": P_SELECTION_PROTOCOL,
        "rows": rows,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return rows
