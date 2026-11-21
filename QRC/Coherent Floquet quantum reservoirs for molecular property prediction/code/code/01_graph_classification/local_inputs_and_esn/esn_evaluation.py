"""Manuscript affine readouts for every matched/wide ESN feature matrix."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from esn_features import ESN_PROTOCOL, load_esn_features
from protocol import (
    LINEAR_BATCH_SIZE,
    LINEAR_EPOCHS,
    LINEAR_LR,
    PROTOCOL_VERSION,
    RESULT_ROOT,
    SELECTION_RULE,
    esn_widths,
    split_path,
    token_cache_path,
)
from readouts import _indices, fit_linear, metric_bundle, metric_bundles


ESN_READOUT_PROTOCOL = "vectorized_independent_affine_width_heads_v1"


def esn_result_dir(
    dataset: str, variant: str, q: int, t: int, width: int, seed: int
) -> Path:
    return (
        RESULT_ROOT
        / "esn"
        / dataset.upper()
        / variant
        / f"Q{int(q):02d}"
        / f"T{int(t):02d}"
        / f"width{int(width):04d}"
        / f"seed{int(seed):03d}"
    )


def evaluate_esn_setting(
    dataset: str,
    variant: str,
    q: int,
    t: int,
    width: int,
    seed: int,
    *,
    device: str = "cpu",
    force: bool = False,
) -> dict:
    destination = esn_result_dir(dataset, variant, q, t, width, seed)
    result_path = destination / "result.json"
    if result_path.exists() and not force:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("protocol_version") == PROTOCOL_VERSION:
            return payload
    with np.load(token_cache_path(dataset, variant), allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    features = load_esn_features(
        dataset,
        variant,
        q,
        t,
        width,
        seed,
        n_samples=len(labels),
        token_sha256=token_sha,
    )
    split = json.loads(split_path(dataset, seed).read_text(encoding="utf-8"))
    metrics, _, _ = fit_linear(
        features,
        labels,
        split,
        seed=seed,
        device=device,
        artifact_dir=destination / "linear",
    )
    payload = {
        **metrics,
        "protocol_version": PROTOCOL_VERSION,
        "esn_protocol": ESN_PROTOCOL,
        "dataset": dataset.upper(),
        "token_variant": variant,
        "system": "ESN",
        "q": int(q),
        "t": int(t),
        "width": int(width),
        "seed": int(seed),
    }
    destination.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def evaluate_esn_width_group(
    dataset: str,
    variant: str,
    q: int,
    t: int,
    seed: int,
    *,
    device: str,
    force: bool = False,
) -> list[dict]:
    """Train all four width heads jointly while preserving independent Adam rows."""
    widths = esn_widths(q)
    paths = {
        width: esn_result_dir(dataset, variant, q, t, width, seed) / "result.json"
        for width in widths
    }
    if not force and all(path.exists() for path in paths.values()):
        payloads = [json.loads(paths[width].read_text(encoding="utf-8")) for width in widths]
        if all(
            payload.get("protocol_version") == PROTOCOL_VERSION
            and payload.get("esn_readout_protocol") == ESN_READOUT_PROTOCOL
            for payload in payloads
        ):
            return payloads
    with np.load(token_cache_path(dataset, variant), allow_pickle=False) as archive:
        labels = archive["labels"].astype(np.int64)
        token_sha = str(archive["tokens_sha256"].item())
    feature_map = {
        width: load_esn_features(
            dataset,
            variant,
            q,
            t,
            width,
            seed,
            n_samples=len(labels),
            token_sha256=token_sha,
        )
        for width in widths
    }
    split = json.loads(split_path(dataset, seed).read_text(encoding="utf-8"))
    train_np, val_np, test_np = _indices(split)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"ESN readout requested unavailable device {device}")
    max_width = max(widths)
    x = torch.zeros(
        (len(widths), len(labels), max_width), dtype=torch.float32, device=compute_device
    )
    initial_weight = torch.zeros(
        (len(widths), max_width), dtype=torch.float32, device=compute_device
    )
    initial_bias = torch.zeros(len(widths), dtype=torch.float32, device=compute_device)
    for index, width in enumerate(widths):
        # A separate manuscript run resets the same seed before constructing
        # each differently sized Linear layer; reproduce that initialization.
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        reference = torch.nn.Linear(int(width), 1, bias=True).to(compute_device)
        initial_weight[index, :width] = reference.weight.detach()[0]
        initial_bias[index] = reference.bias.detach()[0]
        x[index, :, :width] = torch.from_numpy(feature_map[width]).to(compute_device)
        del reference
    weight = torch.nn.Parameter(initial_weight)
    bias = torch.nn.Parameter(initial_bias)
    y = torch.from_numpy(labels.astype(np.float32)).to(compute_device)
    train = torch.from_numpy(train_np).to(compute_device)
    val = torch.from_numpy(val_np).to(compute_device)
    test = torch.from_numpy(test_np).to(compute_device)
    optimizer = torch.optim.Adam((weight, bias), lr=LINEAR_LR, weight_decay=0.0)
    generator = torch.Generator(device=compute_device).manual_seed(int(seed))
    best_keys = [None] * len(widths)
    best_epochs = [-1] * len(widths)
    best_validation = [None] * len(widths)
    best_weights = [None] * len(widths)
    best_biases = [None] * len(widths)
    history_rows = []
    for epoch in range(1, LINEAR_EPOCHS + 1):
        order = train[torch.randperm(len(train), generator=generator, device=compute_device)]
        losses_by_batch = []
        for start in range(0, len(order), LINEAR_BATCH_SIZE):
            batch = order[start : start + LINEAR_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            logits = torch.einsum("pbd,pd->pb", x[:, batch], weight) + bias[:, None]
            targets = y[batch][None].expand(len(widths), -1)
            losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(1)
            losses.sum().backward()
            optimizer.step()
            losses_by_batch.append(losses.detach().cpu())
        with torch.no_grad():
            val_probability = torch.sigmoid(
                torch.einsum("pbd,pd->pb", x[:, val], weight) + bias[:, None]
            ).cpu().numpy()
        mean_losses = torch.stack(losses_by_batch).mean(0).numpy()
        validation_bank = metric_bundles(labels[val_np], val_probability)
        for index, width in enumerate(widths):
            validation = validation_bank[index]
            key = (validation["auc"], -validation["loss"], validation["acc"])
            if best_keys[index] is None or key > best_keys[index]:
                best_keys[index] = key
                best_epochs[index] = epoch
                best_validation[index] = validation
                best_weights[index] = weight[index, :width].detach().cpu().clone()
                best_biases[index] = bias[index].detach().cpu().clone()
            history_rows.append(
                {
                    "width": int(width),
                    "epoch": epoch,
                    "train_loss": float(mean_losses[index]),
                    **{f"val_{name}": value for name, value in validation.items()},
                }
            )
    payloads = []
    history = pd.DataFrame(history_rows)
    for index, width in enumerate(widths):
        if best_validation[index] is None:
            raise RuntimeError(f"no ESN checkpoint for width={width}")
        selected_weight = best_weights[index].to(compute_device)
        selected_bias = best_biases[index].to(compute_device)
        with torch.no_grad():
            test_probability = torch.sigmoid(
                x[index, test, :width] @ selected_weight + selected_bias
            ).cpu().numpy()
        test_metrics = metric_bundle(labels[test_np], test_probability)
        destination = esn_result_dir(dataset, variant, q, t, width, seed)
        linear_dir = destination / "linear"
        linear_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "protocol_version": PROTOCOL_VERSION,
                "esn_readout_protocol": ESN_READOUT_PROTOCOL,
                "model": "torch.nn.Linear",
                "state_dict": {
                    "weight": best_weights[index][None, :],
                    "bias": best_biases[index][None],
                },
                "input_dim": int(width),
                "seed": int(seed),
                "best_epoch": int(best_epochs[index]),
                "selection_rule": SELECTION_RULE,
                "standardized": False,
                "weight_decay": 0.0,
            },
            linear_dir / "best.pt",
        )
        history[history.width == width].drop(columns="width").to_csv(
            linear_dir / "history.csv", index=False
        )
        pd.DataFrame(
            {
                "sample_index": test_np,
                "y_true": labels[test_np],
                "probability": test_probability,
                "prediction": (test_probability >= 0.5).astype(np.int64),
            }
        ).to_csv(linear_dir / "test_predictions.csv", index=False)
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "esn_protocol": ESN_PROTOCOL,
            "esn_readout_protocol": ESN_READOUT_PROTOCOL,
            "dataset": dataset.upper(),
            "token_variant": variant,
            "system": "ESN",
            "q": int(q),
            "t": int(t),
            "width": int(width),
            "seed": int(seed),
            "feature_dim": int(width),
            "epochs": LINEAR_EPOCHS,
            "best_epoch": int(best_epochs[index]),
            "selection_rule": SELECTION_RULE,
            "standardized": False,
            "weight_decay": 0.0,
            **{
                f"val_{name}": value
                for name, value in best_validation[index].items()
            },
            **{f"test_{name}": value for name, value in test_metrics.items()},
        }
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        payloads.append(payload)
    return payloads
