#!/usr/bin/env python3
"""Train domain-matched two-affine-layer MLP readouts on five hardware tasks.

This is a decoder-sensitivity analysis.  It does not overwrite the manuscript
linear-readout result.  The only protocol change is the readout architecture:

    Linear(d, d) -> ReLU -> Linear(d, 1)

Ideal, Baihua, and Shenglian features are fitted separately with identical
splits, seeds, optimization settings, and validation-only checkpoint ranking.
"""

from __future__ import annotations

import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_complete_five_task_analysis as base
import run_local_analysis as common


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "local_results_five_tasks"
OUTPUT_DIR = INPUT_DIR / "mlp_sensitivity"
LINEAR_METRICS_PATH = (
    INPUT_DIR / "five_task_ideal_vs_hardware_metrics_by_seed.csv"
)
EPOCHS = 500
BATCH_SIZE = 32
LEARNING_RATE = 1.0e-3
WEIGHT_DECAY = 0.0
ARCHITECTURE = "Linear(d,d) -> ReLU -> Linear(d,1)"


def resolve_device() -> torch.device:
    requested = os.environ.get("QRC_MLP_DEVICE", "auto").strip().lower()
    if requested in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    return device


def make_model(input_dim: int, hidden_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("hidden", torch.nn.Linear(input_dim, hidden_dim, bias=True)),
                ("activation", torch.nn.ReLU()),
                ("output", torch.nn.Linear(hidden_dim, 1, bias=True)),
            ]
        )
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    seed_everything(seed)
    x = torch.from_numpy(
        np.ascontiguousarray(features, dtype=np.float32)
    ).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    input_dim = int(x.shape[1])
    hidden_dim = input_dim
    model = make_model(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    train_idx = torch.from_numpy(split["train"]).to(device)
    val_idx = torch.from_numpy(split["val"]).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val: dict[str, float] | None = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = train_idx[
            torch.randperm(
                len(train_idx),
                generator=generator,
                device=device,
            )
        ]
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start : start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[index]).squeeze(1), y[index])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x[val_idx]).squeeze(1).cpu().numpy()
        val = common.metrics(labels[split["val"]], val_logits)
        key = (val["auc"], -val["loss"], val["acc"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_val = val

    if best_state is None or best_val is None:
        raise RuntimeError(f"No validation checkpoint for seed={seed}")
    return best_state, {
        "best_epoch": int(best_epoch),
        "val_auc": float(best_val["auc"]),
        "val_loss": float(best_val["loss"]),
        "val_acc": float(best_val["acc"]),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
    }


def mlp_logits(
    features: np.ndarray,
    state: dict[str, torch.Tensor],
) -> np.ndarray:
    input_dim = int(np.asarray(features).shape[1])
    hidden_dim = int(state["hidden.weight"].shape[0])
    model = make_model(input_dim, hidden_dim)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        return (
            model(
                torch.from_numpy(
                    np.ascontiguousarray(features, dtype=np.float32)
                )
            )
            .squeeze(1)
            .numpy()
        )


def load_inputs() -> tuple[
    dict,
    dict[int, dict],
    dict[tuple[str, str], np.ndarray],
]:
    common.DATASET_PATH = base.DATASET_PATH
    bundle = common.load_dataset()
    ideals = {
        layers: base.load_ideal(path)
        for layers, path in base.IDEAL_PATHS.items()
    }
    hardware: dict[tuple[str, str], np.ndarray] = {}
    for backend in base.BACKENDS:
        for task_key in base.TASKS:
            path = (
                INPUT_DIR
                / f"hardware_features_{backend}_{task_key}.npz"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as package:
                if not np.array_equal(package["indices"], bundle["indices"]):
                    raise AssertionError(f"{backend}/{task_key}: index mismatch")
                hardware[(backend, task_key)] = package["features"].astype(
                    np.float32
                )
    return bundle, ideals, hardware


def run() -> pd.DataFrame:
    bundle, ideals, hardware = load_inputs()
    device = resolve_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[protocol] {ARCHITECTURE}; epochs={EPOCHS}; "
        f"selection={common.SELECTION_RULE}; device={device}",
        flush=True,
    )

    rows: list[dict] = []
    predictions: list[dict] = []
    for seed in common.SEEDS:
        split = common.split_rows(bundle, seed)
        test = split["test"]
        ideal_by_layer: dict[int, dict] = {}
        for layers, ideal in ideals.items():
            state, fit = train_mlp(
                ideal["features"],
                bundle["labels"],
                split,
                seed,
                device,
            )
            scores = mlp_logits(ideal["features"][test], state)
            ideal_by_layer[layers] = {
                "fit": fit,
                "scores": scores,
                "metrics": common.metrics(bundle["labels"][test], scores),
            }

        for task_key, task in base.TASKS.items():
            layers = int(task["layers"])
            ideal_result = ideal_by_layer[layers]
            for backend in base.BACKENDS:
                features = hardware[(backend, task_key)]
                state, fit = train_mlp(
                    features,
                    bundle["labels"],
                    split,
                    seed,
                    device,
                )
                scores = mlp_logits(features[test], state)
                test_metrics = common.metrics(
                    bundle["labels"][test],
                    scores,
                )
                rows.append(
                    {
                        "backend": backend,
                        "task_key": task_key,
                        "layers": layers,
                        "fold_factor": int(task["fold_factor"]),
                        "seed": int(seed),
                        "n_test": int(len(test)),
                        "ideal_best_epoch": ideal_result["fit"][
                            "best_epoch"
                        ],
                        "ideal_val_auc": ideal_result["fit"]["val_auc"],
                        "ideal_val_loss": ideal_result["fit"]["val_loss"],
                        "ideal_val_acc": ideal_result["fit"]["val_acc"],
                        "ideal_test_auc": ideal_result["metrics"]["auc"],
                        "ideal_test_loss": ideal_result["metrics"]["loss"],
                        "ideal_test_acc": ideal_result["metrics"]["acc"],
                        "hardware_best_epoch": fit["best_epoch"],
                        "hardware_val_auc": fit["val_auc"],
                        "hardware_val_loss": fit["val_loss"],
                        "hardware_val_acc": fit["val_acc"],
                        "hardware_test_auc": test_metrics["auc"],
                        "hardware_test_loss": test_metrics["loss"],
                        "hardware_test_acc": test_metrics["acc"],
                        "hardware_minus_ideal_auc": (
                            test_metrics["auc"]
                            - ideal_result["metrics"]["auc"]
                        ),
                        "architecture": ARCHITECTURE,
                        "input_dim": fit["input_dim"],
                        "hidden_dim": fit["hidden_dim"],
                        "activation": "ReLU",
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "standardization": False,
                        "selection_rule": common.SELECTION_RULE,
                        "train_device": str(device),
                    }
                )
                predictions.extend(
                    {
                        "backend": backend,
                        "task_key": task_key,
                        "seed": int(seed),
                        "original_index": int(bundle["indices"][row]),
                        "label": int(bundle["labels"][row]),
                        "ideal_score": float(ideal_score),
                        "hardware_score": float(hardware_score),
                    }
                    for row, ideal_score, hardware_score in zip(
                        test,
                        ideal_result["scores"],
                        scores,
                    )
                )
        print(f"[finished] seed={seed}", flush=True)

    metrics = pd.DataFrame(rows)
    if len(metrics) != 100:
        raise AssertionError(f"Expected 100 rows, got {len(metrics)}")
    metrics.to_csv(
        OUTPUT_DIR / "two_layer_mlp_metrics_by_seed.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "two_layer_mlp_test_scores.csv",
        index=False,
    )
    return metrics


def summarize(metrics: pd.DataFrame) -> None:
    summary = (
        metrics.groupby(
            ["backend", "task_key", "layers", "fold_factor"],
            as_index=False,
        )
        .agg(
            nested_seed_count=("seed", "nunique"),
            ideal_auc_mean=("ideal_test_auc", "mean"),
            ideal_auc_sd_nested=("ideal_test_auc", "std"),
            ideal_acc_mean=("ideal_test_acc", "mean"),
            ideal_acc_sd_nested=("ideal_test_acc", "std"),
            hardware_auc_mean=("hardware_test_auc", "mean"),
            hardware_auc_sd_nested=("hardware_test_auc", "std"),
            hardware_acc_mean=("hardware_test_acc", "mean"),
            hardware_acc_sd_nested=("hardware_test_acc", "std"),
            hardware_minus_ideal_auc_mean=(
                "hardware_minus_ideal_auc",
                "mean",
            ),
        )
    )
    summary.to_csv(
        OUTPUT_DIR / "two_layer_mlp_summary.csv",
        index=False,
    )

    linear = pd.read_csv(LINEAR_METRICS_PATH)
    keys = ["backend", "task_key", "layers", "fold_factor", "seed"]
    linear = linear[
        keys
        + [
            "ideal_test_auc",
            "ideal_test_acc",
            "hardware_trained_test_auc",
            "hardware_trained_test_acc",
        ]
    ].rename(
        columns={
            "ideal_test_auc": "linear_ideal_test_auc",
            "ideal_test_acc": "linear_ideal_test_acc",
            "hardware_trained_test_auc": "linear_hardware_test_auc",
            "hardware_trained_test_acc": "linear_hardware_test_acc",
        }
    )
    comparison = metrics.merge(
        linear,
        on=keys,
        validate="one_to_one",
    )
    comparison["mlp_minus_linear_ideal_auc"] = (
        comparison["ideal_test_auc"]
        - comparison["linear_ideal_test_auc"]
    )
    comparison["mlp_minus_linear_hardware_auc"] = (
        comparison["hardware_test_auc"]
        - comparison["linear_hardware_test_auc"]
    )
    comparison["mlp_minus_linear_ideal_acc"] = (
        comparison["ideal_test_acc"]
        - comparison["linear_ideal_test_acc"]
    )
    comparison["mlp_minus_linear_hardware_acc"] = (
        comparison["hardware_test_acc"]
        - comparison["linear_hardware_test_acc"]
    )
    comparison.to_csv(
        OUTPUT_DIR / "two_layer_mlp_vs_linear_by_seed.csv",
        index=False,
    )
    comparison.groupby(
        ["backend", "task_key", "layers", "fold_factor"],
        as_index=False,
    ).agg(
        nested_seed_count=("seed", "nunique"),
        mlp_ideal_auc_mean=("ideal_test_auc", "mean"),
        linear_ideal_auc_mean=("linear_ideal_test_auc", "mean"),
        mlp_minus_linear_ideal_auc_mean=(
            "mlp_minus_linear_ideal_auc",
            "mean",
        ),
        mlp_hardware_auc_mean=("hardware_test_auc", "mean"),
        linear_hardware_auc_mean=("linear_hardware_test_auc", "mean"),
        mlp_minus_linear_hardware_auc_mean=(
            "mlp_minus_linear_hardware_auc",
            "mean",
        ),
    ).to_csv(
        OUTPUT_DIR / "two_layer_mlp_vs_linear_summary.csv",
        index=False,
    )
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    metrics = run()
    summarize(metrics)


if __name__ == "__main__":
    main()
