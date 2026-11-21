#!/usr/bin/env python3
"""Train Z-only, ZZ-only, and Z+ZZ linear readouts for the five Q6 tasks.

The script reuses the immutable ideal and raw-count-derived hardware feature
matrices from the completed canonical-v3 Q6/T6 experiment.  It changes only
the retained observable columns.  Every feature domain receives its own
single Linear(D,1) readout under the existing split, seed, epoch, and
validation-selection protocol.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_local_analysis as common


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "local_results_five_tasks"
OUTPUT_DIR = RESULT_DIR / "observable_family_ablation"
IDEAL_PATHS = {
    1: ROOT / "data" / "ideal_bace_bal400_canonical_v3_q6_t6_l1_seed0.npz",
    3: ROOT / "data" / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz",
}
TASKS = {
    "01_q6_t6_l3_batch01": {"layers": 3, "fold_factor": 1},
    "02_q6_t6_l3_batch02": {"layers": 3, "fold_factor": 1},
    "03_q6_t6_l3_batch03": {"layers": 3, "fold_factor": 1},
    "04_q6_t6_l1_base_batch01": {"layers": 1, "fold_factor": 1},
    "05_q6_t6_l1_fold3_batch01": {"layers": 1, "fold_factor": 3},
}
FAMILIES = {
    "Z": np.arange(0, 6, dtype=np.int64),
    "ZZ": np.arange(6, 21, dtype=np.int64),
    "Z+ZZ": np.arange(0, 21, dtype=np.int64),
}


def load_feature_npz(path: Path, bundle: dict) -> np.ndarray:
    with np.load(path, allow_pickle=False) as package:
        if not np.array_equal(package["indices"], bundle["indices"]):
            raise AssertionError(f"Index mismatch: {path}")
        if not np.array_equal(package["labels"], bundle["labels"]):
            raise AssertionError(f"Label mismatch: {path}")
        features = package["features"].astype(np.float32)
    if features.shape != (400, 21):
        raise AssertionError(f"Expected (400,21), got {features.shape}: {path}")
    if not np.isfinite(features).all():
        raise AssertionError(f"Nonfinite feature value: {path}")
    return features


def metric_row(
    *,
    backend: str,
    domain: str,
    task_key: str,
    task: dict,
    family: str,
    seed: int,
    split: dict,
    fit: dict,
    test_metrics: dict,
) -> dict:
    return {
        "backend_context": backend,
        "domain": domain,
        "task_key": task_key,
        "layers": int(task["layers"]),
        "fold_factor": int(task["fold_factor"]),
        "observable_family": family,
        "feature_dim": int(len(FAMILIES[family])),
        "seed": int(seed),
        "n_test": int(len(split["test"])),
        "best_epoch": int(fit["best_epoch"]),
        "val_auc": float(fit["val_auc"]),
        "val_loss": float(fit["val_loss"]),
        "val_acc": float(fit["val_acc"]),
        "test_auc": float(test_metrics["auc"]),
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "readout": "single torch.nn.Linear(d,1) readout layer",
        "selection_rule": common.SELECTION_RULE,
        "hardware_repeat_unit": (
            "fixed feature matrix; seeds are nested scaffold/readout analyses"
            if domain != "ideal"
            else "matched ideal feature matrix"
        ),
    }


def train_batched_heads(
    specifications: list[dict],
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
) -> list[dict]:
    """Train independent linear heads in one batched GPU calculation.

    Summing the per-head mean BCE losses leaves every head's gradient identical
    to a separate optimization.  Adam states remain elementwise independent in
    the stacked parameter tensors.  All heads use the same minibatch order,
    matching the original per-head function because that function resets its
    generator to the same seed on every call.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_count = len(specifications)
    maximum_dim = max(item["features"].shape[1] for item in specifications)
    sample_count = len(labels)
    feature_stack = np.zeros(
        (model_count, sample_count, maximum_dim),
        dtype=np.float32,
    )
    initial_weight = np.zeros((model_count, maximum_dim), dtype=np.float32)
    initial_bias = np.zeros(model_count, dtype=np.float32)
    for model_index, item in enumerate(specifications):
        features = np.ascontiguousarray(item["features"], dtype=np.float32)
        width = features.shape[1]
        feature_stack[model_index, :, :width] = features

        # Reproduce the initialization of an independently trained head.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        initial_model = torch.nn.Linear(width, 1, bias=True)
        initial_weight[model_index, :width] = (
            initial_model.weight.detach().numpy().reshape(-1)
        )
        initial_bias[model_index] = float(
            initial_model.bias.detach().numpy().reshape(-1)[0]
        )

    x = torch.from_numpy(feature_stack).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    weight = torch.nn.Parameter(torch.from_numpy(initial_weight).to(device))
    bias = torch.nn.Parameter(torch.from_numpy(initial_bias).to(device))
    optimizer = torch.optim.Adam([weight, bias], lr=1e-3, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    train_idx = torch.from_numpy(split["train"]).to(device)
    val_idx = torch.from_numpy(split["val"]).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    best_keys: list[tuple | None] = [None] * model_count
    best_epochs = np.zeros(model_count, dtype=np.int64)
    best_values: list[dict | None] = [None] * model_count
    best_weights = np.zeros_like(initial_weight)
    best_biases = np.zeros_like(initial_bias)

    for epoch in range(1, 501):
        order = train_idx[
            torch.randperm(len(train_idx), generator=generator, device=device)
        ]
        for start in range(0, len(order), 32):
            index = order[start : start + 32]
            optimizer.zero_grad(set_to_none=True)
            logits = (
                torch.einsum("mnd,md->mn", x[:, index, :], weight)
                + bias[:, None]
            )
            targets = y[index][None, :].expand(model_count, -1)
            # Each model contributes its own mean loss without cross-model
            # scaling, matching independent BCEWithLogitsLoss reductions.
            loss = criterion(logits, targets).mean(dim=1).sum()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_logits = (
                torch.einsum("mnd,md->mn", x[:, val_idx, :], weight)
                + bias[:, None]
            ).cpu().numpy()
            weight_now = weight.detach().cpu().numpy()
            bias_now = bias.detach().cpu().numpy()
        for model_index in range(model_count):
            val = common.metrics(labels[split["val"]], val_logits[model_index])
            key = (val["auc"], -val["loss"], val["acc"])
            if best_keys[model_index] is None or key > best_keys[model_index]:
                best_keys[model_index] = key
                best_epochs[model_index] = epoch
                best_values[model_index] = val
                best_weights[model_index] = weight_now[model_index]
                best_biases[model_index] = bias_now[model_index]

    test_idx = split["test"]
    results = []
    for model_index, item in enumerate(specifications):
        width = item["features"].shape[1]
        scores = (
            np.asarray(item["features"][test_idx], dtype=np.float64)
            @ best_weights[model_index, :width].astype(np.float64)
            + float(best_biases[model_index])
        )
        results.append(
            {
                "fit": {
                    "best_epoch": int(best_epochs[model_index]),
                    "val_auc": float(best_values[model_index]["auc"]),
                    "val_loss": float(best_values[model_index]["loss"]),
                    "val_acc": float(best_values[model_index]["acc"]),
                },
                "test_metrics": common.metrics(labels[test_idx], scores),
            }
        )
    return results


def run_backend(backend: str) -> Path:
    bundle = common.load_dataset()
    ideals = {
        layers: load_feature_npz(path, bundle)
        for layers, path in IDEAL_PATHS.items()
    }
    hardware = {
        task_key: load_feature_npz(
            RESULT_DIR / f"hardware_features_{backend}_{task_key}.npz",
            bundle,
        )
        for task_key in TASKS
    }

    rows: list[dict] = []
    for seed in common.SEEDS:
        split = common.split_rows(bundle, seed)
        specifications: list[dict] = []
        for layers, features in ideals.items():
            for family, columns in FAMILIES.items():
                specifications.append(
                    {
                        "kind": "ideal",
                        "layers": layers,
                        "family": family,
                        "features": np.ascontiguousarray(features[:, columns]),
                    }
                )
        for task_key in TASKS:
            for family, columns in FAMILIES.items():
                specifications.append(
                    {
                        "kind": "hardware",
                        "task_key": task_key,
                        "family": family,
                        "features": np.ascontiguousarray(
                            hardware[task_key][:, columns]
                        ),
                    }
                )

        fitted = train_batched_heads(
            specifications,
            bundle["labels"],
            split,
            seed,
        )
        result_by_key = {}
        for specification, result in zip(specifications, fitted):
            if specification["kind"] == "ideal":
                key = (
                    "ideal",
                    int(specification["layers"]),
                    specification["family"],
                )
            else:
                key = (
                    "hardware",
                    specification["task_key"],
                    specification["family"],
                )
            result_by_key[key] = result

        for task_key, task in TASKS.items():
            layers = int(task["layers"])
            for family in FAMILIES:
                ideal_result = result_by_key[("ideal", layers, family)]
                rows.append(
                    metric_row(
                        backend=backend,
                        domain="ideal",
                        task_key=task_key,
                        task=task,
                        family=family,
                        seed=seed,
                        split=split,
                        fit=ideal_result["fit"],
                        test_metrics=ideal_result["test_metrics"],
                    )
                )
                hardware_result = result_by_key[
                    ("hardware", task_key, family)
                ]
                rows.append(
                    metric_row(
                        backend=backend,
                        domain=backend,
                        task_key=task_key,
                        task=task,
                        family=family,
                        seed=seed,
                        split=split,
                        fit=hardware_result["fit"],
                        test_metrics=hardware_result["test_metrics"],
                    )
                )
        print(f"[{backend}] completed seed {seed}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"observable_family_metrics_by_seed_{backend}.csv"
    data = pd.DataFrame(rows).sort_values(
        ["domain", "task_key", "observable_family", "seed"]
    )
    data.to_csv(output, index=False)

    expected_rows = 2 * len(TASKS) * len(FAMILIES) * len(common.SEEDS)
    if len(data) != expected_rows:
        raise AssertionError((len(data), expected_rows))

    reference = pd.read_csv(
        RESULT_DIR / "five_task_ideal_vs_hardware_metrics_by_seed.csv"
    )
    reference = reference[reference["backend"] == backend].copy()
    combined = data[data["observable_family"] == "Z+ZZ"].copy()
    ideal = combined[combined["domain"] == "ideal"][
        ["task_key", "seed", "test_auc"]
    ].rename(columns={"test_auc": "recomputed_ideal"})
    measured = combined[combined["domain"] == backend][
        ["task_key", "seed", "test_auc"]
    ].rename(columns={"test_auc": "recomputed_hardware"})
    audit = (
        reference[
            [
                "task_key",
                "seed",
                "ideal_test_auc",
                "hardware_trained_test_auc",
            ]
        ]
        .merge(ideal, on=["task_key", "seed"], validate="one_to_one")
        .merge(measured, on=["task_key", "seed"], validate="one_to_one")
    )
    max_ideal = float(
        np.max(np.abs(audit["ideal_test_auc"] - audit["recomputed_ideal"]))
    )
    max_hardware = float(
        np.max(
            np.abs(
                audit["hardware_trained_test_auc"]
                - audit["recomputed_hardware"]
            )
        )
    )
    if max(max_ideal, max_hardware) > 1e-12:
        raise AssertionError(
            f"Z+ZZ reproduction failed: ideal={max_ideal}, "
            f"hardware={max_hardware}"
        )
    audit_payload = {
        "backend": backend,
        "row_count": int(len(data)),
        "families": {key: int(len(value)) for key, value in FAMILIES.items()},
        "seed_count": int(len(common.SEEDS)),
        "max_abs_auc_difference_from_primary_z_plus_zz": {
            "ideal": max_ideal,
            "hardware": max_hardware,
        },
        "status": "pass",
    }
    (
        OUTPUT_DIR / f"observable_family_reproduction_audit_{backend}.json"
    ).write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    print(f"[saved] {output}")
    print(json.dumps(audit_payload, indent=2))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("baihua", "shenglian", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backends = ("baihua", "shenglian") if args.backend == "all" else (args.backend,)
    for backend in backends:
        run_backend(backend)


if __name__ == "__main__":
    main()
