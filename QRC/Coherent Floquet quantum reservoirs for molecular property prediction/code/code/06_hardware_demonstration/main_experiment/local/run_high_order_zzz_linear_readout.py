#!/usr/bin/env python3
"""Extract ZZZ moments and run the hardware-aligned linear-readout diagnostic.

The raw cloud archives contain full six-bit Z-basis count distributions, so
all diagonal Z strings can be reconstructed without another hardware run.
This script compares three predeclared observable families:

* ``Z+ZZ``: the existing 21-dimensional low-order readout;
* ``ZZZ``: all 20 third-order raw Pauli expectations;
* ``Z+ZZ+ZZZ``: the 41-dimensional augmented readout.

Ideal, Baihua, and Shenglian features receive separate affine linear heads.
Splits, seeds, optimizer, epochs, and validation-only checkpoint ranking are
identical to the manuscript linear-readout protocol.
"""

from __future__ import annotations

import itertools
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_complete_five_task_analysis as base
import run_local_analysis as common


ROOT = Path(__file__).resolve().parents[1]
QRC_ROOT = ROOT.parent
LINEAR_ROOT = QRC_ROOT / "FINAL" / "2_linear_readout"
INPUT_DIR = ROOT / "local_results_five_tasks"
OUTPUT_DIR = INPUT_DIR / "high_order_zzz_linear"
FEATURE_DIR = OUTPUT_DIR / "features"
BASELINE_METRICS_PATH = (
    INPUT_DIR / "five_task_ideal_vs_hardware_metrics_by_seed.csv"
)

Q = 6
TRIPLES = tuple(itertools.combinations(range(Q), 3))
TRIPLE_NAMES = tuple(
    "ZZZ_" + "_".join(str(index) for index in triple)
    for triple in TRIPLES
)
FAMILY_ORDER = ("Z+ZZ", "ZZZ", "Z+ZZ+ZZZ")
FAMILY_DIMS = {"Z+ZZ": 21, "ZZZ": 20, "Z+ZZ+ZZZ": 41}


def raw_zzz_from_counts(counts: dict, record: dict) -> np.ndarray:
    total = float(sum(float(value) for value in counts.values()))
    if total <= 0:
        raise ValueError("Counts sum is not positive")
    triple = np.zeros(len(TRIPLES), dtype=np.float64)
    for key, count in counts.items():
        bits = common.measured_bits(key, record)
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        probability = float(count) / total
        triple += probability * np.asarray(
            [
                signs[left] * signs[middle] * signs[right]
                for left, middle, right in TRIPLES
            ],
            dtype=np.float64,
        )
    return triple.astype(np.float32)


def build_hardware_high_order_features(
    backend: str,
    task_key: str,
    bundle: dict,
) -> Path:
    manifest = base.locate_manifest(backend, task_key)
    records = common.read_jsonl(manifest)
    if len(records) != 400:
        raise RuntimeError(
            f"{backend}/{task_key}: expected 400 records, got {len(records)}"
        )
    row_by_index = {
        int(original): row
        for row, original in enumerate(bundle["indices"].tolist())
    }
    rows = []
    raw_dir = manifest.parent / f"raw_counts_{backend}_{task_key}"
    for record in records:
        if int(record["q"]) != Q:
            raise AssertionError(f"{backend}/{task_key}: expected Q={Q}")
        raw_path = raw_dir / str(record["raw_file_name"])
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        counts = common.extract_counts(result)
        if not counts:
            raise RuntimeError(f"{backend}/{task_key}: no counts in {raw_path}")
        original = int(record["original_index"])
        dataset_row = row_by_index[original]
        if str(record["token_hash"]) != str(
            bundle["token_hashes"][dataset_row]
        ):
            raise AssertionError(
                f"{backend}/{task_key}: token hash mismatch for {original}"
            )
        low = common.counts_to_features(counts, record)
        zzz = raw_zzz_from_counts(counts, record)
        rows.append(
            (
                dataset_row,
                original,
                low,
                zzz,
                int(round(sum(float(value) for value in counts.values()))),
                str(record["tid"]),
                int(record["submission_rank"]),
            )
        )
    rows.sort(key=lambda value: value[0])
    if [value[1] for value in rows] != bundle["indices"].astype(int).tolist():
        raise AssertionError(f"{backend}/{task_key}: row reconstruction failed")

    low = np.stack([value[2] for value in rows]).astype(np.float32)
    zzz = np.stack([value[3] for value in rows]).astype(np.float32)
    existing_path = (
        INPUT_DIR / f"hardware_features_{backend}_{task_key}.npz"
    )
    with np.load(existing_path, allow_pickle=False) as package:
        existing = package["features"].astype(np.float32)
    maximum_difference = float(np.max(np.abs(low - existing)))
    if maximum_difference > 1.0e-7:
        raise AssertionError(
            f"{backend}/{task_key}: low-order reconstruction mismatch "
            f"{maximum_difference}"
        )

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FEATURE_DIR / f"hardware_{backend}_{task_key}_z_zz_zzz.npz"
    np.savez_compressed(
        output,
        indices=np.asarray([value[1] for value in rows], dtype=np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        z_zz=low,
        zzz=zzz,
        z_zz_zzz=np.concatenate([low, zzz], axis=1).astype(np.float32),
        zzz_names=np.asarray(TRIPLE_NAMES, dtype=str),
        shots=np.asarray([value[4] for value in rows], dtype=np.int64),
        tids=np.asarray([value[5] for value in rows], dtype=str),
        submission_ranks=np.asarray(
            [value[6] for value in rows],
            dtype=np.int64,
        ),
        backend=np.asarray(backend),
        task_key=np.asarray(task_key),
        raw_moment_definition=np.asarray(
            "<product of Pauli-Z signs>; not a connected cumulant"
        ),
        low_order_reconstruction_max_abs_diff=np.asarray(
            maximum_difference,
            dtype=np.float64,
        ),
    )
    print(f"[saved] {output}", flush=True)
    return output


def observable_sign_matrix(
    num_qubits: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = 1 << int(num_qubits)
    basis = torch.arange(dimension, device=device, dtype=torch.long)
    shifts = torch.arange(
        num_qubits - 1,
        -1,
        -1,
        device=device,
        dtype=torch.long,
    )
    signs = 1.0 - 2.0 * (
        (basis[:, None] >> shifts[None, :]) & 1
    ).to(dtype)
    low_rows = [signs[:, index] for index in range(num_qubits)]
    low_rows.extend(
        signs[:, left] * signs[:, right]
        for left, right in itertools.combinations(range(num_qubits), 2)
    )
    triple_rows = [
        signs[:, left] * signs[:, middle] * signs[:, right]
        for left, middle, right in itertools.combinations(
            range(num_qubits),
            3,
        )
    ]
    return (
        torch.stack(low_rows, dim=0).contiguous(),
        torch.stack(triple_rows, dim=0).contiguous(),
    )


@torch.no_grad()
def simulate_ideal_high_order(
    bundle: dict,
    layers: int,
    existing_ideal: dict,
) -> Path:
    module_path = LINEAR_ROOT / "core" / "gpu_qrc_statevector.py"
    module_name = "qrc_gpu_statevector_high_order_standalone"
    specification = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load statevector module from {module_path}")
    statevector_module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = statevector_module
    specification.loader.exec_module(statevector_module)
    DTCQRCConfig = statevector_module.DTCQRCConfig
    evolve_batch_final_state = statevector_module.evolve_batch_final_state
    generate_dtc_params = statevector_module.generate_dtc_params

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = DTCQRCConfig(
        num_qubits=Q,
        features_per_qubit=3,
        g=0.84,
        W_scale=0.6,
        layers_per_injection=int(layers),
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    params = generate_dtc_params(Q, seed=0, W_scale=0.6)
    if not np.allclose(params["J"], existing_ideal["J"], atol=0, rtol=0):
        raise AssertionError(f"L{layers}: J mismatch")
    if not np.allclose(params["h"], existing_ideal["h"], atol=0, rtol=0):
        raise AssertionError(f"L{layers}: h mismatch")

    low_signs, triple_signs = observable_sign_matrix(
        Q,
        device=device,
        dtype=torch.float32,
    )
    tokens = torch.from_numpy(bundle["tokens"].astype(np.float32))
    J = torch.from_numpy(params["J"].astype(np.float32))
    h = torch.from_numpy(params["h"].astype(np.float32))
    low_chunks = []
    triple_chunks = []
    batch_size = 128
    for start in range(0, len(tokens), batch_size):
        end = min(start + batch_size, len(tokens))
        state = evolve_batch_final_state(
            tokens[start:end],
            cfg,
            J,
            h,
            device=device,
        )
        probabilities = torch.abs(state).square().real
        low_chunks.append((probabilities @ low_signs.t()).cpu())
        triple_chunks.append((probabilities @ triple_signs.t()).cpu())
        print(
            f"[ideal L{layers} | {device}] {end}/{len(tokens)}",
            flush=True,
        )
    low = torch.cat(low_chunks, dim=0).numpy().astype(np.float32)
    zzz = torch.cat(triple_chunks, dim=0).numpy().astype(np.float32)
    existing_low = existing_ideal["features"].astype(np.float32)
    maximum_difference = float(np.max(np.abs(low - existing_low)))
    if maximum_difference > 5.0e-5:
        raise AssertionError(
            f"L{layers}: ideal low-order reconstruction mismatch "
            f"{maximum_difference}"
        )

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FEATURE_DIR / f"ideal_l{layers}_z_zz_zzz.npz"
    np.savez_compressed(
        output,
        indices=bundle["indices"].astype(np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        z_zz=low,
        zzz=zzz,
        z_zz_zzz=np.concatenate([low, zzz], axis=1).astype(np.float32),
        zzz_names=np.asarray(TRIPLE_NAMES, dtype=str),
        J=params["J"].astype(np.float32),
        h=params["h"].astype(np.float32),
        layers=np.asarray(layers, dtype=np.int64),
        simulator_device=np.asarray(str(device)),
        raw_moment_definition=np.asarray(
            "<product of Pauli-Z signs>; not a connected cumulant"
        ),
        low_order_reconstruction_max_abs_diff=np.asarray(
            maximum_difference,
            dtype=np.float64,
        ),
    )
    print(f"[saved] {output}", flush=True)
    return output


def load_bundle() -> dict:
    common.DATASET_PATH = base.DATASET_PATH
    bundle = common.load_dataset()
    with np.load(base.DATASET_PATH, allow_pickle=False) as package:
        bundle["tokens"] = package["tokens"].astype(np.float32)
    return bundle


def load_existing_ideals() -> dict[int, dict]:
    ideals = {}
    for layers, path in base.IDEAL_PATHS.items():
        with np.load(path, allow_pickle=False) as package:
            ideals[layers] = {
                "indices": package["indices"].astype(np.int64),
                "labels": package["labels"].astype(np.int64),
                "token_hashes": package["token_hashes"].astype(str),
                "features": package["features"].astype(np.float32),
                "J": package["J"].astype(np.float32),
                "h": package["h"].astype(np.float32),
            }
    return ideals


def build_all_features(
    bundle: dict,
    existing_ideals: dict[int, dict],
) -> tuple[dict[int, Path], dict[tuple[str, str], Path]]:
    ideal_paths = {
        layers: simulate_ideal_high_order(bundle, layers, ideal)
        for layers, ideal in existing_ideals.items()
    }
    hardware_paths = {
        (backend, task_key): build_hardware_high_order_features(
            backend,
            task_key,
            bundle,
        )
        for backend in base.BACKENDS
        for task_key in base.TASKS
    }
    return ideal_paths, hardware_paths


def load_family_arrays(
    ideal_paths: dict[int, Path],
    hardware_paths: dict[tuple[str, str], Path],
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    dict[tuple[str, str], dict[str, np.ndarray]],
]:
    key_map = {"ZZZ": "zzz", "Z+ZZ+ZZZ": "z_zz_zzz"}
    ideals = {}
    for layers, path in ideal_paths.items():
        with np.load(path, allow_pickle=False) as package:
            ideals[layers] = {
                family: package[key].astype(np.float32)
                for family, key in key_map.items()
            }
    hardware = {}
    for domain, path in hardware_paths.items():
        with np.load(path, allow_pickle=False) as package:
            hardware[domain] = {
                family: package[key].astype(np.float32)
                for family, key in key_map.items()
            }
    return ideals, hardware


def train_high_order_families(
    bundle: dict,
    ideal_paths: dict[int, Path],
    hardware_paths: dict[tuple[str, str], Path],
) -> pd.DataFrame:
    ideals, hardware = load_family_arrays(ideal_paths, hardware_paths)
    rows = []
    predictions = []
    for seed in common.SEEDS:
        split = common.split_rows(bundle, seed)
        test = split["test"]
        ideal_results: dict[tuple[int, str], dict] = {}
        for layers in base.IDEAL_PATHS:
            for family in ("ZZZ", "Z+ZZ+ZZZ"):
                features = ideals[layers][family]
                state, fit = common.train_ideal_linear_head(
                    features,
                    bundle["labels"],
                    split,
                    seed,
                )
                scores = common.logits(features[test], state)
                ideal_results[(layers, family)] = {
                    "fit": fit,
                    "scores": scores,
                    "metrics": common.metrics(
                        bundle["labels"][test],
                        scores,
                    ),
                }

        for task_key, task in base.TASKS.items():
            layers = int(task["layers"])
            for backend in base.BACKENDS:
                for family in ("ZZZ", "Z+ZZ+ZZZ"):
                    features = hardware[(backend, task_key)][family]
                    state, fit = common.train_ideal_linear_head(
                        features,
                        bundle["labels"],
                        split,
                        seed,
                    )
                    scores = common.logits(features[test], state)
                    test_metrics = common.metrics(
                        bundle["labels"][test],
                        scores,
                    )
                    ideal = ideal_results[(layers, family)]
                    rows.append(
                        {
                            "backend": backend,
                            "task_key": task_key,
                            "layers": layers,
                            "fold_factor": int(task["fold_factor"]),
                            "seed": int(seed),
                            "observable_family": family,
                            "feature_dim": FAMILY_DIMS[family],
                            "n_test": int(len(test)),
                            "ideal_best_epoch": ideal["fit"]["best_epoch"],
                            "ideal_val_auc": ideal["fit"]["val_auc"],
                            "ideal_val_loss": ideal["fit"]["val_loss"],
                            "ideal_val_acc": ideal["fit"]["val_acc"],
                            "ideal_test_auc": ideal["metrics"]["auc"],
                            "ideal_test_loss": ideal["metrics"]["loss"],
                            "ideal_test_acc": ideal["metrics"]["acc"],
                            "hardware_best_epoch": fit["best_epoch"],
                            "hardware_val_auc": fit["val_auc"],
                            "hardware_val_loss": fit["val_loss"],
                            "hardware_val_acc": fit["val_acc"],
                            "hardware_test_auc": test_metrics["auc"],
                            "hardware_test_loss": test_metrics["loss"],
                            "hardware_test_acc": test_metrics["acc"],
                            "hardware_minus_ideal_auc": (
                                test_metrics["auc"]
                                - ideal["metrics"]["auc"]
                            ),
                            "readout": "single torch.nn.Linear(d,1) readout layer",
                            "selection_rule": common.SELECTION_RULE,
                            "moment_definition": (
                                "raw Pauli-Z string expectation; "
                                "not connected cumulant"
                            ),
                        }
                    )
                    predictions.extend(
                        {
                            "backend": backend,
                            "task_key": task_key,
                            "seed": int(seed),
                            "observable_family": family,
                            "original_index": int(bundle["indices"][row]),
                            "label": int(bundle["labels"][row]),
                            "ideal_score": float(ideal_score),
                            "hardware_score": float(hardware_score),
                        }
                        for row, ideal_score, hardware_score in zip(
                            test,
                            ideal["scores"],
                            scores,
                        )
                    )
        print(f"[trained] seed={seed}", flush=True)

    high_order = pd.DataFrame(rows)
    baseline = pd.read_csv(BASELINE_METRICS_PATH)
    baseline_rows = baseline[
        [
            "backend",
            "task_key",
            "layers",
            "fold_factor",
            "seed",
            "n_test",
            "ideal_best_epoch",
            "ideal_val_auc",
            "ideal_val_loss",
            "ideal_val_acc",
            "ideal_test_auc",
            "ideal_test_loss",
            "ideal_test_acc",
            "hardware_best_epoch",
            "hardware_val_auc",
            "hardware_val_loss",
            "hardware_val_acc",
            "hardware_trained_test_auc",
            "hardware_trained_test_loss",
            "hardware_trained_test_acc",
            "hardware_minus_ideal_auc",
            "selection_rule",
        ]
    ].rename(
        columns={
            "hardware_trained_test_auc": "hardware_test_auc",
            "hardware_trained_test_loss": "hardware_test_loss",
            "hardware_trained_test_acc": "hardware_test_acc",
        }
    )
    baseline_rows["observable_family"] = "Z+ZZ"
    baseline_rows["feature_dim"] = FAMILY_DIMS["Z+ZZ"]
    baseline_rows["readout"] = "single torch.nn.Linear(d,1) readout layer"
    baseline_rows["moment_definition"] = (
        "raw Pauli-Z string expectation; not connected cumulant"
    )
    columns = list(high_order.columns)
    metrics = pd.concat(
        [baseline_rows[columns], high_order[columns]],
        ignore_index=True,
    )
    metrics["observable_family"] = pd.Categorical(
        metrics["observable_family"],
        categories=FAMILY_ORDER,
        ordered=True,
    )
    metrics = metrics.sort_values(
        ["backend", "task_key", "seed", "observable_family"]
    ).reset_index(drop=True)
    if len(metrics) != 300:
        raise AssertionError(f"Expected 300 rows, got {len(metrics)}")
    metrics.to_csv(
        OUTPUT_DIR / "high_order_zzz_linear_metrics_by_seed.csv",
        index=False,
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "high_order_zzz_linear_test_scores.csv",
        index=False,
    )
    return metrics


def summarize(metrics: pd.DataFrame) -> None:
    summary = (
        metrics.groupby(
            [
                "backend",
                "task_key",
                "layers",
                "fold_factor",
                "observable_family",
                "feature_dim",
            ],
            observed=True,
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
        )
    )
    summary.to_csv(
        OUTPUT_DIR / "high_order_zzz_linear_summary.csv",
        index=False,
    )

    keys = ["backend", "task_key", "layers", "fold_factor", "seed"]
    baseline = metrics[
        metrics["observable_family"].astype(str) == "Z+ZZ"
    ][
        keys
        + [
            "ideal_test_auc",
            "ideal_test_acc",
            "hardware_test_auc",
            "hardware_test_acc",
        ]
    ].rename(
        columns={
            "ideal_test_auc": "baseline_ideal_auc",
            "ideal_test_acc": "baseline_ideal_acc",
            "hardware_test_auc": "baseline_hardware_auc",
            "hardware_test_acc": "baseline_hardware_acc",
        }
    )
    comparison = metrics.merge(
        baseline,
        on=keys,
        validate="many_to_one",
    )
    comparison["delta_ideal_auc_vs_z_zz"] = (
        comparison["ideal_test_auc"] - comparison["baseline_ideal_auc"]
    )
    comparison["delta_hardware_auc_vs_z_zz"] = (
        comparison["hardware_test_auc"]
        - comparison["baseline_hardware_auc"]
    )
    comparison["delta_ideal_acc_vs_z_zz"] = (
        comparison["ideal_test_acc"] - comparison["baseline_ideal_acc"]
    )
    comparison["delta_hardware_acc_vs_z_zz"] = (
        comparison["hardware_test_acc"]
        - comparison["baseline_hardware_acc"]
    )
    comparison.to_csv(
        OUTPUT_DIR / "high_order_zzz_vs_z_zz_paired_by_seed.csv",
        index=False,
    )
    comparison.groupby(
        [
            "backend",
            "task_key",
            "layers",
            "fold_factor",
            "observable_family",
            "feature_dim",
        ],
        observed=True,
        as_index=False,
    ).agg(
        nested_seed_count=("seed", "nunique"),
        delta_ideal_auc_mean=("delta_ideal_auc_vs_z_zz", "mean"),
        delta_ideal_auc_sd_nested=("delta_ideal_auc_vs_z_zz", "std"),
        delta_hardware_auc_mean=("delta_hardware_auc_vs_z_zz", "mean"),
        delta_hardware_auc_sd_nested=(
            "delta_hardware_auc_vs_z_zz",
            "std",
        ),
    ).to_csv(
        OUTPUT_DIR / "high_order_zzz_vs_z_zz_paired_summary.csv",
        index=False,
    )
    print(summary.to_string(index=False), flush=True)


def write_protocol_audit(
    ideal_paths: dict[int, Path],
    hardware_paths: dict[tuple[str, str], Path],
) -> None:
    ideal_audit = {}
    for layers, path in ideal_paths.items():
        with np.load(path, allow_pickle=False) as package:
            ideal_audit[f"L{layers}"] = {
                "path": str(path),
                "z_zz_shape": list(package["z_zz"].shape),
                "zzz_shape": list(package["zzz"].shape),
                "baseline_max_abs_difference": float(
                    package["low_order_reconstruction_max_abs_diff"]
                ),
                "simulator_device": str(package["simulator_device"].item()),
            }
    hardware_audit = {}
    for (backend, task_key), path in hardware_paths.items():
        with np.load(path, allow_pickle=False) as package:
            hardware_audit[f"{backend}/{task_key}"] = {
                "path": str(path),
                "z_zz_shape": list(package["z_zz"].shape),
                "zzz_shape": list(package["zzz"].shape),
                "shots_unique": np.unique(package["shots"]).astype(int).tolist(),
                "baseline_max_abs_difference": float(
                    package["low_order_reconstruction_max_abs_diff"]
                ),
            }
    payload = {
        "observable_families": {
            family: FAMILY_DIMS[family] for family in FAMILY_ORDER
        },
        "triple_names": list(TRIPLE_NAMES),
        "moment_definition": (
            "raw product expectation of Pauli-Z signs; not a connected cumulant"
        ),
        "dimension_boundary": (
            "Z+ZZ (D=21) versus ZZZ-only (D=20) is near-matched, "
            "not exactly fixed D or fixed hardware measurement cost"
        ),
        "ideal": ideal_audit,
        "hardware": hardware_audit,
        "readout": "single torch.nn.Linear(d,1)",
        "selection_rule": common.SELECTION_RULE,
    }
    (OUTPUT_DIR / "HIGH_ORDER_ZZZ_PROTOCOL_AUDIT.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    existing_ideals = load_existing_ideals()
    ideal_paths, hardware_paths = build_all_features(
        bundle,
        existing_ideals,
    )
    write_protocol_audit(ideal_paths, hardware_paths)
    metrics = train_high_order_families(
        bundle,
        ideal_paths,
        hardware_paths,
    )
    summarize(metrics)


if __name__ == "__main__":
    main()
