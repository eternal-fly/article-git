#!/usr/bin/env python3
"""Reconstruct and train the completed Shenglian Q6/T6/L3 g scan locally."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
QRC_ROOT = HERE.parent
COMMON_DIR = QRC_ROOT / "Quafu_mainline_Q6T6L3_experiment" / "local"
STATEVECTOR_PATH = (
    QRC_ROOT / "FINAL" / "2_linear_readout" / "core" / "gpu_qrc_statevector.py"
)
DATASET_PATH = HERE / "bace_bal400_canonical_v3_q6_t6.npz"
OUTPUT_DIR = HERE / "local_training_results"
EXTRACT_DIR = HERE / "extracted_results"
SEEDS = tuple(range(0, 100, 10))
G_POINTS = {"040": 0.40, "060": 0.60, "100": 1.00}
PROTOCOL = "shenglian-q6t6l3-gscan-local-linear-readout-v1"

sys.path.insert(0, str(COMMON_DIR))
import run_local_analysis as common  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value: object) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def load_dataset() -> dict:
    with np.load(DATASET_PATH, allow_pickle=False) as package:
        bundle = {
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "tokens": package["tokens"].astype(np.float32),
            "token_hashes": package["token_hashes"].astype(str),
            "splits": json.loads(scalar_text(package["split_indices_json"])),
        }
    if bundle["tokens"].shape != (400, 6, 18):
        raise AssertionError(bundle["tokens"].shape)
    if np.bincount(bundle["labels"], minlength=2).tolist() != [200, 200]:
        raise AssertionError("Expected a balanced 200/200 subset")
    return bundle


def load_statevector_module():
    name = "gscan_gpu_qrc_statevector"
    spec = importlib.util.spec_from_file_location(name, STATEVECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(STATEVECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def locate_manifest(tag: str) -> Path:
    archive = HERE / f"sl_g{tag}_results.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    destination = EXTRACT_DIR / f"sl_g{tag}"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    manifests = sorted(destination.glob(f"**/sl_g{tag}_jobs.jsonl"))
    if len(manifests) != 1:
        raise FileNotFoundError(f"g={tag}: expected one manifest, got {manifests}")
    return manifests[0]


def validate_record(record: dict, expected_g: float) -> None:
    expected = {
        "q": 6,
        "t": 6,
        "layers": 3,
        "reservoir_seed": 0,
        "nominal_two_qubit_gates": 180,
    }
    for key, value in expected.items():
        if int(record[key]) != value:
            raise AssertionError((key, record[key], value))
    if not np.isclose(float(record["g"]), expected_g, atol=1e-8):
        raise AssertionError((record["g"], expected_g))
    if not np.isclose(float(record["W"]), 0.60, atol=1e-8):
        raise AssertionError(record["W"])


def build_hardware_features(tag: str, expected_g: float, bundle: dict) -> Path:
    manifest = locate_manifest(tag)
    records = common.read_jsonl(manifest)
    if len(records) != 400:
        raise RuntimeError(f"g={expected_g}: manifest has {len(records)}/400 records")
    if len({str(record["tid"]) for record in records}) != 400:
        raise AssertionError(f"g={expected_g}: task IDs are not unique")

    row_by_index = {
        int(original): row
        for row, original in enumerate(bundle["indices"].tolist())
    }
    raw_dir = manifest.parent / f"raw_sl_g{tag}"
    rows = []
    for record in records:
        validate_record(record, expected_g)
        raw_name = record.get("raw_file_name")
        if not raw_name:
            raise RuntimeError(f"g={expected_g}: missing raw filename")
        raw_path = raw_dir / raw_name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        counts = common.extract_counts(result)
        if not counts:
            raise RuntimeError(f"g={expected_g}: no counts in {raw_path}")
        original = int(record["original_index"])
        dataset_row = row_by_index[original]
        if str(record["token_hash"]) != str(bundle["token_hashes"][dataset_row]):
            raise AssertionError(f"g={expected_g}: token mismatch for {original}")
        rows.append(
            (
                dataset_row,
                original,
                common.counts_to_features(counts, record),
                int(round(sum(float(value) for value in counts.values()))),
                str(record["tid"]),
                int(record["submission_rank"]),
            )
        )
    rows.sort(key=lambda value: value[0])
    if [value[1] for value in rows] != bundle["indices"].astype(int).tolist():
        raise AssertionError(f"g={expected_g}: restored dataset order differs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"hardware_sl_g{tag}_z_zz.npz"
    np.savez_compressed(
        output,
        indices=np.asarray([value[1] for value in rows], dtype=np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        features=np.stack([value[2] for value in rows]).astype(np.float32),
        shots=np.asarray([value[3] for value in rows], dtype=np.int64),
        tids=np.asarray([value[4] for value in rows], dtype=str),
        submission_ranks=np.asarray([value[5] for value in rows], dtype=np.int64),
        g=np.asarray(expected_g, dtype=np.float32),
        observable_family=np.asarray("Z+ZZ"),
        source_manifest=np.asarray(str(manifest.resolve())),
        protocol=np.asarray(PROTOCOL),
    )
    print(f"[features] g={expected_g:.2f} -> {output}")
    return output


def build_ideal_features(tag: str, g_value: float, bundle: dict) -> Path:
    output = OUTPUT_DIR / f"ideal_g{tag}_z_zz.npz"
    if output.is_file():
        with np.load(output, allow_pickle=False) as cached:
            if (
                np.array_equal(cached["indices"], bundle["indices"])
                and np.array_equal(cached["token_hashes"], bundle["token_hashes"])
                and np.isclose(float(cached["g"]), g_value)
            ):
                return output

    module = load_statevector_module()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config = module.DTCQRCConfig(
        num_qubits=6,
        features_per_qubit=3,
        g=g_value,
        W_scale=0.60,
        layers_per_injection=3,
        use_tanh_before_encoding=True,
        token_dim_policy="strict",
        complex_dtype="complex64",
    )
    features, params = module.compute_qrc_features_gpu(
        torch.from_numpy(bundle["tokens"]),
        cfg=config,
        seed=0,
        batch_size=128,
        device=device,
        progress=False,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=bundle["indices"].astype(np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        features=features.numpy().astype(np.float32),
        J=params["J"].astype(np.float32),
        h=params["h"].astype(np.float32),
        g=np.asarray(g_value, dtype=np.float32),
        W=np.asarray(0.60, dtype=np.float32),
        reservoir_seed=np.asarray(0, dtype=np.int64),
        protocol=np.asarray(PROTOCOL),
        simulator=np.asarray(str(STATEVECTOR_PATH.resolve())),
    )
    print(f"[ideal] g={g_value:.2f} device={device} -> {output}")
    return output


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    denominator = float(
        np.linalg.norm(x.T @ x, ord="fro")
        * np.linalg.norm(y.T @ y, ord="fro")
    )
    return numerator / denominator if denominator > 0 else float("nan")


def geometry_metrics(ideal: np.ndarray, hardware: np.ndarray) -> dict[str, float]:
    correlations = []
    for column in range(ideal.shape[1]):
        if np.std(ideal[:, column]) == 0 or np.std(hardware[:, column]) == 0:
            correlations.append(np.nan)
        else:
            correlations.append(float(pearsonr(ideal[:, column], hardware[:, column]).statistic))
    return {
        "centered_linear_cka": linear_cka(ideal, hardware),
        "pairwise_distance_spearman": float(
            spearmanr(pdist(ideal), pdist(hardware)).statistic
        ),
        "observable_pearson_mean": float(np.nanmean(correlations)),
        "observable_pearson_median": float(np.nanmedian(correlations)),
    }


def load_feature_matrix(path: Path, bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as package:
        if not np.array_equal(package["indices"], bundle["indices"]):
            raise AssertionError(path)
        return package["features"].astype(np.float32), package.get(
            "shots", np.asarray([], dtype=np.int64)
        ).astype(np.int64)


def train_one(tag: str, g_value: float, bundle: dict) -> None:
    hardware_path = build_hardware_features(tag, g_value, bundle)
    ideal_path = build_ideal_features(tag, g_value, bundle)
    hardware, shots = load_feature_matrix(hardware_path, bundle)
    ideal, _ = load_feature_matrix(ideal_path, bundle)
    geometry = geometry_metrics(ideal, hardware)

    rows = []
    for seed in SEEDS:
        split = common.split_rows(bundle, seed)
        ideal_state, ideal_fit = common.train_ideal_linear_head(
            ideal, bundle["labels"], split, seed
        )
        hardware_state, hardware_fit = common.train_ideal_linear_head(
            hardware, bundle["labels"], split, seed
        )
        test = split["test"]
        ideal_scores = common.logits(ideal[test], ideal_state)
        hardware_scores = common.logits(hardware[test], hardware_state)
        frozen_scores = common.logits(hardware[test], ideal_state)
        ideal_test = common.metrics(bundle["labels"][test], ideal_scores)
        hardware_test = common.metrics(bundle["labels"][test], hardware_scores)
        frozen_test = common.metrics(bundle["labels"][test], frozen_scores)
        rows.append(
            {
                "tag": f"g{tag}",
                "g": g_value,
                "seed": seed,
                "n_test": len(test),
                "ideal_best_epoch": ideal_fit["best_epoch"],
                "ideal_val_auc": ideal_fit["val_auc"],
                "ideal_val_loss": ideal_fit["val_loss"],
                "ideal_val_acc": ideal_fit["val_acc"],
                "ideal_test_auc": ideal_test["auc"],
                "ideal_test_loss": ideal_test["loss"],
                "ideal_test_acc": ideal_test["acc"],
                "hardware_best_epoch": hardware_fit["best_epoch"],
                "hardware_val_auc": hardware_fit["val_auc"],
                "hardware_val_loss": hardware_fit["val_loss"],
                "hardware_val_acc": hardware_fit["val_acc"],
                "hardware_test_auc": hardware_test["auc"],
                "hardware_test_loss": hardware_test["loss"],
                "hardware_test_acc": hardware_test["acc"],
                "hardware_minus_ideal_auc": hardware_test["auc"] - ideal_test["auc"],
                "frozen_ideal_head_hardware_auc": frozen_test["auc"],
                "readout": "single torch.nn.Linear(21,1) layer",
                "selection_rule": common.SELECTION_RULE,
                "protocol": PROTOCOL,
            }
        )
        print(
            f"[train] g={g_value:.2f} seed={seed:02d} "
            f"ideal={ideal_test['auc']:.3f} hardware={hardware_test['auc']:.3f}"
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / f"metrics_g{tag}_by_seed.csv", index=False)
    summary = {
        "tag": f"g{tag}",
        "g": g_value,
        "nested_seed_count": len(SEEDS),
        "ideal_auc_mean": float(table["ideal_test_auc"].mean()),
        "ideal_auc_sd": float(table["ideal_test_auc"].std(ddof=1)),
        "hardware_auc_mean": float(table["hardware_test_auc"].mean()),
        "hardware_auc_sd": float(table["hardware_test_auc"].std(ddof=1)),
        "hardware_minus_ideal_auc_mean": float(table["hardware_minus_ideal_auc"].mean()),
        "hardware_minus_ideal_auc_sd": float(table["hardware_minus_ideal_auc"].std(ddof=1)),
        "frozen_ideal_head_hardware_auc_mean": float(
            table["frozen_ideal_head_hardware_auc"].mean()
        ),
        "shot_mean": float(np.mean(shots)),
        "shot_min": int(np.min(shots)),
        "shot_max": int(np.max(shots)),
        **geometry,
        "independent_hardware_repeat_count": 1,
        "inference_note": (
            "ten scaffold/readout seeds reuse one hardware acquisition at this g"
        ),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "protocol": PROTOCOL,
    }
    (OUTPUT_DIR / f"summary_g{tag}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def bootstrap_mean_interval(values: np.ndarray, seed: int = 20260731) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = values[rng.integers(0, len(values), size=(200_000, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize() -> None:
    tables = []
    summaries = []
    for tag, g_value in G_POINTS.items():
        metric_path = OUTPUT_DIR / f"metrics_g{tag}_by_seed.csv"
        summary_path = OUTPUT_DIR / f"summary_g{tag}.json"
        if not metric_path.is_file() or not summary_path.is_file():
            continue
        tables.append(pd.read_csv(metric_path))
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    if not tables:
        raise FileNotFoundError("No completed per-g metric tables")
    metrics = pd.concat(tables, ignore_index=True).sort_values(["g", "seed"])
    metrics.to_csv(OUTPUT_DIR / "gscan_metrics_by_seed.csv", index=False)
    summary = pd.DataFrame(summaries).sort_values("g")
    summary.to_csv(OUTPUT_DIR / "gscan_summary.csv", index=False)

    paired_rows = []
    available_tags = summary["tag"].tolist()
    for left_index, left_tag in enumerate(available_tags):
        for right_tag in available_tags[left_index + 1 :]:
            left = metrics[metrics["tag"] == left_tag][["seed", "hardware_test_auc"]]
            right = metrics[metrics["tag"] == right_tag][["seed", "hardware_test_auc"]]
            paired = left.merge(right, on="seed", suffixes=("_left", "_right"))
            differences = (
                paired["hardware_test_auc_right"] - paired["hardware_test_auc_left"]
            ).to_numpy()
            low, high = bootstrap_mean_interval(differences)
            paired_rows.append(
                {
                    "contrast": f"{right_tag}-{left_tag}",
                    "n_nested_seeds": len(differences),
                    "mean_delta_auc": float(np.mean(differences)),
                    "sd_delta_auc": float(np.std(differences, ddof=1)),
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                    "note": "nested scaffold/readout seeds; not independent hardware repeats",
                }
            )
    pd.DataFrame(paired_rows).to_csv(
        OUTPUT_DIR / "gscan_hardware_paired_effects.csv", index=False
    )

    report_lines = [
        "# Shenglian hardware g-scan training results",
        "",
        "Single `Linear(21,1)` readouts; ten fixed scaffold/readout seeds; ",
        "checkpoint ranking: validation AUC, then validation loss, then validation accuracy.",
        "",
        "| g | Ideal AUC | Hardware AUC | Hardware-ideal | CKA | Distance rho |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        report_lines.append(
            f"| {row.g:.2f} | {row.ideal_auc_mean:.3f} +/- {row.ideal_auc_sd:.3f} "
            f"| {row.hardware_auc_mean:.3f} +/- {row.hardware_auc_sd:.3f} "
            f"| {row.hardware_minus_ideal_auc_mean:+.3f} "
            f"| {row.centered_linear_cka:.3f} "
            f"| {row.pairwise_distance_spearman:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "The ten seeds are nested analyses of one acquired 400-molecule feature batch",
            "per g and are not independent hardware repetitions.",
        ]
    )
    (OUTPUT_DIR / "RESULTS.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    print(summary[["g", "ideal_auc_mean", "ideal_auc_sd", "hardware_auc_mean", "hardware_auc_sd", "hardware_minus_ideal_auc_mean", "centered_linear_cka", "pairwise_distance_spearman"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*", choices=tuple(G_POINTS), default=tuple(G_POINTS))
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summarize_only:
        bundle = load_dataset()
        common.DATASET_PATH = DATASET_PATH
        for tag in args.tags:
            train_one(tag, G_POINTS[tag], bundle)
    summarize()


if __name__ == "__main__":
    main()
