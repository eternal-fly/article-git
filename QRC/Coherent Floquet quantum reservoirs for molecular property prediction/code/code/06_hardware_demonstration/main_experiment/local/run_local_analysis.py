#!/usr/bin/env python3
"""Parse downloaded raw counts and run the complete analysis locally."""

from __future__ import annotations

import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "bace_bal400_canonical_v3_q6_t6.npz"
IDEAL_PATH = (
    ROOT / "data" / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz"
)
DOWNLOAD_DIR = ROOT / "downloaded"
OUTPUT_DIR = ROOT / "local_results"
BACKENDS = ("baihua", "shenglian")
SEEDS = tuple(range(0, 100, 10))
SELECTION_RULE = "max(val_auc) > min(val_loss) > max(val_acc)"


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def load_dataset() -> dict:
    with np.load(DATASET_PATH, allow_pickle=False) as package:
        return {
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "smiles": package["smiles"].astype(str),
            "token_hashes": package["token_hashes"].astype(str),
            "splits": json.loads(scalar_text(package["split_indices_json"])),
        }


def load_ideal() -> dict:
    with np.load(IDEAL_PATH, allow_pickle=False) as package:
        return {
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "token_hashes": package["token_hashes"].astype(str),
            "features": package["features"].astype(np.float32),
        }


def extract_backend_archive(backend: str) -> Path:
    archive = DOWNLOAD_DIR / f"{backend}_q6t6l3_cloud_results.zip"
    if archive.is_file():
        destination = DOWNLOAD_DIR / f"extracted_{backend}"
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(destination)
        return destination
    return DOWNLOAD_DIR


def locate_manifest(backend: str) -> Path:
    root = extract_backend_archive(backend)
    candidates = sorted(root.glob(f"**/{backend}_q6t6l3_job_manifest.jsonl"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {backend} manifest under {root}, got {candidates}"
        )
    return candidates[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_counts(result: Any) -> Mapping[str, int | float] | None:
    if not isinstance(result, dict):
        return None
    for key in ("count", "counts"):
        if result.get(key):
            return result[key]
    for parent in ("data", "result"):
        child = result.get(parent)
        if isinstance(child, dict):
            for key in ("count", "counts"):
                if child.get(key):
                    return child[key]
    return None


def parse_bitstring(key: Any) -> str:
    if isinstance(key, (int, np.integer)):
        return bin(int(key))[2:]
    value = str(key).strip().replace(" ", "")
    if value.startswith("0b"):
        return value[2:]
    if value and set(value).issubset({"0", "1"}):
        return value
    runs = re.findall(r"[01]+", value)
    if not runs:
        raise ValueError(f"Cannot parse count key {key!r}")
    return "".join(runs)


def measured_bits(key: Any, record: dict) -> np.ndarray:
    raw = parse_bitstring(key)
    q = int(record["q"])
    measured = [int(value) for value in record["measured_clbits"]]
    classical_n = int(record["classical_num_bits"])
    # Quafu may return either the full physical classical register or only the
    # compact measured substring. Treat it as full only when the key is long
    # enough to address the largest measured classical bit.
    if max(measured) >= q and len(raw) >= max(measured) + 1:
        full = raw.zfill(max(classical_n, len(raw)))
        return np.asarray([int(full[-1 - c]) for c in measured], dtype=np.int8)
    compact = raw.zfill(q)[-q:]
    return np.asarray([int(compact[-1 - i]) for i in range(q)], dtype=np.int8)


def counts_to_features(
    counts: Mapping[str, int | float], record: dict
) -> np.ndarray:
    q = int(record["q"])
    total = float(sum(float(value) for value in counts.values()))
    if total <= 0:
        raise ValueError("Counts sum is not positive")
    z1 = np.zeros(q, dtype=np.float64)
    z2 = np.zeros((q, q), dtype=np.float64)
    for key, count in counts.items():
        bits = measured_bits(key, record)
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        probability = float(count) / total
        z1 += probability * signs
        z2 += probability * np.outer(signs, signs)
    pairs = [z2[left, right] for left in range(q) for right in range(left + 1, q)]
    return np.concatenate([z1, np.asarray(pairs)]).astype(np.float32)


def build_hardware_features(backend: str, bundle: dict) -> Path:
    manifest = locate_manifest(backend)
    records = read_jsonl(manifest)
    if len(records) != 400:
        raise RuntimeError(f"{backend}: manifest has {len(records)}/400 records")
    row_by_index = {
        int(original): row for row, original in enumerate(bundle["indices"].tolist())
    }
    rows = []
    for record in records:
        raw_name = record.get("raw_file_name")
        if not raw_name:
            raise RuntimeError(
                f"{backend}: missing raw_file_name for id={record['original_index']}"
            )
        raw_path = manifest.parent / f"raw_counts_{backend}_q6t6l3" / raw_name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        result = json.loads(raw_path.read_text(encoding="utf-8"))
        counts = extract_counts(result)
        if not counts:
            raise RuntimeError(f"{backend}: no counts in {raw_path}")
        original = int(record["original_index"])
        expected_row = row_by_index[original]
        if str(record["token_hash"]) != str(bundle["token_hashes"][expected_row]):
            raise AssertionError(f"{backend}: token hash mismatch for {original}")
        rows.append(
            (
                expected_row,
                original,
                counts_to_features(counts, record),
                int(round(sum(float(value) for value in counts.values()))),
                str(record["tid"]),
                int(record["submission_rank"]),
            )
        )
    rows.sort(key=lambda value: value[0])
    if [value[1] for value in rows] != bundle["indices"].astype(int).tolist():
        raise AssertionError(f"{backend}: dataset order reconstruction failed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"hardware_features_{backend}_q6t6l3.npz"
    np.savez_compressed(
        output,
        indices=np.asarray([value[1] for value in rows], dtype=np.int64),
        labels=bundle["labels"].astype(np.int64),
        token_hashes=bundle["token_hashes"].astype(str),
        features=np.stack([value[2] for value in rows]).astype(np.float32),
        shots=np.asarray([value[3] for value in rows], dtype=np.int64),
        tids=np.asarray([value[4] for value in rows], dtype=str),
        submission_ranks=np.asarray([value[5] for value in rows], dtype=np.int64),
        backend=np.asarray(backend),
        source_manifest=np.asarray(str(manifest.resolve())),
    )
    print(f"[saved] {output}")
    return output


def split_rows(bundle: dict, seed: int) -> dict[str, np.ndarray]:
    row_by_index = {
        int(original): row for row, original in enumerate(bundle["indices"].tolist())
    }
    split = bundle["splits"][str(int(seed))]
    return {
        name: np.asarray(
            [row_by_index[int(value)] for value in split[f"{name}_idx"]],
            dtype=np.int64,
        )
        for name in ("train", "val", "test")
    }


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "loss": float(log_loss(labels, probability, labels=[0, 1])),
        "acc": float(accuracy_score(labels, prediction)),
    }


def train_ideal_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    model = torch.nn.Linear(x.shape[1], 1, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_idx = torch.from_numpy(split["train"]).to(device)
    val_idx = torch.from_numpy(split["val"]).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    best_key = None
    best_state = None
    best_epoch = None
    best_val = None
    for epoch in range(1, 501):
        model.train()
        order = train_idx[
            torch.randperm(len(train_idx), generator=generator, device=device)
        ]
        for start in range(0, len(order), 32):
            index = order[start : start + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[index]).squeeze(1), y[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x[val_idx]).squeeze(1).cpu().numpy()
        val = metrics(labels[split["val"]], val_logits)
        key = (val["auc"], -val["loss"], val["acc"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_val = val
    if best_state is None:
        raise RuntimeError("No validation checkpoint")
    return best_state, {
        "best_epoch": int(best_epoch),
        "val_auc": float(best_val["auc"]),
        "val_loss": float(best_val["loss"]),
        "val_acc": float(best_val["acc"]),
    }


def logits(features: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    weight = state["weight"].numpy().reshape(-1).astype(np.float64)
    bias = float(state["bias"].numpy().reshape(-1)[0])
    return np.asarray(features, dtype=np.float64) @ weight + bias


def run_local_training(
    bundle: dict, ideal: dict, feature_paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hardware = {}
    for backend, path in feature_paths.items():
        with np.load(path, allow_pickle=False) as package:
            if not np.array_equal(package["indices"], bundle["indices"]):
                raise AssertionError(f"{backend}: hardware index mismatch")
            hardware[backend] = package["features"].astype(np.float32)
    rows = []
    predictions = []
    for seed in SEEDS:
        split = split_rows(bundle, seed)
        state, fit = train_ideal_linear_head(
            ideal["features"], bundle["labels"], split, seed
        )
        test = split["test"]
        ideal_scores = logits(ideal["features"][test], state)
        ideal_test = metrics(bundle["labels"][test], ideal_scores)
        for backend in BACKENDS:
            hardware_scores = logits(hardware[backend][test], state)
            hardware_test = metrics(bundle["labels"][test], hardware_scores)
            rows.append(
                {
                    "backend": backend,
                    "seed": seed,
                    "n_test": len(test),
                    "ideal_best_epoch": fit["best_epoch"],
                    "ideal_val_auc": fit["val_auc"],
                    "ideal_test_auc": ideal_test["auc"],
                    "frozen_hardware_test_auc": hardware_test["auc"],
                    "hardware_minus_ideal_auc": (
                        hardware_test["auc"] - ideal_test["auc"]
                    ),
                    "frozen_hardware_test_loss": hardware_test["loss"],
                    "frozen_hardware_test_acc": hardware_test["acc"],
                    "task_score_pearson": float(
                        pearsonr(ideal_scores, hardware_scores).statistic
                    ),
                    "task_score_spearman": float(
                        spearmanr(ideal_scores, hardware_scores).statistic
                    ),
                    "readout": (
                        "single torch.nn.Linear(d,1) readout layer, "
                        "trained on ideal features only"
                    ),
                    "selection_rule": SELECTION_RULE,
                }
            )
            predictions.extend(
                {
                    "backend": backend,
                    "seed": seed,
                    "original_index": int(bundle["indices"][row]),
                    "label": int(bundle["labels"][row]),
                    "ideal_score": float(ideal_score),
                    "frozen_hardware_score": float(hardware_score),
                }
                for row, ideal_score, hardware_score in zip(
                    test, ideal_scores, hardware_scores
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(
        OUTPUT_DIR / "ideal_vs_hardware_metrics_by_seed.csv", index=False
    )
    pd.DataFrame(predictions).to_csv(
        OUTPUT_DIR / "ideal_vs_hardware_test_scores.csv", index=False
    )
    summary = (
        table.groupby("backend", as_index=False)
        .agg(
            nested_readout_seed_count=("seed", "nunique"),
            ideal_test_auc_mean=("ideal_test_auc", "mean"),
            ideal_test_auc_sd=("ideal_test_auc", "std"),
            frozen_hardware_auc_mean=("frozen_hardware_test_auc", "mean"),
            frozen_hardware_auc_sd=("frozen_hardware_test_auc", "std"),
            hardware_minus_ideal_auc_mean=("hardware_minus_ideal_auc", "mean"),
            hardware_minus_ideal_auc_sd=("hardware_minus_ideal_auc", "std"),
            task_score_pearson_mean=("task_score_pearson", "mean"),
            task_score_spearman_mean=("task_score_spearman", "mean"),
        )
    )
    summary["independent_device_repeat_count"] = 1
    summary["inference_note"] = (
        "ten readout seeds reuse one hardware feature batch; descriptive, "
        "not ten device repetitions"
    )
    summary.to_csv(OUTPUT_DIR / "ideal_vs_hardware_summary.csv", index=False)
    print(summary.to_string(index=False))
    return table, summary


def main() -> None:
    bundle = load_dataset()
    ideal = load_ideal()
    if not np.array_equal(bundle["indices"], ideal["indices"]):
        raise AssertionError("Dataset and ideal indices differ")
    if not np.array_equal(bundle["token_hashes"], ideal["token_hashes"]):
        raise AssertionError("Dataset and ideal token hashes differ")
    feature_paths = {
        backend: build_hardware_features(backend, bundle) for backend in BACKENDS
    }
    run_local_training(bundle, ideal, feature_paths)


if __name__ == "__main__":
    main()
