#!/usr/bin/env python3
"""Shared Quafu submission, collection, and linear-readout analysis.

The module is copied into every cloud task directory so that each uploaded
folder is self-contained.  It deliberately contains no API credential.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


PROTOCOL_VERSION = "2026-07-28-mainline-canonical-v3-hardware-noise-v1"
EXPECTED_SCHEMA = "local-bfs-canonical-isomeric-reparse-stable-v3"
SELECTION_RULE = "max(val_auc) > min(val_loss) > max(val_acc)"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def load_config(path: str | Path = "config.json") -> dict:
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise AssertionError(
            f"Expected protocol {PROTOCOL_VERSION}, got {config.get('protocol_version')}"
        )
    return config


def load_dataset(config: dict) -> dict:
    path = Path(config["files"]["dataset"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = config["files"].get("dataset_sha256")
    if expected_hash and sha256_file(path) != expected_hash:
        raise AssertionError(f"Dataset SHA-256 mismatch for {path}")
    with np.load(path, allow_pickle=False) as package:
        bundle = {
            "path": path,
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "smiles": package["smiles"].astype(str),
            "tokens": package["tokens"].astype(np.float32),
            "event_counts": package["event_counts"].astype(np.int16),
            "token_hashes": package["token_hashes"].astype(str),
            "splits": json.loads(scalar_text(package["split_indices_json"])),
            "metadata": json.loads(scalar_text(package["metadata_json"])),
            "schema": scalar_text(package["token_schema_version"]),
        }
    return bundle


def load_ideal(config: dict) -> dict:
    path = Path(config["files"]["ideal_features"])
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = config["files"].get("ideal_features_sha256")
    if expected_hash and sha256_file(path) != expected_hash:
        raise AssertionError(f"Ideal feature SHA-256 mismatch for {path}")
    with np.load(path, allow_pickle=False) as package:
        return {
            "path": path,
            "indices": package["indices"].astype(np.int64),
            "labels": package["labels"].astype(np.int64),
            "features": package["features"].astype(np.float32),
            "token_hashes": package["token_hashes"].astype(str),
            "J": package["J"].astype(np.float32),
            "h": package["h"].astype(np.float32),
            "metadata": json.loads(scalar_text(package["metadata_json"])),
        }


def two_qubit_gate_count(config: dict, fold_factor: int) -> int:
    q = int(config["circuit"]["q"])
    t = int(config["circuit"]["t"])
    layers = int(config["circuit"]["layers"])
    return 2 * t * layers * (q - 1) * int(fold_factor)


def validate_dataset_and_config(config_path: str | Path = "config.json") -> dict:
    config = load_config(config_path)
    bundle = load_dataset(config)
    ideal = load_ideal(config)
    q = int(config["circuit"]["q"])
    t = int(config["circuit"]["t"])
    expected_shape = (400, t, 3 * q)
    if bundle["tokens"].shape != expected_shape:
        raise AssertionError(
            f"Expected token shape {expected_shape}, got {bundle['tokens'].shape}"
        )
    if bundle["schema"] != EXPECTED_SCHEMA:
        raise AssertionError(f"Unexpected token schema {bundle['schema']}")
    if bundle["metadata"].get("task_guided_traversal") is not False:
        raise AssertionError("Task-guided traversal is forbidden in the mainline batch")
    if bundle["metadata"].get("global_descriptors") is not False:
        raise AssertionError("Global descriptors are forbidden in the mainline batch")
    if np.bincount(bundle["labels"], minlength=2).tolist() != [200, 200]:
        raise AssertionError("The hardware subset must contain 200 molecules per class")
    if not np.array_equal(bundle["indices"], ideal["indices"]):
        raise AssertionError("Ideal and token indices differ")
    if not np.array_equal(bundle["token_hashes"], ideal["token_hashes"]):
        raise AssertionError("Ideal and token hashes differ")
    expected_dim = q * (q + 1) // 2
    if ideal["features"].shape != (400, expected_dim):
        raise AssertionError(ideal["features"].shape)
    if int(ideal["metadata"]["layers"]) != int(config["circuit"]["layers"]):
        raise AssertionError("Ideal feature depth differs from task depth")
    gate_cap = int(config["cloud"]["max_two_qubit_gates"])
    gate_counts = {}
    for variant in config["variants"]:
        fold = int(variant["fold_factor"])
        if fold not in (1, 3):
            raise ValueError("Only fold_factor 1 or 3 is supported")
        count = two_qubit_gate_count(config, fold)
        if count > gate_cap:
            raise ValueError(
                f"{variant['name']}: nominal CX count {count} exceeds cap {gate_cap}"
            )
        gate_counts[variant["name"]] = count
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "batch_id": config["batch_id"],
        "dataset_sha256": sha256_file(bundle["path"]),
        "ideal_features_sha256": sha256_file(ideal["path"]),
        "token_shape": list(bundle["tokens"].shape),
        "class_counts": [200, 200],
        "split_count": len(bundle["splits"]),
        "gate_counts": gate_counts,
        "gate_cap": gate_cap,
        "linear_readout": config["training"],
    }
    out = Path("dataset_and_config_validation.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def generate_dtc_params(q: int, seed: int, W: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    return {
        "J": rng.uniform(-0.9, 0.9, size=int(q) - 1).astype(np.float32),
        "h": rng.uniform(-float(W), float(W), size=int(q)).astype(np.float32),
    }


def _insert_barrier(qc: Any, qubits: Sequence[int]) -> None:
    barrier = getattr(qc, "barrier", None)
    if barrier is None:
        raise RuntimeError(
            "fold_factor=3 requires quark.circuit.QuantumCircuit.barrier so that "
            "the compiler cannot silently cancel B B^dagger B."
        )
    try:
        barrier(*[int(q) for q in qubits])
    except TypeError:
        barrier([int(q) for q in qubits])


def _rzz_block(qc: Any, left: int, right: int, angle: float) -> None:
    qc.cx(int(left), int(right))
    qc.rz(float(angle), int(right))
    qc.cx(int(left), int(right))


def _folded_rzz(
    qc: Any,
    left: int,
    right: int,
    angle: float,
    fold_factor: int,
    all_qubits: Sequence[int],
) -> None:
    if int(fold_factor) == 1:
        _rzz_block(qc, left, right, angle)
        return
    if int(fold_factor) != 3:
        raise ValueError("Only fold_factor 1 or 3 is supported")
    _rzz_block(qc, left, right, angle)
    _insert_barrier(qc, all_qubits)
    _rzz_block(qc, left, right, -float(angle))
    _insert_barrier(qc, all_qubits)
    _rzz_block(qc, left, right, angle)


def _resolve_layout(config: dict) -> tuple[int, list[int], list[int]]:
    q = int(config["circuit"]["q"])
    mode = config["cloud"].get("mapping_mode", "logical_auto")
    if mode == "logical_auto":
        return q, list(range(q)), list(range(q))
    if mode != "physical_direct":
        raise ValueError("mapping_mode must be logical_auto or physical_direct")
    physical = [int(value) for value in config["cloud"]["physical_qubits"]]
    if len(physical) != q or len(set(physical)) != q:
        raise ValueError("physical_qubits must contain q unique entries")
    backend_q = int(config["cloud"]["backend_num_qubits"])
    if min(physical) < 0 or max(physical) >= backend_q:
        raise ValueError("physical_qubits is outside backend register")
    return backend_q, physical, physical


def build_quantum_circuit(
    token_sequence: np.ndarray,
    config: dict,
    fold_factor: int,
) -> tuple[Any, dict]:
    from quark.circuit import QuantumCircuit

    q = int(config["circuit"]["q"])
    t = int(config["circuit"]["t"])
    layers = int(config["circuit"]["layers"])
    g = float(config["circuit"]["g"])
    W = float(config["circuit"]["W"])
    reservoir_seed = int(config["circuit"]["reservoir_seed"])
    register_q, circuit_qubits, measured_clbits = _resolve_layout(config)
    qc = QuantumCircuit(register_q, register_q)
    sequence = np.tanh(np.asarray(token_sequence, dtype=np.float32))
    if sequence.shape != (t, 3 * q):
        raise ValueError(f"Unexpected token shape {sequence.shape}")
    params = generate_dtc_params(q, reservoir_seed, W)
    for event in range(t):
        event_values = sequence[event].reshape(q, 3)
        for logical, physical in enumerate(circuit_qubits):
            theta, phi, lam = event_values[logical]
            qc.u(float(theta), float(phi), float(lam), int(physical))
        for _ in range(layers):
            for physical in circuit_qubits:
                qc.rx(g * math.pi, int(physical))
            for start in (0, 1):
                for logical in range(start, q - 1, 2):
                    _folded_rzz(
                        qc,
                        circuit_qubits[logical],
                        circuit_qubits[logical + 1],
                        float(params["J"][logical]),
                        int(fold_factor),
                        circuit_qubits,
                    )
            for logical, physical in enumerate(circuit_qubits):
                qc.rz(float(params["h"][logical]) * math.pi, int(physical))
    qc.measure(circuit_qubits, measured_clbits)
    metadata = {
        "q": q,
        "t": t,
        "layers": layers,
        "g": g,
        "W": W,
        "reservoir_seed": reservoir_seed,
        "J": params["J"].tolist(),
        "h": params["h"].tolist(),
        "J_distribution": "Uniform(-0.9,0.9)",
        "interaction_block": "CX-RZ(J_i)-CX",
        "fold_factor": int(fold_factor),
        "nominal_two_qubit_gates": two_qubit_gate_count(config, int(fold_factor)),
        "mapping_mode": config["cloud"].get("mapping_mode", "logical_auto"),
        "circuit_qubits": circuit_qubits,
        "measured_clbits": measured_clbits,
        "classical_num_bits": register_q,
        "full_fixed_horizon": True,
    }
    return qc, metadata


def dry_run(config_path: str | Path = "config.json", preview_items: int = 40) -> dict:
    config = load_config(config_path)
    validate_dataset_and_config(config_path)
    bundle = load_dataset(config)
    variant = config["variants"][0]
    qc, metadata = build_quantum_circuit(
        bundle["tokens"][0], config, int(variant["fold_factor"])
    )
    qlisp = qc.to_qlisp
    print(json.dumps(metadata, indent=2))
    if isinstance(qlisp, (list, tuple)):
        for item in list(qlisp)[: int(preview_items)]:
            print(item)
        print(f"QLISP items: {len(qlisp)}")
    else:
        print(str(qlisp)[:4000])
    return metadata


def _atomic_write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _task_specs(bundle: dict, config: dict) -> list[dict]:
    rng = np.random.default_rng(int(config["submission"]["order_seed"]))
    row_order = np.arange(len(bundle["indices"]), dtype=np.int64)
    rng.shuffle(row_order)
    variants = [dict(value) for value in config["variants"]]
    records = []
    if len(variants) > 1 and config["submission"].get("paired_interleave", False):
        for rank, row in enumerate(row_order.tolist()):
            local_variants = variants.copy()
            rng.shuffle(local_variants)
            for within_pair, variant in enumerate(local_variants):
                records.append(
                    {
                        "row": int(row),
                        "submission_rank": len(records),
                        "molecule_rank": rank,
                        "within_pair_rank": within_pair,
                        "variant": variant["name"],
                        "fold_factor": int(variant["fold_factor"]),
                    }
                )
    else:
        for variant in variants:
            for row in row_order.tolist():
                records.append(
                    {
                        "row": int(row),
                        "submission_rank": len(records),
                        "molecule_rank": len(records),
                        "within_pair_rank": 0,
                        "variant": variant["name"],
                        "fold_factor": int(variant["fold_factor"]),
                    }
                )
    return records


def _task_options(config: dict, metadata: dict) -> dict:
    cloud = config["cloud"]
    options = {
        "compiler": cloud.get("compiler", "quarkcircuit"),
        "correct": bool(cloud.get("correct", False)),
        "open_dd": cloud.get("open_dd"),
    }
    if metadata["mapping_mode"] == "physical_direct":
        options["target_qubits"] = metadata["circuit_qubits"]
    return {key: value for key, value in options.items() if value is not None}


def _task_manager(api_token: str):
    if not api_token or api_token == "REPLACE_WITH_NEW_API_TOKEN":
        raise RuntimeError("Replace API_TOKEN in the notebook before submission/collection")
    from quark import Task

    return Task(api_token)


def submit_jobs(
    api_token: str,
    config_path: str | Path = "config.json",
    max_submit: int | None = None,
) -> Path:
    config = load_config(config_path)
    validate_dataset_and_config(config_path)
    bundle = load_dataset(config)
    run_dir = Path("runs") / config["batch_id"]
    manifest_path = run_dir / "job_manifest.jsonl"
    existing = _read_jsonl(manifest_path)
    by_id = {record["circuit_id"]: record for record in existing}
    manager = _task_manager(api_token)
    submitted = 0
    limit = None if max_submit in (None, 0) else int(max_submit)
    repeat = int(config["cloud"]["repeat"])
    for spec in _task_specs(bundle, config):
        row = int(spec["row"])
        original_index = int(bundle["indices"][row])
        circuit_id = (
            f"{config['batch_id']}__{spec['variant']}__"
            f"seq{spec['submission_rank']:05d}__idx{original_index:05d}"
        )
        old = by_id.get(circuit_id)
        if old is not None and old.get("tid") is not None:
            continue
        if limit is not None and submitted >= limit:
            break
        qc, metadata = build_quantum_circuit(
            bundle["tokens"][row], config, int(spec["fold_factor"])
        )
        task = {
            "chip": config["cloud"]["chip"],
            "name": circuit_id[:120],
            "circuit": qc.to_qlisp,
            "compile": bool(config["cloud"].get("compile", True)),
            "correct": bool(config["cloud"].get("correct", False)),
            "options": _task_options(config, metadata),
        }
        tid = manager.run(task, repeat=repeat)
        record = {
            **spec,
            **metadata,
            "protocol_version": PROTOCOL_VERSION,
            "batch_id": config["batch_id"],
            "circuit_id": circuit_id,
            "original_index": original_index,
            "label_for_offline_analysis_only": int(bundle["labels"][row]),
            "smiles_for_manifest_only": str(bundle["smiles"][row]),
            "token_hash": str(bundle["token_hashes"][row]),
            "tid": tid,
            "status": "Submitted",
            "chip": config["cloud"]["chip"],
            "repeat": repeat,
            "shots_approx": repeat * 1024,
            "compile": bool(config["cloud"].get("compile", True)),
            "correct": bool(config["cloud"].get("correct", False)),
            "task_options": task["options"],
            "api_token_persisted": False,
        }
        by_id[circuit_id] = record
        ordered = sorted(by_id.values(), key=lambda value: int(value["submission_rank"]))
        _atomic_write_jsonl(ordered, manifest_path)
        submitted += 1
        print(
            f"submitted {submitted}: variant={spec['variant']} "
            f"idx={original_index} tid={tid}"
        )
        delay = float(config["submission"].get("sleep_seconds", 0.0))
        if delay > 0:
            time.sleep(delay)
    print(f"Submitted {submitted} new circuits; manifest={manifest_path}")
    return manifest_path


def _extract_counts(result: Any) -> Mapping[str, int | float] | None:
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


def _parse_bitstring(key: Any) -> str:
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


def _measured_bits(key: Any, record: dict) -> np.ndarray:
    raw = _parse_bitstring(key)
    q = int(record["q"])
    measured = [int(value) for value in record["measured_clbits"]]
    classical_n = int(record["classical_num_bits"])
    if max(measured) >= q or len(raw) > q:
        full = raw.zfill(max(classical_n, len(raw)))
        return np.asarray([int(full[-1 - c]) for c in measured], dtype=np.int8)
    compact = raw.zfill(q)[-q:]
    return np.asarray([int(compact[-1 - i]) for i in range(q)], dtype=np.int8)


def counts_to_features(counts: Mapping[str, int | float], record: dict) -> np.ndarray:
    q = int(record["q"])
    total = float(sum(float(value) for value in counts.values()))
    if total <= 0:
        raise ValueError("Counts sum is not positive")
    z1 = np.zeros(q, dtype=np.float64)
    z2 = np.zeros((q, q), dtype=np.float64)
    for key, count in counts.items():
        bits = _measured_bits(key, record)
        signs = 1.0 - 2.0 * bits.astype(np.float64)
        probability = float(count) / total
        z1 += probability * signs
        z2 += probability * np.outer(signs, signs)
    pairs = [z2[i, j] for i in range(q) for j in range(i + 1, q)]
    return np.concatenate([z1, np.asarray(pairs)]).astype(np.float32)


def collect_jobs(
    api_token: str,
    config_path: str | Path = "config.json",
    require_complete: bool = True,
) -> dict[str, Path]:
    config = load_config(config_path)
    bundle = load_dataset(config)
    run_dir = Path("runs") / config["batch_id"]
    manifest_path = run_dir / "job_manifest.jsonl"
    records = _read_jsonl(manifest_path)
    if not records:
        raise FileNotFoundError(manifest_path)
    raw_dir = run_dir / "raw_results"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manager = _task_manager(api_token)
    by_variant: dict[str, list[tuple[dict, np.ndarray]]] = {
        variant["name"]: [] for variant in config["variants"]
    }
    for record in records:
        tid = record.get("tid")
        if tid is None:
            continue
        try:
            result = manager.result(tid)
        except Exception as error:
            record["status"] = f"FetchFailed: {error}"
            print(record["circuit_id"], record["status"])
            continue
        raw_path = raw_dir / f"{record['circuit_id']}__tid{tid}.json"
        raw_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        counts = _extract_counts(result)
        record["raw_result_path"] = str(raw_path)
        record["status"] = (
            result.get("status") or result.get("task_status") or result.get("state") or "Unknown"
            if isinstance(result, dict)
            else "Unknown"
        )
        if counts:
            feature = counts_to_features(counts, record)
            record["feature_cached"] = True
            record["shots_downloaded"] = int(round(sum(float(v) for v in counts.values())))
            by_variant[record["variant"]].append((record, feature))
        else:
            record["feature_cached"] = False
            print(f"no counts: {record['circuit_id']} status={record['status']}")
    _atomic_write_jsonl(
        sorted(records, key=lambda value: int(value["submission_rank"])),
        manifest_path,
    )

    outputs = {}
    dataset_order = {int(value): row for row, value in enumerate(bundle["indices"])}
    for variant in config["variants"]:
        name = variant["name"]
        items = by_variant[name]
        if require_complete and len(items) != len(bundle["indices"]):
            raise RuntimeError(
                f"{name}: collected {len(items)}/400 circuits. Re-run collection later."
            )
        items.sort(key=lambda item: dataset_order[int(item[0]["original_index"])])
        if not items:
            continue
        feature_path = run_dir / f"hardware_features__{name}.npz"
        item_records = [item[0] for item in items]
        np.savez_compressed(
            feature_path,
            indices=np.asarray([item["original_index"] for item in item_records], dtype=np.int64),
            labels=np.asarray(
                [item["label_for_offline_analysis_only"] for item in item_records],
                dtype=np.int64,
            ),
            token_hashes=np.asarray([item["token_hash"] for item in item_records], dtype=str),
            features=np.stack([item[1] for item in items]).astype(np.float32),
            tids=np.asarray([str(item["tid"]) for item in item_records], dtype=str),
            shots=np.asarray([int(item["shots_downloaded"]) for item in item_records], dtype=np.int64),
            variant=np.asarray(name),
            batch_id=np.asarray(config["batch_id"]),
            config_json=np.asarray(json.dumps(config, sort_keys=True)),
        )
        outputs[name] = feature_path
        print(f"[saved] {feature_path} ({len(items)} rows)")
    return outputs


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "loss": float(log_loss(labels, probability, labels=[0, 1])),
        "acc": float(accuracy_score(labels, prediction)),
    }


def _split_rows(bundle: dict, seed: int) -> dict[str, np.ndarray]:
    row_by_index = {
        int(original): row for row, original in enumerate(bundle["indices"].tolist())
    }
    split = bundle["splits"][str(int(seed))]
    return {
        name: np.asarray([row_by_index[int(value)] for value in split[f"{name}_idx"]], dtype=np.int64)
        for name in ("train", "val", "test")
    }


def _train_affine(
    features: np.ndarray,
    labels: np.ndarray,
    split: dict[str, np.ndarray],
    seed: int,
    training: dict,
) -> tuple[dict, dict[str, torch.Tensor], pd.DataFrame]:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    requested = str(training.get("device", "auto"))
    device = torch.device(
        "cuda:0" if requested == "auto" and torch.cuda.is_available()
        else "cpu" if requested == "auto"
        else requested
    )
    x = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)).to(device)
    y = torch.from_numpy(labels.astype(np.float32)).to(device)
    model = torch.nn.Linear(x.shape[1], 1, bias=True).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=0.0,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    train_idx = torch.from_numpy(split["train"]).to(device)
    val_idx = torch.from_numpy(split["val"]).to(device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    best_key = None
    best_state = None
    best_epoch = None
    best_val = None
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        order = train_idx[
            torch.randperm(len(train_idx), generator=generator, device=device)
        ]
        losses = []
        for start in range(0, len(order), int(training["batch_size"])):
            rows = order[start : start + int(training["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[rows]).squeeze(1), y[rows])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits = model(x[val_idx]).squeeze(1).cpu().numpy()
        val = _metrics(labels[split["val"]], val_logits)
        key = (val["auc"], -val["loss"], val["acc"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_val = val
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_auc": val["auc"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
            }
        )
    if best_state is None:
        raise RuntimeError("No valid validation checkpoint")
    return (
        {
            "best_epoch": int(best_epoch),
            "val_auc": float(best_val["auc"]),
            "val_loss": float(best_val["loss"]),
            "val_acc": float(best_val["acc"]),
            "device": str(device),
        },
        best_state,
        pd.DataFrame(history),
    )


def _logits(features: np.ndarray, state: dict[str, torch.Tensor]) -> np.ndarray:
    weight = state["weight"].detach().cpu().numpy().reshape(-1).astype(np.float64)
    bias = float(state["bias"].detach().cpu().numpy().reshape(-1)[0])
    return np.asarray(features, dtype=np.float64) @ weight + bias


def train_and_analyze(config_path: str | Path = "config.json") -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    bundle = load_dataset(config)
    ideal = load_ideal(config)
    if not np.array_equal(bundle["indices"], ideal["indices"]):
        raise AssertionError("Ideal feature row mismatch")
    run_dir = Path("runs") / config["batch_id"]
    feature_bundles = {}
    for variant in config["variants"]:
        path = run_dir / f"hardware_features__{variant['name']}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as package:
            indices = package["indices"].astype(np.int64)
            if not np.array_equal(indices, bundle["indices"]):
                raise AssertionError(f"{path}: incomplete or reordered rows")
            feature_bundles[variant["name"]] = package["features"].astype(np.float32)

    checkpoint_dir = run_dir / "checkpoints"
    history_dir = run_dir / "histories"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = []
    for seed in config["training"]["seeds"]:
        seed = int(seed)
        split = _split_rows(bundle, seed)
        ideal_fit, ideal_state, ideal_history = _train_affine(
            ideal["features"],
            bundle["labels"],
            split,
            seed,
            config["training"],
        )
        torch.save(
            {
                "model": "torch.nn.Linear",
                "state_dict": ideal_state,
                "seed": seed,
                "domain": "ideal",
                "selection_rule": SELECTION_RULE,
                "standardization": False,
                "weight_decay": 0.0,
                "protocol_version": PROTOCOL_VERSION,
            },
            checkpoint_dir / f"ideal_seed{seed:03d}_best.pt",
        )
        ideal_history.to_csv(
            history_dir / f"ideal_seed{seed:03d}.csv", index=False
        )
        test = split["test"]
        ideal_scores = _logits(ideal["features"][test], ideal_state)
        ideal_test = _metrics(bundle["labels"][test], ideal_scores)
        for variant in config["variants"]:
            name = variant["name"]
            hardware = feature_bundles[name]
            frozen_scores = _logits(hardware[test], ideal_state)
            frozen_test = _metrics(bundle["labels"][test], frozen_scores)
            score_pearson = float(pearsonr(ideal_scores, frozen_scores).statistic)
            score_spearman = float(spearmanr(ideal_scores, frozen_scores).statistic)
            hardware_fit, hardware_state, hardware_history = _train_affine(
                hardware,
                bundle["labels"],
                split,
                seed,
                config["training"],
            )
            torch.save(
                {
                    "model": "torch.nn.Linear",
                    "state_dict": hardware_state,
                    "seed": seed,
                    "domain": f"hardware:{name}",
                    "selection_rule": SELECTION_RULE,
                    "standardization": False,
                    "weight_decay": 0.0,
                    "protocol_version": PROTOCOL_VERSION,
                },
                checkpoint_dir / f"hardware_{name}_seed{seed:03d}_best.pt",
            )
            hardware_history.to_csv(
                history_dir / f"hardware_{name}_seed{seed:03d}.csv",
                index=False,
            )
            hardware_refit_scores = _logits(hardware[test], hardware_state)
            hardware_refit_test = _metrics(
                bundle["labels"][test], hardware_refit_scores
            )
            rows.append(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "batch_id": config["batch_id"],
                    "variant": name,
                    "fold_factor": int(variant["fold_factor"]),
                    "seed": seed,
                    "n_test": len(test),
                    "ideal_best_epoch": ideal_fit["best_epoch"],
                    "ideal_val_auc": ideal_fit["val_auc"],
                    "ideal_test_auc": ideal_test["auc"],
                    "frozen_ideal_to_hardware_test_auc": frozen_test["auc"],
                    "frozen_delta_auc": frozen_test["auc"] - ideal_test["auc"],
                    "frozen_test_loss": frozen_test["loss"],
                    "frozen_test_acc": frozen_test["acc"],
                    "task_score_pearson": score_pearson,
                    "task_score_spearman": score_spearman,
                    "hardware_refit_best_epoch": hardware_fit["best_epoch"],
                    "hardware_refit_val_auc": hardware_fit["val_auc"],
                    "hardware_refit_test_auc": hardware_refit_test["auc"],
                    "hardware_refit_test_loss": hardware_refit_test["loss"],
                    "hardware_refit_test_acc": hardware_refit_test["acc"],
                    "readout": "single unstandardized unregularized torch.nn.Linear(d,1)",
                    "selection_rule": SELECTION_RULE,
                }
            )
            predictions.extend(
                {
                    "batch_id": config["batch_id"],
                    "variant": name,
                    "seed": seed,
                    "original_index": int(bundle["indices"][row]),
                    "label": int(bundle["labels"][row]),
                    "ideal_score": float(ideal_score),
                    "frozen_hardware_score": float(frozen_score),
                    "hardware_refit_score": float(refit_score),
                }
                for row, ideal_score, frozen_score, refit_score in zip(
                    test,
                    ideal_scores,
                    frozen_scores,
                    hardware_refit_scores,
                )
            )
    metrics = pd.DataFrame(rows)
    predictions_frame = pd.DataFrame(predictions)
    metrics.to_csv(run_dir / "linear_readout_metrics_by_seed.csv", index=False)
    predictions_frame.to_csv(run_dir / "test_scores.csv", index=False)
    summary = (
        metrics.groupby(["batch_id", "variant", "fold_factor"], as_index=False)
        .agg(
            ideal_test_auc_mean=("ideal_test_auc", "mean"),
            ideal_test_auc_std=("ideal_test_auc", "std"),
            frozen_hardware_auc_mean=("frozen_ideal_to_hardware_test_auc", "mean"),
            frozen_hardware_auc_std=("frozen_ideal_to_hardware_test_auc", "std"),
            frozen_delta_auc_mean=("frozen_delta_auc", "mean"),
            frozen_delta_auc_std=("frozen_delta_auc", "std"),
            task_score_pearson_mean=("task_score_pearson", "mean"),
            task_score_spearman_mean=("task_score_spearman", "mean"),
            hardware_refit_auc_mean=("hardware_refit_test_auc", "mean"),
            hardware_refit_auc_std=("hardware_refit_test_auc", "std"),
        )
    )
    summary.to_csv(run_dir / "linear_readout_summary.csv", index=False)
    print(summary.to_string(index=False))
    return metrics, summary


def _resample_counts(
    counts: Mapping[str, int | float],
    shots: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    states = np.asarray(list(counts.keys()), dtype=str)
    values = np.asarray([float(counts[key]) for key in counts], dtype=np.float64)
    probabilities = values / values.sum()
    sampled = rng.multinomial(int(shots), probabilities)
    return {
        str(state): int(value)
        for state, value in zip(states, sampled)
        if value > 0
    }


def shot_noise_bootstrap(
    config_path: str | Path = "config.json",
    budgets: Sequence[int] = (128, 256, 512, 1024, 2048, 4096, 10240),
    n_boot: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    bundle = load_dataset(config)
    ideal = load_ideal(config)
    run_dir = Path("runs") / config["batch_id"]
    records = _read_jsonl(run_dir / "job_manifest.jsonl")
    complete = [
        record
        for record in records
        if record.get("feature_cached") and record.get("raw_result_path")
    ]
    by_variant: dict[str, list[tuple[dict, Mapping[str, int | float]]]] = {}
    for record in complete:
        payload = json.loads(Path(record["raw_result_path"]).read_text(encoding="utf-8"))
        counts = _extract_counts(payload)
        if not counts:
            continue
        by_variant.setdefault(record["variant"], []).append((record, counts))
    row_by_index = {
        int(index): row for row, index in enumerate(bundle["indices"].tolist())
    }
    draws = []
    for variant_index, variant in enumerate(config["variants"]):
        name = variant["name"]
        items = by_variant.get(name, [])
        if len(items) != 400:
            raise RuntimeError(f"{name}: expected 400 raw count records, got {len(items)}")
        ordered = sorted(items, key=lambda item: row_by_index[int(item[0]["original_index"])])
        minimum_shots = min(
            int(round(sum(float(v) for v in counts.values())))
            for _, counts in ordered
        )
        usable_budgets = [int(value) for value in budgets if int(value) <= minimum_shots]
        ideal_states = {}
        for seed in config["training"]["seeds"]:
            checkpoint = torch.load(
                run_dir / "checkpoints" / f"ideal_seed{int(seed):03d}_best.pt",
                map_location="cpu",
                weights_only=False,
            )
            ideal_states[int(seed)] = checkpoint["state_dict"]
        for budget in usable_budgets:
            for replicate in range(int(n_boot)):
                rng = np.random.default_rng(
                    8_100_000
                    + variant_index * 1_000_000
                    + int(budget) * 1009
                    + replicate
                )
                sampled_features = np.stack(
                    [
                        counts_to_features(
                            _resample_counts(counts, budget, rng),
                            record,
                        )
                        for record, counts in ordered
                    ]
                )
                for seed, state in ideal_states.items():
                    split = _split_rows(bundle, seed)
                    test = split["test"]
                    ideal_scores = _logits(ideal["features"][test], state)
                    hardware_scores = _logits(sampled_features[test], state)
                    ideal_auc = _metrics(bundle["labels"][test], ideal_scores)["auc"]
                    hardware_auc = _metrics(
                        bundle["labels"][test], hardware_scores
                    )["auc"]
                    draws.append(
                        {
                            "batch_id": config["batch_id"],
                            "variant": name,
                            "shot_budget": budget,
                            "bootstrap_replicate": replicate,
                            "seed": seed,
                            "ideal_auc": ideal_auc,
                            "frozen_hardware_auc": hardware_auc,
                            "delta_auc": hardware_auc - ideal_auc,
                        }
                    )
        print(f"shot bootstrap complete: {name}, budgets={usable_budgets}")
    draws_frame = pd.DataFrame(draws)
    draws_frame.to_csv(run_dir / "shot_bootstrap_draws.csv", index=False)
    summary = (
        draws_frame.groupby(["batch_id", "variant", "shot_budget"], as_index=False)
        .agg(
            frozen_hardware_auc_mean=("frozen_hardware_auc", "mean"),
            frozen_hardware_auc_sd=("frozen_hardware_auc", "std"),
            frozen_hardware_auc_q025=(
                "frozen_hardware_auc",
                lambda value: float(np.quantile(value, 0.025)),
            ),
            frozen_hardware_auc_q975=(
                "frozen_hardware_auc",
                lambda value: float(np.quantile(value, 0.975)),
            ),
            delta_auc_mean=("delta_auc", "mean"),
            delta_auc_q025=("delta_auc", lambda value: float(np.quantile(value, 0.025))),
            delta_auc_q975=("delta_auc", lambda value: float(np.quantile(value, 0.975))),
        )
    )
    summary.to_csv(run_dir / "shot_bootstrap_summary.csv", index=False)
    print(summary.to_string(index=False))
    return draws_frame, summary
