#!/usr/bin/env python3
"""Build two self-contained Quafu acquisition notebooks and upload archives."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
QRC_ROOT = HERE.parent
SOURCE_ROOT = QRC_ROOT / "FINAL" / "15_mainline_hardware_noise" / "generated"
SOURCE_DATASET = SOURCE_ROOT / "bace_bal400_canonical_v3_q6_t6.npz"
SOURCE_IDEAL = SOURCE_ROOT / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz"
DATA_DIR = HERE / "data"

DATASET_NAME = SOURCE_DATASET.name
IDEAL_NAME = SOURCE_IDEAL.name
PROTOCOL_VERSION = "2026-07-28-mainline-canonical-v3-q6t6l3-cloud-acquisition-v2"

BACKENDS = {
    "baihua": {
        "chip": "Baihua",
        "backend_num_qubits": 156,
        "chain": [72, 73, 74, 75, 76, 77],
        "calibrated_at": "2026-07-28 19:23:48",
        "order_seed": 20260728,
    },
    "shenglian": {
        "chip": "Shenglian",
        "backend_num_qubits": 84,
        "chain": [27, 34, 41, 48, 54, 61],
        "calibrated_at": "2026-07-28 19:25:54",
        "order_seed": 20260728,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_lines(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(source)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def notebook_payload(backend_key: str, dataset_sha256: str) -> dict:
    backend = BACKENDS[backend_key]
    chip = backend["chip"]
    config = f"""# Edit only ACTION, then run all cells.
# "submit": randomized, resumable hardware submission.
# "collect": fetch raw results and create one download ZIP.
ACTION = "submit"  # "submit" or "collect"

from getpass import getpass
API_TOKEN = getpass("New Quafu API token: ")

BACKEND_KEY = {backend_key!r}
CHIP = {chip!r}
BACKEND_NUM_QUBITS = {backend["backend_num_qubits"]}
PHYSICAL_QUBITS = {backend["chain"]!r}
CALIBRATED_AT = {backend["calibrated_at"]!r}
SUBMISSION_ORDER_SEED = {backend["order_seed"]}

# Frozen manuscript-mainline circuit.
Q = 6
T = 6
LAYERS = 3
G = 0.84
W = 0.60
RESERVOIR_SEED = 0
REPEAT = 1
DATASET_FILE = {DATASET_NAME!r}
EXPECTED_DATASET_SHA256 = {dataset_sha256!r}

RESULT_ROOT_NAME = f"{{BACKEND_KEY}}_q6t6l3_cloud_results"
MANIFEST_NAME = f"{{BACKEND_KEY}}_q6t6l3_job_manifest.jsonl"
RAW_DIR_NAME = f"raw_counts_{{BACKEND_KEY}}_q6t6l3"
DOWNLOAD_ZIP_NAME = f"{{BACKEND_KEY}}_q6t6l3_cloud_results.zip"
"""

    imports = """import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

PROTOCOL_VERSION = "2026-07-28-mainline-canonical-v3-q6t6l3-cloud-acquisition-v2"
RESULT_ROOT = Path(RESULT_ROOT_NAME)
MANIFEST_PATH = RESULT_ROOT / MANIFEST_NAME
RAW_DIR = RESULT_ROOT / RAW_DIR_NAME
ORDER_PATH = RESULT_ROOT / f"{BACKEND_KEY}_q6t6l3_submission_order.json"
SUMMARY_PATH = RESULT_ROOT / f"{BACKEND_KEY}_q6t6l3_collection_summary.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def scalar_text(value):
    array = np.asarray(value)
    return str(array.item()) if array.shape == () else str(value)
"""

    dataset_code = """def load_processed_dataset():
    path = Path(DATASET_FILE)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Upload the notebook and processed NPZ together."
        )
    actual_hash = sha256_file(path)
    if actual_hash != EXPECTED_DATASET_SHA256:
        raise AssertionError(
            f"Dataset SHA-256 mismatch: {actual_hash} != {EXPECTED_DATASET_SHA256}"
        )
    with np.load(path, allow_pickle=False) as package:
        bundle = {
            "indices": package["indices"].astype(np.int64),
            "tokens": package["tokens"].astype(np.float32),
            "token_hashes": package["token_hashes"].astype(str),
            "schema": scalar_text(package["token_schema_version"]),
            "metadata": json.loads(scalar_text(package["metadata_json"])),
        }
    if bundle["tokens"].shape != (400, T, 3 * Q):
        raise AssertionError(bundle["tokens"].shape)
    if bundle["schema"] != "local-bfs-canonical-isomeric-reparse-stable-v3":
        raise AssertionError(bundle["schema"])
    if bundle["metadata"].get("global_descriptors") is not False:
        raise AssertionError("Global descriptors are forbidden")
    if bundle["metadata"].get("task_guided_traversal") is not False:
        raise AssertionError("Task-guided traversal is forbidden")
    return bundle


BUNDLE = load_processed_dataset()
print(
    {
        "backend": CHIP,
        "dataset_shape": BUNDLE["tokens"].shape,
        "physical_chain": PHYSICAL_QUBITS,
        "calibration_timestamp": CALIBRATED_AT,
    }
)
"""

    topology_circuit = """def norm_edge(left, right):
    left, right = int(left), int(right)
    return (left, right) if left < right else (right, left)


def baihua_edges():
    edges = set()
    rows, cols = 12, 13
    for row in range(rows):
        for col in range(cols - 1):
            edges.add(norm_edge(row * cols + col, row * cols + col + 1))
    for row in range(rows - 1):
        vertical_cols = (2, 6, 10) if row % 2 == 0 else (0, 4, 8, 12)
        for col in vertical_cols:
            edges.add(norm_edge(row * cols + col, (row + 1) * cols + col))
    return edges


def shenglian_edges():
    # Twelve seven-qubit rows. Even inter-row bands contain both diagonals;
    # odd bands contain the displayed down-left connection only.
    edges = set()
    rows, cols = 12, 7
    for row in range(rows - 1):
        for col in range(cols):
            upper = row * cols + col
            lower_same = (row + 1) * cols + col
            edges.add(norm_edge(upper, lower_same))
            if row % 2 == 0 and col > 0:
                edges.add(norm_edge(upper, (row + 1) * cols + col - 1))
    return edges


def validate_chain():
    if len(PHYSICAL_QUBITS) != Q or len(set(PHYSICAL_QUBITS)) != Q:
        raise AssertionError("Physical chain must contain Q unique qubits")
    if min(PHYSICAL_QUBITS) < 0 or max(PHYSICAL_QUBITS) >= BACKEND_NUM_QUBITS:
        raise AssertionError("Physical chain is outside the backend register")
    coupling = baihua_edges() if BACKEND_KEY == "baihua" else shenglian_edges()
    missing = [
        norm_edge(left, right)
        for left, right in zip(PHYSICAL_QUBITS[:-1], PHYSICAL_QUBITS[1:])
        if norm_edge(left, right) not in coupling
    ]
    if missing:
        raise AssertionError(f"Nonconnected physical chain; missing={missing}")
    return True


def dtc_parameters():
    rng = np.random.default_rng(RESERVOIR_SEED)
    return {
        "J": rng.uniform(-0.9, 0.9, size=Q - 1).astype(np.float32),
        "h": rng.uniform(-W, W, size=Q).astype(np.float32),
    }


def build_circuit(token_sequence):
    from quark.circuit import QuantumCircuit

    validate_chain()
    sequence = np.tanh(np.asarray(token_sequence, dtype=np.float32))
    if sequence.shape != (T, 3 * Q):
        raise AssertionError(sequence.shape)
    params = dtc_parameters()
    qc = QuantumCircuit(BACKEND_NUM_QUBITS, BACKEND_NUM_QUBITS)
    for event in range(T):
        values = sequence[event].reshape(Q, 3)
        for logical, physical in enumerate(PHYSICAL_QUBITS):
            theta, phi, lam = values[logical]
            qc.u(float(theta), float(phi), float(lam), int(physical))
        for _ in range(LAYERS):
            for physical in PHYSICAL_QUBITS:
                qc.rx(G * math.pi, int(physical))
            for start in (0, 1):
                for logical in range(start, Q - 1, 2):
                    left = int(PHYSICAL_QUBITS[logical])
                    right = int(PHYSICAL_QUBITS[logical + 1])
                    qc.cx(left, right)
                    qc.rz(float(params["J"][logical]), right)
                    qc.cx(left, right)
            for logical, physical in enumerate(PHYSICAL_QUBITS):
                qc.rz(float(params["h"][logical]) * math.pi, int(physical))
    qc.measure(PHYSICAL_QUBITS, PHYSICAL_QUBITS)
    metadata = {
        "q": Q,
        "t": T,
        "layers": LAYERS,
        "g": G,
        "W": W,
        "reservoir_seed": RESERVOIR_SEED,
        "J": params["J"].tolist(),
        "h": params["h"].tolist(),
        "interaction": "CX-RZ(J_i)-CX",
        "circuit_qubits": list(PHYSICAL_QUBITS),
        "measured_clbits": list(PHYSICAL_QUBITS),
        "classical_num_bits": BACKEND_NUM_QUBITS,
        "nominal_two_qubit_gates": 2 * T * LAYERS * (Q - 1),
    }
    if metadata["nominal_two_qubit_gates"] != 180:
        raise AssertionError(metadata["nominal_two_qubit_gates"])
    return qc, metadata


validate_chain()
print("Topology validation passed; nominal CX count = 180")
"""

    acquisition = """def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\\n")
    temporary.replace(path)


def task_manager():
    if not API_TOKEN:
        raise RuntimeError("API token is empty")
    from quark import Task
    return Task(API_TOKEN)


def randomized_specs():
    rng = np.random.default_rng(SUBMISSION_ORDER_SEED)
    rows = rng.permutation(len(BUNDLE["indices"])).astype(int)
    specs = [
        {
            "row": int(row),
            "submission_rank": int(rank),
            "original_index": int(BUNDLE["indices"][row]),
        }
        for rank, row in enumerate(rows)
    ]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ORDER_PATH.write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "backend": CHIP,
                "seed": SUBMISSION_ORDER_SEED,
                "label_blind_random_permutation": True,
                "original_indices_in_submission_order": [
                    value["original_index"] for value in specs
                ],
            },
            indent=2,
        )
        + "\\n",
        encoding="utf-8",
    )
    return specs


def submit_all_randomized():
    manager = task_manager()
    existing = read_jsonl(MANIFEST_PATH)
    by_index = {int(record["original_index"]): record for record in existing}
    submitted = 0
    specs = randomized_specs()
    for spec in specs:
        original_index = int(spec["original_index"])
        old = by_index.get(original_index)
        if old is not None and old.get("tid") is not None:
            continue
        row = int(spec["row"])
        qc, metadata = build_circuit(BUNDLE["tokens"][row])
        circuit_id = (
            f"{BACKEND_KEY}_q6t6l3_rank{spec['submission_rank']:05d}"
            f"_id{original_index:05d}"
        )
        options = {
            "compiler": "quarkcircuit",
            "correct": False,
            "target_qubits": list(PHYSICAL_QUBITS),
        }
        task = {
            "chip": CHIP,
            "name": circuit_id,
            "circuit": qc.to_qlisp,
            "compile": True,
            "correct": False,
            "options": options,
        }
        tid = manager.run(task, repeat=REPEAT)
        record = {
            **spec,
            **metadata,
            "protocol_version": PROTOCOL_VERSION,
            "backend_key": BACKEND_KEY,
            "chip": CHIP,
            "physical_qubits": list(PHYSICAL_QUBITS),
            "calibration_timestamp_from_image": CALIBRATED_AT,
            "dataset_file": DATASET_FILE,
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "token_hash": str(BUNDLE["token_hashes"][row]),
            "circuit_id": circuit_id,
            "tid": str(tid),
            "repeat": REPEAT,
            "status": "Submitted",
            "compile": True,
            "task_options": options,
        }
        by_index[original_index] = record
        ordered = sorted(by_index.values(), key=lambda value: int(value["submission_rank"]))
        atomic_write_jsonl(ordered, MANIFEST_PATH)
        submitted += 1
        print(
            f"[{submitted:03d}] rank={spec['submission_rank']:03d} "
            f"id={original_index} tid={tid}"
        )
    print(f"Submitted {submitted} new jobs; manifest={MANIFEST_PATH}")
    return MANIFEST_PATH


def extract_counts(result):
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


def refresh_download_zip():
    archive_base = str(Path(DOWNLOAD_ZIP_NAME).with_suffix(""))
    archive_path = Path(
        shutil.make_archive(
            archive_base,
            "zip",
            root_dir=RESULT_ROOT.parent,
            base_dir=RESULT_ROOT.name,
        )
    )
    print(f"Download this file: {archive_path.resolve()}")
    return archive_path


def collect_raw_results():
    records = read_jsonl(MANIFEST_PATH)
    if not records:
        raise FileNotFoundError(
            f"{MANIFEST_PATH} is missing. Run ACTION='submit' first."
        )
    manager = task_manager()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    completed = 0
    pending = 0
    failed = 0
    for position, record in enumerate(
        sorted(records, key=lambda value: int(value["submission_rank"]))
    ):
        raw_name = record.get(
            "raw_file_name",
            f"rank{int(record['submission_rank']):05d}"
            f"_id{int(record['original_index']):05d}"
            f"_tid{record['tid']}.json",
        )
        raw_path = RAW_DIR / raw_name
        result = None
        if raw_path.is_file():
            try:
                result = json.loads(raw_path.read_text(encoding="utf-8"))
            except Exception:
                result = None
        if result is None or not extract_counts(result):
            try:
                result = manager.result(record["tid"])
            except Exception as error:
                record["status"] = f"FetchFailed: {error}"
                failed += 1
                atomic_write_jsonl(records, MANIFEST_PATH)
                print(record["circuit_id"], record["status"])
                continue
            raw_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=json_default),
                encoding="utf-8",
            )
        counts = extract_counts(result)
        record["raw_file_name"] = raw_name
        record["raw_result_path_cloud"] = str(raw_path)
        if counts:
            record["status"] = "CountsDownloaded"
            record["shots_downloaded"] = int(
                round(sum(float(value) for value in counts.values()))
            )
            completed += 1
        else:
            record["status"] = (
                result.get("status")
                or result.get("task_status")
                or result.get("state")
                or "PendingNoCounts"
                if isinstance(result, dict)
                else "PendingNoCounts"
            )
            pending += 1
        atomic_write_jsonl(records, MANIFEST_PATH)
        print(
            f"[{position + 1:03d}/{len(records)}] "
            f"rank={record['submission_rank']} status={record['status']}"
        )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "backend": CHIP,
        "physical_qubits": list(PHYSICAL_QUBITS),
        "submitted_records": len(records),
        "counts_downloaded": completed,
        "pending_no_counts": pending,
        "fetch_failed": failed,
        "complete": completed == 400,
        "cloud_processing": "raw result retrieval only; no feature extraction or training",
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\\n", encoding="utf-8"
    )
    archive = refresh_download_zip()
    print(json.dumps(summary, indent=2))
    return archive
"""

    execute = """if ACTION == "submit":
    OUTPUT = submit_all_randomized()
elif ACTION == "collect":
    OUTPUT = collect_raw_results()
else:
    raise ValueError("ACTION must be 'submit' or 'collect'")

OUTPUT
"""

    title = f"""# {chip}: mainline canonical-v3 Q6/T6/L3 acquisition

This notebook is self-contained. It reads the already processed 400-sample
NPZ, uses the fixed image-selected physical chain, submits circuits in a
deterministic label-blind random order, and later downloads raw Quafu results.

It does **not** tokenize molecules, create splits, reconstruct observables,
train a decoder, calculate AUC, or make manuscript figures. Those operations
are reserved for the local analysis folder.
"""
    return {
        "cells": [
            markdown_cell(title),
            code_cell(config),
            code_cell(imports),
            code_cell(dataset_code),
            code_cell(topology_circuit),
            code_cell(acquisition),
            code_cell(execute),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def make_zip(folder: Path) -> Path:
    archive = HERE / f"{folder.name}.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(HERE).as_posix())
    return archive


def make_total_zip() -> Path:
    archive = HERE.parent / f"{HERE.name}.zip"
    excluded_parts = {"__pycache__", "extracted_baihua", "extracted_shenglian"}
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as handle:
        for path in sorted(HERE.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(HERE.parent)
            if any(part in excluded_parts for part in relative.parts):
                continue
            if path.suffix == ".pyc":
                continue
            handle.write(path, relative.as_posix())
    return archive


def main() -> None:
    for source in (SOURCE_DATASET, SOURCE_IDEAL):
        if not source.is_file():
            raise FileNotFoundError(source)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DATASET, DATA_DIR / DATASET_NAME)
    shutil.copy2(SOURCE_IDEAL, DATA_DIR / IDEAL_NAME)
    dataset_hash = sha256_file(SOURCE_DATASET)
    ideal_hash = sha256_file(SOURCE_IDEAL)

    upload_archives = {}
    for backend_key, backend in BACKENDS.items():
        folder = HERE / f"upload_{backend_key}"
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_DATASET, folder / DATASET_NAME)
        notebook_name = (
            f"{backend['chip']}_Q6T6L3_random_submit_collect_raw.ipynb"
        )
        write_notebook(
            folder / notebook_name,
            notebook_payload(backend_key, dataset_hash),
        )
        upload_archives[backend_key] = make_zip(folder)

    topology_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_rule": (
            "image-derived connected Q6 path; darker rendered node/edge means "
            "higher fidelity according to the user-provided maps"
        ),
        "limitation": (
            "relative rendered colors, not exported numerical calibration table; "
            "recheck if live calibration timestamp changes"
        ),
        "backends": BACKENDS,
    }
    (HERE / "topology" / "chain_selection.json").write_text(
        json.dumps(topology_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "processed_dataset": {
            "path": f"data/{DATASET_NAME}",
            "sha256": dataset_hash,
            "shape": [400, 6, 18],
            "class_counts": [200, 200],
        },
        "ideal_features_local_only": {
            "path": f"data/{IDEAL_NAME}",
            "sha256": ideal_hash,
            "shape": [400, 21],
        },
        "cloud_upload_archives": {
            key: {
                "path": path.name,
                "sha256": sha256_file(path),
                "physical_chain": BACKENDS[key]["chain"],
            }
            for key, path in upload_archives.items()
        },
        "cloud_contract": "submit randomized circuits or retrieve raw results only",
        "local_contract": "all count parsing, feature extraction, training, and comparison",
        "code_provenance": {
            "dataset_processing": {
                "path": "dataset_processing/prepare_dataset.py",
                "sha256": sha256_file(HERE / "dataset_processing" / "prepare_dataset.py"),
            },
            "local_analysis": {
                "path": "local/run_local_analysis.py",
                "sha256": sha256_file(HERE / "local" / "run_local_analysis.py"),
            },
            "cloud_notebooks": {
                key: {
                    "path": (
                        f"upload_{key}/"
                        f"{BACKENDS[key]['chip']}_Q6T6L3_random_submit_collect_raw.ipynb"
                    ),
                    "sha256": sha256_file(
                        HERE
                        / f"upload_{key}"
                        / (
                            f"{BACKENDS[key]['chip']}_Q6T6L3_"
                            "random_submit_collect_raw.ipynb"
                        )
                    ),
                }
                for key in BACKENDS
            },
        },
    }
    (HERE / "BUNDLE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    total_archive = make_total_zip()
    print(
        json.dumps(
            {
                "total_archive": str(total_archive),
                "sha256": sha256_file(total_archive),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
