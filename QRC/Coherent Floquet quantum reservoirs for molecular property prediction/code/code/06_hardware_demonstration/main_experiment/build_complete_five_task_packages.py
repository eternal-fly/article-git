#!/usr/bin/env python3
"""Build the complete five-task Baihua and Shenglian Quafu upload packages.

Each backend archive contains five task folders. Every folder contains exactly
one locally processed NPZ and one self-contained submit/collect notebook.
The three L3 folders contain 400 circuits each. The two L1 folders form one
coordinated, interleaved base/fold3 pair containing 800 circuits in total.
"""

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
SOURCE_IDEAL_L1 = SOURCE_ROOT / "ideal_bace_bal400_canonical_v3_q6_t6_l1_seed0.npz"
SOURCE_IDEAL_L3 = SOURCE_ROOT / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz"
DATA_DIR = HERE / "data"
PROTOCOL_VERSION = (
    "2026-07-28-mainline-canonical-v3-q6t6-five-task-cloud-acquisition-v3"
)

BACKENDS = {
    "baihua": {
        "chip": "Baihua",
        "backend_num_qubits": 156,
        "chain": [72, 73, 74, 75, 76, 77],
        "calibrated_at": "2026-07-28 19:23:48",
    },
    "shenglian": {
        "chip": "Shenglian",
        "backend_num_qubits": 84,
        "chain": [27, 34, 41, 48, 54, 61],
        "calibrated_at": "2026-07-28 19:25:54",
    },
}

TASKS = (
    {
        "key": "01_q6_t6_l3_batch01",
        "folder": "01_q6_t6_l3_batch01",
        "layers": 3,
        "fold_factor": 1,
        "order_seed": 31001,
        "role": "independent L3 hardware batch 1",
        "pair_group": None,
    },
    {
        "key": "02_q6_t6_l3_batch02",
        "folder": "02_q6_t6_l3_batch02",
        "layers": 3,
        "fold_factor": 1,
        "order_seed": 31002,
        "role": "independent L3 hardware batch 2",
        "pair_group": None,
    },
    {
        "key": "03_q6_t6_l3_batch03",
        "folder": "03_q6_t6_l3_batch03",
        "layers": 3,
        "fold_factor": 1,
        "order_seed": 31003,
        "role": "independent L3 hardware batch 3",
        "pair_group": None,
    },
    {
        "key": "04_q6_t6_l1_base_batch01",
        "folder": "04_q6_t6_l1_base_batch01",
        "layers": 1,
        "fold_factor": 1,
        "order_seed": 41001,
        "role": "L1 base member of the interleaved fold pair",
        "pair_group": "l1_base_fold3_batch01",
    },
    {
        "key": "05_q6_t6_l1_fold3_batch01",
        "folder": "05_q6_t6_l1_fold3_batch01",
        "layers": 1,
        "fold_factor": 3,
        "order_seed": 41001,
        "role": "L1 fold3 member of the interleaved fold pair",
        "pair_group": "l1_base_fold3_batch01",
    },
)
TASK_BY_KEY = {task["key"]: task for task in TASKS}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_dataset_name(task: dict) -> str:
    return f"bace_bal400_canonical_v3_q6_t6__{task['key']}.npz"


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


def submission_tasks_for(task: dict) -> list[dict]:
    if task["pair_group"] is None:
        return [dict(task)]
    return [dict(value) for value in TASKS if value["pair_group"] == task["pair_group"]]


def notebook_payload(
    backend_key: str,
    task: dict,
    dataset_sha256: str,
) -> dict:
    backend = BACKENDS[backend_key]
    chip = backend["chip"]
    submission_tasks = submission_tasks_for(task)
    is_pair = len(submission_tasks) == 2
    dataset_name = task_dataset_name(task)
    expected_cx = 2 * 6 * int(task["layers"]) * 5 * int(task["fold_factor"])

    title = f"""# {chip}: {task['key']}

This self-contained notebook uses one already processed 400-sample NPZ. It
does not read labels, tokenize molecules, construct data splits, reconstruct
observables, train a decoder, calculate AUC, or make figures.

`ACTION="submit"` performs deterministic label-blind randomized submission.
`ACTION="collect"` retrieves only this folder's 400 raw results and creates a
backend/task-specific ZIP for local processing.
"""
    if is_pair:
        title += """
The L1 base and fold3 notebooks are coordinated. Running `ACTION="submit"` in
either L1 notebook submits the complete 800-circuit base/fold3 pair in an
interleaved per-molecule order and writes separate manifests into the two task
folders. Running submit from the other L1 notebook afterwards is safe and
submits zero duplicates. Run `ACTION="collect"` once in each L1 notebook to
download the two separate 400-result archives.
"""

    config = f"""# Edit only ACTION, then run all cells.
ACTION = "submit"  # "submit" or "collect"

from getpass import getpass
API_TOKEN = getpass("New Quafu API token: ")

PROTOCOL_VERSION = {PROTOCOL_VERSION!r}
BACKEND_KEY = {backend_key!r}
CHIP = {chip!r}
BACKEND_NUM_QUBITS = {backend["backend_num_qubits"]}
PHYSICAL_QUBITS = {backend["chain"]!r}
CALIBRATED_AT = {backend["calibrated_at"]!r}

CURRENT_TASK_KEY = {task["key"]!r}
CURRENT_TASK_FOLDER = {task["folder"]!r}
CURRENT_TASK = {task!r}
SUBMISSION_TASKS = {submission_tasks!r}
IS_INTERLEAVED_L1_PAIR = {is_pair!r}

Q = 6
T = 6
G = 0.84
W = 0.60
RESERVOIR_SEED = 0
REPEAT = 1
DATASET_FILE = {dataset_name!r}
EXPECTED_DATASET_SHA256 = {dataset_sha256!r}
EXPECTED_CURRENT_CX = {expected_cx}
"""

    imports = """import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np


def locate_notebook_dir():
    direct = Path.cwd()
    if (direct / DATASET_FILE).is_file():
        return direct
    matches = list(direct.glob(f"**/{DATASET_FILE}"))
    if len(matches) == 1:
        return matches[0].parent
    raise FileNotFoundError(
        f"Cannot locate {DATASET_FILE}; run from the extracted five-task package."
    )


NOTEBOOK_DIR = locate_notebook_dir()
PACKAGE_DIR = NOTEBOOK_DIR.parent
TASK_BY_KEY = {value["key"]: dict(value) for value in SUBMISSION_TASKS}


def task_paths(task_key):
    spec = TASK_BY_KEY.get(task_key)
    if spec is None and task_key == CURRENT_TASK_KEY:
        spec = dict(CURRENT_TASK)
    if spec is None:
        raise KeyError(task_key)
    task_dir = PACKAGE_DIR / spec["folder"]
    stem = f"{BACKEND_KEY}_{task_key}"
    root = task_dir / f"{stem}_cloud_results"
    return {
        "task_dir": task_dir,
        "root": root,
        "manifest": root / f"{stem}_job_manifest.jsonl",
        "raw_dir": root / f"raw_counts_{stem}",
        "order": root / f"{stem}_submission_order.json",
        "summary": root / f"{stem}_collection_summary.json",
        "zip": task_dir / f"{stem}_cloud_results.zip",
    }


CURRENT_PATHS = task_paths(CURRENT_TASK_KEY)


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
    path = NOTEBOOK_DIR / DATASET_FILE
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
        "task": CURRENT_TASK_KEY,
        "dataset_shape": BUNDLE["tokens"].shape,
        "physical_chain": PHYSICAL_QUBITS,
        "calibration_timestamp": CALIBRATED_AT,
        "submission_circuit_count": 800 if IS_INTERLEAVED_L1_PAIR else 400,
        "collection_circuit_count": 400,
    }
)
"""

    circuit = """def norm_edge(left, right):
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


def dtc_parameters():
    rng = np.random.default_rng(RESERVOIR_SEED)
    return {
        "J": rng.uniform(-0.9, 0.9, size=Q - 1).astype(np.float32),
        "h": rng.uniform(-W, W, size=Q).astype(np.float32),
    }


def insert_barrier(qc):
    barrier = getattr(qc, "barrier", None)
    if barrier is None:
        raise RuntimeError(
            "fold3 requires QuantumCircuit.barrier so B B^dagger B cannot "
            "be silently cancelled before platform compilation."
        )
    try:
        barrier(*[int(value) for value in PHYSICAL_QUBITS])
    except TypeError:
        barrier([int(value) for value in PHYSICAL_QUBITS])


def rzz_block(qc, left, right, angle):
    qc.cx(int(left), int(right))
    qc.rz(float(angle), int(right))
    qc.cx(int(left), int(right))


def folded_rzz(qc, left, right, angle, fold_factor):
    if int(fold_factor) == 1:
        rzz_block(qc, left, right, angle)
        return
    if int(fold_factor) != 3:
        raise ValueError(fold_factor)
    rzz_block(qc, left, right, angle)
    insert_barrier(qc)
    rzz_block(qc, left, right, -float(angle))
    insert_barrier(qc)
    rzz_block(qc, left, right, angle)


def build_circuit(token_sequence, task_spec=None):
    from quark.circuit import QuantumCircuit

    spec = dict(CURRENT_TASK if task_spec is None else task_spec)
    layers = int(spec["layers"])
    fold_factor = int(spec["fold_factor"])
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
        for _ in range(layers):
            for physical in PHYSICAL_QUBITS:
                qc.rx(G * math.pi, int(physical))
            for start in (0, 1):
                for logical in range(start, Q - 1, 2):
                    folded_rzz(
                        qc,
                        PHYSICAL_QUBITS[logical],
                        PHYSICAL_QUBITS[logical + 1],
                        float(params["J"][logical]),
                        fold_factor,
                    )
            for logical, physical in enumerate(PHYSICAL_QUBITS):
                qc.rz(float(params["h"][logical]) * math.pi, int(physical))
    qc.measure(PHYSICAL_QUBITS, PHYSICAL_QUBITS)
    expected_cx = 2 * T * layers * (Q - 1) * fold_factor
    metadata = {
        "q": Q,
        "t": T,
        "layers": layers,
        "fold_factor": fold_factor,
        "g": G,
        "W": W,
        "reservoir_seed": RESERVOIR_SEED,
        "J": params["J"].tolist(),
        "h": params["h"].tolist(),
        "interaction": "CX-RZ(J_i)-CX",
        "folding": "B B^dagger B with barriers" if fold_factor == 3 else "none",
        "circuit_qubits": list(PHYSICAL_QUBITS),
        "measured_clbits": list(PHYSICAL_QUBITS),
        "classical_num_bits": BACKEND_NUM_QUBITS,
        "nominal_two_qubit_gates": expected_cx,
    }
    if expected_cx > 200:
        raise AssertionError(f"Two-qubit gate cap exceeded: {expected_cx}")
    return qc, metadata


validate_chain()
preview_circuit, preview_metadata = build_circuit(BUNDLE["tokens"][0])
preview_cx = sum(
    isinstance(value, (tuple, list)) and value and str(value[0]).lower() == "cx"
    for value in preview_circuit.to_qlisp
)
if preview_cx != EXPECTED_CURRENT_CX:
    raise AssertionError((preview_cx, EXPECTED_CURRENT_CX))
print(
    f"Topology/circuit validation passed; current nominal CX count = "
    f"{EXPECTED_CURRENT_CX}"
)
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
            handle.write(
                json.dumps(record, ensure_ascii=False, default=json_default) + "\\n"
            )
    temporary.replace(path)


def task_manager():
    if not API_TOKEN:
        raise RuntimeError("API token is empty")
    from quark import Task
    return Task(API_TOKEN)


def randomized_specs():
    seed = int(SUBMISSION_TASKS[0]["order_seed"])
    if any(int(value["order_seed"]) != seed for value in SUBMISSION_TASKS):
        raise AssertionError("Coordinated submission tasks must share one seed")
    rng = np.random.default_rng(seed)
    rows = rng.permutation(len(BUNDLE["indices"])).astype(int)
    specs = []
    for molecule_rank, row in enumerate(rows):
        local_tasks = [dict(value) for value in SUBMISSION_TASKS]
        if IS_INTERLEAVED_L1_PAIR:
            rng.shuffle(local_tasks)
        for within_pair_rank, task_spec in enumerate(local_tasks):
            specs.append(
                {
                    "row": int(row),
                    "submission_rank": int(molecule_rank),
                    "global_submission_rank": int(len(specs)),
                    "within_pair_rank": int(within_pair_rank),
                    "original_index": int(BUNDLE["indices"][row]),
                    "task_key": task_spec["key"],
                }
            )
    for task_spec in SUBMISSION_TASKS:
        paths = task_paths(task_spec["key"])
        paths["root"].mkdir(parents=True, exist_ok=True)
        selected = [value for value in specs if value["task_key"] == task_spec["key"]]
        paths["order"].write_text(
            json.dumps(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "backend": CHIP,
                    "task_key": task_spec["key"],
                    "seed": seed,
                    "label_blind_random_permutation": True,
                    "interleaved_pair": IS_INTERLEAVED_L1_PAIR,
                    "records": selected,
                },
                indent=2,
            )
            + "\\n",
            encoding="utf-8",
        )
    return specs


def submit_all_randomized():
    manager = task_manager()
    specs = randomized_specs()
    records_by_task = {}
    by_index_by_task = {}
    for task_spec in SUBMISSION_TASKS:
        key = task_spec["key"]
        records = read_jsonl(task_paths(key)["manifest"])
        records_by_task[key] = records
        by_index_by_task[key] = {
            int(record["original_index"]): record for record in records
        }
    submitted = 0
    for spec in specs:
        task_key = spec["task_key"]
        old = by_index_by_task[task_key].get(int(spec["original_index"]))
        if old is not None and old.get("tid") is not None:
            continue
        task_spec = next(
            value for value in SUBMISSION_TASKS if value["key"] == task_key
        )
        row = int(spec["row"])
        qc, metadata = build_circuit(BUNDLE["tokens"][row], task_spec)
        circuit_id = (
            f"{BACKEND_KEY}_{task_key}_grank{spec['global_submission_rank']:05d}"
            f"_id{int(spec['original_index']):05d}"
        )
        options = {
            "compiler": "quarkcircuit",
            "correct": False,
            "target_qubits": list(PHYSICAL_QUBITS),
        }
        task_payload = {
            "chip": CHIP,
            "name": circuit_id,
            "circuit": qc.to_qlisp,
            "compile": True,
            "correct": False,
            "options": options,
        }
        tid = manager.run(task_payload, repeat=REPEAT)
        record = {
            **spec,
            **metadata,
            "protocol_version": PROTOCOL_VERSION,
            "backend_key": BACKEND_KEY,
            "chip": CHIP,
            "physical_qubits": list(PHYSICAL_QUBITS),
            "calibration_timestamp_from_image": CALIBRATED_AT,
            "dataset_file": (
                f"bace_bal400_canonical_v3_q6_t6__{task_spec['key']}.npz"
            ),
            "dataset_sha256": EXPECTED_DATASET_SHA256,
            "token_hash": str(BUNDLE["token_hashes"][row]),
            "circuit_id": circuit_id,
            "tid": str(tid),
            "repeat": REPEAT,
            "status": "Submitted",
            "compile": True,
            "task_options": options,
        }
        by_index_by_task[task_key][int(spec["original_index"])] = record
        ordered = sorted(
            by_index_by_task[task_key].values(),
            key=lambda value: int(value["submission_rank"]),
        )
        atomic_write_jsonl(ordered, task_paths(task_key)["manifest"])
        submitted += 1
        print(
            f"[{submitted:04d}] global_rank={spec['global_submission_rank']:04d} "
            f"task={task_key} id={spec['original_index']} tid={tid}"
        )
    manifests = {
        value["key"]: str(task_paths(value["key"])["manifest"])
        for value in SUBMISSION_TASKS
    }
    print(
        f"Submitted {submitted} new jobs; coordinated total expected="
        f"{800 if IS_INTERLEAVED_L1_PAIR else 400}"
    )
    return manifests


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
    paths = CURRENT_PATHS
    archive_base = str(paths["zip"].with_suffix(""))
    archive_path = Path(
        shutil.make_archive(
            archive_base,
            "zip",
            root_dir=paths["root"].parent,
            base_dir=paths["root"].name,
        )
    )
    print(f"Download this file: {archive_path.resolve()}")
    return archive_path


def collect_raw_results():
    paths = CURRENT_PATHS
    records = read_jsonl(paths["manifest"])
    if len(records) != 400:
        raise RuntimeError(
            f"Expected 400 submitted records for {CURRENT_TASK_KEY}, got "
            f"{len(records)}. Run ACTION='submit' first."
        )
    manager = task_manager()
    paths["raw_dir"].mkdir(parents=True, exist_ok=True)
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
        raw_path = paths["raw_dir"] / raw_name
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
                atomic_write_jsonl(records, paths["manifest"])
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
                (
                    result.get("status")
                    or result.get("task_status")
                    or result.get("state")
                    or "PendingNoCounts"
                )
                if isinstance(result, dict)
                else "PendingNoCounts"
            )
            pending += 1
        atomic_write_jsonl(records, paths["manifest"])
        print(
            f"[{position + 1:03d}/400] task={CURRENT_TASK_KEY} "
            f"rank={record['submission_rank']} status={record['status']}"
        )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "backend": CHIP,
        "task_key": CURRENT_TASK_KEY,
        "physical_qubits": list(PHYSICAL_QUBITS),
        "submitted_records": len(records),
        "counts_downloaded": completed,
        "pending_no_counts": pending,
        "fetch_failed": failed,
        "complete": completed == 400,
        "cloud_processing": (
            "raw result retrieval only; no label loading, feature extraction, "
            "training, AUC, or plotting"
        ),
    }
    paths["summary"].write_text(
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

    return {
        "cells": [
            markdown_cell(title),
            code_cell(config),
            code_cell(imports),
            code_cell(dataset_code),
            code_cell(circuit),
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


def root_readme(backend_key: str) -> str:
    backend = BACKENDS[backend_key]
    return f"""# {backend['chip']} complete five-task Q6/T6 package

This package contains five task folders and 2,000 hardware circuits:

| Folder | Circuits | L | Fold | Nominal CX/circuit |
|---|---:|---:|---:|---:|
| `01_q6_t6_l3_batch01` | 400 | 3 | 1 | 180 |
| `02_q6_t6_l3_batch02` | 400 | 3 | 1 | 180 |
| `03_q6_t6_l3_batch03` | 400 | 3 | 1 | 180 |
| `04_q6_t6_l1_base_batch01` | 400 | 1 | 1 | 60 |
| `05_q6_t6_l1_fold3_batch01` | 400 | 1 | 3 | 180 |

Every folder contains exactly one processed NPZ and one self-contained
notebook. The cloud code never loads labels or performs feature extraction,
training, AUC calculation, or plotting.

For each L3 folder, open its notebook, set `ACTION="submit"`, enter the API
token, and run all cells. Later use `ACTION="collect"` in the same notebook.
To interpret batch01/02/03 as independent device repetitions, submit them in
three distinct hardware/calibration windows and retain the calibration
metadata for each run. Different folder names and random orders alone do not
make simultaneously acquired circuits independent device repeats.

For the L1 pair, run `ACTION="submit"` in either folder 04 or 05 exactly once.
That call submits all 800 base/fold3 circuits in randomized per-molecule
interleaved order and creates separate manifests in both folders. Repeating
submit from the other notebook is safe and adds no duplicates. Then run
`ACTION="collect"` separately in both folder-04 and folder-05 notebooks.

Physical Q6 chain: `{backend['chain']}`. The chain was selected from the
supplied calibration screenshot; recheck the live numerical calibration table
if the device has been recalibrated.
"""


def build_backend_package(
    backend_key: str,
    dataset_hash: str,
) -> tuple[Path, dict]:
    backend = BACKENDS[backend_key]
    package_root = HERE / f"{backend_key}_complete_5tasks"
    package_root.mkdir(parents=True, exist_ok=True)
    notebook_rows = []
    for task in TASKS:
        folder = package_root / task["folder"]
        folder.mkdir(parents=True, exist_ok=True)
        dataset_name = task_dataset_name(task)
        dataset_target = folder / dataset_name
        shutil.copy2(SOURCE_DATASET, dataset_target)
        notebook_name = (
            f"{backend['chip']}_{task['key']}_submit_collect_raw.ipynb"
        )
        notebook_path = folder / notebook_name
        write_notebook(
            notebook_path,
            notebook_payload(backend_key, task, dataset_hash),
        )
        notebook_rows.append(
            {
                "task_key": task["key"],
                "folder": task["folder"],
                "circuits": 400,
                "layers": task["layers"],
                "fold_factor": task["fold_factor"],
                "nominal_two_qubit_gates": (
                    2
                    * 6
                    * int(task["layers"])
                    * 5
                    * int(task["fold_factor"])
                ),
                "order_seed": task["order_seed"],
                "pair_group": task["pair_group"],
                "dataset": dataset_name,
                "dataset_sha256": sha256_file(dataset_target),
                "notebook": notebook_name,
                "notebook_sha256": sha256_file(notebook_path),
            }
        )
    (package_root / "README.md").write_text(
        root_readme(backend_key), encoding="utf-8"
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "backend_key": backend_key,
        "chip": backend["chip"],
        "physical_qubits": backend["chain"],
        "calibration_timestamp_from_image": backend["calibrated_at"],
        "task_folder_count": 5,
        "npz_count": 5,
        "notebook_count": 5,
        "circuits_per_task": 400,
        "total_circuits": 2000,
        "credential_included": False,
        "cloud_contract": (
            "circuit construction/submission and raw result retrieval only"
        ),
        "tasks": notebook_rows,
    }
    (package_root / "UPLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive = HERE / f"{backend['chip']}_Q6T6_complete_5tasks_2000circuits.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as handle:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(package_root).as_posix())
    return archive, manifest


def main() -> None:
    for source in (SOURCE_DATASET, SOURCE_IDEAL_L1, SOURCE_IDEAL_L3):
        if not source.is_file():
            raise FileNotFoundError(source)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DATASET, DATA_DIR / SOURCE_DATASET.name)
    shutil.copy2(SOURCE_IDEAL_L1, DATA_DIR / SOURCE_IDEAL_L1.name)
    shutil.copy2(SOURCE_IDEAL_L3, DATA_DIR / SOURCE_IDEAL_L3.name)
    dataset_hash = sha256_file(SOURCE_DATASET)
    packages = {}
    for backend_key in BACKENDS:
        archive, manifest = build_backend_package(backend_key, dataset_hash)
        packages[backend_key] = {
            "archive": archive.name,
            "archive_sha256": sha256_file(archive),
            "manifest": manifest,
        }
    complete_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_sha256": dataset_hash,
        "ideal_l1_sha256": sha256_file(SOURCE_IDEAL_L1),
        "ideal_l3_sha256": sha256_file(SOURCE_IDEAL_L3),
        "backend_package_count": 2,
        "circuits_per_backend": 2000,
        "total_circuits_both_backends": 4000,
        "packages": packages,
    }
    (HERE / "COMPLETE_FIVE_TASK_MANIFEST.json").write_text(
        json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(complete_manifest, indent=2))


if __name__ == "__main__":
    main()
