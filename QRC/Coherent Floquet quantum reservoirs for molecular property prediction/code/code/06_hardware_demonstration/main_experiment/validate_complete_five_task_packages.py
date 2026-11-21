#!/usr/bin/env python3
"""Offline validation for the two complete five-task Quafu packages."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TASK_KEYS = (
    "01_q6_t6_l3_batch01",
    "02_q6_t6_l3_batch02",
    "03_q6_t6_l3_batch03",
    "04_q6_t6_l1_base_batch01",
    "05_q6_t6_l1_fold3_batch01",
)
EXPECTED_CX = {
    "01_q6_t6_l3_batch01": 180,
    "02_q6_t6_l3_batch02": 180,
    "03_q6_t6_l3_batch03": 180,
    "04_q6_t6_l1_base_batch01": 60,
    "05_q6_t6_l1_fold3_batch01": 180,
}


class FakeCircuit:
    def __init__(self, quantum_bits: int, classical_bits: int):
        self.quantum_bits = int(quantum_bits)
        self.classical_bits = int(classical_bits)
        self.instructions: list[tuple] = []

    def u(self, theta, phi, lam, wire):
        self.instructions.append(
            ("u", float(theta), float(phi), float(lam), int(wire))
        )

    def rx(self, angle, wire):
        self.instructions.append(("rx", float(angle), int(wire)))

    def rz(self, angle, wire):
        self.instructions.append(("rz", float(angle), int(wire)))

    def cx(self, control, target):
        self.instructions.append(("cx", int(control), int(target)))

    def barrier(self, *qubits):
        if len(qubits) == 1 and isinstance(qubits[0], (list, tuple)):
            qubits = tuple(qubits[0])
        self.instructions.append(("barrier", tuple(map(int, qubits))))

    def measure(self, qubits, classical):
        self.instructions.append(
            ("measure", tuple(map(int, qubits)), tuple(map(int, classical)))
        )

    @property
    def to_qlisp(self):
        return self.instructions


class FakeTask:
    counter = 0

    def __init__(self, api_token: str):
        if api_token != "OFFLINE_TEST_ONLY":
            raise AssertionError("Unexpected offline API token")

    def run(self, task: dict, repeat: int = 1):
        FakeTask.counter += 1
        return f"FAKE_TID_{FakeTask.counter:05d}"

    def result(self, tid: str):
        return {
            "tid": str(tid),
            "status": "Finished",
            "count": {"000000": 600, "101010": 424},
        }


@contextlib.contextmanager
def fake_quark():
    quark_module = types.ModuleType("quark")
    quark_module.Task = FakeTask
    circuit_module = types.ModuleType("quark.circuit")
    circuit_module.QuantumCircuit = FakeCircuit
    old_quark = sys.modules.get("quark")
    old_circuit = sys.modules.get("quark.circuit")
    sys.modules["quark"] = quark_module
    sys.modules["quark.circuit"] = circuit_module
    try:
        yield
    finally:
        if old_quark is None:
            sys.modules.pop("quark", None)
        else:
            sys.modules["quark"] = old_quark
        if old_circuit is None:
            sys.modules.pop("quark.circuit", None)
        else:
            sys.modules["quark.circuit"] = old_circuit


def notebook_path(task_dir: Path) -> Path:
    notebooks = list(task_dir.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise AssertionError(f"{task_dir}: expected one notebook, got {notebooks}")
    return notebooks[0]


def load_namespace(task_dir: Path) -> dict:
    notebook = notebook_path(task_dir)
    text = notebook.read_text(encoding="utf-8")
    if re.search(
        r"(api[_-]?token|password)\s*[:=]\s*['\"][^'\"]+['\"]",
        text,
        re.I,
    ):
        raise AssertionError(f"{notebook}: literal credential found")
    payload = json.loads(text)
    namespace: dict = {}
    current = Path.cwd()
    try:
        os.chdir(task_dir)
        for cell_index in range(1, 6):
            source = "".join(payload["cells"][cell_index]["source"])
            source = re.sub(
                r"API_TOKEN = getpass\([^\n]+\)",
                "API_TOKEN = 'OFFLINE_TEST_ONLY'",
                source,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                exec(
                    compile(
                        source,
                        f"{notebook.name}:cell{cell_index}",
                        "exec",
                    ),
                    namespace,
                )
    finally:
        os.chdir(current)
    return namespace


def apply_one(
    state: np.ndarray,
    gate: np.ndarray,
    wire: int,
    q: int,
) -> np.ndarray:
    tensor = state.reshape([2] * q)
    moved = np.moveaxis(tensor, wire, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    return np.moveaxis(updated, 0, wire).reshape(-1)


def apply_cx(
    state: np.ndarray,
    control: int,
    target: int,
    q: int,
) -> np.ndarray:
    output = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        mapped = index
        if (index >> (q - 1 - control)) & 1:
            mapped ^= 1 << (q - 1 - target)
        output[mapped] += amplitude
    return output


def simulate(instructions: list[tuple], chain: list[int]) -> np.ndarray:
    logical = {physical: index for index, physical in enumerate(chain)}
    q = len(chain)
    state = np.zeros(1 << q, dtype=np.complex128)
    state[0] = 1.0
    for instruction in instructions:
        name = instruction[0]
        if name == "u":
            _, theta, phi, lam, physical = instruction
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            gate = np.asarray(
                [
                    [c, -np.exp(1j * lam) * s],
                    [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
                ],
                dtype=np.complex128,
            )
            state = apply_one(state, gate, logical[physical], q)
        elif name == "rx":
            _, angle, physical = instruction
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            gate = np.asarray(
                [[c, -1j * s], [-1j * s, c]], dtype=np.complex128
            )
            state = apply_one(state, gate, logical[physical], q)
        elif name == "rz":
            _, angle, physical = instruction
            gate = np.diag(
                [np.exp(-0.5j * angle), np.exp(0.5j * angle)]
            )
            state = apply_one(state, gate, logical[physical], q)
        elif name == "cx":
            _, control, target = instruction
            state = apply_cx(
                state,
                logical[control],
                logical[target],
                q,
            )
        elif name in {"barrier", "measure"}:
            continue
        else:
            raise ValueError(instruction)
    return state


def z_zz_features(state: np.ndarray, q: int) -> np.ndarray:
    probability = np.abs(state) ** 2
    basis = np.arange(1 << q)
    signs = np.stack(
        [
            1.0 - 2.0 * ((basis >> (q - 1 - wire)) & 1)
            for wire in range(q)
        ],
        axis=1,
    )
    z = probability @ signs
    pairs = [
        np.sum(probability * signs[:, left] * signs[:, right])
        for left in range(q)
        for right in range(left + 1, q)
    ]
    return np.concatenate([z, np.asarray(pairs)])


def extract_package(backend_key: str, destination: Path) -> Path:
    display = "Baihua" if backend_key == "baihua" else "Shenglian"
    archive = HERE / f"{display}_Q6T6_complete_5tasks_2000circuits.zip"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)
    folders = sorted(
        path.name for path in destination.iterdir() if path.is_dir()
    )
    if folders != list(TASK_KEYS):
        raise AssertionError((display, folders))
    if len(list(destination.glob("*/*.npz"))) != 5:
        raise AssertionError(f"{display}: expected five NPZ files")
    if len(list(destination.glob("*/*.ipynb"))) != 5:
        raise AssertionError(f"{display}: expected five notebooks")
    return destination


def check_package(
    backend_key: str,
    full_roundtrip: bool,
) -> dict[str, tuple[int, ...]]:
    display = "Baihua" if backend_key == "baihua" else "Shenglian"
    orders: dict[str, tuple[int, ...]] = {}
    with tempfile.TemporaryDirectory(prefix=f"{backend_key}-five-task-") as name:
        package = extract_package(backend_key, Path(name))
        with fake_quark():
            namespaces = {
                task_key: load_namespace(package / task_key)
                for task_key in TASK_KEYS
            }
            for task_key, namespace in namespaces.items():
                circuit, metadata = namespace["build_circuit"](
                    namespace["BUNDLE"]["tokens"][0]
                )
                cx_count = sum(
                    instruction[0] == "cx"
                    for instruction in circuit.instructions
                )
                if cx_count != EXPECTED_CX[task_key]:
                    raise AssertionError((display, task_key, cx_count))
                if metadata["nominal_two_qubit_gates"] != EXPECTED_CX[task_key]:
                    raise AssertionError(metadata)
                specs = namespace["randomized_specs"]()
                if task_key.startswith(("04_", "05_")):
                    if len(specs) != 800:
                        raise AssertionError((task_key, len(specs)))
                    for start in range(0, 800, 2):
                        pair = specs[start : start + 2]
                        if pair[0]["original_index"] != pair[1]["original_index"]:
                            raise AssertionError("L1 pair is not molecule-interleaved")
                        if {value["task_key"] for value in pair} != {
                            TASK_KEYS[3],
                            TASK_KEYS[4],
                        }:
                            raise AssertionError("L1 pair variants are incomplete")
                    selected = [
                        value for value in specs if value["task_key"] == task_key
                    ]
                else:
                    if len(specs) != 400:
                        raise AssertionError((task_key, len(specs)))
                    selected = specs
                order = tuple(
                    int(value["original_index"]) for value in selected
                )
                if len(order) != 400 or len(set(order)) != 400:
                    raise AssertionError((display, task_key, "bad permutation"))
                if order == tuple(sorted(order)):
                    raise AssertionError((display, task_key, "not randomized"))
                orders[task_key] = order
                state = simulate(
                    circuit.instructions,
                    namespace["PHYSICAL_QUBITS"],
                )
                reconstructed = z_zz_features(state, 6)
                ideal_name = (
                    "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz"
                    if task_key.startswith(("01_", "02_", "03_"))
                    else "ideal_bace_bal400_canonical_v3_q6_t6_l1_seed0.npz"
                )
                with np.load(HERE / "data" / ideal_name, allow_pickle=False) as data:
                    expected = data["features"][0]
                error = float(np.max(np.abs(reconstructed - expected)))
                if error > 3e-5:
                    raise AssertionError((display, task_key, error))
                print(
                    f"PASS {display} {task_key}: 400 circuits, "
                    f"CX={cx_count}, ideal_error={error:.3g}"
                )

            if orders[TASK_KEYS[3]] != orders[TASK_KEYS[4]]:
                raise AssertionError("L1 base/fold3 molecule orders differ")

            if full_roundtrip:
                FakeTask.counter = 0
                with contextlib.redirect_stdout(io.StringIO()):
                    for task_key in TASK_KEYS[:3]:
                        namespaces[task_key]["submit_all_randomized"]()
                    namespaces[TASK_KEYS[3]]["submit_all_randomized"]()
                if FakeTask.counter != 2000:
                    raise AssertionError(
                        f"{display}: submitted {FakeTask.counter}, expected 2000"
                    )
                before = FakeTask.counter
                with contextlib.redirect_stdout(io.StringIO()):
                    namespaces[TASK_KEYS[4]]["submit_all_randomized"]()
                if FakeTask.counter != before:
                    raise AssertionError("Second L1 submit created duplicates")
                with contextlib.redirect_stdout(io.StringIO()):
                    for task_key in TASK_KEYS:
                        archive = namespaces[task_key]["collect_raw_results"]()
                        if not Path(archive).is_file():
                            raise AssertionError(archive)
                raw_count = sum(
                    len(
                        list(
                            namespaces[task_key]["CURRENT_PATHS"][
                                "raw_dir"
                            ].glob("*.json")
                        )
                    )
                    for task_key in TASK_KEYS
                )
                if raw_count != 2000:
                    raise AssertionError((display, raw_count))
                print(
                    f"PASS {display} full roundtrip: exactly 2000 submits, "
                    "five 400-record collections, no L1 duplicates"
                )
    return orders


def main() -> None:
    baihua = check_package("baihua", full_roundtrip=True)
    shenglian = check_package("shenglian", full_roundtrip=False)
    for task_key in TASK_KEYS:
        if baihua[task_key] != shenglian[task_key]:
            raise AssertionError(
                f"Paired backend order differs for {task_key}"
            )
    print(
        "PASS complete five-task packages: paired per-task orders are "
        "identical across Baihua and Shenglian"
    )


if __name__ == "__main__":
    main()
