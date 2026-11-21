#!/usr/bin/env python3
"""Offline contract checks for both self-contained cloud notebooks."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


class FakeCircuit:
    def __init__(self, quantum_bits: int, classical_bits: int):
        self.quantum_bits = int(quantum_bits)
        self.classical_bits = int(classical_bits)
        self.instructions = []

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


def load_notebook_namespace(folder: Path, notebook: Path) -> dict:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    namespace = {}
    current = Path.cwd()
    try:
        os.chdir(folder)
        for cell_index in range(1, 6):
            source = "".join(payload["cells"][cell_index]["source"])
            source = re.sub(
                r"API_TOKEN = getpass\([^\n]+\)",
                "API_TOKEN = 'OFFLINE_TEST_ONLY'",
                source,
            )
            exec(compile(source, f"{notebook.name}:cell{cell_index}", "exec"), namespace)
    finally:
        os.chdir(current)
    return namespace


def apply_one(
    state: np.ndarray, gate: np.ndarray, wire: int, q: int
) -> np.ndarray:
    tensor = state.reshape([2] * q)
    moved = np.moveaxis(tensor, wire, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    return np.moveaxis(updated, 0, wire).reshape(-1)


def apply_cx(
    state: np.ndarray, control: int, target: int, q: int
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
                state, logical[control], logical[target], q
            )
        elif name == "measure":
            continue
        else:
            raise ValueError(instruction)
    return state


def z_zz_features(state: np.ndarray, q: int) -> np.ndarray:
    probability = np.abs(state) ** 2
    basis = np.arange(1 << q)
    signs = np.stack(
        [1.0 - 2.0 * ((basis >> (q - 1 - wire)) & 1) for wire in range(q)],
        axis=1,
    )
    z = probability @ signs
    pairs = [
        np.sum(probability * signs[:, left] * signs[:, right])
        for left in range(q)
        for right in range(left + 1, q)
    ]
    return np.concatenate([z, np.asarray(pairs)])


def check_backend(key: str, display: str) -> tuple[int, ...]:
    folder = HERE / f"upload_{key}"
    notebook = folder / f"{display}_Q6T6L3_random_submit_collect_raw.ipynb"
    text = notebook.read_text(encoding="utf-8")
    if re.search(r"(api[_-]?token|password)\s*[:=]\s*['\"][^'\"]+['\"]", text, re.I):
        raise AssertionError(f"{display}: literal credential found")
    with fake_quark():
        namespace = load_notebook_namespace(folder, notebook)
        circuit, metadata = namespace["build_circuit"](
            namespace["BUNDLE"]["tokens"][0]
        )
        if key == "baihua":
            current = Path.cwd()
            try:
                with tempfile.TemporaryDirectory(
                    prefix="qrc_cloud_roundtrip_"
                ) as temporary:
                    os.chdir(temporary)
                    with contextlib.redirect_stdout(io.StringIO()):
                        manifest = namespace["submit_all_randomized"]()
                        archive = namespace["collect_raw_results"]()
                    if len(namespace["read_jsonl"](manifest)) != 400:
                        raise AssertionError("Offline submission manifest is incomplete")
                    if not Path(archive).is_file():
                        raise AssertionError("Offline collection archive is missing")
                    raw_files = list(
                        (
                            namespace["RESULT_ROOT"]
                            / namespace["RAW_DIR_NAME"]
                        ).glob("*.json")
                    )
                    if len(raw_files) != 400:
                        raise AssertionError(
                            f"Offline raw-result count is {len(raw_files)}/400"
                        )
                    print(
                        "PASS Baihua offline cloud roundtrip: "
                        "400 submit + 400 collect + ZIP"
                    )
            finally:
                os.chdir(current)
    cx_count = sum(item[0] == "cx" for item in circuit.instructions)
    if cx_count != 180 or metadata["nominal_two_qubit_gates"] != 180:
        raise AssertionError((display, cx_count, metadata))
    current = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix=f"qrc_{key}_order_test_") as temporary:
            os.chdir(temporary)
            specs = namespace["randomized_specs"]()
    finally:
        os.chdir(current)
    order = tuple(value["original_index"] for value in specs)
    sorted_indices = tuple(sorted(order))
    if len(order) != 400 or len(set(order)) != 400 or order == sorted_indices:
        raise AssertionError(f"{display}: submission order is not a permutation shuffle")
    state = simulate(circuit.instructions, namespace["PHYSICAL_QUBITS"])
    reconstructed = z_zz_features(state, 6)
    with np.load(
        HERE
        / "data"
        / "ideal_bace_bal400_canonical_v3_q6_t6_l3_seed0.npz",
        allow_pickle=False,
    ) as package:
        expected = package["features"][0]
    error = float(np.max(np.abs(reconstructed - expected)))
    if error > 3e-5:
        raise AssertionError(f"{display}: ideal feature mismatch {error}")
    print(
        f"PASS {display}: chain={namespace['PHYSICAL_QUBITS']}, "
        f"CX={cx_count}, randomized=True, max_feature_error={error:.3g}"
    )
    return order


def main() -> None:
    baihua = check_backend("baihua", "Baihua")
    shenglian = check_backend("shenglian", "Shenglian")
    if baihua != shenglian:
        raise AssertionError(
            "The two backends should use the same paired randomized molecule order"
        )
    print("PASS paired randomized submission order is identical across backends")


if __name__ == "__main__":
    main()
