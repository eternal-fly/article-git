#!/usr/bin/env python3
"""Dependency-free circuit-contract test using a fake Quark circuit recorder."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path

import numpy as np

import experiment_common as common


HERE = Path(__file__).resolve().parent


class FakeCircuit:
    def __init__(self, quantum_bits: int, classical_bits: int):
        self.quantum_bits = int(quantum_bits)
        self.classical_bits = int(classical_bits)
        self.instructions = []

    def u(self, theta, phi, lam, wire):
        self.instructions.append(("u", float(theta), float(phi), float(lam), int(wire)))

    def rx(self, angle, wire):
        self.instructions.append(("rx", float(angle), int(wire)))

    def rz(self, angle, wire):
        self.instructions.append(("rz", float(angle), int(wire)))

    def cx(self, control, target):
        self.instructions.append(("cx", int(control), int(target)))

    def barrier(self, *wires):
        if len(wires) == 1 and isinstance(wires[0], (list, tuple)):
            wires = tuple(wires[0])
        self.instructions.append(("barrier", *[int(value) for value in wires]))

    def measure(self, qubits, classical):
        self.instructions.append(
            ("measure", tuple(map(int, qubits)), tuple(map(int, classical)))
        )

    @property
    def to_qlisp(self):
        return self.instructions


@contextlib.contextmanager
def fake_quark():
    quark_module = types.ModuleType("quark")
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


def apply_one(state: np.ndarray, gate: np.ndarray, wire: int, q: int) -> np.ndarray:
    tensor = state.reshape([2] * q)
    moved = np.moveaxis(tensor, int(wire), 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    return np.moveaxis(updated, 0, int(wire)).reshape(-1)


def apply_cx(state: np.ndarray, control: int, target: int, q: int) -> np.ndarray:
    output = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        mapped = index
        if (index >> (q - 1 - int(control))) & 1:
            mapped ^= 1 << (q - 1 - int(target))
        output[mapped] += amplitude
    return output


def simulate(instructions: list[tuple], q: int) -> np.ndarray:
    state = np.zeros(1 << q, dtype=np.complex128)
    state[0] = 1.0
    for instruction in instructions:
        name = instruction[0]
        if name == "u":
            _, theta, phi, lam, wire = instruction
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            gate = np.asarray(
                [
                    [c, -np.exp(1j * lam) * s],
                    [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
                ],
                dtype=np.complex128,
            )
            state = apply_one(state, gate, wire, q)
        elif name == "rx":
            _, angle, wire = instruction
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            gate = np.asarray([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
            state = apply_one(state, gate, wire, q)
        elif name == "rz":
            _, angle, wire = instruction
            gate = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
            state = apply_one(state, gate, wire, q)
        elif name == "cx":
            _, control, target = instruction
            state = apply_cx(state, control, target, q)
        elif name in {"barrier", "measure"}:
            continue
        else:
            raise ValueError(instruction)
    return state


def z_zz_features(state: np.ndarray, q: int) -> np.ndarray:
    probabilities = np.abs(state) ** 2
    basis = np.arange(1 << q, dtype=np.int64)
    signs = np.stack(
        [
            1.0 - 2.0 * ((basis >> (q - 1 - wire)) & 1)
            for wire in range(q)
        ],
        axis=1,
    )
    z = probabilities @ signs
    values = [z[wire] for wire in range(q)]
    values.extend(
        float(np.sum(probabilities * signs[:, left] * signs[:, right]))
        for left in range(q)
        for right in range(left + 1, q)
    )
    return np.asarray(values, dtype=np.float64)


def check_task(folder: Path, variant_name: str) -> None:
    config = common.load_config(folder / "config.json")
    current = Path.cwd()
    try:
        # Config file names are task-relative by construction.
        import os

        os.chdir(folder)
        bundle = common.load_dataset(config)
        ideal = common.load_ideal(config)
        variant = next(value for value in config["variants"] if value["name"] == variant_name)
        with fake_quark():
            circuit, metadata = common.build_quantum_circuit(
                bundle["tokens"][0],
                config,
                int(variant["fold_factor"]),
            )
        cx_count = sum(instruction[0] == "cx" for instruction in circuit.instructions)
        assert cx_count == metadata["nominal_two_qubit_gates"]
        state = simulate(circuit.instructions, int(config["circuit"]["q"]))
        reconstructed = z_zz_features(state, int(config["circuit"]["q"]))
        if not np.allclose(reconstructed, ideal["features"][0], atol=3e-5, rtol=3e-5):
            difference = float(np.max(np.abs(reconstructed - ideal["features"][0])))
            raise AssertionError(
                f"{folder.name}/{variant_name}: circuit/ideal mismatch {difference}"
            )
        assert np.allclose(metadata["J"], ideal["J"], atol=1e-7, rtol=0)
        assert np.allclose(metadata["h"], ideal["h"], atol=1e-7, rtol=0)
        print(
            f"PASS {folder.name}/{variant_name}: "
            f"CX={cx_count}, max_feature_error="
            f"{np.max(np.abs(reconstructed - ideal['features'][0])):.3g}"
        )
    finally:
        import os

        os.chdir(current)


def main() -> None:
    check_task(HERE / "experiment" / "01_q6_t6_l3_batch01", "base")
    check_task(HERE / "experiment" / "04_q6_t6_l1_fold_pair_batch01", "base")
    check_task(HERE / "experiment" / "04_q6_t6_l1_fold_pair_batch01", "fold3")


if __name__ == "__main__":
    main()
