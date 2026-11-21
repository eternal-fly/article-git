#!/usr/bin/env python3
"""CPU tests for Task 3 Pauli observables and feature contracts."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from task3_core import (
    Task3Config,
    build_pauli_action_tables,
    compute_ablation_features,
    compute_esn_temporal_features,
    observable_specifications,
    pauli_expectations,
)


class PauliExpectationTests(unittest.TestCase):
    def expectation(self, state, specs):
        gather, phases = build_pauli_action_tables(
            specs, 2, torch.device("cpu"), torch.complex64
        )
        return pauli_expectations(state, gather, phases).numpy()

    def test_product_zero_state(self):
        state = torch.zeros((1, 4), dtype=torch.complex64)
        state[0, 0] = 1
        specs = [
            ((0, "Z"),),
            ((0, "X"),),
            ((0, "Y"),),
            ((0, "Z"), (1, "Z")),
        ]
        values = self.expectation(state, specs)[0]
        np.testing.assert_allclose(values, [1, 0, 0, 1], atol=1e-6)

    def test_bell_state(self):
        state = torch.zeros((1, 4), dtype=torch.complex64)
        state[0, 0] = 1 / np.sqrt(2)
        state[0, 3] = 1 / np.sqrt(2)
        specs = [
            ((0, "X"), (1, "X")),
            ((0, "Y"), (1, "Y")),
            ((0, "Z"), (1, "Z")),
        ]
        values = self.expectation(state, specs)[0]
        np.testing.assert_allclose(values, [1, -1, 1], atol=1e-6)


class FeatureContractTests(unittest.TestCase):
    def test_q8_feature_dimensions(self):
        specs = observable_specifications(8, 6)
        expected = {
            "A": 36,
            "B": 52,
            "C": 276,
            "D": 92,
            "F": 216,
            "FB": 312,
            "FC": 1656,
        }
        self.assertEqual(
            {key: len(value) for key, value in specs["names"].items()},
            expected,
        )

    def test_cpu_extraction_is_finite(self):
        cfg = Task3Config(
            processed_dataset=None,
            task1_feature_dir=None,
            q=2,
            prefix_length=2,
            g=0.84,
            W=0.6,
            layers=1,
            batch_size=2,
            reservoir_seeds=(0,),
            devices=("cpu", "cpu"),
            ridge_alphas=(0.1,),
        )
        rng = np.random.default_rng(7)
        tokens = rng.normal(size=(3, 2, 6)).astype(np.float32)
        variants, _, names = compute_ablation_features(
            tokens, cfg, seed=0, device="cpu"
        )
        for key, values in variants.items():
            self.assertEqual(values.shape[0], len(tokens))
            self.assertEqual(values.shape[1], len(names[key]))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.max(np.abs(values)) <= 1.00001)

        from gpu_qrc_statevector import DTCQRCConfig, compute_qrc_features_gpu

        reference_cfg = DTCQRCConfig(
            num_qubits=2,
            features_per_qubit=3,
            g=0.84,
            W_scale=0.6,
            layers_per_injection=1,
            use_tanh_before_encoding=True,
            token_dim_policy="strict",
            complex_dtype="complex64",
        )
        reference, _ = compute_qrc_features_gpu(
            torch.from_numpy(tokens),
            reference_cfg,
            seed=0,
            batch_size=2,
            device="cpu",
            progress=False,
        )
        np.testing.assert_allclose(
            variants["A"], reference.numpy(), atol=2e-6
        )

    def test_temporal_esn_final_slice_matches_standard_esn(self):
        from esn_baseline import compute_esn_standard_features

        cfg = Task3Config(
            processed_dataset=None,
            task1_feature_dir=None,
            q=2,
            prefix_length=3,
            g=0.84,
            W=0.6,
            layers=2,
            batch_size=4,
            reservoir_seeds=(10,),
            devices=("cpu", "cpu"),
            ridge_alphas=(0.1,),
        )
        rng = np.random.default_rng(11)
        tokens = rng.normal(size=(5, 3, 6)).astype(np.float32)
        temporal, _ = compute_esn_temporal_features(
            tokens, cfg, seed=10, device="cpu"
        )
        final, _ = compute_esn_standard_features(
            tokens,
            q=2,
            layers=2,
            seed=10,
            device="cpu",
        )
        self.assertEqual(temporal.shape, (5, 9))
        np.testing.assert_allclose(temporal[:, -3:], final, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
