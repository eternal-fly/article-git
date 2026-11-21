"""CPU tests for the 18-injection task."""

from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np


TASK2_DIR = Path(__file__).resolve().parent
if str(TASK2_DIR) not in sys.path:
    sys.path.insert(0, str(TASK2_DIR))

from task2_core import (
    Task2Config,
    build_task2_dataset,
    expand_hierarchical_injections,
    load_config,
)


class HierarchicalInjectionTest(unittest.TestCase):
    def test_config_is_locked_and_has_eighteen_injections(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.q, 8)
        self.assertEqual(cfg.g, 0.84)
        self.assertEqual(cfg.W, 0.6)
        self.assertEqual(cfg.total_injections, 18)
        self.assertEqual(cfg.output_dim, 36)
        self.assertEqual(cfg.total_count, 3000)
        self.assertEqual(cfg.gpu_ids, (0, 1))

    def test_cumulative_hierarchical_expansion(self) -> None:
        source = np.asarray(
            [[[0.1, 0.2, 0.3, -0.1, -0.2, -0.3]]],
            dtype=np.float32,
        )
        expanded = expand_hierarchical_injections(source, q=2)
        self.assertEqual(expanded.shape, (1, 3, 6))
        expected = np.asarray(
            [
                [
                    [0.1, 0.0, 0.0, -0.1, 0.0, 0.0],
                    [0.1, 0.2, 0.0, -0.1, -0.2, 0.0],
                    [0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(expanded, expected)

    def test_dataset_derivation_preserves_labels_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.npz"
            output_path = root / "task2.npz"
            rng = np.random.default_rng(12)
            splits = np.asarray(["train", "train", "val", "test"])
            target_rho = rng.normal(size=(4, 2)).astype(np.float32)
            target_energy = rng.normal(size=4)
            np.savez_compressed(
                source_path,
                tokens=rng.normal(size=(4, 2, 6)).astype(np.float32),
                reaction_ids=np.asarray([f"rxn{i}" for i in range(4)]),
                formulas=np.asarray(["A", "A", "B", "C"]),
                splits=splits,
                num_atoms=np.asarray([3, 3, 4, 5]),
                atomic_numbers=np.ones((4, 5), dtype=np.int16),
                target_delta_rho=target_rho,
                target_delta_energy=target_energy,
                pair_slots=np.zeros((4, 2, 2), dtype=np.int16),
                role_ids=np.zeros((4, 2), dtype=np.int8),
                changed_pair_mask=np.ones((4, 2), dtype=bool),
                valid_pair_mask=np.ones((4, 2), dtype=bool),
                relative_energy_paths=np.zeros((4, 10)),
            )
            cfg = Task2Config(
                source_processed_dataset=source_path,
                processed_dataset=output_path,
                train_count=2,
                val_count=1,
                test_count=1,
                sample_seed=1,
                q=2,
                source_images=2,
                hierarchical_stages=("a", "b", "c"),
                injections_per_image=3,
                total_injections=6,
                target_image_index=2,
                g=0.84,
                W=0.6,
                layers=3,
                reservoir_seeds=(0,),
                gpu_ids=(0, 1),
            )
            build_task2_dataset(cfg)
            with np.load(output_path, allow_pickle=False) as data:
                self.assertEqual(data["tokens"].shape, (4, 6, 6))
                np.testing.assert_array_equal(
                    data["target_delta_rho"], target_rho
                )
                np.testing.assert_array_equal(data["splits"], splits)


if __name__ == "__main__":
    unittest.main()
