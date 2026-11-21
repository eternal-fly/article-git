"""Unit tests that do not require the full dataset or GPUs."""

from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path

import h5py
import numpy as np


TASK_DIR = Path(__file__).resolve().parent
DEMO_DIR = TASK_DIR.parent / "demo"
for directory in (TASK_DIR, DEMO_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from reaction_events import generate_synthetic_h_transfer_paths  # noqa: E402
from download_transition1x import (  # noqa: E402
    discover_covered_parts,
    find_gaps,
    repartition_gaps,
)
from task1_core import (  # noqa: E402
    ENERGY_KEY,
    Task1Config,
    encode_pair_event_path,
    evaluate_features,
    load_config,
    preprocess_real_dataset,
)


class NoBFSPairEncoderTest(unittest.TestCase):
    def test_resumable_download_gap_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parts_dir = Path(temporary)
            (parts_dir / "part_00_100_199.bin").write_bytes(b"x" * 25)
            (parts_dir / "part_01_200_299.bin").write_bytes(b"x" * 40)
            covered = discover_covered_parts(parts_dir)
            self.assertEqual(
                [(start, end) for start, end, _ in covered],
                [(100, 124), (200, 239)],
            )
            gaps = find_gaps(100, 299, covered)
            self.assertEqual(gaps, [(125, 199), (240, 299)])
            ranges = repartition_gaps(gaps, 4)
            self.assertEqual(len(ranges), 4)
            self.assertEqual(
                sum(end - start + 1 for start, end in ranges), 135
            )

    def test_prioritizes_changed_pairs_without_traversal(self) -> None:
        atomic_numbers, positions, _, _ = generate_synthetic_h_transfer_paths(
            num_reactions=1, num_frames=10, seed=23
        )
        encoded = encode_pair_event_path(
            atomic_numbers[0], positions[0], q=6
        )
        self.assertEqual(encoded.tokens.shape, (10, 6, 3))
        self.assertEqual(encoded.rho.shape, (10, 6))
        self.assertTrue(np.all(encoded.changed_mask[:2]))
        changed_pairs = {
            tuple(sorted(map(int, pair)))
            for pair, changed in zip(encoded.pair_slots, encoded.changed_mask)
            if changed
        }
        self.assertEqual(changed_pairs, {(0, 2), (1, 2)})
        self.assertTrue(np.isfinite(encoded.tokens).all())
        self.assertLessEqual(float(np.max(np.abs(encoded.tokens))), 1.0)

    def test_task_configuration_is_dimension_aligned(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.q, 8)
        self.assertEqual(cfg.output_dim, 36)
        self.assertEqual(cfg.total_count, 3000)
        self.assertEqual(cfg.gpu_ids, (0, 1))
        self.assertLess(cfg.prefix_length, 10)
        self.assertEqual(cfg.target_image_index, cfg.prefix_length)

    def test_small_hdf5_preprocessing_contract(self) -> None:
        atomic_numbers, positions, energies, _ = generate_synthetic_h_transfer_paths(
            num_reactions=3, num_frames=10, seed=31
        )
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            h5_path = temporary_path / "tiny.h5"
            processed_path = temporary_path / "processed.npz"
            with h5py.File(h5_path, "w") as handle:
                handle.create_group("data")
                for split_index, split in enumerate(("train", "val", "test")):
                    split_group = handle.create_group(split)
                    formula_group = split_group.create_group(f"C2H{split_index + 2}O")
                    reaction = formula_group.create_group(f"rxn{split_index:04d}")
                    reaction.create_dataset(
                        "atomic_numbers", data=atomic_numbers[split_index]
                    )
                    reaction.create_dataset(
                        "positions", data=positions[split_index]
                    )
                    reaction.create_dataset(
                        ENERGY_KEY, data=energies[split_index].astype(np.float64)
                    )
                    reactant = reaction.create_group("reactant")
                    product = reaction.create_group("product")
                    reactant.create_dataset(
                        "positions", data=positions[split_index, :1]
                    )
                    reactant.create_dataset(
                        ENERGY_KEY,
                        data=energies[split_index, :1].astype(np.float64),
                    )
                    product.create_dataset(
                        "positions", data=positions[split_index, -1:]
                    )
                    product.create_dataset(
                        ENERGY_KEY,
                        data=energies[split_index, -1:].astype(np.float64),
                    )
            cfg = Task1Config(
                raw_h5=h5_path,
                processed_dataset=processed_path,
                train_count=1,
                val_count=1,
                test_count=1,
                q=4,
                prefix_length=6,
                target_image_index=6,
                reservoir_seeds=(0,),
            )
            preprocess_real_dataset(cfg)
            with np.load(processed_path, allow_pickle=False) as data:
                self.assertEqual(data["tokens"].shape, (3, 6, 12))
                self.assertEqual(data["target_delta_rho"].shape, (3, 4))
                self.assertEqual(
                    set(data["splits"].tolist()), {"train", "val", "test"}
                )

    def test_common_readout_evaluation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "processed.npz"
            feature_dir = root / "features"
            output_dir = root / "outputs"
            feature_dir.mkdir()
            splits = np.asarray(
                ["train"] * 6 + ["val"] * 3 + ["test"] * 3
            )
            rng = np.random.default_rng(41)
            latent = rng.normal(size=(12, 3)).astype(np.float32)
            target_rho = np.column_stack(
                [0.4 * latent[:, 0], -0.2 * latent[:, 1]]
            ).astype(np.float32)
            target_energy = (
                latent[:, 0] - 0.5 * latent[:, 2]
            ).astype(np.float64)
            np.savez_compressed(
                dataset_path,
                splits=splits,
                reaction_ids=np.asarray([f"rxn{i:04d}" for i in range(12)]),
                pair_slots=np.tile(
                    np.asarray([[[0, 1], [0, 2]]], dtype=np.int16),
                    (12, 1, 1),
                ),
                valid_pair_mask=np.ones((12, 2), dtype=bool),
                changed_pair_mask=np.tile(
                    np.asarray([[True, False]]), (12, 1)
                ),
                target_delta_rho=target_rho,
                target_delta_energy=target_energy,
            )
            cfg = Task1Config(
                raw_h5=root / "unused.h5",
                processed_dataset=dataset_path,
                train_count=6,
                val_count=3,
                test_count=3,
                q=2,
                reservoir_seeds=(0, 10, 20),
            )
            for method_index, method in enumerate(("qrc", "esn")):
                for seed in cfg.reservoir_seeds:
                    features = (
                        latent
                        + 0.01 * method_index
                        + 0.001 * seed
                    )
                    np.savez_compressed(
                        feature_dir / f"{method}_seed{seed:03d}.npz",
                        features=features,
                    )
            report_path = evaluate_features(
                cfg, feature_dir=feature_dir, output_dir=output_dir
            )
            self.assertTrue(report_path.exists())
            self.assertTrue((output_dir / "metrics_by_seed.csv").exists())
            self.assertTrue((output_dir / "paired_differences.csv").exists())


if __name__ == "__main__":
    unittest.main()
