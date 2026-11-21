from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from task7_core import _base_gamma, _esn_parameters, load_config


class Task7ProtocolTest(unittest.TestCase):
    def test_locked_dimensions(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.output_dim, 78)
        self.assertEqual(cfg.temporal_nodes, 3)
        self.assertEqual(cfg.temporal_dim, 234)
        self.assertEqual(cfg.task6.total_injections, 6)

    def test_esn_parameters_are_deterministic(self) -> None:
        first = _esn_parameters(12, 10)
        second = _esn_parameters(12, 10)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first[0].shape, (78, 78))
        self.assertEqual(first[1].shape, (78, 36))

    def test_gamma_is_positive_and_deterministic(self) -> None:
        x = np.random.default_rng(4).normal(size=(600, 20))
        self.assertGreater(_base_gamma(x, 3), 0.0)
        self.assertEqual(_base_gamma(x, 3), _base_gamma(x, 3))


if __name__ == "__main__":
    unittest.main()

