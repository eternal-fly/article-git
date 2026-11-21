from __future__ import annotations

import sys
import unittest
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from task8_core import load_config


class Task8ProtocolTest(unittest.TestCase):
    def test_post_input_tm_contract(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.sample_cycles, (0, 1, 2))
        self.assertEqual(cfg.observable_family, "Z+ZZ")
        self.assertEqual(cfg.output_dim, 78)
        self.assertEqual(cfg.temporal_dim, 234)

    def test_inherited_reservoir_contract(self) -> None:
        task6 = load_config().task6
        self.assertEqual((task6.q, task6.g, task6.W, task6.layers), (12, 0.84, 0.5, 3))
        self.assertEqual(task6.injection_axis, "XZ")
        self.assertEqual(task6.total_injections, 6)


if __name__ == "__main__":
    unittest.main()

