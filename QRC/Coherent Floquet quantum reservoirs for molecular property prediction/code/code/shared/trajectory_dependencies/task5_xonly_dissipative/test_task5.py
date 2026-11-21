from __future__ import annotations

import unittest
import sys
from pathlib import Path
import numpy as np

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from task5_core import mix_suffix_bank, reset_suffix_weights, x_only_field_injections


class Task5ProtocolTest(unittest.TestCase):
    def test_x_only_serialization_preserves_all_three_fields(self) -> None:
        source = np.arange(2 * 4 * 3 * 3, dtype=np.float32).reshape(2, 4, 9)
        output = x_only_field_injections(source, q=3).reshape(2, 12, 3, 3)
        self.assertTrue(np.all(output[..., 1:] == 0))
        original = source.reshape(2, 4, 3, 3)
        for field in range(3):
            np.testing.assert_array_equal(output[:, field::3, :, 0], original[..., field])

    def test_exact_reset_weights(self) -> None:
        np.testing.assert_array_equal(reset_suffix_weights(6, 0.0), [1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(reset_suffix_weights(4, 0.5), [0.125, 0.125, 0.25, 0.5])
        self.assertAlmostEqual(float(reset_suffix_weights(6, 0.8).sum()), 1.0)

    def test_closed_mixture_is_full_trajectory(self) -> None:
        bank = np.arange(6 * 5 * 3, dtype=np.float32).reshape(6, 5, 3)
        np.testing.assert_array_equal(mix_suffix_bank(bank, 0.0), bank[0])


if __name__ == "__main__":
    unittest.main()
