from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from task6_core import q12_xz_tokens, recover_source_values


class Q12XZProtocolTest(unittest.TestCase):
    def test_exact_24_to_12_xz_mapping(self) -> None:
        source = np.arange(2 * 6 * 24, dtype=np.float32).reshape(2, 6, 24)
        encoded = q12_xz_tokens(source).reshape(2, 6, 12, 3)
        self.assertEqual(encoded.shape, (2, 6, 12, 3))
        self.assertEqual(np.count_nonzero(encoded[..., 1]), 0)
        np.testing.assert_array_equal(encoded[..., 0], source.reshape(2, 6, 12, 2)[..., 0])
        np.testing.assert_array_equal(encoded[..., 2], source.reshape(2, 6, 12, 2)[..., 1])
        np.testing.assert_array_equal(recover_source_values(encoded.reshape(2, 6, 36)), source)


if __name__ == "__main__":
    unittest.main()

