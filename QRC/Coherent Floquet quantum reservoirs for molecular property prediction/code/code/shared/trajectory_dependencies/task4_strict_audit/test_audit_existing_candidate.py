import unittest

import numpy as np

from audit_existing_candidate import bootstrap_interval, per_reaction_errors


class AuditHelpersTest(unittest.TestCase):
    def test_per_reaction_errors(self):
        y_rho = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        p_rho = np.asarray([[2.0, 0.0], [3.0, 6.0]])
        mask = np.asarray([[True, True], [True, False]])
        errors = per_reaction_errors(y_rho, np.asarray([1.0, 2.0]), p_rho, np.asarray([0.0, 4.0]), mask, mask)
        np.testing.assert_allclose(errors["rho_all"], [1.5, 0.0])
        np.testing.assert_allclose(errors["energy"], [1.0, 2.0])

    def test_positive_bootstrap_interval(self):
        result = bootstrap_interval(np.ones(20), None, 100, 1, 0.95)
        self.assertEqual(result["lower"], 1.0)
        self.assertEqual(result["upper"], 1.0)


if __name__ == "__main__":
    unittest.main()
