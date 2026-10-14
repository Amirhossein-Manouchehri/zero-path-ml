"""Verify encoding and agreement with the original trial, not solver exactness."""
import unittest
from unittest.mock import patch

import numpy as np

from src.dataset import DEFAULT_SOLVER, generate_bonds, load_solver, pack_bonds, total_percolates, unpack_bonds


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = load_solver(DEFAULT_SOLVER)

    def test_round_trip_and_reference_labels(self):
        rng = np.random.default_rng(123)
        labels = []
        for p in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            edges = generate_bonds(4, p, rng, self.solver)
            features = pack_bonds(edges)
            self.assertEqual(features.shape, (28,))
            reconstructed = unpack_bonds(features, 4)
            np.testing.assert_array_equal(reconstructed[:, :, :2], edges[:, :, :2])
            np.testing.assert_array_equal(reconstructed[:-1, :, 2], edges[:-1, :, 2])
            np.testing.assert_array_equal(reconstructed[1:, :, 3], edges[1:, :, 3])
            # Supply the exact same bonds to the unmodified trial function.
            with patch.object(self.solver.np.random, "choice", return_value=edges.copy()):
                expected, _ = self.solver.run_single_trial_no_pbc(4, p)
            actual = total_percolates(reconstructed, self.solver)
            self.assertEqual(actual, expected)
            labels.append(actual)
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[-1], 0)
        self.assertIn(1, labels)

    def test_reproducible_bonds(self):
        a = generate_bonds(4, 0.3, np.random.default_rng(42), self.solver)
        b = generate_bonds(4, 0.3, np.random.default_rng(42), self.solver)
        np.testing.assert_array_equal(a, b)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            generate_bonds(2, 0.3, np.random.default_rng(42), self.solver)
        with self.assertRaises(ValueError):
            generate_bonds(4, -0.1, np.random.default_rng(42), self.solver)


if __name__ == "__main__":
    unittest.main()
