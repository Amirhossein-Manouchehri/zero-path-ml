import unittest

import numpy as np

from src.train_baseline import remove_duplicate_configurations, train_baselines


class BaselineTests(unittest.TestCase):
    def test_duplicates_are_removed_before_split(self):
        X = np.array([[1, 1], [1, 1], [-1, -1], [-1, 1]], dtype=np.int8)
        y = np.array([1, 1, 0, 0], dtype=np.int8)
        unique_X, unique_y, removed = remove_duplicate_configurations(X, y)
        self.assertEqual(len(unique_X), 3)
        self.assertEqual(len(unique_y), 3)
        self.assertEqual(removed, 1)

    def test_conflicting_duplicate_labels_are_rejected(self):
        X = np.array([[1, -1], [1, -1]], dtype=np.int8)
        y = np.array([0, 1], dtype=np.int8)
        with self.assertRaises(ValueError):
            remove_duplicate_configurations(X, y)

    def test_training_is_reproducible(self):
        rng = np.random.default_rng(10)
        X = rng.choice([-1, 1], size=(100, 8)).astype(np.int8)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int8)
        # Ensure both classes are large enough for a stratified split.
        _, first = np.unique(X, axis=0, return_index=True)
        X, y = X[first], y[first]
        model_a, results_a = train_baselines(X, y, test_size=0.25, seed=7)
        model_b, results_b = train_baselines(X, y, test_size=0.25, seed=7)
        self.assertEqual(results_a, results_b)
        np.testing.assert_array_equal(model_a.predict(X), model_b.predict(X))


if __name__ == "__main__":
    unittest.main()
