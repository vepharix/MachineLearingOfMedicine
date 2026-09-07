from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("remaining_los", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RemainingLengthOfStayTests(unittest.TestCase):
    def test_predict_remaining_applies_smearing_and_nonnegative_bound(self) -> None:
        class Dummy:
            def predict(self, x):
                return np.array([0.0, np.log(3.0)])

        result = MODULE.predict_remaining(Dummy(), np.zeros((2, 1)), smearing=2.0)
        np.testing.assert_allclose(result, np.array([1.0, 5.0]))

    def test_smearing_factor_recovers_multiplicative_residual_mean(self) -> None:
        class Dummy:
            def predict(self, x):
                return np.zeros(len(x))

        y = np.array([0.0, 1.0, 3.0])
        factor = MODULE.smearing_factor(Dummy(), np.zeros((3, 1)), y)
        self.assertAlmostEqual(factor, (1.0 + 2.0 + 4.0) / 3.0)

    def test_metrics_separate_individual_and_aggregate_error(self) -> None:
        result = MODULE.regression_metrics(np.array([1.0, 9.0]), np.array([4.0, 4.0]))
        self.assertEqual(result["mae_days"], 4.0)
        self.assertEqual(result["true_sum_bed_days"], 10.0)
        self.assertEqual(result["sum_bias_percent"], -20.0)

    def test_quantile_split_is_disjoint_and_complete(self) -> None:
        y = np.repeat(np.arange(20.0), 10)
        splits = MODULE.make_quantile_splits(y, 7)
        merged = np.concatenate(list(splits.values()))
        self.assertEqual(len(np.unique(merged)), len(y))
        self.assertEqual(set(merged), set(range(len(y))))


if __name__ == "__main__":
    unittest.main()
