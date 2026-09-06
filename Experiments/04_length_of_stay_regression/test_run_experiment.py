from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("los_regression", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LengthOfStayRegressionTests(unittest.TestCase):
    def test_metrics_are_on_days_scale(self) -> None:
        result = MODULE.regression_metrics(
            np.array([2.0, 5.0, 11.0]), np.array([3.0, 5.0, 8.0])
        )
        self.assertAlmostEqual(result["mae_days"], 4.0 / 3.0)
        self.assertAlmostEqual(result["median_ae_days"], 1.0)
        self.assertEqual(result["n"], 3)

    def test_prediction_is_back_transformed_and_clipped(self) -> None:
        class Dummy:
            def predict(self, x):
                return np.array([np.log1p(1.0), np.log1p(9.0)])

        predicted = MODULE.predict_days(Dummy(), np.zeros((2, 1)))
        np.testing.assert_allclose(predicted, np.array([2.0, 9.0]))

    def test_prediction_respects_training_target_upper_bound(self) -> None:
        class Dummy:
            def predict(self, x):
                return np.array([np.log1p(500.0)])

        predicted = MODULE.predict_days(Dummy(), np.zeros((1, 1)), maximum_days=295.0)
        np.testing.assert_allclose(predicted, np.array([295.0]))

    def test_quantile_clipper_learns_without_test_data(self) -> None:
        clipper = MODULE.QuantileClipper(lower=0.0, upper=0.75).fit(
            np.array([[0.0], [1.0], [2.0], [3.0]])
        )
        transformed = clipper.transform(np.array([[100.0]]))
        self.assertAlmostEqual(float(transformed[0, 0]), 2.25)

    def test_quantile_split_is_disjoint_and_complete(self) -> None:
        y = np.repeat(np.arange(2.0, 22.0), 10)
        splits = MODULE.make_quantile_splits(y, 7)
        merged = np.concatenate(list(splits.values()))
        self.assertEqual(len(np.unique(merged)), len(y))
        self.assertEqual(set(merged), set(range(len(y))))


if __name__ == "__main__":
    unittest.main()
