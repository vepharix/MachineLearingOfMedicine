from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("mortality_robustness", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MortalityRobustnessTests(unittest.TestCase):
    def test_sensitivity_threshold_meets_target_with_minimum_alerts(self) -> None:
        y = np.array([1, 1, 0, 0])
        probability = np.array([0.9, 0.6, 0.7, 0.1])
        threshold = MODULE.sensitivity_threshold(y, probability, 1.0)
        self.assertAlmostEqual(threshold, 0.6)

    def test_physiologic_masker_leaves_non_value_statistics_untouched(self) -> None:
        names = ("HR__min", "HR__count", "Temp__mean")
        x = np.array([[0.0, 4.0, 50.0], [70.0, 2.0, 37.0]])
        result = MODULE.PhysiologicFeatureMasker(names).fit_transform(x)
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertEqual(result[0, 1], 4.0)
        self.assertTrue(np.isnan(result[0, 2]))

    def test_missingness_features_do_not_include_values(self) -> None:
        names = ["Age", "Gender", "Height", "InitialWeight", "HR__count", "HR__mean"]
        x = np.array([[70.0, 1.0, np.nan, 80.0, 3.0, 100.0], [60.0, 0.0, 170.0, 70.0, 0.0, 40.0]])
        presence = MODULE.missingness_features(x, names, include_counts=False)
        with_counts = MODULE.missingness_features(x, names, include_counts=True)
        self.assertEqual(presence.shape, (2, 5))
        self.assertEqual(with_counts.shape, (2, 6))
        self.assertNotIn(100.0, with_counts)

    def test_hard_outcome_invalid_matches_explicit_codes(self) -> None:
        outcomes = pd.DataFrame(
            {"Survival": [-1, -23, 0, 1, 4], "Length_of_stay": [5, 4, 2, 2, 1]}
        )
        np.testing.assert_array_equal(
            MODULE.hard_outcome_invalid(outcomes), np.array([False, True, True, True, True])
        )

    def test_classification_metrics_reports_alert_burden(self) -> None:
        result = MODULE.classification_metrics(
            np.array([1, 1, 0, 0]), np.array([0.9, 0.4, 0.8, 0.1]), 0.5
        )
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["alerts_per_100"], 50.0)


if __name__ == "__main__":
    unittest.main()
