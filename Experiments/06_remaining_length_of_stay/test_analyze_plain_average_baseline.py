from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("analyze_plain_average_baseline.py")
SPEC = importlib.util.spec_from_file_location("plain_average_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlainAverageBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        # Development records 0-5, test records 6-10; every ICU type appears in both.
        self.remaining = np.array([2.0, 4.0, 6.0, 10.0, 20.0, 30.0, 3.0, 5.0, 12.0, 18.0, 33.0])
        self.icu = np.array([1, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4])
        self.development = np.arange(6)
        self.test = np.arange(6, 11)

    def test_own_and_pooled_averages(self) -> None:
        result = MODULE.averages_by_icu(self.remaining, self.icu, self.development, self.test).set_index("subgroup")
        self.assertAlmostEqual(result.loc["all", "own_average_days"], 12.0)
        self.assertAlmostEqual(result.loc["all", "own_average_bias_percent"], 100 * (60 - 71) / 71)
        self.assertAlmostEqual(result.loc["ICUType_1", "own_average_days"], 4.0)
        self.assertAlmostEqual(result.loc["ICUType_1", "own_average_bias_percent"], 0.0)
        self.assertAlmostEqual(result.loc["ICUType_2", "own_average_bias_percent"], 100 * (10 - 12) / 12)
        self.assertAlmostEqual(result.loc["ICUType_1", "pooled_average_bias_percent"], 100 * (24 - 8) / 8)

    def test_split_check_rejects_a_different_test_set(self) -> None:
        by_icu = MODULE.averages_by_icu(self.remaining, self.icu, self.development, self.test)
        by_icu = by_icu[by_icu["subgroup"].isin(["all", "ICUType_1"])]
        primary = pd.DataFrame([{"readout": "cohort_smeared", "n": 5, "true_sum_bed_days": 71.0, "sum_bias_percent": 1.0}])
        subgroups = pd.DataFrame(
            [{"readout": "cohort_smeared", "subgroup": "ICUType_1", "n": 2, "true_sum_bed_days": 8.0, "sum_bias_percent": 2.0}]
        )
        MODULE.check_split_matches(by_icu, primary, subgroups)
        subgroups.loc[0, "true_sum_bed_days"] = 9.0
        with self.assertRaises(ValueError):
            MODULE.check_split_matches(by_icu, primary, subgroups)

    def test_constant_baseline_bias(self) -> None:
        result = MODULE.constant_baselines(self.remaining, self.development, self.test, 50, 72)
        mean_bias = result[(result["readout"] == "development_mean") & (result["metric"] == "sum_bias_percent")]
        self.assertAlmostEqual(float(mean_bias["point_estimate"].iloc[0]), 100 * (60 - 71) / 71)
        median = result[result["readout"] == "development_median"]["constant_days"].iloc[0]
        self.assertAlmostEqual(median, 8.0)


if __name__ == "__main__":
    unittest.main()
