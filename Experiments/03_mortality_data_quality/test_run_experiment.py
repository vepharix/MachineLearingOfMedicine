#!/usr/bin/env python3
"""Focused tests for Experiment 03 preprocessing rules."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("mortality_quality_experiment", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class QualityRuleTests(unittest.TestCase):
    def test_plausibility_bounds_are_broad_but_remove_clear_errors(self):
        self.assertTrue(MODULE.value_is_plausible("Temp", 39.5))
        self.assertFalse(MODULE.value_is_plausible("Temp", -17.8))
        self.assertTrue(MODULE.value_is_plausible("pH", 7.35))
        self.assertFalse(MODULE.value_is_plausible("pH", 735.0))
        self.assertFalse(MODULE.value_is_plausible("SysABP", 0.0))
        self.assertTrue(MODULE.value_is_plausible("SysABP", 220.0))

    def test_quantile_clipper_learns_only_from_fit_data(self):
        clipper = MODULE.QuantileClipper(np.array([True, False]), lower=0.0, upper=0.5)
        clipper.fit(np.array([[1.0, 10.0], [3.0, 20.0]]))
        transformed = clipper.transform(np.array([[100.0, 100.0]]))
        self.assertEqual(float(transformed[0, 0]), 2.0)
        self.assertEqual(float(transformed[0, 1]), 100.0)

    def test_hard_outcome_anomaly_mask(self):
        outcomes = pd.DataFrame(
            {
                "Length_of_stay": [5, 5, 5, 1],
                "Survival": [3, 8, -1, -1],
                "In-hospital_death": [1, 0, 1, 0],
            }
        )
        mask = MODULE.hard_outcome_anomaly_mask(outcomes)
        self.assertEqual(mask.tolist(), [False, False, True, True])


if __name__ == "__main__":
    unittest.main()

