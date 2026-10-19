#!/usr/bin/env python3
"""Tests for the paired-bootstrap and Challenge 2012 Event 1 helpers."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = load("mortality_paired_comparison", "analyze_paired_comparison.py")
CHALLENGE = load("mortality_challenge_score", "analyze_challenge_score.py")


def toy_data(seed: int = 3, n: int = 300):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(np.int8)
    first = np.clip(0.25 * y + rng.random(n) * 0.75, 0, 1).round(2)
    second = np.clip(0.15 * y + rng.random(n) * 0.85, 0, 1).round(2)
    return y, first, second


class PairedBootstrapTests(unittest.TestCase):
    def test_resamples_match_run_experiment_intervals(self):
        y, first, _ = toy_data()
        samples, valid = PAIRED.paired_bootstrap(y, {"hist_gradient_boosting": first}, 90, 200)
        reference = CHALLENGE.experiment.bootstrap_intervals(y, first, 0.5, 90, 200)
        self.assertEqual(valid, 200)
        low, high = np.quantile(samples[("hist_gradient_boosting", "auroc")], [0.025, 0.975])
        self.assertAlmostEqual(low, reference["auroc_ci_low"])
        self.assertAlmostEqual(high, reference["auroc_ci_high"])

    def test_identical_models_have_zero_difference(self):
        y, first, _ = toy_data()
        probabilities = {"hist_gradient_boosting": first, "logistic": first.copy()}
        samples, valid = PAIRED.paired_bootstrap(y, probabilities, 7, 100)
        summary = PAIRED.summarize(
            y, probabilities, samples, valid, 100, 7, {"hist_gradient_boosting": "a", "logistic": "b"}
        )
        self.assertTrue((summary["difference_hgb_minus_lr"] == 0).all())
        self.assertTrue((summary["difference_ci_2_5"] == 0).all())
        self.assertTrue((summary["share_resamples_hgb_better"] == 0).all())


class Event1Tests(unittest.TestCase):
    def test_curve_matches_brute_force(self):
        y, first, _ = toy_data()
        curve = CHALLENGE.event1_curve(y, first)
        for row in curve.itertuples(index=False):
            direct = CHALLENGE.event1_score(y, first, row.threshold)
            self.assertAlmostEqual(row.sensitivity, direct["sensitivity"])
            self.assertAlmostEqual(row.ppv, direct["ppv"])
            self.assertAlmostEqual(row.event1_score, direct["event1_score"])

    def test_threshold_maximises_score_and_breaks_ties_upwards(self):
        y = np.array([1, 0, 1, 0], dtype=np.int8)
        probability = np.array([0.1, 0.2, 0.3, 0.4])
        # 0.3 and 0.1 both give min(Se, PPV) = 0.5; the higher threshold wins.
        self.assertEqual(CHALLENGE.event1_threshold(y, probability), 0.3)
        y, first, _ = toy_data()
        threshold = CHALLENGE.event1_threshold(y, first)
        best = CHALLENGE.event1_curve(y, first)["event1_score"].max()
        self.assertAlmostEqual(CHALLENGE.event1_score(y, first, threshold)["event1_score"], best)

    def test_no_alerts_scores_zero(self):
        result = CHALLENGE.event1_score(np.array([0, 1], dtype=np.int8), np.array([0.1, 0.2]), 0.9)
        self.assertEqual(result["alerts"], 0)
        self.assertEqual(result["event1_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
