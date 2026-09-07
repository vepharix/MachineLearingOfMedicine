#!/usr/bin/env python3
"""Lightweight tests for Experiment 03 operating-point helpers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parent / "analyze_operating_points.py"
SPEC = importlib.util.spec_from_file_location("mortality_operating_points", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sensitivity_threshold_uses_validation_target():
    y = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    probability = np.array([0.05, 0.10, 0.40, 0.35, 0.80])
    threshold = MODULE.sensitivity_threshold(y, probability, 1.0)
    prediction = probability >= threshold
    assert prediction[y == 1].mean() == 1.0
    assert threshold == 0.35


def test_logit_clips_extreme_probabilities():
    transformed = MODULE.logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(transformed).all()
    assert transformed[0] < 0 < transformed[-1]


if __name__ == "__main__":
    test_sensitivity_threshold_uses_validation_target()
    test_logit_clips_extreme_probabilities()
    print("All operating-point helper tests passed.")
