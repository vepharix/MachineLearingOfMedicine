from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("mortality_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FeatureExtractionTests(unittest.TestCase):
    def test_elapsed_time_validation(self) -> None:
        self.assertEqual(MODULE.parse_elapsed_minutes("47:59"), 2879)
        with self.assertRaises(ValueError):
            MODULE.parse_elapsed_minutes("01:60")

    def test_same_minute_values_are_collapsed_before_summarizing(self) -> None:
        summary, duplicate_rows = MODULE.summarize_events(
            [(60, 80.0), (60, 100.0), (420, 120.0)], horizon_hours=12
        )
        self.assertEqual(duplicate_rows, 1)
        self.assertEqual(summary["count"], 2.0)
        self.assertEqual(summary["first"], 90.0)
        self.assertEqual(summary["last"], 120.0)
        self.assertAlmostEqual(summary["mean"], 105.0)
        self.assertAlmostEqual(summary["slope_per_hour"], 5.0)

    def test_horizon_is_strict_and_missing_rows_are_ignored(self) -> None:
        strict_summary, _ = MODULE.summarize_events([(359, 70.0), (360, 90.0)], horizon_hours=6)
        self.assertEqual(strict_summary["count"], 1.0)
        self.assertEqual(strict_summary["last"], 70.0)

    def test_read_record_drops_blank_and_minus_one_rows(self) -> None:
        content = """Time,Parameter,Value
00:00,RecordID,100001
00:00,Age,60
00:00,Gender,-1
00:00,ICUType,3
00:00,Weight,70
01:00,HR,80
01:00,,3.2
02:00,Temp,-1
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "100001.csv"
            path.write_text(content, encoding="utf-8")
            static, events, quality = MODULE.read_record(path)
        self.assertEqual(static["Age"], 60.0)
        self.assertEqual(static["InitialWeight"], 70.0)
        self.assertNotIn("Gender", static)
        self.assertEqual(events["Weight"], [(0, 70.0)])
        self.assertEqual(events["HR"], [(60, 80.0)])
        self.assertNotIn("Temp", events)
        self.assertEqual(quality["blank_parameter_rows_skipped"], 1)
        self.assertEqual(quality["minus_one_rows_treated_as_missing"], 2)


if __name__ == "__main__":
    unittest.main()
