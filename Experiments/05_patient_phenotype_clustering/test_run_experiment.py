from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("phenotype_clustering", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PhenotypeClusteringTests(unittest.TestCase):
    def test_combined_pressure_and_urine_features(self) -> None:
        columns = [source for _, source in MODULE.FEATURE_SPECS]
        columns += [
            "MAP__min",
            "NIMAP__min",
            "SysABP__min",
            "NISysABP__min",
            "Urine__mean",
            "Urine__count",
            "MechVent__count",
        ]
        matrix = np.zeros((3, len(columns)), dtype=float)
        position = {name: index for index, name in enumerate(columns)}
        matrix[:, position["MAP__min"]] = [60.0, np.nan, 0.0]
        matrix[:, position["NIMAP__min"]] = [65.0, 55.0, 70.0]
        matrix[:, position["SysABP__min"]] = [90.0, np.nan, 0.0]
        matrix[:, position["NISysABP__min"]] = [95.0, 85.0, 110.0]
        matrix[:, position["Urine__mean"]] = [100.0, 50.0, 75.0]
        matrix[:, position["Urine__count"]] = [3.0, 4.0, 2.0]
        matrix[:, position["MechVent__count"]] = [0.0, 2.0, 0.0]
        names, result = MODULE.build_phenotype_matrix(columns, matrix)
        self.assertEqual(result[0, names.index("MAP_min_combined")], 60.0)
        self.assertEqual(result[1, names.index("MAP_min_combined")], 55.0)
        self.assertEqual(result[1, names.index("Urine_sum_24h")], 200.0)
        self.assertEqual(result[0, names.index("MechVent_documented_24h")], 0.0)
        self.assertEqual(result[1, names.index("MechVent_documented_24h")], 1.0)
        self.assertEqual(result[2, names.index("MAP_min_combined")], 70.0)
        self.assertEqual(result[2, names.index("SysBP_min_combined")], 110.0)

    def test_choose_k_applies_stability_and_size_filters(self) -> None:
        metrics = pd.DataFrame(
            {
                "k": [2, 3, 4],
                "holdout_silhouette": [0.20, 0.30, 0.40],
                "stability_ari_median": [0.80, 0.78, 0.60],
                "minimum_train_cluster_fraction": [0.20, 0.10, 0.08],
            }
        )
        self.assertEqual(MODULE.choose_k(metrics), 3)

    def test_relabeling_is_ordered_by_pc1(self) -> None:
        labels = np.array([5, 5, 2, 2])
        scores = np.array([[2.0, 0.0], [3.0, 0.0], [-2.0, 0.0], [-1.0, 0.0]])
        relabeled, mapping = MODULE.relabel_by_pc1(labels, scores)
        np.testing.assert_array_equal(relabeled, np.array([2, 2, 1, 1]))
        self.assertEqual(mapping, {2: 1, 5: 2})


if __name__ == "__main__":
    unittest.main()
