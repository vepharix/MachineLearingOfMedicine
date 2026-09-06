#!/usr/bin/env python3
"""Secondary analyses for Experiment 03 using the cached feature matrices.

This script does not rescan raw ICU files. It reproduces the fixed split,
computes calibration and subgroup summaries from frozen test predictions, and
fits pre-specified feature ablations for the two selected model pipelines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
MAIN_PATH = EXPERIMENT_DIR / "run_experiment.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("mortality_quality_main", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = load_main_module()


def calibration_summary(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(probability.astype(float), 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
    calibrator.fit(logit, y)
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0])


def calibration_bins(
    y: np.ndarray, probability: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y, "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, duplicates="drop")
    result = (
        frame.groupby("bin", observed=True)
        .agg(
            n=("outcome", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_event_rate=("outcome", "mean"),
        )
        .reset_index(drop=True)
    )
    result.insert(0, "calibration_bin", np.arange(1, len(result) + 1))
    return result


def safe_metric_row(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    if len(y) == 0 or np.unique(y).size < 2:
        return {
            "n": int(len(y)),
            "prevalence": float(np.mean(y)) if len(y) else np.nan,
            **{
                key: np.nan
                for key in (
                    "auroc",
                    "auprc",
                    "brier",
                    "log_loss",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "f1",
                )
            },
        }
    return experiment.baseline.metric_row(y, probability, threshold)


def subgroup_definitions(columns: list[str], matrix: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    column_index = {column: index for index, column in enumerate(columns)}
    age = matrix[:, column_index["Age"]]
    gender = matrix[:, column_index["Gender"]]
    icu = np.full(len(matrix), np.nan)
    for icu_type in experiment.baseline.ICU_TYPES:
        present = matrix[:, column_index[f"ICUType_{icu_type}"]] == 1
        icu[present] = icu_type
    return {
        "ICU类型": {
            "CCU": icu == 1,
            "CSRU": icu == 2,
            "MICU": icu == 3,
            "SICU": icu == 4,
            "未知": np.isnan(icu),
        },
        "性别": {
            "女性": gender == 0,
            "男性": gender == 1,
            "未知": ~np.isin(gender, [0, 1]),
        },
        "年龄": {
            "<45岁": age < 45,
            "45–64岁": (age >= 45) & (age < 65),
            "65–79岁": (age >= 65) & (age < 80),
            "≥80岁": age >= 80,
            "未知": np.isnan(age),
        },
    }


def prediction_analyses(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    columns: list[str],
    raw_matrix: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = outcomes["In-hospital_death"].to_numpy(dtype=np.int8)
    test_ids = outcomes.iloc[test_indices]["RecordID"].to_numpy(dtype=int)
    definitions = subgroup_definitions(columns, raw_matrix)
    calibration_rows = []
    curve_rows = []
    subgroup_rows = []
    for (model_name, horizon), group in predictions.groupby(["model", "horizon_hours"]):
        ordered = group.set_index("RecordID").loc[test_ids]
        probability = ordered["probability"].to_numpy(dtype=float)
        threshold = float(ordered["threshold"].iloc[0])
        test_y = y[test_indices]
        intercept, slope = calibration_summary(test_y, probability)
        calibration_rows.append(
            {
                "model": model_name,
                "horizon_hours": int(horizon),
                "calibration_intercept": intercept,
                "calibration_slope": slope,
            }
        )
        curve = calibration_bins(test_y, probability)
        curve.insert(0, "horizon_hours", int(horizon))
        curve.insert(0, "model", model_name)
        curve_rows.extend(curve.to_dict("records"))

        for subgroup_type, levels in definitions.items():
            for subgroup_level, all_mask in levels.items():
                mask = all_mask[test_indices]
                if not mask.any():
                    continue
                row = {
                    "model": model_name,
                    "horizon_hours": int(horizon),
                    "subgroup_type": subgroup_type,
                    "subgroup_level": subgroup_level,
                    "threshold": threshold,
                }
                row.update(safe_metric_row(test_y[mask], probability[mask], threshold))
                subgroup_rows.append(row)
    return (
        pd.DataFrame(calibration_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(subgroup_rows),
    )


def feature_groups(columns: list[str]) -> dict[str, np.ndarray]:
    static = np.array(["__" not in column for column in columns])
    process = np.array(
        [column.endswith("__count") or column.endswith("__hours_since_last") for column in columns]
    )
    value = ~process
    return {
        "static_only": np.flatnonzero(static),
        "clinical_values_no_indicator": np.flatnonzero(value),
        "clinical_values_plus_indicator": np.flatnonzero(value),
        "measurement_process_only": np.flatnonzero(static | process),
        "full_selected": np.arange(len(columns)),
    }


def ablation_analysis(
    matrices,
    columns: list[str],
    y: np.ndarray,
    split_indices,
    selected: dict[str, str],
    random_state: int,
) -> pd.DataFrame:
    configs = {config.name: config for config in experiment.CONFIGS}
    groups = feature_groups(columns)
    rows = []
    for model_name, selected_name in selected.items():
        selected_config = configs[selected_name]
        for group_name, selected_columns in groups.items():
            config = selected_config
            if group_name == "clinical_values_no_indicator":
                config = replace(config, add_indicator=False)
            elif group_name in {"clinical_values_plus_indicator", "measurement_process_only"}:
                config = replace(config, add_indicator=True)
            subset_columns = [columns[index] for index in selected_columns]
            for horizon in (6, 12, 24, 48):
                x = matrices[selected_config.feature_profile][horizon][:, selected_columns]
                model = experiment.build_model(model_name, config, subset_columns, random_state)
                model.fit(x[split_indices["train"]], y[split_indices["train"]])
                validation_probability = model.predict_proba(x[split_indices["validation"]])[:, 1]
                threshold = experiment.baseline.choose_youden_threshold(
                    y[split_indices["validation"]], validation_probability
                )
                for split_name in ("validation", "test"):
                    indices = split_indices[split_name]
                    probability = (
                        validation_probability
                        if split_name == "validation"
                        else model.predict_proba(x[indices])[:, 1]
                    )
                    row = {
                        "model": model_name,
                        "preprocess": selected_name,
                        "feature_group": group_name,
                        "feature_count": int(len(selected_columns)),
                        "horizon_hours": horizon,
                        "split": split_name,
                        "threshold": threshold,
                    }
                    row.update(experiment.baseline.metric_row(y[indices], probability, threshold))
                    rows.append(row)
                print(f"Ablation {model_name} {group_name} {horizon}h complete", flush=True)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "release")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.output_dir / "data" / "feature_profiles.npz"
    prediction_path = args.output_dir / "data" / "test_predictions.csv"
    metadata_path = args.output_dir / "run_metadata.json"
    if not cache_path.exists() or not prediction_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run run_experiment.py before this secondary analysis.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache = np.load(cache_path)
    columns = cache["columns"].tolist()
    y = cache["y"].astype(np.int8)
    record_ids = cache["RecordID"].astype(int)
    matrices = {
        profile: {horizon: cache[f"{profile}_{horizon}h"] for horizon in (6, 12, 24, 48)}
        for profile in ("raw", "clean")
    }
    outcomes = pd.read_csv(args.data_dir / "outcomes.csv")
    outcomes = outcomes[outcomes["RecordID"].isin(record_ids)].copy()
    outcomes = outcomes.sort_values("RecordID").reset_index(drop=True)
    if not np.array_equal(outcomes["RecordID"].to_numpy(dtype=int), record_ids):
        raise ValueError("Cached RecordID order does not match outcomes.csv")
    predictions = pd.read_csv(prediction_path)
    split_indices = experiment.fixed_splits(y, int(metadata["random_state"]))

    calibration, calibration_curve, subgroups = prediction_analyses(
        predictions,
        outcomes,
        columns,
        matrices["raw"][48],
        split_indices["test"],
    )
    ablations = ablation_analysis(
        matrices,
        columns,
        y,
        split_indices,
        metadata["selected_configs"],
        int(metadata["random_state"]),
    )
    calibration.to_csv(args.output_dir / "calibration_metrics.csv", index=False)
    calibration_curve.to_csv(args.output_dir / "calibration_curve.csv", index=False)
    subgroups.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    ablations.to_csv(args.output_dir / "feature_ablation_metrics.csv", index=False)

    hard_mask = experiment.hard_outcome_anomaly_mask(outcomes)
    metadata["hard_outcome_anomalies_by_split"] = {
        split: int(hard_mask[indices].sum()) for split, indices in split_indices.items()
    }
    metadata["secondary_outputs"] = [
        "calibration_metrics.csv",
        "calibration_curve.csv",
        "subgroup_metrics.csv",
        "feature_ablation_metrics.csv",
    ]
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Secondary outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
