#!/usr/bin/env python3
"""Incremental deployment-oriented analyses for Experiment 03.

This script deliberately does not repeat the preprocessing screen, subgroup
tables, feature ablation, bootstrap, or label sensitivity analyses. It reuses
Experiment 03's cached 48-hour raw feature matrix and selected gradient-
boosting configuration to add three analyses: probability recalibration,
sensitivity-constrained operating points, and leave-one-ICU-type-out stress
tests.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


EXPERIMENT_DIR = Path(__file__).resolve().parent
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

INK = "#20313B"
GRID = "#D9E0E4"
BLUE = "#1F5A7A"
ORANGE = "#D46A3A"
GREEN = "#4F8A5B"


def font(size: int, bold: bool = False):
    filename = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / filename
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def calibration_summary(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibrator.fit(logit(probability).reshape(-1, 1), y)
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0])


def calibration_bins(y: np.ndarray, probability: np.ndarray, method: str) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y, "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], q=10, duplicates="drop")
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
    result.insert(0, "method", method)
    return result


def sensitivity_threshold(y: np.ndarray, probability: np.ndarray, target: float) -> float:
    """Return the highest validation threshold whose sensitivity reaches target."""
    _, sensitivities, thresholds = roc_curve(y, probability)
    eligible = np.isfinite(thresholds) & (sensitivities >= target)
    if not eligible.any():
        return 0.0
    return float(np.max(thresholds[eligible]))


def icu_types_from_matrix(matrix: np.ndarray, columns: list[str]) -> np.ndarray:
    column_index = {column: index for index, column in enumerate(columns)}
    result = np.full(len(matrix), -1, dtype=np.int8)
    for value in experiment.baseline.ICU_TYPES:
        result[matrix[:, column_index[f"ICUType_{value}"]] == 1] = value
    return result


def selected_hgb_model(metadata: dict, columns: list[str], random_state: int):
    selected_name = metadata["selected_configs"]["hist_gradient_boosting"]
    configs = {config.name: config for config in experiment.CONFIGS}
    return experiment.build_model(
        "hist_gradient_boosting", configs[selected_name], columns, random_state
    )


def recalibration_analysis(
    y: np.ndarray,
    validation_probability: np.ndarray,
    test_probability: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_y = y[validation_indices]
    test_y = y[test_indices]
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    platt.fit(logit(validation_probability).reshape(-1, 1), validation_y)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(validation_probability, validation_y)
    methods = {
        "uncalibrated": (validation_probability, test_probability),
        "platt": (
            platt.predict_proba(logit(validation_probability).reshape(-1, 1))[:, 1],
            platt.predict_proba(logit(test_probability).reshape(-1, 1))[:, 1],
        ),
        "isotonic": (isotonic.predict(validation_probability), isotonic.predict(test_probability)),
    }

    metric_rows = []
    curve_frames = []
    for method, (validation_prob, test_prob) in methods.items():
        threshold = experiment.baseline.choose_youden_threshold(validation_y, validation_prob)
        row = {"method": method}
        row.update(experiment.baseline.metric_row(test_y, test_prob, threshold))
        intercept, slope = calibration_summary(test_y, test_prob)
        row.update({"calibration_intercept": intercept, "calibration_slope": slope})
        metric_rows.append(row)
        curve_frames.append(calibration_bins(test_y, test_prob, method))
    return pd.DataFrame(metric_rows), pd.concat(curve_frames, ignore_index=True)


def operating_point_analysis(
    y: np.ndarray,
    validation_probability: np.ndarray,
    test_probability: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for target in (0.70, 0.80, 0.90, 0.95):
        threshold = sensitivity_threshold(y[validation_indices], validation_probability, target)
        test_prediction = test_probability >= threshold
        row = {"target_sensitivity": target}
        row.update(experiment.baseline.metric_row(y[test_indices], test_probability, threshold))
        row["alerts_per_100"] = float(100 * test_prediction.mean())
        row["false_negatives"] = int(((y[test_indices] == 1) & ~test_prediction).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_icu_out_analysis(
    matrix: np.ndarray,
    columns: list[str],
    y: np.ndarray,
    metadata: dict,
    random_state: int,
) -> pd.DataFrame:
    icu_types = icu_types_from_matrix(matrix, columns)
    icu_feature_indices = [columns.index(f"ICUType_{value}") for value in experiment.baseline.ICU_TYPES]
    matrix_without_icu = np.delete(matrix, icu_feature_indices, axis=1)
    columns_without_icu = [
        column for index, column in enumerate(columns) if index not in icu_feature_indices
    ]
    selected_name = metadata["selected_configs"]["hist_gradient_boosting"]
    configs = {config.name: config for config in experiment.CONFIGS}
    rows = []
    for held_out in experiment.baseline.ICU_TYPES:
        source = np.flatnonzero(icu_types != held_out)
        source_train, source_validation = train_test_split(
            source,
            test_size=0.20,
            random_state=random_state + held_out,
            stratify=y[source],
        )
        destination = np.flatnonzero(icu_types == held_out)
        model = experiment.build_model(
            "hist_gradient_boosting",
            configs[selected_name],
            columns_without_icu,
            random_state + held_out,
        )
        model.fit(matrix_without_icu[source_train], y[source_train])
        validation_probability = model.predict_proba(matrix_without_icu[source_validation])[:, 1]
        threshold = experiment.baseline.choose_youden_threshold(
            y[source_validation], validation_probability
        )
        probability = model.predict_proba(matrix_without_icu[destination])[:, 1]
        row = {"held_out_icu": held_out, "predicted_mean": float(probability.mean())}
        row.update(experiment.baseline.metric_row(y[destination], probability, threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_calibration(curves: pd.DataFrame, destination: Path) -> None:
    width, height = 920, 700
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 105, 75, 855, 590
    maximum = 0.65
    for tick in np.linspace(0, maximum, 6):
        x = left + (right - left) * tick / maximum
        y = bottom - (bottom - top) * tick / maximum
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((x - 18, bottom + 12), f"{tick:.2f}", font=font(16), fill=INK)
        draw.text((left - 54, y - 9), f"{tick:.2f}", font=font(16), fill=INK)
    draw.line((left, bottom, right, top), fill="#7A8792", width=2)
    colors = {"uncalibrated": BLUE, "platt": ORANGE, "isotonic": GREEN}
    labels = {"uncalibrated": "Uncalibrated", "platt": "Platt", "isotonic": "Isotonic"}
    for method, group in curves.groupby("method", sort=False):
        points = []
        for row in group.itertuples():
            predicted = min(max(float(row.mean_predicted_probability), 0.0), maximum)
            observed = min(max(float(row.observed_event_rate), 0.0), maximum)
            x = left + (right - left) * predicted / maximum
            y = bottom - (bottom - top) * observed / maximum
            points.append((x, y))
        draw.line(points, fill=colors[method], width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[method])
    draw.rectangle((left, top, right, bottom), outline=INK, width=2)
    draw.text((285, 25), "48-hour probability calibration", font=font(25, True), fill=INK)
    draw.text((335, 635), "Mean predicted probability", font=font(19), fill=INK)
    draw.text((18, 35), "Observed event rate", font=font(18), fill=INK)
    legend_x = 575
    for index, method in enumerate(("uncalibrated", "platt", "isotonic")):
        y = 92 + index * 30
        draw.line((legend_x, y, legend_x + 28, y), fill=colors[method], width=4)
        draw.text((legend_x + 38, y - 10), labels[method], font=font(16), fill=INK)
    image.save(destination)


def plot_operating_points(metrics: pd.DataFrame, destination: Path) -> None:
    width, height = 920, 610
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 95, 85, 860, 500
    maximum = max(60.0, float(metrics[["alerts_per_100", "false_negatives"]].to_numpy().max()) * 1.1)
    for tick in np.linspace(0, maximum, 5):
        y = bottom - (bottom - top) * tick / maximum
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 48, y - 9), f"{tick:.0f}", font=font(16), fill=INK)
    group_width = (right - left) / len(metrics)
    bar_width = group_width * 0.26
    for index, row in enumerate(metrics.itertuples()):
        center_x = left + group_width * (index + 0.5)
        for offset, value, color in (
            (-bar_width, row.alerts_per_100, BLUE),
            (0, row.false_negatives, ORANGE),
        ):
            x1 = center_x + offset
            x2 = x1 + bar_width
            y = bottom - (bottom - top) * value / maximum
            draw.rectangle((x1, y, x2, bottom), fill=color)
            draw.text((x1, y - 24), f"{value:.0f}", font=font(16, True), fill=color)
        draw.text((center_x - 22, bottom + 14), f"{row.target_sensitivity:.0%}", font=font(17), fill=INK)
    draw.rectangle((left, top, right, bottom), outline=INK, width=2)
    draw.text((245, 26), "Sensitivity target and review burden", font=font(25, True), fill=INK)
    draw.text((335, 555), "Validation target sensitivity", font=font(19), fill=INK)
    draw.rectangle((610, 100, 628, 118), fill=BLUE)
    draw.text((638, 98), "Alerts / 100", font=font(16), fill=INK)
    draw.rectangle((610, 130, 628, 148), fill=ORANGE)
    draw.text((638, 128), "Missed deaths (of 256)", font=font(16), fill=INK)
    image.save(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = args.source_output_dir / "data" / "feature_profiles.npz"
    metadata_path = args.source_output_dir / "run_metadata.json"
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Run Experiment 03's run_experiment.py before this extension.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cache = np.load(cache_path)
    columns = cache["columns"].tolist()
    y = cache["y"].astype(np.int8)
    matrix = cache["raw_48h"]
    random_state = int(metadata["random_state"])
    splits = experiment.fixed_splits(y, random_state)

    model = selected_hgb_model(metadata, columns, random_state)
    model.fit(matrix[splits["train"]], y[splits["train"]])
    validation_probability = model.predict_proba(matrix[splits["validation"]])[:, 1]
    test_probability = model.predict_proba(matrix[splits["test"]])[:, 1]

    calibration_metrics, calibration_curve = recalibration_analysis(
        y,
        validation_probability,
        test_probability,
        splits["validation"],
        splits["test"],
    )
    operating_points = operating_point_analysis(
        y,
        validation_probability,
        test_probability,
        splits["validation"],
        splits["test"],
    )
    leave_one_icu = leave_one_icu_out_analysis(matrix, columns, y, metadata, random_state)

    output = args.output_dir
    figure_output = output / "report_figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)
    calibration_metrics.to_csv(output / "calibration_method_metrics.csv", index=False)
    calibration_curve.to_csv(output / "calibration_method_curve.csv", index=False)
    operating_points.to_csv(output / "threshold_operating_points.csv", index=False)
    leave_one_icu.to_csv(output / "leave_one_icu_out.csv", index=False)
    plot_calibration(calibration_curve, figure_output / "calibration_method_comparison.png")
    plot_operating_points(operating_points, figure_output / "threshold_tradeoff.png")

    extension_metadata = {
        "source_experiment": "03_mortality_data_quality",
        "horizon_hours": 48,
        "model": "hist_gradient_boosting",
        "preprocess": metadata["selected_configs"]["hist_gradient_boosting"],
        "random_state": random_state,
        "analyses": ["recalibration", "sensitivity_operating_points", "leave_one_icu_out"],
    }
    (output / "operating_point_metadata.json").write_text(
        json.dumps(extension_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Incremental operating-point outputs written to {output}", flush=True)


if __name__ == "__main__":
    main()
