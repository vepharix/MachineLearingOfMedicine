#!/usr/bin/env python3
"""Experiment 02: reproducible multi-horizon in-hospital mortality baselines.

The source ICU files are read-only. Patient-level features, split assignments,
and predictions are written under ``output/data`` (ignored by Git). Aggregate
metrics, coverage summaries, and run metadata are safe to keep in the repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

STATIC_NUMERIC = ("Age", "Gender", "Height", "InitialWeight")
ICU_TYPES = (1, 2, 3, 4)
TIME_VARYING = (
    "Albumin",
    "ALP",
    "ALT",
    "AST",
    "Bilirubin",
    "BUN",
    "Cholesterol",
    "Creatinine",
    "DiasABP",
    "FiO2",
    "GCS",
    "Glucose",
    "HCO3",
    "HCT",
    "HR",
    "K",
    "Lactate",
    "MAP",
    "MechVent",
    "Mg",
    "Na",
    "NIDiasABP",
    "NIMAP",
    "NISysABP",
    "PaCO2",
    "PaO2",
    "pH",
    "Platelets",
    "RespRate",
    "SaO2",
    "SysABP",
    "Temp",
    "TroponinI",
    "TroponinT",
    "Urine",
    "WBC",
    "Weight",
)
SUMMARY_STATS = (
    "first",
    "last",
    "min",
    "max",
    "mean",
    "std",
    "count",
    "slope_per_hour",
    "hours_since_last",
)
STATIC_SOURCE_PARAMS = {"RecordID", "Age", "Gender", "Height", "ICUType"}


def parse_elapsed_minutes(value: str) -> int:
    """Convert an HH:MM elapsed-time value into integer minutes."""

    hour, minute = (int(part) for part in value.split(":"))
    if hour < 0 or minute < 0 or minute >= 60:
        raise ValueError(f"invalid elapsed time: {value!r}")
    return hour * 60 + minute


def build_feature_columns() -> list[str]:
    columns = list(STATIC_NUMERIC)
    columns.extend(f"ICUType_{icu_type}" for icu_type in ICU_TYPES)
    columns.extend(f"{parameter}__{stat}" for parameter in TIME_VARYING for stat in SUMMARY_STATS)
    return columns


def collapse_same_minute(events: Iterable[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse conflicting same-minute readings using their median."""

    grouped: dict[int, list[float]] = defaultdict(list)
    for minute, value in events:
        grouped[minute].append(value)
    minutes = np.asarray(sorted(grouped), dtype=np.float64)
    values = np.asarray([np.median(grouped[int(minute)]) for minute in minutes], dtype=np.float64)
    duplicate_extra_rows = sum(len(items) - 1 for items in grouped.values())
    return minutes, values, duplicate_extra_rows


def summarize_events(events: Iterable[tuple[int, float]], horizon_hours: int) -> tuple[dict[str, float], int]:
    """Summarize valid readings strictly before a prediction horizon."""

    cutoff = horizon_hours * 60
    eligible = [(minute, value) for minute, value in events if 0 <= minute < cutoff]
    if not eligible:
        return {stat: (0.0 if stat == "count" else math.nan) for stat in SUMMARY_STATS}, 0

    minutes, values, duplicate_extra_rows = collapse_same_minute(eligible)
    if len(minutes) >= 2 and float(np.var(minutes)) > 0:
        centered_minutes = minutes - float(np.mean(minutes))
        centered_values = values - float(np.mean(values))
        slope_per_minute = float(np.dot(centered_minutes, centered_values) / np.dot(centered_minutes, centered_minutes))
        slope_per_hour = slope_per_minute * 60.0
    else:
        slope_per_hour = math.nan

    summary = {
        "first": float(values[0]),
        "last": float(values[-1]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)) if len(values) >= 2 else 0.0,
        "count": float(len(values)),
        "slope_per_hour": slope_per_hour,
        "hours_since_last": float(horizon_hours - minutes[-1] / 60.0),
    }
    return summary, duplicate_extra_rows


def read_record(path: Path) -> tuple[dict[str, float], dict[str, list[tuple[int, float]]], Counter]:
    """Read one long-format ICU record without modifying the source file."""

    static: dict[str, float] = {}
    events: dict[str, list[tuple[int, float]]] = defaultdict(list)
    quality = Counter()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Time", "Parameter", "Value"]:
            raise ValueError(f"unexpected schema in {path}: {reader.fieldnames}")
        for row in reader:
            quality["source_rows"] += 1
            parameter = row["Parameter"].strip()
            if not parameter:
                quality["blank_parameter_rows_skipped"] += 1
                continue
            try:
                minute = parse_elapsed_minutes(row["Time"])
                value = float(row["Value"])
            except (TypeError, ValueError):
                quality["unparseable_rows_skipped"] += 1
                continue
            if minute == 48 * 60:
                quality["rows_at_48_00_excluded"] += 1
            if value == -1:
                quality["minus_one_rows_treated_as_missing"] += 1
                continue
            if parameter == "Weight" and minute == 0 and "InitialWeight" not in static:
                static["InitialWeight"] = value
            if parameter in STATIC_SOURCE_PARAMS:
                if parameter not in static:
                    static[parameter] = value
            elif parameter in TIME_VARYING:
                events[parameter].append((minute, value))
            else:
                quality["unknown_parameter_rows_skipped"] += 1
    return static, events, quality


def extract_feature_matrices(
    record_paths: list[Path], horizons: list[int]
) -> tuple[list[str], dict[int, np.ndarray], Counter, list[dict[str, float]]]:
    """Extract fixed-width patient matrices for every requested horizon."""

    columns = build_feature_columns()
    column_index = {column: index for index, column in enumerate(columns)}
    matrices = {
        horizon: np.full((len(record_paths), len(columns)), np.nan, dtype=np.float32)
        for horizon in horizons
    }
    count_indices = [column_index[f"{parameter}__count"] for parameter in TIME_VARYING]
    for matrix in matrices.values():
        matrix[:, count_indices] = 0.0

    quality = Counter()
    coverage_counts = {horizon: Counter() for horizon in horizons}

    for row_index, path in enumerate(record_paths):
        static, events, record_quality = read_record(path)
        quality.update(record_quality)
        expected_record_id = int(path.stem)
        first_record_id = int(static["RecordID"]) if "RecordID" in static else None
        if first_record_id != expected_record_id:
            raise ValueError(f"RecordID mismatch in {path}: {first_record_id}")

        for horizon in horizons:
            matrix = matrices[horizon]
            for parameter in STATIC_NUMERIC:
                if parameter in static:
                    matrix[row_index, column_index[parameter]] = static[parameter]
            icu_type = int(static["ICUType"]) if "ICUType" in static else None
            if icu_type in ICU_TYPES:
                matrix[row_index, column_index[f"ICUType_{icu_type}"]] = 1.0
                for other_type in ICU_TYPES:
                    if other_type != icu_type:
                        matrix[row_index, column_index[f"ICUType_{other_type}"]] = 0.0

            for parameter in TIME_VARYING:
                summary, duplicate_extra_rows = summarize_events(events.get(parameter, ()), horizon)
                quality[f"same_minute_extra_rows_collapsed_{horizon}h"] += duplicate_extra_rows
                if summary["count"] > 0:
                    coverage_counts[horizon][parameter] += 1
                for stat, value in summary.items():
                    matrix[row_index, column_index[f"{parameter}__{stat}"]] = value

        if (row_index + 1) % 1000 == 0 or row_index + 1 == len(record_paths):
            print(f"Extracted {row_index + 1:,}/{len(record_paths):,} records", flush=True)

    coverage_rows = []
    for horizon in horizons:
        for parameter in TIME_VARYING:
            observed = int(coverage_counts[horizon][parameter])
            coverage_rows.append(
                {
                    "horizon_hours": horizon,
                    "parameter": parameter,
                    "patients_observed": observed,
                    "coverage_rate": observed / len(record_paths),
                }
            )
    return columns, matrices, quality, coverage_rows


def choose_youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    score = np.where(finite, tpr - fpr, -np.inf)
    return float(thresholds[int(np.argmax(score))])


def metric_row(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
    }


def logistic_model(random_state: int):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=2500, solver="lbfgs", random_state=random_state),
    )


def gradient_boosting_model(random_state: int):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=random_state,
        ),
    )


def fit_and_evaluate(
    *,
    experiment_name: str,
    feature_set: str,
    horizon_hours: int | None,
    model,
    x: np.ndarray,
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    metric_rows: list[dict[str, object]],
    prediction_frame: pd.DataFrame,
) -> None:
    model.fit(x[split_indices["train"]], y[split_indices["train"]])
    probabilities = {
        split: model.predict_proba(x[indices])[:, 1]
        for split, indices in split_indices.items()
        if split != "train"
    }
    threshold = choose_youden_threshold(y[split_indices["validation"]], probabilities["validation"])
    for split in ("validation", "test"):
        row: dict[str, object] = {
            "experiment": experiment_name,
            "feature_set": feature_set,
            "horizon_hours": horizon_hours,
            "model": type(model[-1]).__name__ if hasattr(model, "__getitem__") else type(model).__name__,
            "split": split,
            "threshold_selection": "maximum validation Youden J",
        }
        row.update(metric_row(y[split_indices[split]], probabilities[split], threshold))
        metric_rows.append(row)

        column = f"{experiment_name}_probability"
        prediction_frame.loc[split_indices[split], column] = probabilities[split]


def markdown_test_table(metrics: pd.DataFrame) -> str:
    test = metrics[metrics["split"] == "test"].copy()
    lines = [
        "| 特征集 | 模型 | AUROC | AUPRC | Brier | 灵敏度 | 特异度 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in test.itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {row.model} | {row.auroc:.3f} | {row.auprc:.3f} | "
            f"{row.brier:.3f} | {row.sensitivity:.3f} | {row.specificity:.3f} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    split_sizes: dict[str, int],
    horizons: list[int],
) -> None:
    report = f"""# 实验 02：多时间窗院内死亡预测基线

## 目的

使用入 ICU 后前 {"、".join(str(value) for value in horizons)} 小时的原始静态信息、生命体征和实验室检查预测院内死亡。所有时间特征都严格使用预测时点之前的观测，`RecordID` 不进入模型。SAPS-I 与 SOFA 只组成一个不与早期时间窗严格匹配的参照模型，不混入原始时序模型。

## 方法

每个时序变量在各时间窗内生成首次值、末次值、最小值、最大值、均值、标准差、有效时间点数、每小时线性趋势和距末次测量时间。同一分钟的重复测量先取中位数；`-1` 按缺失处理；空参数和未知参数排除；48:00 的记录不进入“前48小时”特征。

数据按 ICU 住院记录固定分为训练集 {split_sizes['train']:,} 例、验证集 {split_sizes['validation']:,} 例和测试集 {split_sizes['test']:,} 例，并按院内死亡分层。两类模型都使用仅从训练集学习的中位数填补并保留缺失指示；逻辑回归额外进行标准化。分类阈值仅在验证集上按 Youden J 选择，随后原样应用到测试集。

## 测试集结果

{markdown_test_table(metrics)}

这里的单次固定划分用于建立可复现基线，并不等同于外部验证。AUPRC 应结合测试集死亡率理解；Brier 分数越低越好。灵敏度和特异度依赖验证集选出的阈值，不能直接解释为临床决策阈值。

## 边界与后续工作

数据采用仓库文档规定的 `release/outcomes.csv` 与 `release/icu_records/` 布局。数据没有患者级身份、医院或入院日期，因此无法排除同一患者多次住院，也不能执行医院外或时间外验证。后续应增加重复分层交叉验证、置信区间、校准曲线与 ICU 类型/年龄/性别亚组评估，再考虑更复杂的非规则时间序列模型。

汇总结果保存在 `output/metrics.csv`、`output/feature_coverage.csv` 和 `output/run_metadata.json`；患者级划分与预测位于被 Git 忽略的 `output/data/`。
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "release")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 24, 48])
    parser.add_argument("--random-state", type=int, default=20260904)
    parser.add_argument("--max-records", type=int, default=None, help="Optional deterministic subset for smoke tests.")
    parser.add_argument("--save-features", action="store_true", help="Save patient-level feature matrices under output/data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = sorted(set(args.horizons))
    if not horizons or any(horizon <= 0 or horizon > 48 for horizon in horizons):
        raise ValueError("horizons must be unique integers between 1 and 48")

    outcomes_path = args.data_dir / "outcomes.csv"
    record_dir = args.data_dir / "icu_records"
    if not outcomes_path.exists() or not record_dir.is_dir():
        raise FileNotFoundError(
            f"Expected outcomes.csv and icu_records/ under {args.data_dir}. "
            "Pass --data-dir when the private dataset is stored outside this clone."
        )

    outcomes = pd.read_csv(outcomes_path)
    required = ["RecordID", "SAPS-I", "SOFA", "In-hospital_death"]
    missing_columns = sorted(set(required) - set(outcomes.columns))
    if missing_columns:
        raise ValueError(f"outcomes.csv is missing columns: {missing_columns}")
    outcomes[required] = outcomes[required].apply(pd.to_numeric, errors="coerce")
    outcomes = outcomes[outcomes["In-hospital_death"].isin([0, 1])].copy()
    outcomes["RecordID"] = outcomes["RecordID"].astype(int)
    outcomes["In-hospital_death"] = outcomes["In-hospital_death"].astype(int)
    outcomes = outcomes.sort_values("RecordID").reset_index(drop=True)
    if args.max_records is not None:
        outcomes = outcomes.head(args.max_records).copy()
    if outcomes["In-hospital_death"].nunique() != 2:
        raise ValueError("selected records must contain both mortality classes")

    record_paths = [record_dir / f"{record_id}.csv" for record_id in outcomes["RecordID"]]
    missing_files = [str(path) for path in record_paths if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"missing {len(missing_files)} ICU files; first: {missing_files[0]}")

    feature_columns, matrices, quality, coverage_rows = extract_feature_matrices(record_paths, horizons)
    y = outcomes["In-hospital_death"].to_numpy(dtype=np.int8)
    all_indices = np.arange(len(outcomes))
    train_indices, remaining = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=args.random_state,
        stratify=y,
    )
    validation_indices, test_indices = train_test_split(
        remaining,
        test_size=0.50,
        random_state=args.random_state,
        stratify=y[remaining],
    )
    split_indices = {
        "train": np.sort(train_indices),
        "validation": np.sort(validation_indices),
        "test": np.sort(test_indices),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_output = args.output_dir / "data"
    data_output.mkdir(parents=True, exist_ok=True)
    split_labels = np.full(len(outcomes), "", dtype=object)
    for split, indices in split_indices.items():
        split_labels[indices] = split
    prediction_frame = pd.DataFrame(
        {
            "RecordID": outcomes["RecordID"].to_numpy(),
            "In-hospital_death": y,
            "split": split_labels,
        }
    )

    metric_rows: list[dict[str, object]] = []
    static_column_count = len(STATIC_NUMERIC) + len(ICU_TYPES)
    first_matrix = matrices[horizons[0]]
    fit_and_evaluate(
        experiment_name="static_logistic",
        feature_set="静态信息",
        horizon_hours=None,
        model=logistic_model(args.random_state),
        x=first_matrix[:, :static_column_count],
        y=y,
        split_indices=split_indices,
        metric_rows=metric_rows,
        prediction_frame=prediction_frame,
    )

    score_features = outcomes[["SAPS-I", "SOFA"]].replace(-1, np.nan).to_numpy(dtype=np.float64)
    fit_and_evaluate(
        experiment_name="scores_reference_logistic",
        feature_set="SAPS-I+SOFA参照（非时间匹配）",
        horizon_hours=None,
        model=logistic_model(args.random_state),
        x=score_features,
        y=y,
        split_indices=split_indices,
        metric_rows=metric_rows,
        prediction_frame=prediction_frame,
    )

    for horizon in horizons:
        matrix = matrices[horizon]
        for short_name, model in (
            ("logistic", logistic_model(args.random_state)),
            ("hist_gradient_boosting", gradient_boosting_model(args.random_state)),
        ):
            fit_and_evaluate(
                experiment_name=f"{horizon}h_{short_name}",
                feature_set=f"前{horizon}小时原始记录",
                horizon_hours=horizon,
                model=model,
                x=matrix,
                y=y,
                split_indices=split_indices,
                metric_rows=metric_rows,
                prediction_frame=prediction_frame,
            )

        if args.save_features:
            frame = pd.DataFrame(matrix, columns=feature_columns)
            frame.insert(0, "RecordID", outcomes["RecordID"].to_numpy())
            frame.to_csv(data_output / f"features_{horizon}h.csv", index=False)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "feature_coverage.csv", index=False)
    prediction_frame.to_csv(data_output / "predictions_and_splits.csv", index=False)

    metadata = {
        "data_layout": "release/outcomes.csv + release/icu_records/<RecordID>.csv",
        "record_count": int(len(outcomes)),
        "death_count": int(y.sum()),
        "death_rate": float(y.mean()),
        "horizons_hours": horizons,
        "feature_count_per_horizon": len(feature_columns),
        "split_sizes": {split: int(len(indices)) for split, indices in split_indices.items()},
        "random_state": args.random_state,
        "scikit_learn_version": sklearn.__version__,
        "quality_counters": dict(sorted(quality.items())),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        EXPERIMENT_DIR / "README.md",
        metrics,
        metadata["split_sizes"],
        horizons,
    )
    print(metrics[metrics["split"] == "test"].to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
