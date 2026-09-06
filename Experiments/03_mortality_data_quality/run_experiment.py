#!/usr/bin/env python3
"""Experiment 03: mortality data-cleaning and imputation robustness study.

The experiment preserves Experiment 02's cohort, horizons, random seed and
train/validation/test split. Candidate preprocessing strategies are screened
with five-fold CV inside the original training set. The validation set selects
one strategy per model; the test set is evaluated only after selection.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "Experiments" / "02_mortality_prediction_baseline" / "run_experiment.py"


def load_baseline_module():
    spec = importlib.util.spec_from_file_location("mortality_baseline", BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


baseline = load_baseline_module()


@dataclass(frozen=True)
class PreprocessConfig:
    name: str
    feature_profile: str
    imputer_strategy: str | None
    add_indicator: bool
    winsorize: bool


CONFIGS = (
    PreprocessConfig("raw_median", "raw", "median", False, False),
    PreprocessConfig("raw_median_indicator", "raw", "median", True, False),
    PreprocessConfig("raw_mean_indicator", "raw", "mean", True, False),
    PreprocessConfig("raw_native_missing", "raw", None, False, False),
    PreprocessConfig("raw_winsor_median_indicator", "raw", "median", True, True),
    PreprocessConfig("clean_median_indicator", "clean", "median", True, False),
    PreprocessConfig("clean_winsor_median_indicator", "clean", "median", True, True),
)


STRICT_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "Height": (100.0, 250.0, True),
    "InitialWeight": (20.0, 300.0, True),
    "Weight": (20.0, 300.0, True),
    "HR": (0.0, 250.0, False),
    "Temp": (25.0, 45.0, True),
    "GCS": (3.0, 15.0, True),
    "pH": (6.5, 8.0, True),
    "RespRate": (0.0, 80.0, False),
    "FiO2": (0.0, 1.0, False),
    "SaO2": (0.0, 100.0, False),
    "DiasABP": (0.0, 300.0, False),
    "MAP": (0.0, 300.0, False),
    "SysABP": (0.0, 300.0, False),
    "NIDiasABP": (0.0, 300.0, False),
    "NIMAP": (0.0, 300.0, False),
    "NISysABP": (0.0, 300.0, False),
}


def value_is_plausible(parameter: str, value: float) -> bool:
    """Apply deliberately broad technical plausibility bounds, not normal ranges."""

    if not math.isfinite(value):
        return False
    if parameter == "Gender":
        return value in (0.0, 1.0)
    if parameter == "ICUType":
        return value in (1.0, 2.0, 3.0, 4.0)
    if parameter == "MechVent":
        return value in (0.0, 1.0)
    if parameter == "Urine":
        return value >= 0.0
    if parameter in STRICT_BOUNDS:
        lower, upper, inclusive_lower = STRICT_BOUNDS[parameter]
        lower_ok = value >= lower if inclusive_lower else value > lower
        return lower_ok and value <= upper
    return value >= 0.0


def read_record_profiles(
    path: Path,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, list[tuple[int, float]]]],
    Counter,
]:
    static = {"raw": {}, "clean": {}}
    events = {
        "raw": defaultdict(list),
        "clean": defaultdict(list),
    }
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
                minute = baseline.parse_elapsed_minutes(row["Time"])
                value = float(row["Value"])
            except (TypeError, ValueError):
                quality["unparseable_rows_skipped"] += 1
                continue
            if value == -1:
                quality["minus_one_rows_treated_as_missing"] += 1
                continue
            if minute == 48 * 60:
                quality["rows_at_48_00_excluded"] += 1

            if parameter == "Weight" and minute == 0:
                if "InitialWeight" not in static["raw"]:
                    static["raw"]["InitialWeight"] = value
                if value_is_plausible("InitialWeight", value) and "InitialWeight" not in static["clean"]:
                    static["clean"]["InitialWeight"] = value
                elif not value_is_plausible("InitialWeight", value):
                    quality["cleaned_InitialWeight_rows"] += 1

            if parameter in baseline.STATIC_SOURCE_PARAMS:
                if parameter not in static["raw"]:
                    static["raw"][parameter] = value
                if value_is_plausible(parameter, value):
                    if parameter not in static["clean"]:
                        static["clean"][parameter] = value
                else:
                    quality[f"cleaned_{parameter}_rows"] += 1
            elif parameter in baseline.TIME_VARYING:
                events["raw"][parameter].append((minute, value))
                if value_is_plausible(parameter, value):
                    events["clean"][parameter].append((minute, value))
                else:
                    quality[f"cleaned_{parameter}_rows"] += 1
            else:
                quality["unknown_parameter_rows_skipped"] += 1
    return static, events, quality


def extract_profile_matrices(
    record_paths: list[Path], horizons: list[int]
) -> tuple[list[str], dict[str, dict[int, np.ndarray]], Counter]:
    columns = baseline.build_feature_columns()
    column_index = {column: index for index, column in enumerate(columns)}
    matrices = {
        profile: {
            horizon: np.full((len(record_paths), len(columns)), np.nan, dtype=np.float32)
            for horizon in horizons
        }
        for profile in ("raw", "clean")
    }
    count_indices = [column_index[f"{parameter}__count"] for parameter in baseline.TIME_VARYING]
    for profile_matrices in matrices.values():
        for matrix in profile_matrices.values():
            matrix[:, count_indices] = 0.0

    quality = Counter()
    for row_index, path in enumerate(record_paths):
        statics, event_profiles, record_quality = read_record_profiles(path)
        quality.update(record_quality)
        expected_record_id = int(path.stem)
        if int(statics["raw"].get("RecordID", -1)) != expected_record_id:
            raise ValueError(f"RecordID mismatch in {path}")

        for profile in ("raw", "clean"):
            static = statics[profile]
            events = event_profiles[profile]
            for horizon in horizons:
                matrix = matrices[profile][horizon]
                for parameter in baseline.STATIC_NUMERIC:
                    if parameter in static:
                        matrix[row_index, column_index[parameter]] = static[parameter]
                icu_type = int(static["ICUType"]) if "ICUType" in static else None
                if icu_type in baseline.ICU_TYPES:
                    for candidate in baseline.ICU_TYPES:
                        matrix[row_index, column_index[f"ICUType_{candidate}"]] = float(candidate == icu_type)
                for parameter in baseline.TIME_VARYING:
                    summary, duplicate_rows = baseline.summarize_events(events.get(parameter, ()), horizon)
                    if profile == "raw":
                        quality[f"same_minute_extra_rows_collapsed_{horizon}h"] += duplicate_rows
                    for stat, value in summary.items():
                        matrix[row_index, column_index[f"{parameter}__{stat}"]] = value
        if (row_index + 1) % 1000 == 0 or row_index + 1 == len(record_paths):
            print(f"Extracted raw+clean features {row_index + 1:,}/{len(record_paths):,}", flush=True)
    return columns, matrices, quality


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip selected columns using quantiles learned from the current training fold."""

    def __init__(self, column_mask: np.ndarray, lower: float = 0.005, upper: float = 0.995):
        self.column_mask = column_mask
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        data = np.asarray(x, dtype=np.float64)
        mask = np.asarray(self.column_mask, dtype=bool)
        selected = data[:, mask]
        with np.errstate(all="ignore"):
            self.lower_bounds_ = np.nanquantile(selected, self.lower, axis=0)
            self.upper_bounds_ = np.nanquantile(selected, self.upper, axis=0)
        self.lower_bounds_ = np.where(np.isfinite(self.lower_bounds_), self.lower_bounds_, -np.inf)
        self.upper_bounds_ = np.where(np.isfinite(self.upper_bounds_), self.upper_bounds_, np.inf)
        return self

    def transform(self, x):
        data = np.asarray(x, dtype=np.float64).copy()
        mask = np.asarray(self.column_mask, dtype=bool)
        data[:, mask] = np.clip(data[:, mask], self.lower_bounds_, self.upper_bounds_)
        return data


def winsor_column_mask(columns: list[str]) -> np.ndarray:
    excluded = {"Age", "Gender"}
    result = []
    for column in columns:
        if column in excluded or column.startswith("ICUType_"):
            result.append(False)
        elif column.endswith("__count") or column.endswith("__hours_since_last"):
            result.append(False)
        else:
            result.append(True)
    return np.asarray(result, dtype=bool)


def build_model(model_name: str, config: PreprocessConfig, columns: list[str], random_state: int):
    if model_name == "logistic" and config.imputer_strategy is None:
        raise ValueError("Logistic regression cannot use native missing values")
    steps = []
    if config.winsorize:
        steps.append(("winsor", QuantileClipper(winsor_column_mask(columns))))
    if config.imputer_strategy is not None:
        steps.append(
            (
                "imputer",
                SimpleImputer(
                    strategy=config.imputer_strategy,
                    add_indicator=config.add_indicator,
                    keep_empty_features=True,
                ),
            )
        )
    if model_name == "logistic":
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2500, solver="lbfgs", random_state=random_state),
                ),
            ]
        )
    elif model_name == "hist_gradient_boosting":
        steps.append(
            (
                "model",
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
        )
    else:
        raise ValueError(model_name)
    return Pipeline(steps)


def probability_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "auprc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
    }


def fixed_splits(y: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    all_indices = np.arange(len(y))
    train, remaining = train_test_split(
        all_indices, test_size=0.30, random_state=random_state, stratify=y
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, random_state=random_state, stratify=y[remaining]
    )
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}


def hard_outcome_anomaly_mask(outcomes: pd.DataFrame) -> np.ndarray:
    los = outcomes["Length_of_stay"]
    survival = outcomes["Survival"]
    death = outcomes["In-hospital_death"]
    return (
        (los.ge(0) & los.lt(2))
        | (survival.ge(0) & survival.lt(2))
        | survival.lt(-1)
        | (death.eq(1) & ~(survival.ge(2) & survival.le(los)))
        | (death.eq(0) & ~(survival.eq(-1) | survival.gt(los)))
    ).to_numpy()


def bootstrap_intervals(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    random_state: int,
    iterations: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    values = {key: [] for key in ["auroc", "auprc", "brier", "sensitivity", "specificity"]}
    for _ in range(iterations):
        sample = rng.integers(0, len(y), len(y))
        ys = y[sample]
        ps = probability[sample]
        if np.unique(ys).size < 2:
            continue
        prediction = ps >= threshold
        tp = int(((ys == 1) & prediction).sum())
        fn = int(((ys == 1) & ~prediction).sum())
        tn = int(((ys == 0) & ~prediction).sum())
        fp = int(((ys == 0) & prediction).sum())
        values["auroc"].append(roc_auc_score(ys, ps))
        values["auprc"].append(average_precision_score(ys, ps))
        values["brier"].append(brier_score_loss(ys, ps))
        values["sensitivity"].append(tp / (tp + fn))
        values["specificity"].append(tn / (tn + fp))
    intervals = {}
    for metric, samples in values.items():
        intervals[f"{metric}_ci_low"] = float(np.quantile(samples, 0.025))
        intervals[f"{metric}_ci_high"] = float(np.quantile(samples, 0.975))
    return intervals


def stage_one_cv(
    matrices,
    columns,
    y,
    train_indices,
    random_state,
    folds,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    split_pairs = list(splitter.split(train_indices, y[train_indices]))
    for model_name in ("logistic", "hist_gradient_boosting"):
        for config in CONFIGS:
            if model_name == "logistic" and config.imputer_strategy is None:
                continue
            x = matrices[config.feature_profile][24]
            for fold, (fit_local, score_local) in enumerate(split_pairs, start=1):
                fit_indices = train_indices[fit_local]
                score_indices = train_indices[score_local]
                model = build_model(model_name, config, columns, random_state + fold)
                started = time.perf_counter()
                model.fit(x[fit_indices], y[fit_indices])
                probability = model.predict_proba(x[score_indices])[:, 1]
                row = {
                    "stage": "24h_train_cv",
                    "model": model_name,
                    "preprocess": config.name,
                    "fold": fold,
                    "fit_seconds": time.perf_counter() - started,
                }
                row.update(probability_metrics(y[score_indices], probability))
                rows.append(row)
                print(
                    f"CV {model_name} {config.name} fold {fold}/{folds}: "
                    f"AUPRC={row['auprc']:.4f} Brier={row['brier']:.4f}",
                    flush=True,
                )
    fold_frame = pd.DataFrame(rows)
    summary = (
        fold_frame.groupby(["model", "preprocess"], as_index=False)
        .agg(
            folds=("fold", "count"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            brier_mean=("brier", "mean"),
            brier_sd=("brier", "std"),
            fit_seconds_total=("fit_seconds", "sum"),
        )
        .sort_values(["model", "auprc_mean", "brier_mean"], ascending=[True, False, True])
    )
    return fold_frame, summary


def select_cv_candidates(summary: pd.DataFrame) -> dict[str, list[str]]:
    result = {}
    baseline_name = "raw_median_indicator"
    for model_name in ("logistic", "hist_gradient_boosting"):
        ranked = summary[summary["model"].eq(model_name)].sort_values(
            ["auprc_mean", "brier_mean"], ascending=[False, True]
        )
        names = ranked["preprocess"].head(2).tolist()
        if baseline_name not in names:
            names.append(baseline_name)
        result[model_name] = names
    return result


def validation_stage(
    candidates,
    configs_by_name,
    matrices,
    columns,
    y,
    split_indices,
    horizons,
    random_state,
) -> pd.DataFrame:
    rows = []
    for model_name, candidate_names in candidates.items():
        for config_name in candidate_names:
            config = configs_by_name[config_name]
            for horizon in horizons:
                x = matrices[config.feature_profile][horizon]
                model = build_model(model_name, config, columns, random_state)
                model.fit(x[split_indices["train"]], y[split_indices["train"]])
                probability = model.predict_proba(x[split_indices["validation"]])[:, 1]
                threshold = baseline.choose_youden_threshold(
                    y[split_indices["validation"]], probability
                )
                row = {
                    "stage": "validation_confirmation",
                    "model": model_name,
                    "preprocess": config.name,
                    "horizon_hours": horizon,
                    "threshold": threshold,
                }
                row.update(probability_metrics(y[split_indices["validation"]], probability))
                rows.append(row)
                print(
                    f"Validation {model_name} {config.name} {horizon}h: "
                    f"AUPRC={row['auprc']:.4f} Brier={row['brier']:.4f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def select_final_configs(validation: pd.DataFrame) -> dict[str, str]:
    selected = {}
    for model_name in ("logistic", "hist_gradient_boosting"):
        subset = validation[validation["model"].eq(model_name)]
        ranking = (
            subset.groupby("preprocess", as_index=False)
            .agg(auprc_mean=("auprc", "mean"), brier_mean=("brier", "mean"))
            .sort_values(["auprc_mean", "brier_mean"], ascending=[False, True])
        )
        selected[model_name] = str(ranking.iloc[0]["preprocess"])
    return selected


def final_evaluation(
    selected,
    configs_by_name,
    matrices,
    columns,
    outcomes,
    y,
    split_indices,
    horizons,
    random_state,
    bootstrap_iterations,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    sensitivity_rows = []
    prediction_rows = []
    hard_mask = hard_outcome_anomaly_mask(outcomes)
    los_gap = outcomes["Length_of_stay"].to_numpy() - outcomes["Survival"].to_numpy()
    death = y == 1
    valid_death_time = death & outcomes["Survival"].ge(2).to_numpy()

    for model_name, config_name in selected.items():
        config = configs_by_name[config_name]
        for horizon in horizons:
            x = matrices[config.feature_profile][horizon]
            model = build_model(model_name, config, columns, random_state)
            model.fit(x[split_indices["train"]], y[split_indices["train"]])
            validation_probability = model.predict_proba(x[split_indices["validation"]])[:, 1]
            threshold = baseline.choose_youden_threshold(
                y[split_indices["validation"]], validation_probability
            )
            test_indices = split_indices["test"]
            test_probability = model.predict_proba(x[test_indices])[:, 1]
            row = {
                "model": model_name,
                "preprocess": config.name,
                "horizon_hours": horizon,
                "split": "test",
                "threshold": threshold,
            }
            row.update(baseline.metric_row(y[test_indices], test_probability, threshold))
            row.update(
                bootstrap_intervals(
                    y[test_indices],
                    test_probability,
                    threshold,
                    random_state + horizon,
                    bootstrap_iterations,
                )
            )
            metric_rows.append(row)
            prediction_rows.extend(
                {
                    "RecordID": int(outcomes.iloc[index]["RecordID"]),
                    "In-hospital_death": int(y[index]),
                    "model": model_name,
                    "preprocess": config.name,
                    "horizon_hours": horizon,
                    "probability": float(probability),
                    "threshold": threshold,
                }
                for index, probability in zip(test_indices, test_probability)
            )

            subsets = {
                "all_test": np.ones(len(test_indices), dtype=bool),
                "exclude_hard_rule_anomalies": ~hard_mask[test_indices],
            }
            for cutoff in (2, 5, 10):
                gap_too_large = valid_death_time & (los_gap > cutoff)
                subsets[f"exclude_hard_and_death_gap_gt{cutoff}"] = ~(
                    hard_mask[test_indices] | gap_too_large[test_indices]
                )
            for subset_name, mask in subsets.items():
                subset_row = {
                    "model": model_name,
                    "preprocess": config.name,
                    "horizon_hours": horizon,
                    "training_population": "all_official_labels",
                    "evaluation_subset": subset_name,
                    "threshold": threshold,
                }
                subset_row.update(
                    baseline.metric_row(
                        y[test_indices][mask], test_probability[mask], threshold
                    )
                )
                sensitivity_rows.append(subset_row)

            clean_train = split_indices["train"][~hard_mask[split_indices["train"]]]
            clean_validation = split_indices["validation"][
                ~hard_mask[split_indices["validation"]]
            ]
            clean_model = build_model(model_name, config, columns, random_state)
            clean_model.fit(x[clean_train], y[clean_train])
            clean_validation_probability = clean_model.predict_proba(
                x[clean_validation]
            )[:, 1]
            clean_threshold = baseline.choose_youden_threshold(
                y[clean_validation], clean_validation_probability
            )
            clean_test_probability = clean_model.predict_proba(x[test_indices])[:, 1]
            for subset_name, mask in subsets.items():
                if subset_name == "all_test":
                    continue
                subset_row = {
                    "model": model_name,
                    "preprocess": config.name,
                    "horizon_hours": horizon,
                    "training_population": "exclude_hard_rule_anomalies",
                    "evaluation_subset": subset_name,
                    "threshold": clean_threshold,
                }
                subset_row.update(
                    baseline.metric_row(
                        y[test_indices][mask], clean_test_probability[mask], clean_threshold
                    )
                )
                sensitivity_rows.append(subset_row)
            print(
                f"Final test {model_name} {config.name} {horizon}h: "
                f"AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} Brier={row['brier']:.4f}",
                flush=True,
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(sensitivity_rows), pd.DataFrame(prediction_rows)


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without requiring optional dependencies."""

    def format_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_readme(
    path: Path,
    cv_summary: pd.DataFrame,
    validation: pd.DataFrame,
    selected: dict[str, str],
    final_metrics: pd.DataFrame,
) -> None:
    cv_table = cv_summary[
        ["model", "preprocess", "auroc_mean", "auprc_mean", "brier_mean"]
    ].copy()
    final_table = final_metrics[
        [
            "model",
            "preprocess",
            "horizon_hours",
            "auroc",
            "auprc",
            "brier",
            "sensitivity",
            "specificity",
        ]
    ].copy()
    report = f"""# 实验 03：院内死亡模型的数据填补与清洗稳健性

## 研究问题

本实验不改变院内死亡二分类目标、患者划分、时间窗或模型家族，只比较缺失填补、缺失指示、宽松异常值清洗和训练集分位数截断。24 小时训练集五折交叉验证用于筛选，验证集确认候选，测试集在方案冻结后评价。

## 设计

- 原始基线：只处理 `-1`、空参数、同分钟重复和时间边界。
- 宽松清洗：只把明显违反单位或技术范围的值改为缺失，不把临床异常值当作错误。
- 填补：中位数、均值、是否加入缺失指示，以及梯度提升的原生缺失分支。
- Winsorization：只由当前训练折估计 0.5% 和 99.5% 分位数。
- 主要筛选指标：AUPRC；Brier 和 AUROC 作为共同约束。

## 24 小时训练集五折结果

{dataframe_to_markdown(cv_table)}

## 验证集选定方案

- 逻辑回归：`{selected['logistic']}`
- 梯度提升：`{selected['hist_gradient_boosting']}`

候选方案在 6、12、24、48 小时验证集上的完整结果见 `output/validation_candidates.csv`。

## 冻结后的测试集结果

{dataframe_to_markdown(final_table)}

测试集已经在实验 02 中用于建立基线，因此这里称为“基线后重新锁定的确认集”，不把它描述成研究全过程中从未查看过的盲测集。置信区间、标签敏感性和清洗数量分别保存在 `output/test_metrics.csv`、`output/label_sensitivity_metrics.csv` 和 `output/cleaning_counts.csv`。
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "release")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    parser.add_argument("--random-state", type=int, default=20260904)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [6, 12, 24, 48]
    outcomes_path = args.data_dir / "outcomes.csv"
    record_dir = args.data_dir / "icu_records"
    if not outcomes_path.exists() or not record_dir.is_dir():
        raise FileNotFoundError("Pass --data-dir containing outcomes.csv and icu_records/.")

    outcomes = pd.read_csv(outcomes_path)
    numeric = ["RecordID", "Length_of_stay", "Survival", "In-hospital_death"]
    outcomes[numeric] = outcomes[numeric].apply(pd.to_numeric, errors="coerce")
    outcomes = outcomes[outcomes["In-hospital_death"].isin([0, 1])].copy()
    outcomes = outcomes.sort_values("RecordID").reset_index(drop=True)
    if args.max_records is not None:
        outcomes = outcomes.head(args.max_records).copy()
    outcomes["RecordID"] = outcomes["RecordID"].astype(int)
    outcomes["In-hospital_death"] = outcomes["In-hospital_death"].astype(int)
    y = outcomes["In-hospital_death"].to_numpy(dtype=np.int8)
    if np.unique(y).size != 2:
        raise ValueError("Selected records must contain both classes")

    record_paths = [record_dir / f"{record_id}.csv" for record_id in outcomes["RecordID"]]
    missing = [path for path in record_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} record files; first: {missing[0]}")

    started = time.perf_counter()
    columns, matrices, quality = extract_profile_matrices(record_paths, horizons)
    extraction_seconds = time.perf_counter() - started
    split_indices = fixed_splits(y, args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_output = args.output_dir / "data"
    data_output.mkdir(parents=True, exist_ok=True)

    cache = {"columns": np.asarray(columns), "RecordID": outcomes["RecordID"].to_numpy(), "y": y}
    for profile in ("raw", "clean"):
        for horizon in horizons:
            cache[f"{profile}_{horizon}h"] = matrices[profile][horizon]
    np.savez_compressed(data_output / "feature_profiles.npz", **cache)

    fold_metrics, cv_summary = stage_one_cv(
        matrices,
        columns,
        y,
        split_indices["train"],
        args.random_state,
        args.folds,
    )
    configs_by_name = {config.name: config for config in CONFIGS}
    candidates = select_cv_candidates(cv_summary)
    validation = validation_stage(
        candidates,
        configs_by_name,
        matrices,
        columns,
        y,
        split_indices,
        horizons,
        args.random_state,
    )
    selected = select_final_configs(validation)
    final_metrics, label_sensitivity, predictions = final_evaluation(
        selected,
        configs_by_name,
        matrices,
        columns,
        outcomes,
        y,
        split_indices,
        horizons,
        args.random_state,
        args.bootstrap_iterations,
    )

    fold_metrics.to_csv(args.output_dir / "cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    validation.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    final_metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)
    label_sensitivity.to_csv(args.output_dir / "label_sensitivity_metrics.csv", index=False)
    predictions.to_csv(data_output / "test_predictions.csv", index=False)
    cleaning_counts = pd.DataFrame(
        [
            {"quality_item": key, "row_count": int(value)}
            for key, value in sorted(quality.items())
            if key.startswith("cleaned_")
        ]
    )
    cleaning_counts.to_csv(args.output_dir / "cleaning_counts.csv", index=False)
    metadata = {
        "record_count": int(len(outcomes)),
        "death_count": int(y.sum()),
        "random_state": args.random_state,
        "horizons_hours": horizons,
        "screening_horizon_hours": 24,
        "cv_folds": args.folds,
        "bootstrap_iterations": args.bootstrap_iterations,
        "split_sizes": {key: int(len(value)) for key, value in split_indices.items()},
        "feature_columns": len(columns),
        "selected_configs": selected,
        "candidate_configs": candidates,
        "configs": [asdict(config) for config in CONFIGS],
        "strict_bounds": STRICT_BOUNDS,
        "hard_outcome_anomalies": int(hard_outcome_anomaly_mask(outcomes).sum()),
        "extraction_seconds": extraction_seconds,
        "scikit_learn_version": sklearn.__version__,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = (
        EXPERIMENT_DIR / "RUN_SUMMARY.md"
        if args.max_records is None
        else args.output_dir / "SMOKE_README.md"
    )
    write_readme(report_path, cv_summary, validation, selected, final_metrics)
    print(f"Selected configs: {selected}", flush=True)
    print(f"Outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

