#!/usr/bin/env python3
"""Experiment 06: mortality preprocessing, calibration and transportability checks."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from PIL import Image, ImageDraw, ImageFont
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Experiments.shared.icu_feature_cache import (  # noqa: E402
    default_data_dir,
    load_or_extract_features,
    read_outcomes,
)


BG = "#F7F8FA"
WHITE = "#FFFFFF"
INK = "#17212B"
MUTED = "#5B6773"
GRID = "#D9DEE5"
BLUE = "#2474B5"
ORANGE = "#D97824"
GREEN = "#37805B"
RED = "#B84747"

VALUE_STATS = {"first", "last", "min", "max", "mean"}
PHYSIOLOGIC_BOUNDS = {
    "Height": (100.0, 250.0),
    "InitialWeight": (20.0, 300.0),
    "DiasABP": (10.0, 200.0),
    "NIDiasABP": (10.0, 200.0),
    "MAP": (10.0, 250.0),
    "NIMAP": (10.0, 250.0),
    "SysABP": (20.0, 300.0),
    "NISysABP": (20.0, 300.0),
    "Temp": (25.0, 45.0),
    "pH": (6.5, 8.0),
    "HR": (0.0, 250.0),
    "RespRate": (0.0, 80.0),
}


def font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str, used_font, fill=INK) -> None:
    box = draw.textbbox((0, 0), label, font=used_font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        label,
        font=used_font,
        fill=fill,
    )


def make_splits(y: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    train, remainder = train_test_split(
        indices, test_size=0.30, random_state=random_state, stratify=y
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=y[remainder],
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature using quantiles learned only from the fitted partition."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        lows, highs = [], []
        for column in values.T:
            observed = column[np.isfinite(column)]
            lows.append(float(np.quantile(observed, self.lower)) if len(observed) else math.nan)
            highs.append(float(np.quantile(observed, self.upper)) if len(observed) else math.nan)
        self.lower_bounds_ = np.asarray(lows)
        self.upper_bounds_ = np.asarray(highs)
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float).copy()
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


class PhysiologicFeatureMasker(BaseEstimator, TransformerMixin):
    """Mask clearly implausible cached value summaries without using outcomes."""

    def __init__(self, feature_names: tuple[str, ...]):
        self.feature_names = feature_names

    def fit(self, x, y=None):
        self.masked_by_feature_ = {}
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float).copy()
        counts: dict[str, int] = {}
        for index, feature in enumerate(self.feature_names):
            parameter = feature
            if "__" in feature:
                parameter, stat = feature.split("__", 1)
                if stat not in VALUE_STATS:
                    continue
            if parameter not in PHYSIOLOGIC_BOUNDS:
                continue
            lower, upper = PHYSIOLOGIC_BOUNDS[parameter]
            selected = np.isfinite(values[:, index]) & (
                (values[:, index] <= lower) | (values[:, index] > upper)
            )
            counts[feature] = int(selected.sum())
            values[selected, index] = np.nan
        self.masked_by_feature_ = counts
        return values


def classifier(random_state: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=random_state,
    )


def build_model(name: str, feature_names: list[str], random_state: int):
    imputer_options = dict(add_indicator=True, keep_empty_features=True)
    if name == "median_indicator":
        return make_pipeline(SimpleImputer(strategy="median", **imputer_options), classifier(random_state))
    if name == "median_only":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True),
            classifier(random_state),
        )
    if name == "mean_indicator":
        return make_pipeline(SimpleImputer(strategy="mean", **imputer_options), classifier(random_state))
    if name == "native_missing":
        return classifier(random_state)
    if name == "winsor_median_indicator":
        return make_pipeline(
            QuantileClipper(),
            SimpleImputer(strategy="median", **imputer_options),
            classifier(random_state),
        )
    if name == "physiologic_median_indicator":
        return make_pipeline(
            PhysiologicFeatureMasker(tuple(feature_names)),
            SimpleImputer(strategy="median", **imputer_options),
            classifier(random_state),
        )
    if name == "physiologic_winsor_indicator":
        return make_pipeline(
            PhysiologicFeatureMasker(tuple(feature_names)),
            QuantileClipper(),
            SimpleImputer(strategy="median", **imputer_options),
            classifier(random_state),
        )
    raise ValueError(f"Unknown preprocessing strategy: {name}")


def youden_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    score = np.where(finite, tpr - fpr, -np.inf)
    return float(thresholds[int(np.argmax(score))])


def sensitivity_threshold(y_true: np.ndarray, probability: np.ndarray, target: float) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    eligible = np.isfinite(thresholds) & (tpr >= target)
    if not eligible.any():
        return float(np.min(probability))
    candidates = np.flatnonzero(eligible)
    best_fpr = np.min(fpr[candidates])
    candidates = candidates[np.isclose(fpr[candidates], best_fpr)]
    return float(np.max(thresholds[candidates]))


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
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
        "alerts_per_100": float(100.0 * (tp + fp) / len(y_true)),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
    }


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def calibration_intercept_slope(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logit(probability).reshape(-1, 1), y_true)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_table(
    y_true: np.ndarray, probability: np.ndarray, method: str, bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y_true, "probability": probability})
    frame["bin"] = pd.qcut(frame["probability"], q=bins, labels=False, duplicates="drop")
    result = (
        frame.groupby("bin", observed=True)
        .agg(n=("outcome", "size"), predicted_mean=("probability", "mean"), observed_rate=("outcome", "mean"))
        .reset_index()
    )
    result.insert(0, "method", method)
    return result


def bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    samples = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(repeats):
        selected = rng.integers(0, len(y_true), size=len(y_true))
        observed = y_true[selected]
        predicted = probability[selected]
        if np.unique(observed).size < 2:
            continue
        samples["auroc"].append(float(roc_auc_score(observed, predicted)))
        samples["auprc"].append(float(average_precision_score(observed, predicted)))
        samples["brier"].append(float(brier_score_loss(observed, predicted)))
    point = {
        "auroc": roc_auc_score(y_true, probability),
        "auprc": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }
    rows = []
    for metric, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "point_estimate": float(point[metric]),
                "ci_2.5%": float(low),
                "ci_97.5%": float(high),
                "bootstrap_repeats": len(values),
            }
        )
    return pd.DataFrame(rows)


def missingness_features(matrix: np.ndarray, feature_names: list[str], include_counts: bool) -> np.ndarray:
    presence_columns, count_columns = [], []
    for parameter in sorted({name.split("__", 1)[0] for name in feature_names if "__" in name}):
        count_name = f"{parameter}__count"
        if count_name not in feature_names:
            continue
        count = matrix[:, feature_names.index(count_name)].astype(float)
        presence_columns.append((count > 0).astype(float))
        if include_counts:
            count_columns.append(np.log1p(np.maximum(count, 0.0)))
    static_names = [name for name in ("Age", "Gender", "Height", "InitialWeight") if name in feature_names]
    static_missing = [~np.isfinite(matrix[:, feature_names.index(name)]) for name in static_names]
    columns = presence_columns + [item.astype(float) for item in static_missing] + count_columns
    return np.column_stack(columns).astype(np.float32)


def hard_outcome_invalid(outcomes: pd.DataFrame) -> np.ndarray:
    survival = outcomes["Survival"].to_numpy(float)
    los = outcomes["Length_of_stay"].to_numpy(float)
    return np.isin(survival, [-23.0, 0.0, 1.0]) | np.isin(los, [0.0, 1.0])


def icu_types_from_matrix(matrix: np.ndarray, feature_names: list[str]) -> np.ndarray:
    indices = [feature_names.index(f"ICUType_{value}") for value in range(1, 5)]
    encoded = matrix[:, indices]
    if not np.allclose(np.nansum(encoded, axis=1), 1.0):
        raise ValueError("ICU type one-hot columns are incomplete")
    return np.argmax(encoded, axis=1).astype(int) + 1


def draw_preprocessing(metrics: pd.DataFrame, destination: Path) -> None:
    labels = {
        "median_indicator": "中位数+缺失指示",
        "median_only": "仅中位数",
        "mean_indicator": "均值+缺失指示",
        "native_missing": "模型原生缺失",
        "winsor_median_indicator": "分位截尾",
        "physiologic_median_indicator": "生理范围遮蔽",
        "physiologic_winsor_indicator": "范围+截尾",
    }
    data = metrics[metrics["split"] == "validation"].sort_values("auprc", ascending=False)
    image = Image.new("RGB", (1700, 980), BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 38), "24 小时死亡模型：预处理敏感性", font=font(38, True), fill=INK)
    draw.text((60, 94), "候选方案按验证集 AUPRC 比较；测试集不参与方案选择。", font=font(22), fill=MUTED)
    x0, y0, x1, y1 = 500, 175, 1600, 870
    draw.rounded_rectangle((45, 145, 1650, 920), radius=20, fill=WHITE)
    for tick in np.linspace(0, 0.65, 6):
        px = x0 + tick / 0.65 * (x1 - x0)
        draw.line((px, y0, px, y1), fill=GRID, width=1)
        center(draw, (px, y1 + 28), f"{tick:.2f}", font(17), MUTED)
    row_height = (y1 - y0) / max(len(data), 1)
    for row_index, row in enumerate(data.itertuples(index=False)):
        cy = y0 + (row_index + 0.5) * row_height
        draw.text((75, cy - 15), labels.get(row.strategy, row.strategy), font=font(20), fill=INK)
        width = row.auprc / 0.65 * (x1 - x0)
        draw.rounded_rectangle((x0, cy - 16, x0 + width, cy + 16), radius=8, fill=BLUE)
        draw.text((x0 + width + 12, cy - 13), f"{row.auprc:.3f}", font=font(18, True), fill=INK)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def draw_thresholds(metrics: pd.DataFrame, destination: Path) -> None:
    image = Image.new("RGB", (1600, 920), BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 38), "灵敏度目标与测试集告警负担", font=font(38, True), fill=INK)
    draw.text((60, 94), "阈值只在验证集选择；每根柱表示测试集中每 100 人需要复核的人数。", font=font(22), fill=MUTED)
    draw.rounded_rectangle((55, 145, 1545, 850), radius=20, fill=WHITE)
    x0, y0, x1, y1 = 170, 215, 1480, 745
    max_alert = max(60.0, float(metrics["alerts_per_100"].max()) + 5)
    for tick in np.linspace(0, max_alert, 7):
        py = y1 - tick / max_alert * (y1 - y0)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((95, py - 10), f"{tick:.0f}", font=font(17), fill=MUTED)
    width = (x1 - x0) / len(metrics)
    for index, row in enumerate(metrics.itertuples(index=False)):
        cx = x0 + (index + 0.5) * width
        bar_width = width * 0.45
        top = y1 - row.alerts_per_100 / max_alert * (y1 - y0)
        draw.rounded_rectangle((cx - bar_width / 2, top, cx + bar_width / 2, y1), radius=8, fill=ORANGE)
        center(draw, (cx, top - 22), f"{row.alerts_per_100:.1f}", font(19, True))
        center(draw, (cx, y1 + 34), f"目标 {row.target_sensitivity:.2f}", font(18))
        center(draw, (cx, y1 + 67), f"实测 sens {row.sensitivity:.2f}", font(16), MUTED)
        center(draw, (cx, y1 + 94), f"漏诊 {row.false_negatives}", font(16), RED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def draw_calibration(calibration: pd.DataFrame, destination: Path) -> None:
    image = Image.new("RGB", (1100, 980), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "48 小时模型测试集校准", font=font(38, True), fill=INK)
    draw.text((55, 90), "横轴为平均预测风险，纵轴为实际死亡率。", font=font(21), fill=MUTED)
    draw.rounded_rectangle((50, 140, 1050, 920), radius=20, fill=WHITE)
    x0, y0, x1, y1 = 150, 200, 960, 820
    for tick in np.linspace(0, 0.8, 5):
        px = x0 + tick / 0.8 * (x1 - x0)
        py = y1 - tick / 0.8 * (y1 - y0)
        draw.line((px, y0, px, y1), fill=GRID, width=1)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((px - 15, y1 + 15), f"{tick:.1f}", font=font(16), fill=MUTED)
        draw.text((95, py - 10), f"{tick:.1f}", font=font(16), fill=MUTED)
    draw.line((x0, y1, x1, y0), fill=INK, width=2)
    palette = {"uncalibrated": BLUE, "platt": ORANGE, "isotonic": GREEN}
    for method, subset in calibration.groupby("method"):
        points = []
        for row in subset.itertuples(index=False):
            px = x0 + min(row.predicted_mean, 0.8) / 0.8 * (x1 - x0)
            py = y1 - min(row.observed_rate, 0.8) / 0.8 * (y1 - y0)
            points.append((px, py))
        if len(points) > 1:
            draw.line(points, fill=palette[method], width=4)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=palette[method])
    legend = [("未校准", BLUE), ("Platt", ORANGE), ("Isotonic", GREEN)]
    for index, (label, color) in enumerate(legend):
        x = 185 + index * 260
        draw.line((x, 875, x + 38, 875), fill=color, width=5)
        draw.text((x + 50, 861), label, font=font(18), fill=INK)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], decimals: int = 3) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        values = []
        for key, _ in columns:
            value = getattr(row, key)
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    destination: Path,
    preprocessing: pd.DataFrame,
    preprocessing_cv: pd.DataFrame,
    selected_strategy: str,
    missingness: pd.DataFrame,
    calibration: pd.DataFrame,
    thresholds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    subgroup: pd.DataFrame,
    leave_one_icu: pd.DataFrame,
    label_sensitivity: pd.DataFrame,
    gap_sensitivity: pd.DataFrame,
) -> None:
    prep_test = preprocessing[preprocessing["split"] == "test"]
    miss_test = missingness[missingness["split"] == "test"]
    cv_summary = (
        preprocessing_cv.groupby("strategy", as_index=False)
        .agg(cv_auprc_mean=("auprc", "mean"), cv_auprc_sd=("auprc", "std"), cv_brier_mean=("brier", "mean"))
        .sort_values("cv_auprc_mean", ascending=False)
    )
    test_auprc_range = (float(prep_test["auprc"].min()), float(prep_test["auprc"].max()))
    report = f"""# 实验 06：院内死亡模型稳健性、校准与可迁移性

## 为什么需要这个实验

实验 02 已经给出多时间窗死亡预测基线，但最高 AUROC 并不能回答模型对缺失填补和异常处理是否敏感，也不能直接决定临床阈值。本实验固定患者划分、特征定义和梯度提升参数，先在 24 小时验证集比较预处理方案，再把选定方案扩展到 48 小时，并补充缺失模式、校准、告警负担、亚组、跨 ICU 类型和标签质量分析。

## 预处理敏感性

训练集五折交叉验证结果如下。选择采用一标准误规则：先找到平均 AUPRC 最高的方案，再把与它相差不超过该方案一个标准误的候选视为性能无法明确区分，按照预先排列的简单方案顺序选择。

{markdown_table(cv_summary, [("strategy", "方案"), ("cv_auprc_mean", "CV AUPRC均值"), ("cv_auprc_sd", "CV AUPRC标准差"), ("cv_brier_mean", "CV Brier均值")])}

{markdown_table(prep_test, [("strategy", "方案"), ("horizon_hours", "时间窗"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier")])}

选中方案为 `{selected_strategy}`。七种方案在 24 小时测试集上的 AUPRC 范围仅为 {test_auprc_range[0]:.3f}—{test_auprc_range[1]:.3f}，差异没有形成稳定的复杂度收益，因此不进入 KNN 或迭代填补。医学范围方案仅遮蔽报告中已经审计过的明显技术异常，对缓存中的首次值、末次值、最小值、最大值和均值进行特征级敏感性处理，不能替代回到原始逐条记录重新计算摘要。

![预处理敏感性](output/report_figures/preprocessing_comparison.png)

## 缺失模式消融

{markdown_table(miss_test, [("feature_set", "特征"), ("horizon_hours", "时间窗"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier")])}

这里只使用项目是否被测量、静态字段是否缺失，并在第二组加入测量次数，不使用任何实际生理数值。结果衡量诊疗流程本身能提供多少预测信息，不能解释为缺失导致死亡。

## 校准与工作点

{markdown_table(calibration, [("method", "方法"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier"), ("calibration_intercept", "校准截距"), ("calibration_slope", "校准斜率")])}

{markdown_table(thresholds, [("target_sensitivity", "验证集目标灵敏度"), ("sensitivity", "测试集灵敏度"), ("specificity", "特异度"), ("precision", "PPV"), ("alerts_per_100", "每100人告警"), ("false_negatives", "漏诊死亡")], decimals=2)}

![测试集校准](output/report_figures/calibration_curve.png)

![阈值与告警负担](output/report_figures/threshold_tradeoff.png)

最终未校准模型的患者记录级 Bootstrap 区间如下：

{markdown_table(bootstrap, [("metric", "指标"), ("point_estimate", "点估计"), ("ci_2_5", "95% CI下限"), ("ci_97_5", "95% CI上限")])}

## 亚组、病区和标签敏感性

{markdown_table(subgroup, [("subgroup", "亚组"), ("n", "n"), ("prevalence", "死亡率"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier")])}

跨 ICU 类型内部压力测试在三类 ICU 训练、第四类 ICU 评价，并移除 ICU 类型特征：

{markdown_table(leave_one_icu, [("held_out_icu", "留出ICU"), ("n", "n"), ("prevalence", "实际死亡率"), ("predicted_mean", "平均预测风险"), ("auroc", "AUROC"), ("auprc", "AUPRC")])}

{markdown_table(label_sensitivity, [("analysis", "分析"), ("n", "n"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("brier", "Brier")])}

只保留死亡时间与住院结束相差不超过不同阈值的死亡记录、同时保留所有硬规则有效的存活记录，冻结模型后的结果如下：

{markdown_table(gap_sensitivity, [("maximum_death_los_gap_days", "最大差值（天）"), ("n", "n"), ("prevalence", "死亡率"), ("auroc", "AUROC"), ("auprc", "AUPRC")])}

亚组差异是探索性结果；跨 ICU 类型测试仍来自同一数据集，只能称为内部可迁移性压力测试。标签敏感性把 `Survival=-23/0/1` 或 `Length_of_stay=0/1` 的 95 条硬性异常排除后重新训练和评价，用来检查少量明显错误是否推动总体结果，不用于擅自改写官方主标签。

## 解释边界

本实验没有把 KNN 和迭代填补放入首轮比较：在 12,000×341 的高缺失矩阵上，它们成本更高，也可能生成缺少临床约束的变量组合；只有当简单方案表现不稳定时才值得进入第二轮。校准方法使用验证集拟合，所以校准后的验证集本身不作性能结论，测试集结果仍是内部确认。当前固定测试集此前已经用于基线报告，因此应称为基线后重新锁定的确认集，而不是研究全过程中从未查看的盲测集。
"""
    destination.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Experiments" / "shared" / "output" / "data")
    parser.add_argument("--random-state", type=int, default=20260904)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes = read_outcomes(args.data_dir)
    outcomes = outcomes[outcomes["In-hospital_death"].isin([0, 1])].copy().reset_index(drop=True)
    y = outcomes["In-hospital_death"].to_numpy(np.int8)
    feature_names, matrices, quality, _, cache_hit = load_or_extract_features(
        data_dir=args.data_dir,
        record_ids=outcomes["RecordID"].to_numpy(np.int64),
        horizons=[6, 12, 24, 48],
        cache_dir=args.cache_dir,
    )
    splits = make_splits(y, args.random_state)
    output = args.output_dir
    data_output = output / "data"
    figure_output = output / "report_figures"
    output.mkdir(parents=True, exist_ok=True)
    data_output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)

    strategies = [
        "median_indicator",
        "median_only",
        "mean_indicator",
        "native_missing",
        "winsor_median_indicator",
        "physiologic_median_indicator",
        "physiologic_winsor_indicator",
    ]
    preprocessing_rows = []
    cv_rows = []
    fitted_24h: dict[str, object] = {}
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    for strategy in strategies:
        train_pool = splits["train"]
        for fold, (fold_train_local, fold_valid_local) in enumerate(
            cross_validation.split(train_pool, y[train_pool]), start=1
        ):
            fold_train = train_pool[fold_train_local]
            fold_valid = train_pool[fold_valid_local]
            fold_model = build_model(strategy, feature_names, args.random_state + fold)
            fold_model.fit(matrices[24][fold_train], y[fold_train])
            fold_probability = fold_model.predict_proba(matrices[24][fold_valid])[:, 1]
            fold_row = {"strategy": strategy, "fold": fold}
            fold_row.update(
                classification_metrics(y[fold_valid], fold_probability, 0.5)
            )
            cv_rows.append(fold_row)
        model = build_model(strategy, feature_names, args.random_state)
        model.fit(matrices[24][splits["train"]], y[splits["train"]])
        fitted_24h[strategy] = model
        val_probability = model.predict_proba(matrices[24][splits["validation"]])[:, 1]
        threshold = youden_threshold(y[splits["validation"]], val_probability)
        for split in ("validation", "test"):
            probability = model.predict_proba(matrices[24][splits[split]])[:, 1]
            row = {"strategy": strategy, "horizon_hours": 24, "split": split}
            row.update(classification_metrics(y[splits[split]], probability, threshold))
            preprocessing_rows.append(row)
    preprocessing = pd.DataFrame(preprocessing_rows)
    preprocessing_cv = pd.DataFrame(cv_rows)
    cv_summary = (
        preprocessing_cv.groupby("strategy", as_index=False)
        .agg(auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"))
        .sort_values("auprc_mean", ascending=False)
    )
    best = cv_summary.iloc[0]
    one_standard_error_floor = float(best["auprc_mean"] - best["auprc_sd"] / math.sqrt(5))
    eligible = set(cv_summary.loc[cv_summary["auprc_mean"] >= one_standard_error_floor, "strategy"])
    selected_strategy = next(strategy for strategy in strategies if strategy in eligible)

    horizon_rows = []
    final_models: dict[int, object] = {}
    final_probabilities: dict[int, dict[str, np.ndarray]] = {}
    for horizon in (6, 12, 24, 48):
        model = fitted_24h[selected_strategy] if horizon == 24 else build_model(
            selected_strategy, feature_names, args.random_state
        )
        if horizon != 24:
            model.fit(matrices[horizon][splits["train"]], y[splits["train"]])
        probabilities = {
            split: model.predict_proba(matrices[horizon][indices])[:, 1]
            for split, indices in splits.items()
            if split != "train"
        }
        threshold = youden_threshold(y[splits["validation"]], probabilities["validation"])
        for split in ("validation", "test"):
            row = {"strategy": selected_strategy, "horizon_hours": horizon, "split": split}
            row.update(classification_metrics(y[splits[split]], probabilities[split], threshold))
            horizon_rows.append(row)
        final_models[horizon] = model
        final_probabilities[horizon] = probabilities
    horizon_metrics = pd.DataFrame(horizon_rows)

    missingness_rows = []
    for horizon in (24, 48):
        for include_counts, feature_set in ((False, "presence_only"), (True, "presence_plus_counts")):
            x_missing = missingness_features(matrices[horizon], feature_names, include_counts)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, solver="lbfgs", max_iter=2500, random_state=args.random_state),
            )
            model.fit(x_missing[splits["train"]], y[splits["train"]])
            val_probability = model.predict_proba(x_missing[splits["validation"]])[:, 1]
            threshold = youden_threshold(y[splits["validation"]], val_probability)
            for split in ("validation", "test"):
                probability = model.predict_proba(x_missing[splits[split]])[:, 1]
                row = {"feature_set": feature_set, "horizon_hours": horizon, "split": split}
                row.update(classification_metrics(y[splits[split]], probability, threshold))
                missingness_rows.append(row)
    missingness = pd.DataFrame(missingness_rows)

    val_probability = final_probabilities[48]["validation"]
    test_probability = final_probabilities[48]["test"]
    platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    platt.fit(logit(val_probability).reshape(-1, 1), y[splits["validation"]])
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_probability, y[splits["validation"]])
    calibrated = {
        "uncalibrated": (val_probability, test_probability),
        "platt": (
            platt.predict_proba(logit(val_probability).reshape(-1, 1))[:, 1],
            platt.predict_proba(logit(test_probability).reshape(-1, 1))[:, 1],
        ),
        "isotonic": (isotonic.predict(val_probability), isotonic.predict(test_probability)),
    }
    calibration_rows, calibration_bins = [], []
    for method, (val_prob, test_prob) in calibrated.items():
        threshold = youden_threshold(y[splits["validation"]], val_prob)
        row = {"method": method}
        row.update(classification_metrics(y[splits["test"]], test_prob, threshold))
        intercept, slope = calibration_intercept_slope(y[splits["test"]], test_prob)
        row.update({"calibration_intercept": intercept, "calibration_slope": slope})
        calibration_rows.append(row)
        calibration_bins.append(calibration_table(y[splits["test"]], test_prob, method))
    calibration_metrics = pd.DataFrame(calibration_rows)
    calibration_bins_frame = pd.concat(calibration_bins, ignore_index=True)

    threshold_rows = []
    for target in (0.70, 0.80, 0.90, 0.95):
        threshold = sensitivity_threshold(y[splits["validation"]], val_probability, target)
        row = {"target_sensitivity": target}
        row.update(classification_metrics(y[splits["test"]], test_probability, threshold))
        threshold_rows.append(row)
    threshold_metrics = pd.DataFrame(threshold_rows)

    bootstrap = bootstrap_intervals(
        y[splits["test"]], test_probability, args.bootstrap_repeats, args.random_state + 17
    )
    bootstrap = bootstrap.rename(columns={"ci_2.5%": "ci_2_5", "ci_97.5%": "ci_97_5"})

    final_threshold = youden_threshold(y[splits["validation"]], val_probability)
    icu_types = icu_types_from_matrix(matrices[48], feature_names)
    subgroup_rows = []
    test_indices = splits["test"]
    for icu_type in range(1, 5):
        local = test_indices[icu_types[test_indices] == icu_type]
        row = {"subgroup": f"ICUType_{icu_type}"}
        row.update(classification_metrics(y[local], final_models[48].predict_proba(matrices[48][local])[:, 1], final_threshold))
        subgroup_rows.append(row)
    ages = matrices[48][:, feature_names.index("Age")]
    for label, selected in (
        ("age_<65", ages < 65),
        ("age_65_79", (ages >= 65) & (ages < 80)),
        ("age_80+", ages >= 80),
    ):
        local = test_indices[selected[test_indices]]
        row = {"subgroup": label}
        row.update(classification_metrics(y[local], final_models[48].predict_proba(matrices[48][local])[:, 1], final_threshold))
        subgroup_rows.append(row)
    gender = matrices[48][:, feature_names.index("Gender")]
    for label, selected in (("female", gender == 0), ("male", gender == 1)):
        local = test_indices[selected[test_indices]]
        row = {"subgroup": label}
        row.update(classification_metrics(y[local], final_models[48].predict_proba(matrices[48][local])[:, 1], final_threshold))
        subgroup_rows.append(row)
    subgroup = pd.DataFrame(subgroup_rows)

    icu_feature_indices = [feature_names.index(f"ICUType_{value}") for value in range(1, 5)]
    x_without_icu = np.delete(matrices[48], icu_feature_indices, axis=1)
    names_without_icu = [name for index, name in enumerate(feature_names) if index not in icu_feature_indices]
    leave_rows = []
    for held_out in range(1, 5):
        source = np.flatnonzero(icu_types != held_out)
        source_train, source_validation = train_test_split(
            source, test_size=0.20, random_state=args.random_state + held_out, stratify=y[source]
        )
        destination_indices = np.flatnonzero(icu_types == held_out)
        model = build_model(selected_strategy, names_without_icu, args.random_state + held_out)
        model.fit(x_without_icu[source_train], y[source_train])
        source_probability = model.predict_proba(x_without_icu[source_validation])[:, 1]
        threshold = youden_threshold(y[source_validation], source_probability)
        probability = model.predict_proba(x_without_icu[destination_indices])[:, 1]
        row = {"held_out_icu": held_out, "predicted_mean": float(np.mean(probability))}
        row.update(classification_metrics(y[destination_indices], probability, threshold))
        leave_rows.append(row)
    leave_one_icu = pd.DataFrame(leave_rows)

    invalid = hard_outcome_invalid(outcomes)
    label_rows = []
    valid_test = splits["test"][~invalid[splits["test"]]]
    for analysis, indices, probability in (
        ("official_all", splits["test"], test_probability),
        ("official_model_hard_valid_test", valid_test, final_models[48].predict_proba(matrices[48][valid_test])[:, 1]),
    ):
        row = {"analysis": analysis}
        row.update(classification_metrics(y[indices], probability, final_threshold))
        label_rows.append(row)
    strict_model = build_model(selected_strategy, feature_names, args.random_state)
    strict_train = splits["train"][~invalid[splits["train"]]]
    strict_validation = splits["validation"][~invalid[splits["validation"]]]
    strict_model.fit(matrices[48][strict_train], y[strict_train])
    strict_val_probability = strict_model.predict_proba(matrices[48][strict_validation])[:, 1]
    strict_threshold = youden_threshold(y[strict_validation], strict_val_probability)
    strict_test_probability = strict_model.predict_proba(matrices[48][valid_test])[:, 1]
    row = {"analysis": "retrained_without_hard_invalid"}
    row.update(classification_metrics(y[valid_test], strict_test_probability, strict_threshold))
    label_rows.append(row)
    label_sensitivity = pd.DataFrame(label_rows)

    gap_rows = []
    survival = outcomes["Survival"].to_numpy(float)
    los = outcomes["Length_of_stay"].to_numpy(float)
    for maximum_gap in (2, 5, 10, math.inf):
        acceptable_death = (y == 0) | (
            (y == 1) & (survival >= 2) & (los >= survival) & ((los - survival) <= maximum_gap)
        )
        local = splits["test"][(~invalid & acceptable_death)[splits["test"]]]
        probability = final_models[48].predict_proba(matrices[48][local])[:, 1]
        row = {"maximum_death_los_gap_days": "all" if math.isinf(maximum_gap) else maximum_gap}
        row.update(classification_metrics(y[local], probability, final_threshold))
        gap_rows.append(row)
    gap_sensitivity = pd.DataFrame(gap_rows)

    split_labels = np.full(len(y), "", dtype=object)
    for split, indices in splits.items():
        split_labels[indices] = split
    pd.DataFrame(
        {
            "RecordID": outcomes["RecordID"].to_numpy(int),
            "In-hospital_death": y,
            "split": split_labels,
            "ICUType": icu_types,
        }
    ).to_csv(data_output / "split_assignments.csv", index=False)

    preprocessing.to_csv(output / "preprocessing_metrics.csv", index=False)
    preprocessing_cv.to_csv(output / "preprocessing_cv_metrics.csv", index=False)
    horizon_metrics.to_csv(output / "selected_strategy_horizon_metrics.csv", index=False)
    missingness.to_csv(output / "missingness_ablation.csv", index=False)
    calibration_metrics.to_csv(output / "calibration_metrics.csv", index=False)
    calibration_bins_frame.to_csv(output / "calibration_bins.csv", index=False)
    threshold_metrics.to_csv(output / "threshold_operating_points.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_confidence_intervals.csv", index=False)
    subgroup.to_csv(output / "subgroup_metrics.csv", index=False)
    leave_one_icu.to_csv(output / "leave_one_icu_out.csv", index=False)
    label_sensitivity.to_csv(output / "label_sensitivity.csv", index=False)
    gap_sensitivity.to_csv(output / "outcome_gap_sensitivity.csv", index=False)

    draw_preprocessing(preprocessing, figure_output / "preprocessing_comparison.png")
    draw_thresholds(threshold_metrics, figure_output / "threshold_tradeoff.png")
    draw_calibration(calibration_bins_frame, figure_output / "calibration_curve.png")

    metadata = {
        "record_count": int(len(y)),
        "death_count": int(y.sum()),
        "death_rate": float(y.mean()),
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "screening_horizon_hours": 24,
        "selected_strategy": selected_strategy,
        "selection_rule": "one-standard-error rule on five-fold training CV AUPRC; predefined strategy order breaks practical ties",
        "cross_validation": "five-fold stratified CV within the training partition",
        "one_standard_error_floor": one_standard_error_floor,
        "strategies": strategies,
        "hard_outcome_invalid_count": int(invalid.sum()),
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_state": args.random_state,
        "cache_hit": bool(cache_hit),
        "feature_count": len(feature_names),
        "scikit_learn_version": sklearn.__version__,
        "quality_counters": quality,
        "omitted_first_round": {
            "KNNImputer": "computationally expensive and unconstrained synthetic combinations",
            "IterativeImputer": "computationally expensive and requires a separate convergence/sanity study",
            "additional_clustering_algorithms": "lower presentation priority than supervised robustness",
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        EXPERIMENT_DIR / "README.md",
        preprocessing,
        preprocessing_cv,
        selected_strategy,
        missingness,
        calibration_metrics,
        threshold_metrics,
        bootstrap,
        subgroup,
        leave_one_icu,
        label_sensitivity,
        gap_sensitivity,
    )
    print(f"Selected preprocessing: {selected_strategy}")
    print(horizon_metrics[horizon_metrics["split"] == "test"].to_string(index=False))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
