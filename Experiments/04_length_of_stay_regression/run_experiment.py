#!/usr/bin/env python3
"""Experiment 04: multi-horizon hospital length-of-stay regression."""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
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


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, used_font, fill=INK) -> None:
    bounds = draw.textbbox((0, 0), text, font=used_font)
    draw.text(
        (xy[0] - (bounds[2] - bounds[0]) / 2, xy[1] - (bounds[3] - bounds[1]) / 2),
        text,
        font=used_font,
        fill=fill,
    )


def make_quantile_splits(y: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    """Create a fixed 70/15/15 split with similar LOS distributions."""

    indices = np.arange(len(y))
    q = min(10, max(2, len(y) // 100))
    bins = np.asarray(pd.qcut(y, q=q, labels=False, duplicates="drop"), dtype=int)
    train, remainder = train_test_split(
        indices, test_size=0.30, random_state=random_state, stratify=bins
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=bins[remainder],
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each feature using bounds learned from the training partition only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        lower_bounds = []
        upper_bounds = []
        for column in values.T:
            observed = column[np.isfinite(column)]
            if len(observed):
                lower_bounds.append(float(np.quantile(observed, self.lower)))
                upper_bounds.append(float(np.quantile(observed, self.upper)))
            else:
                lower_bounds.append(math.nan)
                upper_bounds.append(math.nan)
        self.lower_bounds_ = np.asarray(lower_bounds)
        self.upper_bounds_ = np.asarray(upper_bounds)
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def build_model(name: str, random_state: int):
    if name == "ridge_log_target":
        return make_pipeline(
            QuantileClipper(),
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    if name == "hist_gradient_boosting_log_target":
        return make_pipeline(
            QuantileClipper(),
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=random_state,
            ),
        )
    raise ValueError(f"Unknown model: {name}")


def predict_days(model, x: np.ndarray, maximum_days: float | None = None) -> np.ndarray:
    upper = np.inf if maximum_days is None else float(maximum_days)
    return np.clip(np.expm1(model.predict(x)), 2.0, upper)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rank_true = pd.Series(y_true).rank(method="average").to_numpy()
    rank_prediction = pd.Series(prediction).rank(method="average").to_numpy()
    if np.std(rank_prediction) == 0:
        spearman = math.nan
    else:
        spearman = float(np.corrcoef(rank_true, rank_prediction)[0, 1])
    return {
        "n": int(len(y_true)),
        "mae_days": float(mean_absolute_error(y_true, prediction)),
        "rmse_days": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "median_ae_days": float(median_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
        "spearman": spearman,
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    prediction: np.ndarray,
    *,
    repeats: int,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    samples = {"mae_days": [], "rmse_days": [], "median_ae_days": []}
    for _ in range(repeats):
        selected = rng.integers(0, len(y_true), size=len(y_true))
        observed = y_true[selected]
        predicted = prediction[selected]
        samples["mae_days"].append(float(mean_absolute_error(observed, predicted)))
        samples["rmse_days"].append(float(np.sqrt(mean_squared_error(observed, predicted))))
        samples["median_ae_days"].append(float(median_absolute_error(observed, predicted)))
    rows = []
    point = regression_metrics(y_true, prediction)
    for metric, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_2.5%": float(low),
                "ci_97.5%": float(high),
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


def draw_performance(metrics: pd.DataFrame, destination: Path) -> None:
    validation = metrics[(metrics["split"] == "validation") & metrics["horizon_hours"].notna()].copy()
    horizons = sorted(validation["horizon_hours"].astype(int).unique())
    image = Image.new("RGB", (1600, 900), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "住院时长回归：验证集 MAE 随时间窗变化", font=font(38, True), fill=INK)
    draw.text((70, 100), "纵轴为平均绝对误差（天），越低越好；测试集未参与模型与时间窗选择。", font=font(22), fill=MUTED)
    panel = (85, 175, 1515, 790)
    draw.rounded_rectangle(panel, radius=18, fill=WHITE)
    x0, y0, x1, y1 = 190, 245, 1450, 690
    values = validation["mae_days"].to_numpy(float)
    baseline = float(metrics[(metrics["model"] == "training_median") & (metrics["split"] == "validation")]["mae_days"].iloc[0])
    ymin = max(0.0, min(values.min(), baseline) - 0.4)
    ymax = max(values.max(), baseline) + 0.4
    for tick in np.linspace(ymin, ymax, 6):
        py = y1 - (tick - ymin) / (ymax - ymin) * (y1 - y0)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((105, py - 12), f"{tick:.1f}", font=font(17), fill=MUTED)
    draw.rectangle((x0, y0, x1, y1), outline=INK, width=2)
    xpos = {h: x0 + i * (x1 - x0) / (len(horizons) - 1) for i, h in enumerate(horizons)}
    for horizon in horizons:
        center(draw, (xpos[horizon], y1 + 34), f"{horizon}h", font(18))
    series = [
        ("ridge_log_target", "岭回归（log 结局）", BLUE),
        ("hist_gradient_boosting_log_target", "梯度提升（log 结局）", ORANGE),
    ]
    for model, label, color in series:
        subset = validation[validation["model"] == model].set_index("horizon_hours")
        points = []
        for horizon in horizons:
            value = float(subset.loc[float(horizon), "mae_days"])
            py = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
            points.append((xpos[horizon], py, value))
        draw.line([(x, y) for x, y, _ in points], fill=color, width=5)
        for x, y, value in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
            center(draw, (x, y - 22), f"{value:.2f}", font(16, True), color)
    baseline_y = y1 - (baseline - ymin) / (ymax - ymin) * (y1 - y0)
    draw.line((x0, baseline_y, x1, baseline_y), fill=GREEN, width=3)
    draw.text((x1 - 235, baseline_y + 8), f"训练集中位数基线 {baseline:.2f}", font=font(16), fill=GREEN)
    for i, (_, label, color) in enumerate(series):
        lx = 420 + i * 390
        draw.line((lx, 835, lx + 45, 835), fill=color, width=5)
        draw.text((lx + 58, 820), label, font=font(19), fill=INK)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def draw_observed_vs_predicted(
    y_true: np.ndarray,
    prediction: np.ndarray,
    title: str,
    destination: Path,
) -> None:
    image = Image.new("RGB", (1100, 1030), BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 36), "最终模型：观察值与预测值", font=font(36, True), fill=INK)
    draw.text((60, 91), title, font=font(20), fill=MUTED)
    x0, y0, x1, y1 = 145, 180, 1015, 820
    draw.rounded_rectangle((70, 140, 1050, 975), radius=18, fill=WHITE)
    upper = float(max(np.quantile(y_true, 0.995), np.quantile(prediction, 0.995), 10.0))
    max_log = math.log1p(upper)
    for day in (2, 5, 10, 20, 50, 100, 200):
        if day > upper:
            continue
        px = x0 + math.log1p(day) / max_log * (x1 - x0)
        py = y1 - math.log1p(day) / max_log * (y1 - y0)
        draw.line((px, y0, px, y1), fill=GRID, width=1)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        center(draw, (px, y1 + 30), str(day), font(15), MUTED)
        draw.text((92, py - 10), str(day), font=font(15), fill=MUTED)
    draw.rectangle((x0, y0, x1, y1), outline=INK, width=2)
    draw.line((x0, y1, x1, y0), fill=GREEN, width=3)
    rng = np.random.default_rng(20260906)
    selected = rng.choice(len(y_true), size=min(1800, len(y_true)), replace=False)
    for observed, predicted in zip(y_true[selected], prediction[selected]):
        px = x0 + min(math.log1p(float(observed)), max_log) / max_log * (x1 - x0)
        py = y1 - min(math.log1p(float(predicted)), max_log) / max_log * (y1 - y0)
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=BLUE)
    center(draw, ((x0 + x1) / 2, y1 + 72), "观察住院时长（天，对数刻度）", font(19))
    draw.text((73, 460), "预测值\n（天）", font=font(18), fill=INK, spacing=4)
    draw.text((145, 930), "绿色对角线表示预测与观察完全一致；右侧长住院患者仍是主要误差来源。", font=font(17), fill=MUTED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def markdown_test_table(metrics: pd.DataFrame) -> str:
    test = metrics[(metrics["split"] == "test") & metrics["horizon_hours"].notna()].copy()
    lines = [
        "| 时间窗 | 模型 | MAE（天） | RMSE（天） | 中位绝对误差（天） | R² | Spearman |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "ridge_log_target": "岭回归",
        "hist_gradient_boosting_log_target": "直方图梯度提升",
    }
    for row in test.sort_values(["horizon_hours", "model"]).itertuples(index=False):
        lines.append(
            f"| {int(row.horizon_hours)} 小时 | {labels[row.model]} | {row.mae_days:.2f} | "
            f"{row.rmse_days:.2f} | {row.median_ae_days:.2f} | {row.r2:.3f} | {row.spearman:.3f} |"
        )
    return "\n".join(lines)


def write_report(
    *,
    metrics: pd.DataFrame,
    primary: pd.Series,
    intervals: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    ci = intervals.set_index("metric")
    model_label = {
        "ridge_log_target": "岭回归",
        "hist_gradient_boosting_log_target": "直方图梯度提升",
    }[str(primary["model"])]
    report = rf"""# 实验 04：住院时长回归

## 研究问题与数据边界

本实验使用 ICU 入院后前 6、12、24、48 小时的静态信息、生命体征和化验记录，预测从 ICU 入院到住院结束的总天数。`Length_of_stay` 缺失或记为 `-1` 的 {metadata['los_missing_or_minus_one']:,} 例被排除；另有 {metadata['los_below_two_days']:,} 例小于 2 天，与数据集“ICU 停留不足 48 小时者已排除”的纳入条件不一致，也不进入训练。最终纳入 {metadata['valid_los_records']:,} 例。模型不使用 `RecordID`、SAPS-I、SOFA、院内死亡或生存时间，因此不会直接把结局信息作为特征。

这里预测的是“观察到的住院结束时间”，而不等同于康复所需时间。院内死亡会提前结束住院，转院和不同医院的出院制度也会影响结果；数据又没有医院、日期或患者身份，因此暂时无法处理中心差异、时间漂移和同一患者重复入院。

## 训练设计

数据按住院时长十分位固定分为训练集 {metadata['split_sizes']['train']:,} 例、验证集 {metadata['split_sizes']['validation']:,} 例和测试集 {metadata['split_sizes']['test']:,} 例。每个时序变量仍采用实验 02 的九类摘要特征；连续输入按训练集 0.5% 和 99.5% 分位截尾，所有中位数填补、缺失指示和标准化参数也只从训练集学习。住院时长呈明显右偏，因此两类模型都拟合 `log(1 + LOS)`，预测后再变换回天数，并限制在 2 天至训练集最大 LOS 之间，防止线性模型对未覆盖极端值产生无依据外推。岭回归提供线性、可审计的基线，直方图梯度提升允许阈值、非线性和变量交互。时间窗与模型只按验证集 MAE 选择，最终模型再用训练集与验证集合并拟合，并在此前未使用的测试集上评价。

数学上，岭回归在对数结局上寻找加权和 $z=\beta_0+x^T\beta$，同时最小化预测残差与 $\alpha\|\beta\|_2^2$，后一个惩罚项会压缩不稳定的大系数；天数预测为 $\exp(z)-1$。梯度提升则从一个初始预测开始，按 $F_m(x)=F_{{m-1}}(x)+\eta h_m(x)$ 逐棵加入小树来修正残差，因此能表示“某项指标超过阈值后风险变化”以及变量之间的组合。两者使用相同输入和同一划分，差异主要来自函数形式。

## 全部候选模型的测试集结果

{markdown_test_table(metrics)}

训练集中位数基线的测试集 MAE 为 {float(metrics[(metrics['model'] == 'training_median') & (metrics['split'] == 'test')]['mae_days'].iloc[0]):.2f} 天。验证集选择出的方案为前 {int(primary['horizon_hours'])} 小时 {model_label}。合并训练集和验证集后，其测试集 MAE 为 {primary['mae_days']:.2f} 天（bootstrap 95% CI {ci.loc['mae_days', 'ci_2.5%']:.2f}–{ci.loc['mae_days', 'ci_97.5%']:.2f}），RMSE 为 {primary['rmse_days']:.2f} 天（95% CI {ci.loc['rmse_days', 'ci_2.5%']:.2f}–{ci.loc['rmse_days', 'ci_97.5%']:.2f}），R² 为 {primary['r2']:.3f}。MAE 便于解释为平均偏差多少天；RMSE 对少数长住院的大误差更加敏感，因此通常高于 MAE。

按最终结局作簇后误差检查时，存活出院者的 MAE 为 {float(subgroup_metrics.loc[subgroup_metrics['level'] == 'survived_to_discharge', 'mae_days'].iloc[0]):.2f} 天，院内死亡者为 {float(subgroup_metrics.loc[subgroup_metrics['level'] == 'in_hospital_death', 'mae_days'].iloc[0]):.2f} 天。死亡结局从未进入模型；这项差异提示“死亡提前结束住院”的竞争事件需要在下一阶段单独处理。

![不同时间窗的验证集 MAE](output/report_figures/validation_mae_by_horizon.png)

![最终模型观察值与预测值](output/report_figures/observed_vs_predicted.png)

## 解释与下一步

这个结果应被视为内部预测基线，而不是可直接部署的床位周转工具。固定测试集评估回答的是同一数据源内的复现能力；它不能替代医院外验证。后续可以在不触碰当前测试集的前提下，对填补方式、异常值截尾、目标函数和长住院分层损失做重复交叉验证，并把死亡作为竞争结局单独建模。患者级预测、划分和共享特征缓存位于 Git 忽略的 `output/data/`，仓库只保存汇总指标、代码、图和元数据。
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Experiments" / "shared" / "output" / "data")
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 24, 48])
    parser.add_argument("--random-state", type=int, default=20260906)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = sorted(set(args.horizons))
    outcomes = read_outcomes(args.data_dir)
    if args.max_records is not None:
        outcomes = outcomes.head(args.max_records).copy()
    record_ids = outcomes["RecordID"].to_numpy(np.int64)
    columns, matrices, quality, coverage, cache_hit = load_or_extract_features(
        data_dir=args.data_dir,
        record_ids=record_ids,
        horizons=horizons,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )

    los_raw = outcomes["Length_of_stay"].to_numpy(float)
    missing_los = ~np.isfinite(los_raw) | (los_raw == -1)
    below_two = np.isfinite(los_raw) & (los_raw != -1) & (los_raw < 2)
    valid = ~(missing_los | below_two)
    if valid.sum() < 100:
        raise ValueError("Too few valid LOS outcomes for a 70/15/15 regression split")
    y = los_raw[valid]
    valid_outcomes = outcomes.loc[valid].reset_index(drop=True)
    split_indices = make_quantile_splits(y, args.random_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_output = args.output_dir / "data"
    figure_output = args.output_dir / "report_figures"
    data_output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)

    split_labels = np.full(len(y), "", dtype=object)
    for split, indices in split_indices.items():
        split_labels[indices] = split
    prediction_frame = pd.DataFrame(
        {
            "RecordID": valid_outcomes["RecordID"].to_numpy(int),
            "Length_of_stay": y,
            "In-hospital_death": valid_outcomes["In-hospital_death"].to_numpy(float),
            "split": split_labels,
        }
    )

    metric_rows: list[dict[str, object]] = []
    train_median = float(np.median(y[split_indices["train"]]))
    for split in ("validation", "test"):
        prediction = np.full(len(split_indices[split]), train_median)
        row: dict[str, object] = {
            "model": "training_median",
            "horizon_hours": math.nan,
            "split": split,
            "training_target": "Length_of_stay_days",
        }
        row.update(regression_metrics(y[split_indices[split]], prediction))
        metric_rows.append(row)

    model_names = ("ridge_log_target", "hist_gradient_boosting_log_target")
    fitted: dict[tuple[int, str], object] = {}
    for horizon in horizons:
        x = matrices[horizon][valid]
        for name in model_names:
            model = build_model(name, args.random_state)
            model.fit(x[split_indices["train"]], np.log1p(y[split_indices["train"]]))
            fitted[(horizon, name)] = model
            for split in ("validation", "test"):
                prediction = predict_days(
                    model,
                    x[split_indices[split]],
                    maximum_days=float(np.max(y[split_indices["train"]])),
                )
                row = {
                    "model": name,
                    "horizon_hours": horizon,
                    "split": split,
                    "training_target": "log1p_Length_of_stay",
                }
                row.update(regression_metrics(y[split_indices[split]], prediction))
                metric_rows.append(row)
                prediction_frame.loc[
                    split_indices[split], f"{horizon}h_{name}_prediction_days"
                ] = prediction

    metrics = pd.DataFrame(metric_rows)
    candidates = metrics[(metrics["split"] == "validation") & metrics["horizon_hours"].notna()]
    selected = candidates.sort_values(["mae_days", "rmse_days", "horizon_hours", "model"]).iloc[0]
    selected_horizon = int(selected["horizon_hours"])
    selected_name = str(selected["model"])
    train_validation = np.sort(
        np.concatenate([split_indices["train"], split_indices["validation"]])
    )
    final_model = build_model(selected_name, args.random_state)
    final_x = matrices[selected_horizon][valid]
    final_model.fit(final_x[train_validation], np.log1p(y[train_validation]))
    final_prediction = predict_days(
        final_model,
        final_x[split_indices["test"]],
        maximum_days=float(np.max(y[train_validation])),
    )
    primary_row: dict[str, object] = {
        "model": selected_name,
        "horizon_hours": selected_horizon,
        "split": "test",
        "fit_sample": "train_plus_validation_after_validation_selection",
        "training_target": "log1p_Length_of_stay",
    }
    primary_row.update(regression_metrics(y[split_indices["test"]], final_prediction))
    primary = pd.DataFrame([primary_row])
    prediction_frame.loc[split_indices["test"], "primary_final_prediction_days"] = final_prediction

    intervals = bootstrap_intervals(
        y[split_indices["test"]],
        final_prediction,
        repeats=args.bootstrap_repeats,
        random_state=args.random_state + 1,
    )
    subgroup_rows = []
    test_outcomes = valid_outcomes.iloc[split_indices["test"]]
    for value, label in ((0, "survived_to_discharge"), (1, "in_hospital_death")):
        mask = test_outcomes["In-hospital_death"].to_numpy() == value
        if mask.any():
            row = {"subgroup": "In-hospital_death", "level": label}
            row.update(regression_metrics(y[split_indices["test"]][mask], final_prediction[mask]))
            subgroup_rows.append(row)
    subgroup_metrics = pd.DataFrame(subgroup_rows)

    metrics.to_csv(args.output_dir / "candidate_metrics.csv", index=False)
    primary.to_csv(args.output_dir / "primary_test_metrics.csv", index=False)
    intervals.to_csv(args.output_dir / "bootstrap_confidence_intervals.csv", index=False)
    subgroup_metrics.to_csv(args.output_dir / "subgroup_metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output_dir / "feature_coverage.csv", index=False)
    prediction_frame.to_csv(data_output / "predictions_and_splits.csv", index=False)

    draw_performance(metrics, figure_output / "validation_mae_by_horizon.png")
    draw_observed_vs_predicted(
        y[split_indices["test"]],
        final_prediction,
        f"前 {selected_horizon} 小时；直方图梯度提升（log 结局）；测试集 n={len(final_prediction):,}",
        figure_output / "observed_vs_predicted.png",
    )

    metadata: dict[str, object] = {
        "data_layout": "release/outcomes.csv + release/icu_records/<RecordID>.csv",
        "source_record_count": int(len(outcomes)),
        "los_missing_or_minus_one": int(missing_los.sum()),
        "los_below_two_days": int(below_two.sum()),
        "valid_los_records": int(valid.sum()),
        "horizons_hours": horizons,
        "feature_count_per_horizon": len(columns),
        "split_sizes": {key: int(len(value)) for key, value in split_indices.items()},
        "selection_rule": "minimum validation MAE; RMSE, horizon and model name break ties",
        "feature_winsorization": "training 0.5th and 99.5th percentiles",
        "prediction_bounds": "2 days to the maximum LOS observed in the fitting partition",
        "selected_model": selected_name,
        "selected_horizon_hours": selected_horizon,
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_state": args.random_state,
        "feature_cache_hit": cache_hit,
        "scikit_learn_version": sklearn.__version__,
        "quality_counters": quality,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        metrics=metrics,
        primary=primary.iloc[0],
        intervals=intervals,
        subgroup_metrics=subgroup_metrics,
        metadata=metadata,
        path=EXPERIMENT_DIR / "README.md",
    )
    print(primary.to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
