#!/usr/bin/env python3
"""Experiment 07: remaining hospital length of stay after the 48-hour landmark."""

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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
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


def center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], label: str, used_font, fill=INK) -> None:
    box = draw.textbbox((0, 0), label, font=used_font)
    draw.text(
        (xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2),
        label,
        font=used_font,
        fill=fill,
    )


class QuantileClipper(BaseEstimator, TransformerMixin):
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
        return np.clip(np.asarray(x, dtype=float), self.lower_bounds_, self.upper_bounds_)


def make_quantile_splits(y: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    bins = np.asarray(pd.qcut(y, q=10, labels=False, duplicates="drop"), dtype=int)
    train, remainder = train_test_split(
        indices, test_size=0.30, random_state=random_state, stratify=bins
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=random_state, stratify=bins[remainder]
    )
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}


def build_model(model_name: str, parameter: float, random_state: int):
    common = [
        QuantileClipper(),
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
    ]
    if model_name == "ridge":
        return make_pipeline(*common, StandardScaler(), Ridge(alpha=parameter))
    if model_name == "lasso":
        return make_pipeline(
            *common,
            StandardScaler(),
            Lasso(alpha=parameter, max_iter=15000, selection="cyclic"),
        )
    if model_name == "hist_gradient_boosting":
        return make_pipeline(
            *common,
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=parameter,
                early_stopping=True,
                random_state=random_state,
            ),
        )
    if model_name == "random_forest":
        return make_pipeline(
            *common,
            RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=int(parameter),
                max_features=0.5,
                n_jobs=-1,
                random_state=random_state,
            ),
        )
    raise ValueError(f"Unknown model: {model_name}")


def smearing_factor(model, x_train: np.ndarray, y_train: np.ndarray) -> float:
    residual = np.log1p(y_train) - model.predict(x_train)
    return float(np.mean(np.exp(residual)))


def predict_remaining(model, x: np.ndarray, smearing: float = 1.0) -> np.ndarray:
    return np.maximum(np.exp(model.predict(x)) * smearing - 1.0, 0.0)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    rank_true = pd.Series(y_true).rank(method="average").to_numpy()
    rank_prediction = pd.Series(prediction).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rank_true, rank_prediction)[0, 1]) if np.std(rank_prediction) else math.nan
    true_sum = float(np.sum(y_true))
    predicted_sum = float(np.sum(prediction))
    return {
        "n": int(len(y_true)),
        "mae_days": float(mean_absolute_error(y_true, prediction)),
        "rmse_days": float(np.sqrt(mean_squared_error(y_true, prediction))),
        "median_ae_days": float(median_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
        "spearman": spearman,
        "true_sum_bed_days": true_sum,
        "predicted_sum_bed_days": predicted_sum,
        "sum_bias_percent": float(100.0 * (predicted_sum - true_sum) / true_sum),
    }


def bootstrap_intervals(
    y_true: np.ndarray, prediction: np.ndarray, repeats: int, random_state: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    samples = {"mae_days": [], "rmse_days": [], "sum_bias_percent": []}
    for _ in range(repeats):
        selected = rng.integers(0, len(y_true), size=len(y_true))
        metrics = regression_metrics(y_true[selected], prediction[selected])
        for key in samples:
            samples[key].append(metrics[key])
    point = regression_metrics(y_true, prediction)
    rows = []
    for metric, values in samples.items():
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "point_estimate": point[metric],
                "ci_2_5": float(low),
                "ci_97_5": float(high),
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(rows)


def icu_types_from_matrix(matrix: np.ndarray, feature_names: list[str]) -> np.ndarray:
    indices = [feature_names.index(f"ICUType_{value}") for value in range(1, 5)]
    encoded = matrix[:, indices]
    if not np.allclose(np.nansum(encoded, axis=1), 1.0):
        raise ValueError("ICU type one-hot columns are incomplete")
    return np.argmax(encoded, axis=1).astype(int) + 1


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]], decimals: int = 3) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "---|" * len(columns)]
    for row in frame.itertuples(index=False):
        values = []
        for key, _ in columns:
            value = getattr(row, key)
            values.append(f"{value:.{decimals}f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def draw_model_comparison(metrics: pd.DataFrame, destination: Path) -> None:
    validation = metrics[(metrics["split"] == "validation") & (metrics["readout"] == "direct")]
    validation = validation.sort_values("mae_days")
    image = Image.new("RGB", (1650, 980), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "48 小时后剩余住院时间：验证集个体 MAE", font=font(38, True), fill=INK)
    draw.text((55, 92), "所有模型拟合 log(1+剩余天数)；越低越好。", font=font(22), fill=MUTED)
    draw.rounded_rectangle((45, 145, 1605, 920), radius=20, fill=WHITE)
    x0, x1 = 530, 1540
    y0, y1 = 185, 850
    maximum = max(8.0, float(validation["mae_days"].max()) + 0.5)
    for tick in np.linspace(0, maximum, 5):
        px = x0 + tick / maximum * (x1 - x0)
        draw.line((px, y0, px, y1), fill=GRID, width=1)
        center(draw, (px, y1 + 28), f"{tick:.1f}", font(17), MUTED)
    row_height = (y1 - y0) / len(validation)
    for index, row in enumerate(validation.itertuples(index=False)):
        cy = y0 + (index + 0.5) * row_height
        label = f"{row.model} ({row.parameter:g})"
        draw.text((75, cy - 14), label, font=font(19), fill=INK)
        end = x0 + row.mae_days / maximum * (x1 - x0)
        draw.rounded_rectangle((x0, cy - 15, end, cy + 15), radius=7, fill=BLUE)
        draw.text((end + 12, cy - 13), f"{row.mae_days:.2f}", font=font(18, True), fill=INK)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def draw_readout_comparison(final_metrics: pd.DataFrame, destination: Path) -> None:
    image = Image.new("RGB", (1500, 900), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "个体误差与队列总床日是两个读数", font=font(38, True), fill=INK)
    draw.text((55, 92), "直接反变换偏向个体中位数；smearing 用训练残差校正队列均值。", font=font(22), fill=MUTED)
    draw.rounded_rectangle((45, 145, 1455, 830), radius=20, fill=WHITE)
    colors = {"individual_direct": BLUE, "cohort_smeared": ORANGE}
    x_positions = {"individual_direct": 460, "cohort_smeared": 1040}
    for row in final_metrics.itertuples(index=False):
        x = x_positions[row.readout]
        draw.rounded_rectangle((x - 220, 225, x + 220, 700), radius=18, outline=colors[row.readout], width=5)
        center(draw, (x, 280), "个体直接读数" if row.readout == "individual_direct" else "队列 Smearing 读数", font(27, True))
        center(draw, (x, 390), f"个体 MAE {row.mae_days:.2f} 天", font(25), colors[row.readout])
        center(draw, (x, 485), f"队列总床日偏差 {row.sum_bias_percent:+.1f}%", font(25), colors[row.readout])
        center(draw, (x, 580), f"R² {row.r2:.3f}", font(23), MUTED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def write_report(
    destination: Path,
    candidate_metrics: pd.DataFrame,
    selected_individual_model: str,
    selected_individual_parameter: float,
    selected_cohort_model: str,
    selected_cohort_parameter: float,
    final_metrics: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    candidate_test = candidate_metrics[
        (candidate_metrics["split"] == "test") & (candidate_metrics["readout"] == "direct")
    ].sort_values("mae_days")
    report = f"""# 实验 07：48 小时后的剩余住院时间

## 研究问题

实验 04 预测从 ICU 入院开始计算的总住院时长；在 48 小时预测时点，这个目标包含了患者已经住满两天这一已知事实。本实验改为 `剩余住院时间=Length_of_stay-2`，只纳入住院时长有效且至少为 2 天的 11,828 次住院。目标仍是医院住院结束时间，不是 ICU 停留时间，也不是康复所需时间。

数据沿用按住院时长十分位划分的 70%/15%/15% 训练、验证和测试结构。中位数填补、缺失指示、标准化和 0.5%—99.5% 分位截尾只从训练数据学习。岭回归、Lasso、直方图梯度提升和随机森林都拟合 `log(1+剩余天数)`；超参数只按验证集直接反变换后的 MAE 选择。

## 候选模型

{markdown_table(candidate_test, [("model", "模型"), ("parameter", "参数"), ("mae_days", "MAE"), ("rmse_days", "RMSE"), ("r2", "R²"), ("sum_bias_percent", "总床日偏差%")])}

个体预测按验证集直接反变换 MAE 选中 `{selected_individual_model}`，参数为 {selected_individual_parameter:g}；队列床日按验证集 smearing 后总量绝对偏差另选中 `{selected_cohort_model}`，参数为 {selected_cohort_parameter:g}。模型间差异需要结合 Bootstrap 区间理解，不能只按小数点后的最低 MAE 宣称一种算法稳定优于另一种算法。

![候选模型验证集 MAE](output/report_figures/model_comparison.png)

## 两种读数

{markdown_table(final_metrics, [("readout", "读数"), ("mae_days", "MAE"), ("median_ae_days", "中位绝对误差"), ("rmse_days", "RMSE"), ("r2", "R²"), ("sum_bias_percent", "总床日偏差%")])}

个体读数使用验证集 MAE 最低模型的直接反变换；队列读数使用验证集总床日偏差最小模型的 Duan smearing 校正。后者使用训练残差估计乘法因子，更接近条件均值。两者的模型和选择标准都不同，不应把总量校准较好解释成个体出院日期已经准确。

![个体误差与总床日](output/report_figures/readout_comparison.png)

最终直接读数的患者记录级 Bootstrap 区间：

{markdown_table(bootstrap, [("readout", "读数"), ("metric", "指标"), ("point_estimate", "点估计"), ("ci_2_5", "95% CI下限"), ("ci_97_5", "95% CI上限")])}

## 亚组与边界

{markdown_table(subgroup_metrics, [("readout", "读数"), ("subgroup", "亚组"), ("n", "n"), ("mae_days", "MAE"), ("true_sum_bed_days", "实际总床日"), ("predicted_sum_bed_days", "预测总床日"), ("sum_bias_percent", "总床日偏差%")], decimals=2)}

死亡会提前结束住院，长住院患者又形成右侧长尾，因此总体均值可能掩盖重要亚组误差。队列总床日接近真实值也可能来自不同 ICU 的正负误差抵消。当前结果只用于内部方法比较；没有医院和日期字段，无法进行跨医院或时间外容量验证，不能用于承诺个人出院日期。
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
    all_outcomes = read_outcomes(args.data_dir)
    record_ids = all_outcomes["RecordID"].to_numpy(np.int64)
    feature_names, matrices, _, _, cache_hit = load_or_extract_features(
        data_dir=args.data_dir,
        record_ids=record_ids,
        horizons=[48],
        cache_dir=args.cache_dir,
    )
    los = all_outcomes["Length_of_stay"].to_numpy(float)
    valid = np.isfinite(los) & (los >= 2)
    outcomes = all_outcomes.loc[valid].reset_index(drop=True)
    x = matrices[48][valid]
    remaining = los[valid] - 2.0
    death = outcomes["In-hospital_death"].to_numpy(int)
    splits = make_quantile_splits(remaining, args.random_state)
    candidates = {
        "ridge": [1.0, 10.0, 100.0, 1000.0],
        "lasso": [0.001, 0.01, 0.05, 0.1],
        "hist_gradient_boosting": [1.0],
        "random_forest": [5.0, 15.0],
    }
    rows = []
    fitted: dict[tuple[str, float], object] = {}
    smearing: dict[tuple[str, float], float] = {}
    for model_name, parameters in candidates.items():
        for parameter in parameters:
            model = build_model(model_name, parameter, args.random_state)
            model.fit(x[splits["train"]], np.log1p(remaining[splits["train"]]))
            factor = smearing_factor(model, x[splits["train"]], remaining[splits["train"]])
            fitted[(model_name, parameter)] = model
            smearing[(model_name, parameter)] = factor
            for split in ("validation", "test"):
                for readout, multiplier in (("direct", 1.0), ("smeared", factor)):
                    prediction = predict_remaining(model, x[splits[split]], multiplier)
                    row = {
                        "model": model_name,
                        "parameter": parameter,
                        "split": split,
                        "readout": readout,
                        "smearing_factor": multiplier,
                    }
                    row.update(regression_metrics(remaining[splits[split]], prediction))
                    rows.append(row)
    for split in ("validation", "test"):
        baseline = float(np.median(remaining[splits["train"]]))
        prediction = np.full(len(splits[split]), baseline)
        row = {
            "model": "training_median",
            "parameter": 0.0,
            "split": split,
            "readout": "direct",
            "smearing_factor": 1.0,
        }
        row.update(regression_metrics(remaining[splits[split]], prediction))
        rows.append(row)
    candidate_metrics = pd.DataFrame(rows)
    ranking = candidate_metrics[
        (candidate_metrics["split"] == "validation")
        & (candidate_metrics["readout"] == "direct")
        & (candidate_metrics["model"] != "training_median")
    ].sort_values(["mae_days", "rmse_days", "model", "parameter"])
    selected_individual_model = str(ranking.iloc[0]["model"])
    selected_individual_parameter = float(ranking.iloc[0]["parameter"])
    aggregate_ranking = candidate_metrics[
        (candidate_metrics["split"] == "validation")
        & (candidate_metrics["readout"] == "smeared")
        & (candidate_metrics["model"] != "training_median")
    ].copy()
    aggregate_ranking["absolute_sum_bias"] = aggregate_ranking["sum_bias_percent"].abs()
    aggregate_ranking = aggregate_ranking.sort_values(
        ["absolute_sum_bias", "mae_days", "model", "parameter"]
    )
    selected_cohort_model = str(aggregate_ranking.iloc[0]["model"])
    selected_cohort_parameter = float(aggregate_ranking.iloc[0]["parameter"])

    development = np.sort(np.concatenate([splits["train"], splits["validation"]]))
    individual_model = build_model(selected_individual_model, selected_individual_parameter, args.random_state)
    individual_model.fit(x[development], np.log1p(remaining[development]))
    cohort_model = build_model(selected_cohort_model, selected_cohort_parameter, args.random_state)
    cohort_model.fit(x[development], np.log1p(remaining[development]))
    cohort_smearing = smearing_factor(cohort_model, x[development], remaining[development])
    test_index = splits["test"]
    final_predictions = {
        "individual_direct": predict_remaining(individual_model, x[test_index], 1.0),
        "cohort_smeared": predict_remaining(cohort_model, x[test_index], cohort_smearing),
    }
    final_rows = []
    for readout, prediction in final_predictions.items():
        if readout == "individual_direct":
            model_name, parameter = selected_individual_model, selected_individual_parameter
        else:
            model_name, parameter = selected_cohort_model, selected_cohort_parameter
        row = {"model": model_name, "parameter": parameter, "readout": readout}
        row.update(regression_metrics(remaining[test_index], prediction))
        final_rows.append(row)
    final_metrics = pd.DataFrame(final_rows)

    icu_type = icu_types_from_matrix(x, feature_names)
    subgroup_rows = []
    direct = final_predictions["individual_direct"]
    smeared = final_predictions["cohort_smeared"]
    test_death = death[test_index]
    test_los = los[valid][test_index]
    for readout, prediction in (("individual_direct", direct), ("cohort_smeared", smeared)):
        groups = {
            "survived_to_discharge": test_death == 0,
            "in_hospital_death": test_death == 1,
            "long_stay_total_LOS_25+": test_los >= 25,
        }
        for value in range(1, 5):
            groups[f"ICUType_{value}"] = icu_type[test_index] == value
        for label, selected in groups.items():
            row = {"readout": readout, "subgroup": label}
            row.update(regression_metrics(remaining[test_index][selected], prediction[selected]))
            subgroup_rows.append(row)
    subgroup_metrics = pd.DataFrame(subgroup_rows)
    bootstrap_frames = []
    for offset, (readout, prediction) in enumerate(final_predictions.items()):
        frame = bootstrap_intervals(
            remaining[test_index], prediction, args.bootstrap_repeats, args.random_state + 29 + offset
        )
        frame.insert(0, "readout", readout)
        bootstrap_frames.append(frame)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)

    output = args.output_dir
    data_output = output / "data"
    figure_output = output / "report_figures"
    output.mkdir(parents=True, exist_ok=True)
    data_output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)
    candidate_metrics.to_csv(output / "candidate_metrics.csv", index=False)
    final_metrics.to_csv(output / "primary_test_metrics.csv", index=False)
    subgroup_metrics.to_csv(output / "subgroup_metrics.csv", index=False)
    bootstrap.to_csv(output / "bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(
        {
            "RecordID": outcomes.loc[test_index, "RecordID"].to_numpy(int),
            "remaining_los_days": remaining[test_index],
            "prediction_individual_direct": direct,
            "prediction_cohort_smeared": smeared,
        }
    ).to_csv(data_output / "test_predictions.csv", index=False)
    draw_model_comparison(candidate_metrics, figure_output / "model_comparison.png")
    draw_readout_comparison(final_metrics, figure_output / "readout_comparison.png")
    metadata = {
        "landmark_hours": 48,
        "target": "Length_of_stay - 2 days",
        "eligible_records": int(valid.sum()),
        "excluded_missing_los": int((~np.isfinite(los) | (los == -1)).sum()),
        "excluded_los_below_2": int((np.isfinite(los) & (los >= 0) & (los < 2)).sum()),
        "split_sizes": {name: int(len(indices)) for name, indices in splits.items()},
        "selected_individual_model": selected_individual_model,
        "selected_individual_parameter": selected_individual_parameter,
        "selected_cohort_model": selected_cohort_model,
        "selected_cohort_parameter": selected_cohort_parameter,
        "individual_selection_rule": "minimum validation direct-inverse MAE, then RMSE",
        "cohort_selection_rule": "minimum absolute validation smeared sum bias, then MAE",
        "cohort_smearing_factor": cohort_smearing,
        "random_state": args.random_state,
        "bootstrap_repeats": args.bootstrap_repeats,
        "cache_hit": bool(cache_hit),
        "feature_count": len(feature_names),
        "scikit_learn_version": sklearn.__version__,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        EXPERIMENT_DIR / "README.md",
        candidate_metrics,
        selected_individual_model,
        selected_individual_parameter,
        selected_cohort_model,
        selected_cohort_parameter,
        final_metrics,
        subgroup_metrics,
        bootstrap,
    )
    print(f"Selected individual: {selected_individual_model} ({selected_individual_parameter:g})")
    print(f"Selected cohort: {selected_cohort_model} ({selected_cohort_parameter:g})")
    print(final_metrics.to_string(index=False))
    print(f"Outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
