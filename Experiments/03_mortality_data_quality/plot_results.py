#!/usr/bin/env python3
"""Create static report figures for Experiment 03 from aggregate CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_DIR = Path(__file__).resolve().parent


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


BG = "#F7F8FA"
WHITE = "#FFFFFF"
INK = "#17212B"
MUTED = "#5B6773"
GRID = "#D9DEE5"
BLUE = "#2474B5"
ORANGE = "#D97824"
GREEN = "#37805B"
PURPLE = "#7455A6"
GRAY = "#7A858F"


PREPROCESS_LABELS = {
    "raw_median": "原始＋中位数",
    "raw_median_indicator": "原始＋中位数＋缺失指示",
    "raw_mean_indicator": "原始＋均值＋缺失指示",
    "raw_native_missing": "原始＋原生缺失",
    "raw_winsor_median_indicator": "原始＋截断＋中位数＋缺失指示",
    "clean_median_indicator": "宽松清洗＋中位数＋缺失指示",
    "clean_winsor_median_indicator": "宽松清洗＋截断＋中位数＋缺失指示",
}

ABLATION_LABELS = {
    "static_only": "仅静态信息",
    "clinical_values_no_indicator": "临床数值，无缺失指示",
    "clinical_values_plus_indicator": "临床数值＋缺失指示",
    "measurement_process_only": "仅测量过程",
    "full_selected": "完整特征",
}


def text_width(draw: ImageDraw.ImageDraw, value: str, used_font) -> float:
    bounds = draw.textbbox((0, 0), value, font=used_font)
    return bounds[2] - bounds[0]


def save(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True)


def preprocess_screen(cv: pd.DataFrame, destination: Path) -> None:
    image = Image.new("RGB", (1800, 1160), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), "24 小时训练集：不同清洗与填补方案的五折 AUPRC", font=font(39, True), fill=INK)
    draw.text((70, 94), "圆点为五折均值，横线为折间标准差；两个模型使用各自横轴范围。", font=font(22), fill=MUTED)

    panels = [
        ("logistic", "逻辑回归", (65, 165, 875, 1080), ORANGE),
        ("hist_gradient_boosting", "直方图梯度提升", (925, 165, 1735, 1080), BLUE),
    ]
    for model_name, title, box, color in panels:
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(box, radius=18, fill=WHITE)
        draw.text((x0 + 28, y0 + 22), title, font=font(28, True), fill=INK)
        subset = cv[cv["model"].eq(model_name)].sort_values("auprc_mean")
        lower = float((subset["auprc_mean"] - subset["auprc_sd"]).min()) - 0.006
        upper = float((subset["auprc_mean"] + subset["auprc_sd"]).max()) + 0.006
        label_right = x0 + 365
        chart_left, chart_right = x0 + 390, x1 - 35
        chart_top, chart_bottom = y0 + 105, y1 - 65
        for tick in np.linspace(lower, upper, 5):
            x = chart_left + (tick - lower) / (upper - lower) * (chart_right - chart_left)
            draw.line((x, chart_top, x, chart_bottom), fill=GRID, width=1)
            label = f"{tick:.2f}"
            draw.text((x - text_width(draw, label, font(18)) / 2, chart_bottom + 14), label, font=font(18), fill=MUTED)
        row_gap = (chart_bottom - chart_top) / len(subset)
        for row_number, row in enumerate(subset.itertuples(index=False)):
            y = chart_top + (row_number + 0.5) * row_gap
            label = PREPROCESS_LABELS[row.preprocess]
            width = text_width(draw, label, font(18))
            draw.text((label_right - width, y - 13), label, font=font(18), fill=INK)
            mean = float(row.auprc_mean)
            sd = float(row.auprc_sd)
            left = chart_left + (mean - sd - lower) / (upper - lower) * (chart_right - chart_left)
            right = chart_left + (mean + sd - lower) / (upper - lower) * (chart_right - chart_left)
            center = chart_left + (mean - lower) / (upper - lower) * (chart_right - chart_left)
            draw.line((left, y, right, y), fill=color, width=4)
            draw.ellipse((center - 7, y - 7, center + 7, y + 7), fill=color)
            draw.text((min(center + 12, chart_right - 52), y - 13), f"{mean:.3f}", font=font(17, True), fill=color)
        axis_label = "AUPRC（越高越好）"
        draw.text(((chart_left + chart_right - text_width(draw, axis_label, font(19))) / 2, y1 - 36), axis_label, font=font(19), fill=MUTED)
    save(image, destination)


def line_chart(
    draw: ImageDraw.ImageDraw,
    box,
    title: str,
    metric: str,
    series,
    y_domain,
    lower_is_better: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=WHITE)
    draw.text((x0 + 25, y0 + 20), title, font=font(27, True), fill=INK)
    left, right, top, bottom = x0 + 82, x1 - 35, y0 + 80, y1 - 75
    low, high = y_domain
    for tick in np.linspace(low, high, 5):
        y = bottom - (tick - low) / (high - low) * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = f"{tick:.2f}"
        draw.text((left - 14 - text_width(draw, label, font(17)), y - 11), label, font=font(17), fill=MUTED)
    horizons = [6, 12, 24, 48]
    x_values = {h: left + index * (right - left) / 3 for index, h in enumerate(horizons)}
    for horizon, x in x_values.items():
        draw.text((x - text_width(draw, str(horizon), font(18)) / 2, bottom + 15), str(horizon), font=font(18), fill=MUTED)
    for values, color, width, marker, dashed in series:
        points = [(x_values[h], bottom - (values[h] - low) / (high - low) * (bottom - top)) for h in horizons]
        for first, second in zip(points[:-1], points[1:]):
            if dashed:
                length = int(abs(second[0] - first[0]))
                for offset in range(0, length, 18):
                    ratio1 = offset / length
                    ratio2 = min((offset + 10) / length, 1)
                    a = (first[0] + ratio1 * (second[0] - first[0]), first[1] + ratio1 * (second[1] - first[1]))
                    b = (first[0] + ratio2 * (second[0] - first[0]), first[1] + ratio2 * (second[1] - first[1]))
                    draw.line((*a, *b), fill=color, width=width)
            else:
                draw.line((*first, *second), fill=color, width=width)
        for x, y in points:
            if marker == "square":
                draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill=color)
            else:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    direction = "越低越好" if lower_is_better else "越高越好"
    draw.text(((left + right - text_width(draw, "观察时间（小时）", font(18))) / 2, y1 - 37), "观察时间（小时）", font=font(18), fill=MUTED)
    draw.text((x0 + 20, y1 - 37), direction, font=font(16), fill=MUTED)


def test_performance(final: pd.DataFrame, baseline: pd.DataFrame, destination: Path) -> None:
    image = Image.new("RGB", (1800, 850), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "确认集结果：第二阶段所选方案与第一阶段基线", font=font(39, True), fill=INK)
    draw.text((70, 90), "实线为第二阶段所选方案，虚线为第一阶段基线；测试集为同一组 1,800 次住院。", font=font(21), fill=MUTED)

    legend = [
        ("逻辑回归：第一阶段", GRAY, "circle", True),
        ("逻辑回归：第二阶段", ORANGE, "circle", False),
        ("梯度提升：第一阶段", PURPLE, "square", True),
        ("梯度提升：第二阶段", BLUE, "square", False),
    ]
    x = 75
    for label, color, marker, dashed in legend:
        if dashed:
            for start in (0, 18, 36):
                draw.line((x + start, 148, x + start + 10, 148), fill=color, width=4)
        else:
            draw.line((x, 148, x + 46, 148), fill=color, width=4)
        if marker == "square":
            draw.rectangle((x + 18, 142, x + 30, 154), fill=color)
        else:
            draw.ellipse((x + 18, 142, x + 30, 154), fill=color)
        draw.text((x + 57, 135), label, font=font(18), fill=INK)
        x += 405

    baseline = baseline[baseline["split"].eq("test") & baseline["horizon_hours"].notna()].copy()
    model_maps = {
        "logistic": ("LogisticRegression", ORANGE, GRAY, "circle"),
        "hist_gradient_boosting": ("HistGradientBoostingClassifier", BLUE, PURPLE, "square"),
    }
    panels = [
        ((60, 185, 590, 790), "AUROC", "auroc", (0.74, 0.90), False),
        ((635, 185, 1165, 790), "AUPRC", "auprc", (0.32, 0.63), False),
        ((1210, 185, 1740, 790), "Brier 分数", "brier", (0.075, 0.115), True),
    ]
    for box, title, metric, domain, lower_better in panels:
        plot_series = []
        for model_name, (baseline_name, selected_color, baseline_color, marker) in model_maps.items():
            selected_values = {
                int(row.horizon_hours): float(getattr(row, metric))
                for row in final[final["model"].eq(model_name)].itertuples(index=False)
            }
            baseline_values = {
                int(float(row.horizon_hours)): float(getattr(row, metric))
                for row in baseline[baseline["model"].eq(baseline_name)].itertuples(index=False)
            }
            plot_series.append((baseline_values, baseline_color, 3, marker, True))
            plot_series.append((selected_values, selected_color, 4, marker, False))
        line_chart(draw, box, title, metric, plot_series, domain, lower_better)
    save(image, destination)


def calibration_and_ablation(
    calibration_curve: pd.DataFrame,
    calibration: pd.DataFrame,
    ablations: pd.DataFrame,
    destination: Path,
) -> None:
    image = Image.new("RGB", (1800, 1060), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "48 小时模型：校准与特征消融", font=font(39, True), fill=INK)
    draw.text((70, 90), "校准图使用确认集十分位；消融图比较 24 与 48 小时 AUPRC。", font=font(21), fill=MUTED)

    left_box = (60, 155, 855, 995)
    draw.rounded_rectangle(left_box, radius=18, fill=WHITE)
    draw.text((90, 180), "预测概率与实际死亡率", font=font(28, True), fill=INK)
    chart = (155, 275, 790, 875)
    cx0, cy0, cx1, cy1 = chart
    for tick in np.linspace(0, 0.7, 8):
        x = cx0 + tick / 0.7 * (cx1 - cx0)
        y = cy1 - tick / 0.7 * (cy1 - cy0)
        draw.line((x, cy0, x, cy1), fill=GRID, width=1)
        draw.line((cx0, y, cx1, y), fill=GRID, width=1)
        label = f"{tick:.1f}"
        draw.text((x - text_width(draw, label, font(16)) / 2, cy1 + 12), label, font=font(16), fill=MUTED)
        draw.text((cx0 - 13 - text_width(draw, label, font(16)), y - 10), label, font=font(16), fill=MUTED)
    draw.line((cx0, cy1, cx1, cy0), fill=GRAY, width=3)
    for model_name, label, color, marker in (
        ("logistic", "逻辑回归", ORANGE, "circle"),
        ("hist_gradient_boosting", "梯度提升", BLUE, "square"),
    ):
        subset = calibration_curve[
            calibration_curve["model"].eq(model_name)
            & calibration_curve["horizon_hours"].eq(48)
        ].sort_values("mean_predicted_probability")
        points = []
        for row in subset.itertuples(index=False):
            x = cx0 + float(row.mean_predicted_probability) / 0.7 * (cx1 - cx0)
            y = cy1 - float(row.observed_event_rate) / 0.7 * (cy1 - cy0)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            if marker == "square":
                draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill=color)
            else:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        row = calibration[
            calibration["model"].eq(model_name) & calibration["horizon_hours"].eq(48)
        ].iloc[0]
        summary = f"{label}：截距 {row.calibration_intercept:+.2f}，斜率 {row.calibration_slope:.2f}"
        y_text = 910 if model_name == "logistic" else 946
        draw.text((145, y_text), summary, font=font(19), fill=color)
    draw.text(((cx0 + cx1 - text_width(draw, "平均预测概率", font(19))) / 2, 886), "平均预测概率", font=font(19), fill=MUTED)
    draw.text((75, 550), "实际死亡率", font=font(19), fill=MUTED)

    test = ablations[ablations["split"].eq("test") & ablations["horizon_hours"].isin([24, 48])]
    for panel_number, (model_name, title, color, box) in enumerate(
        [
            ("logistic", "逻辑回归特征消融", ORANGE, (910, 155, 1740, 555)),
            ("hist_gradient_boosting", "梯度提升特征消融", BLUE, (910, 595, 1740, 995)),
        ]
    ):
        x0, y0, x1, y1 = box
        draw.rounded_rectangle(box, radius=18, fill=WHITE)
        draw.text((x0 + 28, y0 + 20), title, font=font(27, True), fill=INK)
        left, right = x0 + 300, x1 - 40
        top, bottom = y0 + 83, y1 - 52
        x_min, x_max = (0.22, 0.58) if model_name == "logistic" else (0.22, 0.63)
        for tick in np.linspace(x_min, x_max, 5):
            x = left + (tick - x_min) / (x_max - x_min) * (right - left)
            draw.line((x, top, x, bottom), fill=GRID, width=1)
            draw.text((x - text_width(draw, f"{tick:.2f}", font(15)) / 2, bottom + 9), f"{tick:.2f}", font=font(15), fill=MUTED)
        order = list(ABLATION_LABELS)
        row_gap = (bottom - top) / len(order)
        subset = test[test["model"].eq(model_name)]
        for row_number, group_name in enumerate(order):
            y = top + (row_number + 0.5) * row_gap
            label = ABLATION_LABELS[group_name]
            draw.text((x0 + 25, y - 11), label, font=font(17), fill=INK)
            for horizon, offset, marker in ((24, -7, "circle"), (48, 7, "square")):
                value = float(
                    subset[
                        subset["feature_group"].eq(group_name)
                        & subset["horizon_hours"].eq(horizon)
                    ]["auprc"].iloc[0]
                )
                x = left + (value - x_min) / (x_max - x_min) * (right - left)
                if marker == "square":
                    draw.rectangle((x - 5, y + offset - 5, x + 5, y + offset + 5), fill=color)
                else:
                    draw.ellipse((x - 5, y + offset - 5, x + 5, y + offset + 5), fill=color)
        draw.text((x1 - 230, y0 + 27), "● 24小时   ■ 48小时", font=font(16), fill=MUTED)
    save(image, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=EXPERIMENT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.experiment_dir / "output"
    figures = output / "report_figures"
    cv = pd.read_csv(output / "cv_summary.csv")
    final = pd.read_csv(output / "test_metrics.csv")
    baseline = pd.read_csv(
        args.experiment_dir.parent / "02_mortality_prediction_baseline" / "output" / "metrics.csv"
    )
    calibration = pd.read_csv(output / "calibration_metrics.csv")
    calibration_curve = pd.read_csv(output / "calibration_curve.csv")
    ablations = pd.read_csv(output / "feature_ablation_metrics.csv")
    preprocess_screen(cv, figures / "preprocessing_screen.png")
    test_performance(final, baseline, figures / "test_performance_comparison.png")
    calibration_and_ablation(
        calibration_curve,
        calibration,
        ablations,
        figures / "calibration_and_ablation.png",
    )
    print(f"Figures written to {figures}")


if __name__ == "__main__":
    main()
