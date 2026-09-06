#!/usr/bin/env python3
"""Generate reproducible report figures from Experiment 02 outputs."""

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
RED = "#B73A3A"
PURPLE = "#7455A6"


def center(draw: ImageDraw.ImageDraw, xy, value: str, used_font, fill=INK):
    box = draw.textbbox((0, 0), value, font=used_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=used_font, fill=fill)


def multiline_center(draw, box, value, used_font, fill=INK, spacing=8):
    bounds = draw.multiline_textbbox((0, 0), value, font=used_font, spacing=spacing, align="center")
    x = (box[0] + box[2] - (bounds[2] - bounds[0])) / 2
    y = (box[1] + box[3] - (bounds[3] - bounds[1])) / 2
    draw.multiline_text((x, y), value, font=used_font, fill=fill, spacing=spacing, align="center")


def arrow(draw, start, end, fill=BLUE, width=5):
    draw.line((*start, *end), fill=fill, width=width)
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    for offset in (2.55, -2.55):
        tip = (
            end[0] + 18 * np.cos(angle + offset),
            end[1] + 18 * np.sin(angle + offset),
        )
        draw.line((*end, *tip), fill=fill, width=width)


def training_pipeline(destination: Path) -> None:
    image = Image.new("RGB", (1800, 930), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "院内死亡基线的完整训练流程", font=font(42, True), fill=INK)
    draw.text((70, 102), "学习型预处理只在训练集拟合；验证集选择阈值，测试集始终封存到最后。", font=font(23), fill=MUTED)

    boxes = [
        ((80, 205, 390, 390), "① 原始长表\n12,000次住院\n5,285,818行"),
        ((500, 205, 810, 390), "② 质量规则\n-1转缺失\n空字段排除\n同分钟取中位数"),
        ((920, 205, 1230, 390), "③ 严格时间窗\n6 / 12 / 24 / 48小时\n右端点不纳入"),
        ((1340, 205, 1650, 390), "④ 生成基础特征\n37变量×9统计\n+静态信息=341列"),
        ((1340, 560, 1650, 745), "⑤ 固定分层划分\n训练 8,400\n验证 1,800\n测试 1,800"),
        ((920, 560, 1230, 745), "⑥ 训练集内预处理\n中位数填补\n缺失指示\n逻辑回归再标准化"),
        ((500, 560, 810, 745), "⑦ 拟合两类模型\n逻辑回归\n梯度提升\n预设参数+早停"),
        ((80, 560, 390, 745), "⑧ 评价\n验证集选Youden阈值\n测试集报告概率指标\n与阈值指标"),
    ]
    fills = ["#E9F2FA", "#EDF5F0", "#F9F0E8", "#F2EDF8"] * 2
    for index, ((x0, y0, x1, y1), label) in enumerate(boxes):
        draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=fills[index], outline=GRID, width=2)
        multiline_center(draw, (x0 + 18, y0 + 12, x1 - 18, y1 - 12), label, font(24, index == 0), spacing=11)
    for left, right in [(boxes[0], boxes[1]), (boxes[1], boxes[2]), (boxes[2], boxes[3])]:
        arrow(draw, (left[0][2] + 15, (left[0][1] + left[0][3]) / 2), (right[0][0] - 15, (right[0][1] + right[0][3]) / 2))
    arrow(draw, (1495, 410), (1495, 540))
    for left, right in [(boxes[4], boxes[5]), (boxes[5], boxes[6]), (boxes[6], boxes[7])]:
        arrow(draw, (left[0][0] - 15, (left[0][1] + left[0][3]) / 2), (right[0][2] + 15, (right[0][1] + right[0][3]) / 2))
    draw.text((80, 820), "防泄漏检查：RecordID、结局、住院时长和生存时间不进入模型；SAPS-I/SOFA只作为独立参照。", font=font(22), fill=MUTED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def model_intuition(destination: Path) -> None:
    image = Image.new("RGB", (1800, 1040), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), "两个基线模型的数学直观", font=font(42, True), fill=INK)
    draw.text((70, 100), "两者都输出死亡概率，但形成风险分数的方式不同。", font=font(23), fill=MUTED)
    panels = [(70, 165, 865, 950), (935, 165, 1730, 950)]
    for panel in panels:
        draw.rounded_rectangle(panel, radius=20, fill=WHITE)

    # Logistic regression panel.
    draw.text((115, 205), "逻辑回归：所有特征形成一个加权总分", font=font(30, True), fill=INK)
    draw.text((145, 270), "z = b0 + b1*x1 + ... + bp*xp", font=font(29), fill=BLUE)
    draw.text((145, 320), "p(死亡) = 1 / (1 + exp(−z))", font=font(29), fill=BLUE)
    x0, y0, x1, y1 = 150, 430, 790, 745
    draw.line((x0, y1, x1, y1), fill=INK, width=2)
    draw.line((x0, y0, x0, y1), fill=INK, width=2)
    for p in [0, 0.5, 1]:
        py = y1 - (y1 - y0) * p
        draw.line((x0 - 8, py, x1, py), fill=GRID if p == 0.5 else INK, width=1)
        draw.text((105, py - 12), f"{p:.1f}", font=font(17), fill=MUTED)
    points = []
    for z in np.linspace(-6, 6, 160):
        probability = 1 / (1 + np.exp(-z))
        px = x0 + (z + 6) / 12 * (x1 - x0)
        py = y1 - probability * (y1 - y0)
        points.append((px, py))
    draw.line(points, fill=BLUE, width=5)
    center(draw, ((x0 + x1) / 2, y1 + 38), "加权风险分数 z", font(20))
    draw.text((105, 392), "概率", font=font(19), fill=INK)
    draw.text((115, 805), "直观理解：每个特征把风险分数向上或向下推，", font=font(22), fill=INK)
    draw.text((115, 842), "最后由 S 形函数压缩为 0–1 概率。关系主要是加性的。", font=font(22), fill=INK)

    # Histogram gradient boosting panel.
    draw.text((980, 205), "梯度提升：许多小树逐步修正前一步的错误", font=font(30, True), fill=INK)
    draw.text((1010, 270), "F_m(x) = F_(m-1)(x) + eta*h_m(x)", font=font(29), fill=ORANGE)
    draw.text((1010, 320), "p(死亡) = sigmoid(F_M(x))", font=font(29), fill=ORANGE)
    stages = [
        (1015, 420, 1175, 545, "初始风险\nF_0"),
        (1255, 400, 1435, 565, "小树 h_1\n若 x_j<t\n走左/右分支"),
        (1510, 400, 1690, 565, "小树 h_2...h_M\n继续修正\n剩余误差"),
    ]
    for x0b, y0b, x1b, y1b, label in stages:
        draw.rounded_rectangle((x0b, y0b, x1b, y1b), radius=16, fill="#F9F0E8", outline=GRID, width=2)
        multiline_center(draw, (x0b, y0b, x1b, y1b), label, font(21), spacing=8)
    arrow(draw, (1188, 482), (1238, 482), fill=ORANGE, width=4)
    arrow(draw, (1448, 482), (1493, 482), fill=ORANGE, width=4)
    sx0, sy0, sx1, sy1 = 1015, 650, 1675, 760
    draw.line((sx0, sy1, sx1, sy1), fill=INK, width=2)
    draw.line((sx0, sy0, sx0, sy1), fill=INK, width=2)
    steps = [(sx0, 735), (1120, 735), (1120, 700), (1260, 700), (1260, 660), (1440, 660), (1440, 715), (1570, 715), (1570, 675), (sx1, 675)]
    draw.line(steps, fill=ORANGE, width=5)
    center(draw, ((sx0 + sx1) / 2, sy1 + 35), "某个特征及其与其他特征的组合", font(19))
    draw.text((980, 805), "直观理解：树先把特征空间切成许多局部区域，", font=font(22), fill=INK)
    draw.text((980, 842), "再累加每棵树的修正，因此能表示阈值、非线性和交互。", font=font(22), fill=INK)
    draw.text((115, 905), "逻辑回归更容易审计；梯度提升表达能力更强，但需要更严格的稳定性、校准和解释验证。", font=font(21), fill=MUTED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def draw_metric_panel(draw, panel, title, horizons, series, references, y_range, lower_is_better=False):
    x0, y0, x1, y1 = panel
    draw.rounded_rectangle(panel, radius=18, fill=WHITE)
    draw.text((x0 + 30, y0 + 24), title, font=font(26, True), fill=INK)
    px0, py0, px1, py1 = x0 + 85, y0 + 95, x1 - 35, y1 - 85
    ymin, ymax = y_range
    for tick in np.linspace(ymin, ymax, 5):
        py = py1 - (tick - ymin) / (ymax - ymin) * (py1 - py0)
        draw.line((px0, py, px1, py), fill=GRID, width=1)
        draw.text((x0 + 15, py - 10), f"{tick:.2f}", font=font(16), fill=MUTED)
    draw.rectangle((px0, py0, px1, py1), outline=INK, width=2)
    x_positions = [px0 + i * (px1 - px0) / (len(horizons) - 1) for i in range(len(horizons))]
    for xp, horizon in zip(x_positions, horizons):
        center(draw, (xp, py1 + 28), f"{horizon}h", font(17))
    for name, values, color in series:
        points = []
        for xp, value in zip(x_positions, values):
            py = py1 - (value - ymin) / (ymax - ymin) * (py1 - py0)
            points.append((xp, py))
        draw.line(points, fill=color, width=4)
        for (xp, py), value in zip(points, values):
            draw.ellipse((xp - 6, py - 6, xp + 6, py + 6), fill=color)
            center(draw, (xp, py - 19), f"{value:.3f}", font(15, True), color)
    for label, value, color in references:
        py = py1 - (value - ymin) / (ymax - ymin) * (py1 - py0)
        draw.line((px0, py, px1, py), fill=color, width=2)
        draw.text((px1 - 190, py + 5), f"{label} {value:.3f}", font=font(14), fill=color)
    center(draw, ((px0 + px1) / 2, y1 - 28), "预测时间窗", font(18), MUTED)


def performance_figure(metrics: pd.DataFrame, destination: Path) -> None:
    test = metrics[metrics["split"].eq("test")].copy()
    horizons = [6, 12, 24, 48]

    def values(model: str, metric: str):
        return [
            float(test[(test["horizon_hours"].eq(float(h))) & (test["model"].eq(model))][metric].iloc[0])
            for h in horizons
        ]

    static = test[test["experiment"].eq("static_logistic")].iloc[0]
    scores = test[test["experiment"].eq("scores_reference_logistic")].iloc[0]
    image = Image.new("RGB", (1800, 1080), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "不同预测时间窗的测试集表现", font=font(42, True), fill=INK)
    draw.text((70, 102), "同一测试集、同一患者划分；AUPRC基线约等于死亡率0.142，Brier越低越好。", font=font(23), fill=MUTED)
    panels = [(60, 180, 590, 965), (635, 180, 1165, 965), (1210, 180, 1740, 965)]
    draw_metric_panel(
        draw,
        panels[0],
        "AUROC（排序能力）",
        horizons,
        [("逻辑回归", values("LogisticRegression", "auroc"), BLUE), ("梯度提升", values("HistGradientBoostingClassifier", "auroc"), ORANGE)],
        [("静态", float(static.auroc), PURPLE), ("评分", float(scores.auroc), GREEN)],
        (0.62, 0.90),
    )
    draw_metric_panel(
        draw,
        panels[1],
        "AUPRC（阳性识别）",
        horizons,
        [("逻辑回归", values("LogisticRegression", "auprc"), BLUE), ("梯度提升", values("HistGradientBoostingClassifier", "auprc"), ORANGE)],
        [("静态", float(static.auprc), PURPLE), ("评分", float(scores.auprc), GREEN)],
        (0.12, 0.65),
    )
    draw_metric_panel(
        draw,
        panels[2],
        "Brier（概率误差）",
        horizons,
        [("逻辑回归", values("LogisticRegression", "brier"), BLUE), ("梯度提升", values("HistGradientBoostingClassifier", "brier"), ORANGE)],
        [("静态", float(static.brier), PURPLE), ("评分", float(scores.brier), GREEN)],
        (0.075, 0.125),
        lower_is_better=True,
    )
    legend_y = 1020
    for index, (label, color) in enumerate([("逻辑回归", BLUE), ("梯度提升", ORANGE), ("静态参照", PURPLE), ("SAPS-I+SOFA参照", GREEN)]):
        x = 430 + index * 260
        draw.line((x, legend_y, x + 38, legend_y), fill=color, width=5)
        draw.text((x + 50, legend_y - 14), label, font=font(18), fill=INK)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def best_model_diagnostics(metrics: pd.DataFrame, predictions: pd.DataFrame, destination: Path) -> None:
    row = metrics[(metrics["experiment"].eq("48h_hist_gradient_boosting")) & (metrics["split"].eq("test"))].iloc[0]
    threshold = float(row.threshold)
    test = predictions[predictions["split"].eq("test")].copy()
    y = test["In-hospital_death"].astype(int).to_numpy()
    probability = test["48h_hist_gradient_boosting_probability"].astype(float).to_numpy()
    predicted = probability >= threshold
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tp = int(((y == 1) & predicted).sum())

    order = np.argsort(probability)
    groups = np.array_split(order, 10)
    calibration = [(float(probability[g].mean()), float(y[g].mean()), len(g)) for g in groups]

    image = Image.new("RGB", (1800, 970), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "最佳基线：48小时梯度提升模型", font=font(42, True), fill=INK)
    draw.text((70, 102), f"测试集 n=1,800；验证集阈值={threshold:.3f}；AUROC={row.auroc:.3f}，AUPRC={row.auprc:.3f}，Brier={row.brier:.3f}", font=font(23), fill=MUTED)
    panels = [(70, 180, 850, 875), (925, 180, 1730, 875)]
    for panel in panels:
        draw.rounded_rectangle(panel, radius=20, fill=WHITE)

    draw.text((115, 220), "阈值对应的混淆矩阵", font=font(30, True), fill=INK)
    mx0, my0, cell = 270, 360, 220
    labels = [[("TN", tn, "实际存活，预测低风险", GREEN), ("FP", fp, "实际存活，预测高风险", ORANGE)], [("FN", fn, "实际死亡，预测低风险", RED), ("TP", tp, "实际死亡，预测高风险", BLUE)]]
    for i in range(2):
        for j in range(2):
            tag, count, note, color = labels[i][j]
            box = (mx0 + j * cell, my0 + i * cell, mx0 + (j + 1) * cell - 8, my0 + (i + 1) * cell - 8)
            draw.rounded_rectangle(box, radius=14, fill=color)
            center(draw, ((box[0] + box[2]) / 2, box[1] + 62), f"{tag}  {count}", font(31, True), WHITE)
            multiline_center(draw, (box[0] + 12, box[1] + 100, box[2] - 12, box[3] - 15), note, font(17), WHITE, spacing=5)
    center(draw, (mx0 + cell, my0 - 42), "模型预测", font(21))
    draw.text((122, my0 + cell - 15), "实际结局", font=font(21), fill=INK)
    draw.text((150, 817), f"灵敏度 {row.sensitivity:.3f}：识别 {tp}/{tp+fn} 名死亡患者", font=font(19), fill=MUTED)
    draw.text((150, 850), f"特异度 {row.specificity:.3f}：正确排除 {tn}/{tn+fp} 名存活患者", font=font(19), fill=MUTED)

    draw.text((970, 220), "测试集概率校准（按预测风险十分位）", font=font(30, True), fill=INK)
    x0, y0, x1, y1 = 1045, 330, 1655, 735
    for tick in np.linspace(0, 1, 6):
        px = x0 + tick * (x1 - x0)
        py = y1 - tick * (y1 - y0)
        draw.line((px, y0, px, y1), fill=GRID, width=1)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        center(draw, (px, y1 + 27), f"{tick:.1f}", font(16), MUTED)
        draw.text((x0 - 50, py - 10), f"{tick:.1f}", font=font(16), fill=MUTED)
    draw.rectangle((x0, y0, x1, y1), outline=INK, width=2)
    draw.line((x0, y1, x1, y0), fill=MUTED, width=3)
    points = []
    for predicted_mean, observed_rate, _ in calibration:
        px = x0 + predicted_mean * (x1 - x0)
        py = y1 - observed_rate * (y1 - y0)
        points.append((px, py))
    draw.line(points, fill=BLUE, width=4)
    for index, (px, py) in enumerate(points, start=1):
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=BLUE)
        draw.text((px + 9, py - 18), str(index), font=font(14), fill=BLUE)
    center(draw, ((x0 + x1) / 2, y1 + 65), "平均预测死亡概率", font(19))
    draw.text((957, 490), "实际\n死亡率", font=font(19), fill=INK)
    draw.text((970, 837), "对角线表示理想校准；十分位点只作初步检查，正式结论仍需置信区间和外部验证。", font=font(18), fill=MUTED)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    args = parser.parse_args()
    metrics_path = args.output_dir / "metrics.csv"
    predictions_path = args.output_dir / "data" / "predictions_and_splits.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Run Experiment 02 before generating report figures.")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    destination = args.output_dir / "report_figures"
    training_pipeline(destination / "training_pipeline.png")
    model_intuition(destination / "model_intuition.png")
    performance_figure(metrics, destination / "model_performance_by_horizon.png")
    best_model_diagnostics(metrics, predictions, destination / "best_model_diagnostics.png")
    print(f"Report figures written to {destination}")


if __name__ == "__main__":
    main()
