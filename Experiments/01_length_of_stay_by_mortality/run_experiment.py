#!/usr/bin/env python3
"""Experiment 01: split length of stay by in-hospital mortality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "release" / "outcomes.csv"
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_length_of_stay_scatter(group: pd.DataFrame, title: str, color: tuple[int, int, int], path: Path) -> None:
    lengths = np.sort(group["Length_of_stay"].to_numpy(dtype=float))
    ranks = np.arange(1, len(lengths) + 1)
    width, height = 1200, 720
    left, right, top, bottom = 110, 55, 115, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), title, fill="#172033", font=font(30, True))
    summary = f"n={len(lengths):,}   median={np.median(lengths):.1f} days   mean={np.mean(lengths):.1f} days   p90={np.quantile(lengths, .9):.1f} days"
    draw.text((42, 72), summary, fill="#4b5669", font=font(17))
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline="#b9c1cd", width=2)

    x_max = max(len(lengths), 1)
    y_min = np.log10(2.0)
    y_max = np.log10(max(float(lengths.max()), 2.0))

    def px(value: float) -> float:
        return left + (value - 1) / max(x_max - 1, 1) * plot_w

    def py(value: float) -> float:
        transformed = np.log10(value)
        return top + plot_h - (transformed - y_min) / max(y_max - y_min, 1e-9) * plot_h

    for fraction in np.linspace(0, 1, 6):
        rank = 1 + fraction * (x_max - 1)
        x = px(rank)
        draw.line((x, top, x, top + plot_h), fill="#e2e6ec", width=1)
        draw.text((x - 22, top + plot_h + 12), f"{rank:.0f}", fill="#5f6879", font=font(14))
    for value in [2, 5, 10, 20, 50, 100, 300]:
        if value <= lengths.max():
            y = py(value)
            draw.line((left, y, left + plot_w, y), fill="#e2e6ec", width=1)
            draw.text((55, y - 9), str(value), fill="#5f6879", font=font(14))

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer, "RGBA")
    for rank, length in zip(ranks, lengths):
        x, y = px(float(rank)), py(float(length))
        layer_draw.ellipse((x - 2.3, y - 2.3, x + 2.3, y + 2.3), fill=(*color, 85))
    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(image)
    median_y = py(float(np.median(lengths)))
    draw.line((left, median_y, left + plot_w, median_y), fill="#172033", width=2)
    draw.text((left + 10, median_y - 25), "median", fill="#172033", font=font(14, True))
    draw.text((width / 2 - 70, height - 52), "Rank within group", fill="#273247", font=font(18, True))
    draw.text((24, 88), "Length of stay (days, log scale)", fill="#273247", font=font(15, True))
    image.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    outcomes = pd.read_csv(SOURCE)
    outcomes.columns = outcomes.columns.str.strip()
    required = ["RecordID", "Length_of_stay", "In-hospital_death"]
    data = outcomes[required].apply(pd.to_numeric, errors="coerce")

    missing_los = int((data["Length_of_stay"] == -1).sum() + data["Length_of_stay"].isna().sum())
    invalid_los = int(((data["Length_of_stay"] >= 0) & (data["Length_of_stay"] < 2)).sum())
    valid = data[(data["Length_of_stay"] >= 2) & data["In-hospital_death"].isin([0, 1])].copy()
    survived = valid[valid["In-hospital_death"] == 0].sort_values(["Length_of_stay", "RecordID"])
    died = valid[valid["In-hospital_death"] == 1].sort_values(["Length_of_stay", "RecordID"])

    survived.to_csv(DATA_DIR / "survived_hospitalizations.csv", index=False)
    died.to_csv(DATA_DIR / "in_hospital_deaths.csv", index=False)

    summaries = []
    for label, frame in [("Survived hospitalization", survived), ("Died in hospital", died)]:
        lengths = frame["Length_of_stay"]
        summaries.append({
            "group": label,
            "n": len(frame),
            "mean_days": lengths.mean(),
            "std_days": lengths.std(),
            "min_days": lengths.min(),
            "p25_days": lengths.quantile(0.25),
            "median_days": lengths.median(),
            "p75_days": lengths.quantile(0.75),
            "p90_days": lengths.quantile(0.90),
            "max_days": lengths.max(),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT_DIR / "group_summary.csv", index=False)

    survivor_plot = OUTPUT_DIR / "survived_length_of_stay_scatter.png"
    death_plot = OUTPUT_DIR / "death_length_of_stay_scatter.png"
    draw_length_of_stay_scatter(survived, "Length of stay — survived hospitalization", (43, 112, 203), survivor_plot)
    draw_length_of_stay_scatter(died, "Length of stay — died in hospital", (210, 82, 82), death_plot)

    left = Image.open(survivor_plot)
    right = Image.open(death_plot)
    combined = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "#f7f8fa")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    combined.save(OUTPUT_DIR / "length_of_stay_by_mortality.png")

    survivor_stats = summary.iloc[0]
    death_stats = summary.iloc[1]
    report = f"""# 实验 01：按院内死亡状态拆分住院时间

## 目的

将 `Length_of_stay` 按 `In-hospital_death` 拆分为存活出院组和院内死亡组，观察两组住院时间的分布。这个实验只做描述性比较，不进行因果解释或预测建模。

## 数据处理

- 数据源：`release/outcomes.csv`。
- `Length_of_stay=-1` 或缺失的 {missing_los:,} 条记录不参与分析。
- 住院时间小于 2 天、与数据集纳入条件不一致的 {invalid_los:,} 条记录作为异常值排除。
- 两份患者级拆分数据生成在 `output/data/`，为避免重新分发原始患者级数据，不上传 GitHub；代码、汇总统计和图表上传。
- 散点图横轴是组内按住院时间排序后的样本序号，纵轴是住院时间。由于住院时间明显右偏，纵轴使用对数尺度。

## 结果

| 分组 | 有效样本量 | 平均住院时间 | 中位数 | 第25–75百分位 | 第90百分位 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 存活出院 | {int(survivor_stats['n']):,} | {survivor_stats['mean_days']:.2f}天 | {survivor_stats['median_days']:.1f}天 | {survivor_stats['p25_days']:.1f}–{survivor_stats['p75_days']:.1f}天 | {survivor_stats['p90_days']:.1f}天 | {survivor_stats['max_days']:.0f}天 |
| 院内死亡 | {int(death_stats['n']):,} | {death_stats['mean_days']:.2f}天 | {death_stats['median_days']:.1f}天 | {death_stats['p25_days']:.1f}–{death_stats['p75_days']:.1f}天 | {death_stats['p90_days']:.1f}天 | {death_stats['max_days']:.0f}天 |

![两组住院时间散点图](output/length_of_stay_by_mortality.png)

## 初步解释

两组都具有明显的右偏和长尾，少数患者的住院时间远高于大多数患者。这里的 `Length_of_stay` 是从进入 ICU 到整个住院结束的时间，不等同于 ICU 内停留时间。是否院内死亡也不是住院时间的唯一决定因素，后续实验需要加入年龄、ICU 类型、SAPS-I、SOFA 和前 48 小时临床状态进行控制或建模。
"""
    (EXPERIMENT_DIR / "README.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Missing LOS excluded: {missing_los}; invalid LOS excluded: {invalid_los}")


if __name__ == "__main__":
    main()
