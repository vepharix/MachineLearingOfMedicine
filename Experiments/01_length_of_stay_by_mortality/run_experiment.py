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


def draw_score_los_scatter(
    group: pd.DataFrame,
    score: str,
    group_title: str,
    color: tuple[int, int, int],
    path: Path,
) -> None:
    plot_data = group[[score, "Length_of_stay"]].dropna()
    plot_data = plot_data[plot_data[score] >= 0]
    scores = plot_data[score].to_numpy(dtype=float)
    lengths = plot_data["Length_of_stay"].to_numpy(dtype=float)
    rng = np.random.default_rng(20260903)
    display_scores = scores + rng.normal(0, 0.08, len(scores))
    width, height = 1200, 720
    left, right, top, bottom = 110, 55, 115, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), f"{score} vs length of stay — {group_title}", fill="#172033", font=font(30, True))
    pearson = plot_data.corr(method="pearson").iloc[0, 1]
    spearman = plot_data.corr(method="spearman").iloc[0, 1]
    summary = f"n={len(lengths):,}   Pearson r={pearson:+.3f}   Spearman rho={spearman:+.3f}"
    draw.text((42, 72), summary, fill="#4b5669", font=font(17))
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline="#b9c1cd", width=2)

    x_min = min(float(scores.min()), 0.0)
    x_max = float(scores.max())
    x_pad = max((x_max - x_min) * 0.04, 0.5)
    x_min -= x_pad
    x_max += x_pad
    y_min = np.log10(2.0)
    y_max = np.log10(max(float(lengths.max()), 2.0))

    def px(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def py(value: float) -> float:
        transformed = np.log10(value)
        return top + plot_h - (transformed - y_min) / max(y_max - y_min, 1e-9) * plot_h

    for fraction in np.linspace(0, 1, 6):
        score_value = x_min + fraction * (x_max - x_min)
        x = px(score_value)
        draw.line((x, top, x, top + plot_h), fill="#e2e6ec", width=1)
        draw.text((x - 16, top + plot_h + 12), f"{score_value:.0f}", fill="#5f6879", font=font(14))
    for value in [2, 5, 10, 20, 50, 100, 300]:
        if value <= lengths.max():
            y = py(value)
            draw.line((left, y, left + plot_w, y), fill="#e2e6ec", width=1)
            draw.text((55, y - 9), str(value), fill="#5f6879", font=font(14))

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer, "RGBA")
    for score_value, length in zip(display_scores, lengths):
        x, y = px(float(score_value)), py(float(length))
        layer_draw.ellipse((x - 2.3, y - 2.3, x + 2.3, y + 2.3), fill=(*color, 85))
    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.text((width / 2 - 35, height - 52), score, fill="#273247", font=font(18, True))
    draw.text((24, 88), "Length of stay (days, log scale)", fill="#273247", font=font(15, True))
    draw.text((left + 12, top + plot_h - 28), "Small horizontal jitter added for visibility", fill="#5f6879", font=font(14))
    image.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    outcomes = pd.read_csv(SOURCE)
    outcomes.columns = outcomes.columns.str.strip()
    required = ["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "In-hospital_death"]
    data = outcomes[required].apply(pd.to_numeric, errors="coerce")
    data[["SAPS-I", "SOFA"]] = data[["SAPS-I", "SOFA"]].replace(-1, np.nan)

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

    plot_specs = [
        (survived, "SAPS-I", "survived hospitalization", (43, 112, 203), "survived_saps_i_vs_los.png"),
        (died, "SAPS-I", "died in hospital", (210, 82, 82), "death_saps_i_vs_los.png"),
        (survived, "SOFA", "survived hospitalization", (43, 112, 203), "survived_sofa_vs_los.png"),
        (died, "SOFA", "died in hospital", (210, 82, 82), "death_sofa_vs_los.png"),
    ]
    correlation_rows = []
    panels = []
    for frame, score, group_title, color, filename in plot_specs:
        pair = frame[[score, "Length_of_stay"]].dropna()
        pair = pair[pair[score] >= 0]
        correlation_rows.append({
            "group": group_title,
            "score": score,
            "n": len(pair),
            "pearson_r": pair.corr(method="pearson").iloc[0, 1],
            "spearman_rho": pair.corr(method="spearman").iloc[0, 1],
        })
        plot_path = OUTPUT_DIR / filename
        draw_score_los_scatter(frame, score, group_title, color, plot_path)
        panels.append(Image.open(plot_path))
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(OUTPUT_DIR / "score_los_correlations_by_mortality.csv", index=False)

    combined = Image.new("RGB", (2400, 1440), "#f7f8fa")
    for index, panel in enumerate(panels):
        combined.paste(panel, ((index % 2) * 1200, (index // 2) * 720))
    combined.save(OUTPUT_DIR / "saps_sofa_vs_los_by_mortality.png")

    survivor_stats = summary.iloc[0]
    death_stats = summary.iloc[1]
    report = f"""# 实验 01：SAPS-I、SOFA 与住院时间

## 目的

将 `Length_of_stay` 按 `In-hospital_death` 拆分为存活出院组和院内死亡组，分别观察 SAPS-I、SOFA 与住院时间的关系。这个实验只做描述性比较，不进行因果解释或预测建模。

## 数据处理

- 数据源：`release/outcomes.csv`。
- `Length_of_stay=-1` 或缺失的 {missing_los:,} 条记录不参与分析。
- 住院时间小于 2 天、与数据集纳入条件不一致的 {invalid_los:,} 条记录作为异常值排除。
- 两份患者级拆分数据生成在 `output/data/`，为避免重新分发原始患者级数据，不上传 GitHub；代码、汇总统计和图表上传。
- 分别以 SAPS-I、SOFA 为横轴，以住院时间为纵轴，并进一步拆分成存活出院组和院内死亡组，共四幅散点图。
- SAPS-I 和 SOFA 是整数评分，为减少点的完全重叠，图中只添加少量横向抖动；相关系数仍使用未经抖动的原始评分计算。
- 由于住院时间明显右偏，纵轴使用对数尺度。

## 结果

| 分组 | 有效样本量 | 平均住院时间 | 中位数 | 第25–75百分位 | 第90百分位 | 最大值 |
|---|---:|---:|---:|---:|---:|---:|
| 存活出院 | {int(survivor_stats['n']):,} | {survivor_stats['mean_days']:.2f}天 | {survivor_stats['median_days']:.1f}天 | {survivor_stats['p25_days']:.1f}–{survivor_stats['p75_days']:.1f}天 | {survivor_stats['p90_days']:.1f}天 | {survivor_stats['max_days']:.0f}天 |
| 院内死亡 | {int(death_stats['n']):,} | {death_stats['mean_days']:.2f}天 | {death_stats['median_days']:.1f}天 | {death_stats['p25_days']:.1f}–{death_stats['p75_days']:.1f}天 | {death_stats['p90_days']:.1f}天 | {death_stats['max_days']:.0f}天 |

### 评分与住院时间的组内相关性

| 分组 | 评分 | 有效样本量 | Pearson r | Spearman rho |
|---|---|---:|---:|---:|
| 存活出院 | SAPS-I | {int(correlations.iloc[0]['n']):,} | {correlations.iloc[0]['pearson_r']:+.3f} | {correlations.iloc[0]['spearman_rho']:+.3f} |
| 院内死亡 | SAPS-I | {int(correlations.iloc[1]['n']):,} | {correlations.iloc[1]['pearson_r']:+.3f} | {correlations.iloc[1]['spearman_rho']:+.3f} |
| 存活出院 | SOFA | {int(correlations.iloc[2]['n']):,} | {correlations.iloc[2]['pearson_r']:+.3f} | {correlations.iloc[2]['spearman_rho']:+.3f} |
| 院内死亡 | SOFA | {int(correlations.iloc[3]['n']):,} | {correlations.iloc[3]['pearson_r']:+.3f} | {correlations.iloc[3]['spearman_rho']:+.3f} |

![SAPS-I、SOFA 与住院时间散点图](output/saps_sofa_vs_los_by_mortality.png)

## 初步解释

散点图用于观察两项评分与住院时间的组内关系。这里的 `Length_of_stay` 是从进入 ICU 到整个住院结束的时间，不等同于 ICU 内停留时间。相关性不能解释为评分导致住院时间改变；后续实验还需要加入年龄、ICU 类型和前 48 小时临床状态进行控制或建模。
"""
    (EXPERIMENT_DIR / "README.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Missing LOS excluded: {missing_los}; invalid LOS excluded: {invalid_los}")


if __name__ == "__main__":
    main()
