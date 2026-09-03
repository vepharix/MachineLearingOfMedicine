#!/usr/bin/env python3
"""Correlation matrices and scatter plots for SAPS-I and SOFA outcomes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "release" / "outcomes.csv"
OUTPUT = Path(__file__).resolve().parent / "output" / "correlation"
VARIABLES = ["SAPS-I", "SOFA", "Survival", "Length_of_stay", "In-hospital_death"]
DISPLAY = {
    "SAPS-I": "SAPS-I",
    "SOFA": "SOFA",
    "Survival": "Survival (days)",
    "Length_of_stay": "Length of stay (days)",
    "In-hospital_death": "In-hospital death",
}
MATRIX_DISPLAY = {
    "SAPS-I": "SAPS-I",
    "SOFA": "SOFA",
    "Survival": "Survival",
    "Length_of_stay": "LOS",
    "In-hospital_death": "Hospital death",
}


def font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def color_for_correlation(value: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        return (225, 228, 233)
    neutral = np.array([247, 248, 250], dtype=float)
    target = np.array([214, 78, 78] if value >= 0 else [49, 120, 190], dtype=float)
    return tuple((neutral * (1 - abs(value)) + target * abs(value)).astype(int))


def draw_heatmap(matrix: pd.DataFrame, title: str, path: Path) -> None:
    labels = [MATRIX_DISPLAY[c] for c in matrix.columns]
    size, left, top, cell = 1180, 285, 130, 150
    image = Image.new("RGB", (size, 980), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text((35, 28), title, fill="#172033", font=font(30, True))
    for idx, label in enumerate(labels):
        draw.text((left + idx * cell + 10, 88), label, fill="#273247", font=font(15, True))
        draw.text((35, top + idx * cell + 60), label, fill="#273247", font=font(17, True))
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = float(matrix.iloc[row, col])
            x0, y0 = left + col * cell, top + row * cell
            draw.rounded_rectangle((x0 + 3, y0 + 3, x0 + cell - 3, y0 + cell - 3), radius=8, fill=color_for_correlation(value))
            text = "NA" if not np.isfinite(value) else f"{value:.3f}"
            text_color = "white" if np.isfinite(value) and abs(value) >= 0.58 else "#172033"
            box = draw.textbbox((0, 0), text, font=font(23, True))
            draw.text((x0 + (cell - (box[2] - box[0])) / 2, y0 + 57), text, fill=text_color, font=font(23, True))
    draw.text((left, 910), "Blue = negative correlation    Red = positive correlation", fill="#5f6879", font=font(16))
    image.save(path)


def blend_scatter_point(layer: Image.Image, x: int, y: int, color: tuple[int, int, int], radius: int = 2) -> None:
    draw = ImageDraw.Draw(layer, "RGBA")
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 45))


def draw_scatter(df: pd.DataFrame, x_col: str, y_col: str, path: Path) -> None:
    pair = df[[x_col, y_col]].dropna()
    x_raw = pair[x_col].to_numpy(dtype=float)
    y_raw = pair[y_col].to_numpy(dtype=float)
    binary = y_col == "In-hospital_death"
    log_y = y_col in {"Survival", "Length_of_stay"}
    rng = np.random.default_rng(20260903)
    y_plot = y_raw + rng.normal(0, 0.055, len(y_raw)) if binary else (np.log10(y_raw) if log_y else y_raw)

    width, height = 960, 700
    left, right, top, bottom = 105, 45, 90, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    title = f"{DISPLAY[x_col]} vs {DISPLAY[y_col]}"
    draw.text((35, 25), title, fill="#172033", font=font(28, True))

    x_min, x_max = float(np.min(x_raw)), float(np.max(x_raw))
    y_min, y_max = float(np.min(y_plot)), float(np.max(y_plot))
    x_pad = max((x_max - x_min) * 0.04, 0.5)
    y_pad = max((y_max - y_min) * 0.06, 0.08)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def px(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def py(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    draw.rectangle((left, top, left + plot_w, top + plot_h), outline="#b9c1cd", width=2)
    for fraction in np.linspace(0, 1, 6):
        xv = x_min + fraction * (x_max - x_min)
        xp = px(xv)
        draw.line((xp, top, xp, top + plot_h), fill="#e2e6ec", width=1)
        draw.text((xp - 15, top + plot_h + 12), f"{xv:.0f}", fill="#5f6879", font=font(14))

    if binary:
        y_ticks = [(0.0, "0"), (1.0, "1")]
    elif log_y:
        low_exp = math.ceil(math.log10(float(np.min(y_raw))))
        high_exp = math.floor(math.log10(float(np.max(y_raw))))
        candidates = sorted(set([float(np.min(y_raw)), float(np.max(y_raw))] + [10.0**e for e in range(low_exp, high_exp + 1)]))
        y_ticks = [(math.log10(v), f"{v:g}") for v in candidates]
    else:
        y_ticks = [(y_min + f * (y_max - y_min), f"{y_min + f * (y_max - y_min):.1f}") for f in np.linspace(0, 1, 6)]
    for yv, label in y_ticks:
        yp = py(yv)
        draw.line((left, yp, left + plot_w, yp), fill="#e2e6ec", width=1)
        draw.text((45, yp - 9), label, fill="#5f6879", font=font(14))

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    for xv, yv in zip(x_raw, y_plot):
        blend_scatter_point(layer, int(px(xv)), int(py(yv)), (43, 112, 203), 2)
    image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
    draw = ImageDraw.Draw(image)

    pearson = pair.corr(method="pearson").iloc[0, 1]
    spearman = pair.corr(method="spearman").iloc[0, 1]
    annotation = f"n={len(pair):,}   Pearson r={pearson:+.3f}   Spearman rho={spearman:+.3f}"
    draw.rounded_rectangle((left + 12, top + 12, left + 530, top + 48), radius=7, fill="#ffffff", outline="#d4dae3")
    draw.text((left + 23, top + 20), annotation, fill="#172033", font=font(15, True))
    draw.text((left + plot_w / 2 - 35, height - 56), DISPLAY[x_col], fill="#273247", font=font(18, True))
    y_label = DISPLAY[y_col] + (" (log scale)" if log_y else "")
    draw.text((25, 63), y_label, fill="#273247", font=font(15, True))
    if binary:
        draw.text((left + 12, top + plot_h - 30), "Vertical jitter added only for visibility", fill="#5f6879", font=font(14))
    image.save(path)


def markdown_matrix(matrix: pd.DataFrame) -> str:
    labels = list(matrix.columns)
    lines = ["| | " + " | ".join(labels) + " |", "|---|" + "|".join(["---:"] * len(labels)) + "|"]
    for row in labels:
        lines.append("| " + row + " | " + " | ".join(f"{matrix.loc[row, col]:.3f}" for col in labels) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE)
    raw.columns = raw.columns.str.strip()
    data = raw[VARIABLES].apply(pd.to_numeric, errors="coerce")
    data[["SAPS-I", "SOFA", "Length_of_stay", "Survival"]] = data[["SAPS-I", "SOFA", "Length_of_stay", "Survival"]].replace(-1, np.nan)
    data.loc[data["Length_of_stay"] < 2, "Length_of_stay"] = np.nan
    data.loc[data["Survival"] < 2, "Survival"] = np.nan
    data.loc[~data["In-hospital_death"].isin([0, 1]), "In-hospital_death"] = np.nan

    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    counts = data.notna().astype(int).T.dot(data.notna().astype(int))
    pearson.to_csv(OUTPUT / "pearson_correlation_matrix.csv")
    spearman.to_csv(OUTPUT / "spearman_correlation_matrix.csv")
    counts.to_csv(OUTPUT / "pairwise_sample_counts.csv")

    long_rows = []
    for score in ["SAPS-I", "SOFA"]:
        for outcome in ["Survival", "Length_of_stay", "In-hospital_death"]:
            pair = data[[score, outcome]].dropna()
            long_rows.append({
                "score": score,
                "outcome": outcome,
                "n": len(pair),
                "pearson_r": pair.corr(method="pearson").iloc[0, 1],
                "spearman_rho": pair.corr(method="spearman").iloc[0, 1],
            })
            draw_scatter(data, score, outcome, OUTPUT / f"scatter_{score.lower().replace('-', '_')}_vs_{outcome.lower()}.png")
    results = pd.DataFrame(long_rows)
    results.to_csv(OUTPUT / "score_outcome_correlations.csv", index=False)

    draw_heatmap(pearson, "Pearson correlation matrix (pairwise complete)", OUTPUT / "pearson_correlation_matrix.png")
    draw_heatmap(spearman, "Spearman correlation matrix (pairwise complete)", OUTPUT / "spearman_correlation_matrix.png")

    panels = []
    for score in ["SAPS-I", "SOFA"]:
        for outcome in ["Survival", "Length_of_stay", "In-hospital_death"]:
            panels.append(Image.open(OUTPUT / f"scatter_{score.lower().replace('-', '_')}_vs_{outcome.lower()}.png"))
    combined = Image.new("RGB", (2880, 1400), "#f7f8fa")
    for index, panel in enumerate(panels):
        combined.paste(panel, ((index % 3) * 960, (index // 3) * 700))
    combined.save(OUTPUT / "score_outcome_scatterplots.png")

    result_lines = []
    for row in results.itertuples(index=False):
        result_lines.append(f"| {row.score} | {row.outcome} | {row.n:,} | {row.pearson_r:+.3f} | {row.spearman_rho:+.3f} |")
    report = f"""# SAPS-I、SOFA 与结局变量的相关性

数据源为 `release/outcomes.csv`。`SAPS-I=-1`、`SOFA=-1` 和 `Length_of_stay=-1` 按缺失处理；`Survival=-1` 表示没有记录死亡时间，不进入死亡时间相关性。依据数据说明，死亡时间和住院时间小于 2 天的异常值也不参与相应计算。所有相关系数均采用成对完整样本，因此不同单元格的样本量可能不同。

## Pearson 相关性矩阵

{markdown_matrix(pearson)}

![Pearson 相关性矩阵](output/correlation/pearson_correlation_matrix.png)

## Spearman 相关性矩阵

{markdown_matrix(spearman)}

![Spearman 相关性矩阵](output/correlation/spearman_correlation_matrix.png)

## 两项评分与三个结局变量

| 评分 | 结局 | 有效样本量 | Pearson r | Spearman rho |
|---|---|---:|---:|---:|
{chr(10).join(result_lines)}

![六组散点图](output/correlation/score_outcome_scatterplots.png)

死亡时间和住院时间使用对数纵轴显示，但图中标注的相关系数是在原始天数上计算的。是否死亡为二元变量，散点图仅为显示而加入了少量纵向抖动；其 Pearson 系数等价于点二列相关系数。

`Survival` 的相关性只描述“有有效死亡时间记录”的患者，不能代表全部 12,000 位患者的总体生存关系。相关性也不等于因果关系，SAPS-I 与 SOFA 之间共享部分生理测量，因此二者本身存在结构性相关。
"""
    (Path(__file__).resolve().parent / "CORRELATION_REPORT.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False))
    print(f"Outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
