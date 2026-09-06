#!/usr/bin/env python3
"""Audit outcome-field consistency and draw an aggregate diagnostic figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "Distribution"


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


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, used_font, fill):
    box = draw.textbbox((0, 0), value, font=used_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), value, font=used_font, fill=fill)


def audit(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = outcomes.copy()
    for column in ["Length_of_stay", "Survival", "In-hospital_death"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    los = frame["Length_of_stay"]
    survival = frame["Survival"]
    death = frame["In-hospital_death"]

    flags = pd.DataFrame(index=frame.index)
    flags["Length_of_stay为0或1"] = los.ge(0) & los.lt(2)
    flags["Survival为0或1"] = survival.ge(0) & survival.lt(2)
    flags["Survival小于-1"] = survival.lt(-1)
    flags["死亡标签不满足2≤Survival≤LOS"] = death.eq(1) & ~(
        survival.ge(2) & survival.le(los)
    )
    flags["存活标签不满足Survival=-1或Survival>LOS"] = death.eq(0) & ~(
        survival.eq(-1) | survival.gt(los)
    )
    hard_any = flags.any(axis=1)

    valid_deaths = frame[
        death.eq(1) & survival.ge(2) & survival.le(los)
    ][["RecordID", "Length_of_stay", "Survival", "In-hospital_death"]].copy()
    valid_deaths["LOS_minus_Survival"] = (
        valid_deaths["Length_of_stay"] - valid_deaths["Survival"]
    )

    rows = [
        ("明确违反任一官方约束（唯一记录）", int(hard_any.sum()), "互斥总数"),
        ("Survival为0或1", int(flags["Survival为0或1"].sum()), "可与其他项重叠"),
        ("Length_of_stay为0或1", int(flags["Length_of_stay为0或1"].sum()), "可与其他项重叠"),
        ("Survival小于-1", int(flags["Survival小于-1"].sum()), "可与其他项重叠"),
        (
            "死亡标签不满足2≤Survival≤LOS",
            int(flags["死亡标签不满足2≤Survival≤LOS"].sum()),
            "可与其他项重叠",
        ),
        (
            "存活标签不满足Survival=-1或Survival>LOS",
            int(flags["存活标签不满足Survival=-1或Survival>LOS"].sum()),
            "可与其他项重叠",
        ),
    ]
    for cutoff in [2, 5, 10, 30]:
        rows.append(
            (
                f"规则内死亡且LOS-Survival>{cutoff}天",
                int(valid_deaths["LOS_minus_Survival"].gt(cutoff).sum()),
                "官方规则允许，但需做敏感性分析",
            )
        )
    summary = pd.DataFrame(rows, columns=["audit_item", "record_count", "interpretation"])
    return summary, valid_deaths


def draw_figure(
    outcomes: pd.DataFrame,
    valid_deaths: pd.DataFrame,
    summary: pd.DataFrame,
    destination: Path,
) -> None:
    width, height = 1800, 980
    bg = "#F7F8FA"
    ink = "#17212B"
    muted = "#5B6773"
    grid = "#D9DEE5"
    blue = "#2474B5"
    orange = "#D97824"
    red = "#B73A3A"
    green = "#37805B"
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    draw.text((70, 40), "结局字段一致性审计", font=font(42, True), fill=ink)
    draw.text(
        (70, 100),
        "明确规则错误与规则内大差值需要分开处理；下图不把所有大差值直接判定为错误。",
        font=font(23),
        fill=muted,
    )

    # Left: LOS vs survival for in-hospital deaths.
    left = (100, 190, 880, 850)
    draw.rounded_rectangle(left, radius=18, fill="#FFFFFF")
    draw.text((140, 220), "院内死亡记录：死亡时间与住院时长", font=font(28, True), fill=ink)
    plot = (220, 300, 810, 730)
    x0, y0, x1, y1 = plot
    max_day = 150
    for value in range(0, max_day + 1, 25):
        px = x0 + (x1 - x0) * value / max_day
        py = y1 - (y1 - y0) * value / max_day
        draw.line((px, y0, px, y1), fill=grid, width=1)
        draw.line((x0, py, x1, py), fill=grid, width=1)
        text_center(draw, (px, y1 + 25), str(value), font(16), muted)
        box = draw.textbbox((0, 0), str(value), font=font(16))
        draw.text((x0 - 18 - (box[2] - box[0]), py - 9), str(value), font=font(16), fill=muted)
    draw.rectangle(plot, outline=ink, width=2)
    draw.line((x0, y1, x1, y0), fill=green, width=3)

    death_rows = outcomes[outcomes["In-hospital_death"].eq(1)].copy()
    for row in death_rows.itertuples(index=False):
        los = float(row.Length_of_stay)
        survival = float(row.Survival)
        if los < 0 or survival < 0:
            continue
        px = x0 + (x1 - x0) * min(los, max_day) / max_day
        py = y1 - (y1 - y0) * min(survival, max_day) / max_day
        gap = los - survival
        color = red if survival < 2 else (orange if gap > 10 else blue)
        radius = 4 if gap > 10 or survival < 2 else 2
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    text_center(draw, ((x0 + x1) / 2, y1 + 48), "Length_of_stay（天）", font(20), ink)
    draw.text((220, 268), "纵轴：Survival（天）", font=font(18), fill=muted)
    legend = [
        (blue, "规则内，差值≤10天"),
        (orange, "规则内，差值>10天"),
        (red, "Survival<2等硬规则问题"),
        (green, "Survival=LOS参考线"),
    ]
    for index, (color, label) in enumerate(legend):
        lx = 225 + (index % 2) * 285
        ly = 802 + (index // 2) * 34
        draw.ellipse((lx, ly, lx + 14, ly + 14), fill=color)
        draw.text((lx + 22, ly - 5), label, font=font(17), fill=ink)

    # Right: gap distribution for rule-valid in-hospital deaths.
    right = (930, 190, 1700, 850)
    draw.rounded_rectangle(right, radius=18, fill="#FFFFFF")
    draw.text((970, 220), "规则内死亡记录的 LOS−Survival 差值", font=font(28, True), fill=ink)
    gaps = valid_deaths["LOS_minus_Survival"]
    bins = [
        ("0–1天", gaps.between(0, 1).sum()),
        ("2天", gaps.eq(2).sum()),
        ("3–5天", gaps.between(3, 5).sum()),
        ("6–10天", gaps.between(6, 10).sum()),
        ("11–30天", gaps.between(11, 30).sum()),
        (">30天", gaps.gt(30).sum()),
    ]
    bx0, by0, bx1, by1 = 1020, 330, 1640, 700
    max_count = max(count for _, count in bins)
    for tick in np.linspace(0, max_count, 5):
        py = by1 - (by1 - by0) * tick / max_count
        draw.line((bx0, py, bx1, py), fill=grid, width=1)
        label = str(int(round(tick)))
        box = draw.textbbox((0, 0), label, font=font(16))
        draw.text((bx0 - 15 - (box[2] - box[0]), py - 9), label, font=font(16), fill=muted)
    draw.line((bx0, by0, bx0, by1), fill=ink, width=2)
    draw.line((bx0, by1, bx1, by1), fill=ink, width=2)
    bar_width = 70
    gap_width = (bx1 - bx0) / len(bins)
    colors = [blue, blue, blue, orange, orange, red]
    for index, ((label, count), color) in enumerate(zip(bins, colors)):
        cx = bx0 + gap_width * (index + 0.5)
        top = by1 - (by1 - by0) * count / max_count
        draw.rectangle((cx - bar_width / 2, top, cx + bar_width / 2, by1), fill=color)
        text_center(draw, (cx, top - 18), str(int(count)), font(18, True), ink)
        text_center(draw, (cx, by1 + 30), label, font(17), ink)
    text_center(draw, ((bx0 + bx1) / 2, by1 + 65), "住院时长减死亡时间", font(20), ink)
    draw.text((980, 805), f"规则内死亡记录：{len(valid_deaths):,} 例；最大差值：{int(gaps.max())} 天", font=font(19), fill=muted)

    hard_count = int(summary.loc[summary["audit_item"].str.startswith("明确违反"), "record_count"].iloc[0])
    draw.text(
        (70, 915),
        f"注：硬规则异常共 {hard_count} 个唯一记录；分项计数存在重叠。大差值符合官方标签规则，但可能反映结局日期或行政记录的不一致。",
        font=font(20),
        fill=muted,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "release")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.data_dir / "outcomes.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing {source}; pass --data-dir for the private dataset.")
    outcomes = pd.read_csv(source)
    summary, valid_deaths = audit(outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "outcome_consistency_summary.csv", index=False)
    draw_figure(
        outcomes,
        valid_deaths,
        summary,
        args.output_dir / "outcome_consistency_audit.png",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
