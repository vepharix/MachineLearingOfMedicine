#!/usr/bin/env python3
"""Reproducible EDA for the ICU records in ../release.

The script is deliberately read-only with respect to the source dataset.  It
writes summaries and figures to ./output and treats every numeric -1 as the
dataset's missing-value sentinel.
"""

from __future__ import annotations

import csv
import math
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "release"
RECORD_DIR = DATA_DIR / "icu_records"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "Distribution"

STATIC_PARAMS = {"RecordID", "Age", "Gender", "Height", "ICUType"}
OUTCOME_COLUMNS = ["SAPS-I", "SOFA", "Length_of_stay", "Survival"]
KEY_VARIABLES = ["GCS", "HR", "MAP", "RespRate", "Temp", "BUN", "Creatinine", "Lactate", "Platelets", "Urine"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def save_horizontal_bars(path: Path, title: str, items: list[tuple[str, float]], value_suffix: str = "") -> None:
    width = 1280
    left, right, top, row_h = 260, 90, 100, 34
    height = top + max(1, len(items)) * row_h + 70
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), title, fill="#172033", font=font(28, True))
    max_value = max((v for _, v in items), default=1.0) or 1.0
    bar_width = width - left - right
    for index, (label, value) in enumerate(items):
        y = top + index * row_h
        draw.text((40, y + 3), label, fill="#273247", font=font(16))
        draw.rounded_rectangle((left, y, left + bar_width, y + 21), radius=5, fill="#e6eaf0")
        filled = bar_width * max(0.0, value) / max_value
        draw.rounded_rectangle((left, y, left + filled, y + 21), radius=5, fill="#3977d6")
        draw.text((left + filled + 8, y + 1), f"{value:.1f}{value_suffix}", fill="#172033", font=font(15, True))
    image.save(path)


def save_effect_bars(path: Path, title: str, items: list[tuple[str, float]]) -> None:
    width = 1280
    left, right, top, row_h = 220, 80, 105, 38
    height = top + max(1, len(items)) * row_h + 70
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), title, fill="#172033", font=font(28, True))
    chart_left, chart_right = left, width - right
    center = (chart_left + chart_right) / 2
    max_abs = max((abs(v) for _, v in items), default=1.0) or 1.0
    draw.line((center, top - 15, center, height - 45), fill="#758096", width=2)
    draw.text((chart_left, 70), "Lower in deaths", fill="#5f6879", font=font(15))
    draw.text((chart_right - 115, 70), "Higher in deaths", fill="#5f6879", font=font(15))
    half = (chart_right - chart_left) / 2
    for index, (label, value) in enumerate(items):
        y = top + index * row_h
        draw.text((35, y + 2), label, fill="#273247", font=font(16))
        length = half * abs(value) / max_abs
        x0, x1 = (center - length, center) if value < 0 else (center, center + length)
        color = "#2a9d8f" if value < 0 else "#d95d5d"
        draw.rounded_rectangle((x0, y, x1, y + 23), radius=4, fill=color)
        value_x = x0 - 60 if value < 0 else x1 + 8
        draw.text((value_x, y + 1), f"{value:+.2f}", fill="#172033", font=font(15, True))
    image.save(path)


def save_histogram_panels(path: Path, panels: list[tuple[str, np.ndarray]], bins: int = 25) -> None:
    width, height = 1280, 760
    image = Image.new("RGB", (width, height), "#f7f8fa")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), "Patient-level distributions", fill="#172033", font=font(28, True))
    positions = [(50, 100), (660, 100), (50, 420), (660, 420)]
    panel_w, panel_h = 550, 245
    for (title, values), (x, y) in zip(panels, positions):
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        lo, hi = np.quantile(values, [0.01, 0.99])
        clipped = values[(values >= lo) & (values <= hi)]
        counts, edges = np.histogram(clipped, bins=bins)
        max_count = max(counts.max(), 1)
        draw.text((x, y - 32), f"{title} (1st–99th pct)", fill="#273247", font=font(20, True))
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline="#ccd2dc", width=2)
        for i, count in enumerate(counts):
            bx0 = x + i * panel_w / bins + 2
            bx1 = x + (i + 1) * panel_w / bins - 2
            by1 = y + panel_h - 25
            by0 = by1 - (panel_h - 45) * count / max_count
            draw.rectangle((bx0, by0, bx1, by1), fill="#3977d6")
        draw.text((x, y + panel_h - 20), f"{lo:.1f}", fill="#5f6879", font=font(14))
        hi_text = f"{hi:.1f}"
        draw.text((x + panel_w - 50, y + panel_h - 20), hi_text, fill="#5f6879", font=font(14))
    image.save(path)


def quantile(values: array, q: float) -> float:
    return float(np.quantile(np.frombuffer(values, dtype=np.float64), q)) if values else math.nan


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outcomes = pd.read_csv(DATA_DIR / "outcomes.csv")
    outcomes.columns = outcomes.columns.str.strip()
    for column in outcomes.columns:
        outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce")
    outcome_by_id = outcomes.set_index("RecordID").to_dict("index")

    values_by_param: dict[str, array] = defaultdict(lambda: array("d"))
    patient_counts_by_param: dict[str, array] = defaultdict(lambda: array("I"))
    patient_means_by_group: dict[str, dict[int, array]] = defaultdict(lambda: {0: array("d"), 1: array("d")})
    patient_coverage_by_group: dict[str, Counter] = defaultdict(Counter)
    observed_patients = Counter()
    observation_count = Counter()
    missing_count = Counter()
    hour_count = Counter()
    static_rows: list[dict[str, float]] = []
    record_ids_seen: set[int] = set()
    blank_parameter_rows = 0
    files_with_blank_parameter: set[str] = set()

    record_paths = sorted(RECORD_DIR.glob("*.csv"))
    for path in record_paths:
        per_values: dict[str, list[float]] = defaultdict(list)
        per_last: dict[str, float] = {}
        static: dict[str, float] = {}
        record_id = int(path.stem)
        record_ids_seen.add(record_id)
        death = int(outcome_by_id[record_id]["In-hospital_death"])
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                parameter = row["Parameter"].strip()
                if not parameter:
                    blank_parameter_rows += 1
                    files_with_blank_parameter.add(path.name)
                    continue
                try:
                    value = float(row["Value"])
                except ValueError:
                    missing_count[parameter] += 1
                    continue
                if parameter in STATIC_PARAMS or (parameter == "Weight" and row["Time"] == "00:00"):
                    static[parameter] = value
                if value == -1:
                    missing_count[parameter] += 1
                    continue
                if parameter not in STATIC_PARAMS:
                    per_values[parameter].append(value)
                    per_last[parameter] = value
                    values_by_param[parameter].append(value)
                    observation_count[parameter] += 1
                    try:
                        hour_count[int(row["Time"].split(":", 1)[0])] += 1
                    except (ValueError, IndexError):
                        pass
        static["RecordID"] = float(record_id)
        static["In-hospital_death"] = float(death)
        static_rows.append(static)
        for parameter, parameter_values in per_values.items():
            observed_patients[parameter] += 1
            patient_counts_by_param[parameter].append(len(parameter_values))
            patient_means_by_group[parameter][death].append(float(np.mean(parameter_values)))
            patient_coverage_by_group[parameter][death] += 1

    static_df = pd.DataFrame(static_rows).sort_values("RecordID")
    for column in ["Age", "Gender", "Height", "ICUType", "Weight"]:
        if column in static_df:
            static_df.loc[static_df[column] == -1, column] = np.nan

    n_records = len(record_paths)
    outcome_ids = set(outcomes["RecordID"].astype(int))
    integrity = {
        "record_files": n_records,
        "outcome_rows": len(outcomes),
        "duplicate_outcome_ids": int(outcomes["RecordID"].duplicated().sum()),
        "records_without_outcome": len(record_ids_seen - outcome_ids),
        "outcomes_without_record": len(outcome_ids - record_ids_seen),
        "blank_parameter_rows": blank_parameter_rows,
        "files_with_blank_parameter": len(files_with_blank_parameter),
    }

    outcomes_missing = (outcomes[OUTCOME_COLUMNS] == -1).sum().rename("missing_count").to_frame()
    outcomes_missing["missing_rate"] = outcomes_missing["missing_count"] / len(outcomes)
    outcomes_missing.to_csv(OUTPUT_DIR / "outcomes_missingness.csv")

    anomaly_rows = []
    for row in outcomes.itertuples(index=False):
        issues = []
        if row.Survival < -1:
            issues.append("Survival is below -1")
        if row.Length_of_stay != -1 and row.Length_of_stay < 2:
            issues.append("Length_of_stay is below 2 days")
        if row.Survival != -1 and row.Survival < 2:
            issues.append("Survival is below 2 days")
        if issues:
            anomaly_rows.append({"RecordID": int(row.RecordID), "issues": "; ".join(issues)})
    outcome_anomalies = pd.DataFrame(anomaly_rows)
    outcome_anomalies.to_csv(OUTPUT_DIR / "outcome_anomalies.csv", index=False)

    outcome_clean = outcomes.copy()
    outcome_clean[OUTCOME_COLUMNS] = outcome_clean[OUTCOME_COLUMNS].replace(-1, np.nan)
    outcome_clean.loc[outcome_clean["Survival"] < 0, "Survival"] = np.nan
    outcome_describe = outcome_clean[OUTCOME_COLUMNS + ["In-hospital_death"]].describe().T
    outcome_describe.to_csv(OUTPUT_DIR / "outcomes_summary.csv")

    rows = []
    for parameter in sorted(values_by_param):
        vals = values_by_param[parameter]
        counts = np.frombuffer(patient_counts_by_param[parameter], dtype=np.uint32)
        rows.append({
            "Parameter": parameter,
            "patients_observed": observed_patients[parameter],
            "patient_coverage_rate": observed_patients[parameter] / n_records,
            "valid_observations": observation_count[parameter],
            "missing_sentinel_rows": missing_count[parameter],
            "median_observations_per_observed_patient": float(np.median(counts)),
            "p01": quantile(vals, 0.01),
            "p25": quantile(vals, 0.25),
            "median": quantile(vals, 0.50),
            "p75": quantile(vals, 0.75),
            "p99": quantile(vals, 0.99),
            "min": min(vals),
            "max": max(vals),
        })
    variable_summary = pd.DataFrame(rows).sort_values("patient_coverage_rate", ascending=False)
    variable_summary.to_csv(OUTPUT_DIR / "time_series_variable_summary.csv", index=False)

    group_rows = []
    n_by_death = outcomes["In-hospital_death"].value_counts().to_dict()
    for parameter in sorted(patient_means_by_group):
        survivor = np.frombuffer(patient_means_by_group[parameter][0], dtype=np.float64)
        death = np.frombuffer(patient_means_by_group[parameter][1], dtype=np.float64)
        pooled_var = ((len(survivor) - 1) * np.var(survivor, ddof=1) + (len(death) - 1) * np.var(death, ddof=1))
        pooled_denom = max(len(survivor) + len(death) - 2, 1)
        pooled_sd = math.sqrt(pooled_var / pooled_denom) if pooled_var > 0 else math.nan
        smd = (float(np.mean(death)) - float(np.mean(survivor))) / pooled_sd if pooled_sd and np.isfinite(pooled_sd) else math.nan
        coverage_survivor = patient_coverage_by_group[parameter][0] / n_by_death[0]
        coverage_death = patient_coverage_by_group[parameter][1] / n_by_death[1]
        group_rows.append({
            "Parameter": parameter,
            "survivor_patients": len(survivor),
            "death_patients": len(death),
            "survivor_patient_mean_median": float(np.median(survivor)),
            "death_patient_mean_median": float(np.median(death)),
            "standardized_mean_difference_death_minus_survivor": smd,
            "survivor_coverage_rate": coverage_survivor,
            "death_coverage_rate": coverage_death,
            "coverage_rate_difference_death_minus_survivor": coverage_death - coverage_survivor,
        })
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(OUTPUT_DIR / "variable_comparison_by_outcome.csv", index=False)

    static_summary = static_df[["Age", "Gender", "Height", "ICUType", "Weight", "In-hospital_death"]].describe(include="all").T
    static_summary.to_csv(OUTPUT_DIR / "static_variable_summary.csv")
    static_by_outcome = static_df.groupby("In-hospital_death")[["Age", "Height", "Weight"]].agg(["count", "mean", "median", "std"])
    static_by_outcome.to_csv(OUTPUT_DIR / "static_comparison_by_outcome.csv")

    hour_df = pd.DataFrame({"hour": range(48), "valid_observations": [hour_count[h] for h in range(48)]})
    hour_df.to_csv(OUTPUT_DIR / "observation_count_by_hour.csv", index=False)

    death_counts = outcomes["In-hospital_death"].value_counts().sort_index()
    save_horizontal_bars(
        OUTPUT_DIR / "outcome_balance.png",
        "In-hospital mortality class balance",
        [("Survived hospitalization", 100 * death_counts.get(0, 0) / len(outcomes)), ("Died in hospital", 100 * death_counts.get(1, 0) / len(outcomes))],
        "%",
    )
    coverage_items = [(row.Parameter, 100 * row.patient_coverage_rate) for row in variable_summary.itertuples()]
    save_horizontal_bars(OUTPUT_DIR / "variable_coverage.png", "Patient coverage of time-series variables", coverage_items, "%")
    top_effects = group_summary.dropna().assign(abs_smd=lambda x: x["standardized_mean_difference_death_minus_survivor"].abs()).nlargest(15, "abs_smd")
    effect_items = [(row.Parameter, row.standardized_mean_difference_death_minus_survivor) for row in top_effects.itertuples()]
    save_effect_bars(OUTPUT_DIR / "largest_group_differences.png", "Largest unadjusted differences: death minus survivor (SMD)", effect_items)
    save_histogram_panels(
        OUTPUT_DIR / "patient_distributions.png",
        [
            ("Age (years)", static_df["Age"].dropna().to_numpy()),
            ("SAPS-I", outcome_clean["SAPS-I"].dropna().to_numpy()),
            ("SOFA", outcome_clean["SOFA"].dropna().to_numpy()),
            ("Length of stay (days)", outcome_clean["Length_of_stay"].dropna().to_numpy()),
        ],
    )

    top_coverage = variable_summary.head(8)
    low_coverage = variable_summary.tail(8).sort_values("patient_coverage_rate")
    top_effect_text = top_effects.head(10)
    death_rate = outcomes["In-hospital_death"].mean()
    gender_counts = static_df["Gender"].value_counts(dropna=False)
    icu_counts = static_df["ICUType"].value_counts(dropna=False).sort_index()

    def md_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
        formats = formats or {}
        header = "| " + " | ".join(columns) + " |"
        divider = "|" + "|".join(["---"] * len(columns)) + "|"
        lines = [header, divider]
        for _, row in frame.iterrows():
            cells = []
            for column in columns:
                value = row[column]
                cells.append(formats[column].format(value) if column in formats else str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    report = f"""# ICU 数据探索性分析（EDA）

本报告由 `eda.py` 从 `release/` 原始文件生成。分析只读取原始数据；所有数值 `-1` 均按缺失处理。时序指标的组间比较基于“每位患者该指标在前 48 小时内的平均值”，属于未经混杂因素调整的描述性结果，不能解释为因果关系。

## 数据完整性

- ICU 记录文件：{integrity['record_files']:,}
- 结局记录：{integrity['outcome_rows']:,}
- 重复结局 ID：{integrity['duplicate_outcome_ids']}
- 找不到结局的 ICU 记录：{integrity['records_without_outcome']}
- 找不到 ICU 文件的结局记录：{integrity['outcomes_without_record']}
- `Parameter` 为空、无法归类并已排除的观测：{integrity['blank_parameter_rows']:,} 行，涉及 {integrity['files_with_blank_parameter']:,} 个文件
- 结局表中违反数据说明取值约束的记录：{len(outcome_anomalies):,} 条（详见 `outcome_anomalies.csv`）
- 院内死亡：{int(death_counts.get(1, 0)):,}（{death_rate:.2%}）；存活出院：{int(death_counts.get(0, 0)):,}（{1-death_rate:.2%}）

![院内死亡标签分布](output/Distribution/outcome_balance.png)

## 结局与评分字段

{md_table(outcomes_missing.reset_index().rename(columns={'index': '字段'}), ['字段', 'missing_count', 'missing_rate'], {'missing_count': '{:.0f}', 'missing_rate': '{:.2%}'})}

`In-hospital_death` 没有缺失。SAPS-I、SOFA 和住院时长存在少量缺失；`Survival=-1` 的含义是没有记录到死亡时间，与其他列的一般缺失语义需要分开理解。

此外，`RecordID=100348` 的 `Survival=-23`，另有少数 `Length_of_stay` 或 `Survival` 为 0/1，与数据说明中“住院少于 48 小时者已排除”的约束不一致。本报告在生存时间分布中排除了负数异常值，但未改写原始文件。

![患者级分布](output/Distribution/patient_distributions.png)

## 静态变量

- 年龄中位数：{static_df['Age'].median():.1f} 岁；1%–99% 分位数：{static_df['Age'].quantile(.01):.1f}–{static_df['Age'].quantile(.99):.1f} 岁。
- 女性：{int(gender_counts.get(0, 0)):,}；男性：{int(gender_counts.get(1, 0)):,}；性别缺失：{int(static_df['Gender'].isna().sum()):,}。
- ICU 类型计数：CCU={int(icu_counts.get(1, 0)):,}，CSRU={int(icu_counts.get(2, 0)):,}，MICU={int(icu_counts.get(3, 0)):,}，SICU={int(icu_counts.get(4, 0)):,}。
- 身高缺失率：{static_df['Height'].isna().mean():.2%}；入院体重缺失率：{static_df['Weight'].isna().mean():.2%}。

## 时序指标覆盖率

覆盖率表示 12,000 位患者中至少出现一次有效测量的比例。测量密集程度与临床状态、ICU 类型和医生检查决策有关，因此“是否测量”本身可能携带预测信息。

覆盖率最高的指标：

{md_table(top_coverage, ['Parameter', 'patients_observed', 'patient_coverage_rate', 'median_observations_per_observed_patient'], {'patients_observed': '{:,.0f}', 'patient_coverage_rate': '{:.2%}', 'median_observations_per_observed_patient': '{:.1f}'})}

覆盖率最低的指标：

{md_table(low_coverage, ['Parameter', 'patients_observed', 'patient_coverage_rate', 'median_observations_per_observed_patient'], {'patients_observed': '{:,.0f}', 'patient_coverage_rate': '{:.2%}', 'median_observations_per_observed_patient': '{:.1f}'})}

![时序指标覆盖率](output/Distribution/variable_coverage.png)

## 院内死亡与存活组的描述性差异

下表按标准化均值差（SMD）的绝对值排序。正值表示死亡组的患者级平均测量值更高，负值表示更低。SMD 只用于描述和筛选候选变量，并未控制年龄、ICU 类型、测量频率或病情严重程度。

{md_table(top_effect_text, ['Parameter', 'survivor_patient_mean_median', 'death_patient_mean_median', 'standardized_mean_difference_death_minus_survivor', 'coverage_rate_difference_death_minus_survivor'], {'survivor_patient_mean_median': '{:.3g}', 'death_patient_mean_median': '{:.3g}', 'standardized_mean_difference_death_minus_survivor': '{:+.3f}', 'coverage_rate_difference_death_minus_survivor': '{:+.2%}'})}

![组间标准化差异](output/Distribution/largest_group_differences.png)

## 建模前需要保留的结论

1. 标签存在明显但不极端的类别不平衡，后续评价应优先报告 AUROC、AUPRC、召回率与校准，而不是只报告准确率。
2. 数据是不规则采样的长时序，不宜直接把缺失值填成 0。建议同时保留数值、缺失指示、距上次测量时间和测量次数。
3. 极少测量的变量可能仍有强临床价值，但其缺失模式很可能带有检查选择偏倚。
4. 本报告保留原始极值并在分布图中使用 1%–99% 范围显示；正式建模前应逐项制定生理合理范围，而不是统一截断。
5. SAPS-I 和 SOFA 是已计算的综合评分。如果目标是公平比较机器学习模型与传统评分，应把它们作为基线而非普通输入特征。
6. 字段名为空的 2,403 行不能根据数值猜测其检测项目，后续特征工程应继续排除，并保留数据质量记录。
7. 原始极值中存在明显可疑值，例如 `Temp=-17.8`、`pH=735`、`WBC=12500`。此外，`MechVent` 的有效记录只有 1，没有显式的 0；不能把“未记录”直接解释为“没有机械通气”。

## 输出文件

- `time_series_variable_summary.csv`：每个时序指标的覆盖率、测量次数、缺失行和分位数。
- `variable_comparison_by_outcome.csv`：死亡组与存活组的患者级均值、覆盖率和 SMD。
- `static_variable_summary.csv`、`static_comparison_by_outcome.csv`：静态变量摘要。
- `outcomes_summary.csv`、`outcomes_missingness.csv`：结局字段摘要。
- `outcome_anomalies.csv`：违反结局字段取值约束的记录。
- `observation_count_by_hour.csv`：前 48 小时逐小时有效观测数。
"""
    (Path(__file__).resolve().parent / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"EDA complete: {n_records} records, {len(variable_summary)} time-series variables")
    print(f"Report: {Path(__file__).resolve().parent / 'REPORT.md'}")


if __name__ == "__main__":
    main()
