#!/usr/bin/env python3
"""Experiment 05: early ICU phenotype discovery with stability checks."""

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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
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
PURPLE = "#7455A6"
RED = "#B73A3A"
PALETTE = [BLUE, ORANGE, GREEN, PURPLE, RED, "#2E8E8E", "#9B6A36", "#6675A6"]


FEATURE_SPECS = [
    ("Age", "Age"),
    ("Gender", "Gender"),
    ("InitialWeight", "InitialWeight"),
    ("GCS_min", "GCS__min"),
    ("HR_mean", "HR__mean"),
    ("Temp_mean", "Temp__mean"),
    ("BUN_mean", "BUN__mean"),
    ("Creatinine_mean", "Creatinine__mean"),
    ("HCO3_min", "HCO3__min"),
    ("HCT_mean", "HCT__mean"),
    ("Platelets_mean", "Platelets__mean"),
    ("WBC_mean", "WBC__mean"),
    ("Na_mean", "Na__mean"),
    ("K_mean", "K__mean"),
    ("Glucose_mean", "Glucose__mean"),
    ("Mg_mean", "Mg__mean"),
    ("pH_min", "pH__min"),
    ("PaO2_min", "PaO2__min"),
    ("PaCO2_max", "PaCO2__max"),
    ("FiO2_max", "FiO2__max"),
]


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


def build_phenotype_matrix(
    feature_columns: list[str], matrix_24h: np.ndarray
) -> tuple[list[str], np.ndarray]:
    """Build a compact clinical-state matrix without outcomes or care counts."""

    position = {column: index for index, column in enumerate(feature_columns)}
    missing = [source for _, source in FEATURE_SPECS if source not in position]
    extra_required = [
        "MAP__min",
        "NIMAP__min",
        "SysABP__min",
        "NISysABP__min",
        "Urine__mean",
        "Urine__count",
        "MechVent__count",
    ]
    missing.extend(source for source in extra_required if source not in position)
    if missing:
        raise ValueError(f"Feature matrix is missing required columns: {sorted(set(missing))}")

    names = [name for name, _ in FEATURE_SPECS]
    arrays = [matrix_24h[:, position[source]].astype(float) for _, source in FEATURE_SPECS]
    invasive_map = matrix_24h[:, position["MAP__min"]].astype(float).copy()
    noninvasive_map = matrix_24h[:, position["NIMAP__min"]].astype(float).copy()
    invasive_map[invasive_map < 10] = np.nan
    noninvasive_map[noninvasive_map < 10] = np.nan
    arrays.append(np.fmin(invasive_map, noninvasive_map))
    names.append("MAP_min_combined")
    invasive_sysbp = matrix_24h[:, position["SysABP__min"]].astype(float).copy()
    noninvasive_sysbp = matrix_24h[:, position["NISysABP__min"]].astype(float).copy()
    invasive_sysbp[invasive_sysbp < 20] = np.nan
    noninvasive_sysbp[noninvasive_sysbp < 20] = np.nan
    arrays.append(np.fmin(invasive_sysbp, noninvasive_sysbp))
    names.append("SysBP_min_combined")
    urine_sum = (
        matrix_24h[:, position["Urine__mean"]].astype(float)
        * matrix_24h[:, position["Urine__count"]].astype(float)
    )
    arrays.append(urine_sum)
    names.append("Urine_sum_24h")
    arrays.append((matrix_24h[:, position["MechVent__count"]] > 0).astype(float))
    names.append("MechVent_documented_24h")
    return names, np.column_stack(arrays)


def icu_type_from_matrix(feature_columns: list[str], matrix: np.ndarray) -> np.ndarray:
    positions = [feature_columns.index(f"ICUType_{value}") for value in (1, 2, 3, 4)]
    values = matrix[:, positions]
    valid = np.isfinite(values).any(axis=1)
    result = np.zeros(len(matrix), dtype=int)
    result[valid] = np.nanargmax(values[valid], axis=1) + 1
    return result


def winsorize(
    train: np.ndarray, other: np.ndarray, lower: float = 0.01, upper: float = 0.99
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low = np.nanquantile(train, lower, axis=0)
    high = np.nanquantile(train, upper, axis=0)
    return np.clip(train, low, high), np.clip(other, low, high), low, high


def stability_score(
    x: np.ndarray,
    reference_labels: np.ndarray,
    *,
    k: int,
    repeats: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state + k)
    scores = []
    for repeat in range(repeats):
        selected = rng.choice(len(x), size=max(k * 20, int(0.8 * len(x))), replace=False)
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state + 1000 * k + repeat)
        model.fit(x[selected])
        scores.append(adjusted_rand_score(reference_labels, model.predict(x)))
    return float(np.median(scores)), float(np.quantile(scores, 0.10))


def evaluate_k_values(
    train: np.ndarray,
    holdout: np.ndarray,
    *,
    k_values: list[int],
    repeats: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    rows = []
    models = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=30, random_state=random_state)
        train_labels = model.fit_predict(train)
        holdout_labels = model.predict(holdout)
        models[k] = model
        stability_median, stability_p10 = stability_score(
            train,
            train_labels,
            k=k,
            repeats=repeats,
            random_state=random_state,
        )
        train_counts = np.bincount(train_labels, minlength=k)
        holdout_silhouette = (
            float(silhouette_score(holdout, holdout_labels, sample_size=min(5000, len(holdout)), random_state=random_state))
            if np.unique(holdout_labels).size > 1
            else math.nan
        )
        rows.append(
            {
                "k": k,
                "train_silhouette": float(
                    silhouette_score(train, train_labels, sample_size=min(5000, len(train)), random_state=random_state)
                ),
                "holdout_silhouette": holdout_silhouette,
                "calinski_harabasz": float(calinski_harabasz_score(train, train_labels)),
                "davies_bouldin": float(davies_bouldin_score(train, train_labels)),
                "stability_ari_median": stability_median,
                "stability_ari_p10": stability_p10,
                "minimum_train_cluster_fraction": float(train_counts.min() / len(train)),
            }
        )
    return pd.DataFrame(rows), models


def choose_k(metrics: pd.DataFrame) -> int:
    eligible = metrics[
        (metrics["stability_ari_median"] >= 0.75)
        & (metrics["minimum_train_cluster_fraction"] >= 0.05)
    ]
    if eligible.empty:
        eligible = metrics[metrics["minimum_train_cluster_fraction"] >= 0.05]
    if eligible.empty:
        eligible = metrics
    return int(
        eligible.sort_values(
            ["holdout_silhouette", "stability_ari_median", "k"],
            ascending=[False, False, True],
        ).iloc[0]["k"]
    )


def relabel_by_pc1(labels: np.ndarray, pc_scores: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    order = sorted(np.unique(labels), key=lambda value: float(np.mean(pc_scores[labels == value, 0])))
    mapping = {int(old): new for new, old in enumerate(order, start=1)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=int), mapping


def draw_selection(metrics: pd.DataFrame, selected_k: int, destination: Path) -> None:
    image = Image.new("RGB", (1500, 850), BG)
    draw = ImageDraw.Draw(image)
    draw.text((65, 38), "聚类数选择：分离度、留出集复现与稳定性", font=font(36, True), fill=INK)
    draw.text((65, 94), "选择规则先要求稳定性和最小簇比例，再比较未参与拟合的留出集轮廓系数。", font=font(21), fill=MUTED)
    draw.rounded_rectangle((75, 155, 1425, 745), radius=18, fill=WHITE)
    x0, y0, x1, y1 = 170, 225, 1360, 665
    draw.rectangle((x0, y0, x1, y1), outline=INK, width=2)
    for tick in np.linspace(0, 1, 6):
        py = y1 - tick * (y1 - y0)
        draw.line((x0, py, x1, py), fill=GRID, width=1)
        draw.text((104, py - 11), f"{tick:.1f}", font=font(16), fill=MUTED)
    k_values = metrics["k"].astype(int).tolist()
    xpos = {k: x0 + i * (x1 - x0) / (len(k_values) - 1) for i, k in enumerate(k_values)}
    for k in k_values:
        center(draw, (xpos[k], y1 + 30), str(k), font(17))
    for column, label, color in (
        ("train_silhouette", "训练集轮廓系数", BLUE),
        ("holdout_silhouette", "留出集轮廓系数", ORANGE),
        ("stability_ari_median", "子样本稳定性 ARI", GREEN),
    ):
        points = []
        for row in metrics.itertuples(index=False):
            value = float(getattr(row, column))
            points.append((xpos[int(row.k)], y1 - value * (y1 - y0), value))
        draw.line([(x, y) for x, y, _ in points], fill=color, width=4)
        for x, y, _ in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    sx = xpos[selected_k]
    draw.rectangle((sx - 28, y0 - 10, sx + 28, y1 + 10), outline=PURPLE, width=3)
    draw.text((sx - 48, y0 - 46), f"选择 k={selected_k}", font=font(17, True), fill=PURPLE)
    for i, (_, label, color) in enumerate(
        (("a", "训练集轮廓系数", BLUE), ("b", "留出集轮廓系数", ORANGE), ("c", "稳定性 ARI", GREEN))
    ):
        lx = 280 + i * 350
        draw.line((lx, 795, lx + 40, 795), fill=color, width=5)
        draw.text((lx + 52, 779), label, font=font(18), fill=INK)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def draw_pca_scatter(
    scores: np.ndarray,
    labels: np.ndarray,
    explained: np.ndarray,
    destination: Path,
) -> None:
    image = Image.new("RGB", (1350, 1000), BG)
    draw = ImageDraw.Draw(image)
    draw.text((60, 36), "24 小时临床表型在前两个主成分上的分布", font=font(35, True), fill=INK)
    draw.text((60, 90), "二维图只用于观察重叠程度；聚类拟合使用达到 80% 累计解释方差的全部主成分。", font=font(20), fill=MUTED)
    draw.rounded_rectangle((60, 140, 1290, 910), radius=18, fill=WHITE)
    x0, y0, x1, y1 = 145, 195, 1235, 820
    xlow, xhigh = np.quantile(scores[:, 0], [0.005, 0.995])
    ylow, yhigh = np.quantile(scores[:, 1], [0.005, 0.995])
    draw.rectangle((x0, y0, x1, y1), outline=INK, width=2)
    rng = np.random.default_rng(20260906)
    selected = rng.choice(len(scores), size=min(5000, len(scores)), replace=False)
    for index in selected:
        px = x0 + np.clip((scores[index, 0] - xlow) / (xhigh - xlow), 0, 1) * (x1 - x0)
        py = y1 - np.clip((scores[index, 1] - ylow) / (yhigh - ylow), 0, 1) * (y1 - y0)
        color = PALETTE[(labels[index] - 1) % len(PALETTE)]
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    for cluster in sorted(np.unique(labels)):
        centroid = scores[labels == cluster, :2].mean(axis=0)
        px = x0 + np.clip((centroid[0] - xlow) / (xhigh - xlow), 0, 1) * (x1 - x0)
        py = y1 - np.clip((centroid[1] - ylow) / (yhigh - ylow), 0, 1) * (y1 - y0)
        color = PALETTE[(cluster - 1) % len(PALETTE)]
        draw.ellipse((px - 14, py - 14, px + 14, py + 14), fill=WHITE, outline=color, width=5)
        center(draw, (px, py), str(cluster), font(17, True), color)
    center(draw, ((x0 + x1) / 2, y1 + 48), f"PC1（解释 {explained[0]:.1%} 方差）", font(19))
    draw.text((72, 475), f"PC2\n({explained[1]:.1%})", font=font(17), fill=INK, spacing=4)
    for i, cluster in enumerate(sorted(np.unique(labels))):
        lx = 310 + i * 185
        color = PALETTE[(cluster - 1) % len(PALETTE)]
        draw.ellipse((lx, 947, lx + 18, 965), fill=color)
        draw.text((lx + 28, 940), f"簇 {cluster}", font=font(17), fill=INK)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def color_for_heatmap(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value / 1.5, -1, 1))
    neutral = np.array([247, 248, 250], dtype=float)
    target = np.array([36, 116, 181] if value < 0 else [183, 58, 58], dtype=float)
    color = neutral * (1 - abs(value)) + target * abs(value)
    return tuple(color.astype(int))


def draw_profile_heatmap(profile: pd.DataFrame, destination: Path) -> None:
    clusters = profile.columns.tolist()
    features = profile.index.tolist()
    row_height = 31
    image = Image.new("RGB", (1150, 210 + row_height * len(features)), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), "各表型的标准化临床特征均值", font=font(34, True), fill=INK)
    draw.text((55, 81), "蓝色低于总体平均，红色高于总体平均；颜色限制在 ±1.5 个标准差。", font=font(19), fill=MUTED)
    left, top, cell_width = 330, 145, 155
    for j, cluster in enumerate(clusters):
        center(draw, (left + j * cell_width + cell_width / 2, top - 22), f"簇 {cluster}", font(18, True))
    for i, feature in enumerate(features):
        y0 = top + i * row_height
        draw.text((55, y0 + 5), feature, font=font(15), fill=INK)
        for j, cluster in enumerate(clusters):
            value = float(profile.loc[feature, cluster])
            x0 = left + j * cell_width
            draw.rectangle((x0, y0, x0 + cell_width - 3, y0 + row_height - 3), fill=color_for_heatmap(value))
            text_fill = WHITE if abs(value) > 0.85 else INK
            center(draw, (x0 + cell_width / 2, y0 + row_height / 2), f"{value:+.2f}", font(14, True), text_fill)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def markdown_k_table(metrics: pd.DataFrame) -> str:
    lines = [
        "| k | 训练轮廓系数 | 留出集轮廓系数 | 稳定性 ARI 中位数 | 最小簇比例 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.k} | {row.train_silhouette:.3f} | {row.holdout_silhouette:.3f} | "
            f"{row.stability_ari_median:.3f} | {row.minimum_train_cluster_fraction:.1%} |"
        )
    return "\n".join(lines)


def markdown_cluster_table(characteristics: pd.DataFrame) -> str:
    lines = [
        "| 簇 | 样本数 | 占比 | 院内死亡率 | 住院时长中位数（IQR，天） | 平均特征缺失率 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in characteristics.itertuples(index=False):
        lines.append(
            f"| {row.cluster} | {row.n:,} | {row.fraction:.1%} | {row.in_hospital_death_rate:.1%} | "
            f"{row.los_median_days:.1f}（{row.los_q25_days:.1f}–{row.los_q75_days:.1f}） | "
            f"{row.mean_feature_missing_rate:.1%} |"
        )
    return "\n".join(lines)


def profile_summary(profile: pd.DataFrame) -> str:
    sentences = []
    for cluster in profile.columns:
        values = profile[cluster].sort_values()
        lower = "、".join(f"`{name}`" for name in values.head(3).index)
        higher = "、".join(f"`{name}`" for name in values.tail(3).index[::-1])
        sentences.append(f"簇 {cluster} 相对较高的特征为 {higher}，相对较低的特征为 {lower}")
    return "；".join(sentences) + "。"


def write_report(
    *,
    selection_metrics: pd.DataFrame,
    characteristics: pd.DataFrame,
    standardized_profiles: pd.DataFrame,
    metadata: dict[str, object],
    path: Path,
) -> None:
    report = rf"""# 实验 05：24 小时患者表型聚类

## 研究问题与变量选择

本实验尝试回答：在不使用任何结局标签的情况下，患者入 ICU 后 24 小时的临床状态能否形成相对稳定的亚群。输入由年龄、性别、初始体重，以及神经状态、生命体征、血压、肾功能、血常规、电解质、血气、机械通气和记录尿量等共 {metadata['phenotype_feature_count']} 个特征组成。有创和无创血压先分别清除低于宽松技术合理性下限的近零值（MAP 10 mmHg、收缩压 20 mmHg），再取可用通道中的较低值。机械通气原始变量在本数据中只出现数值 1，因此这里把它解释为“24 小时内是否记录到机械通气”，不能据此确认未记录者一定没有接受通气。SAPS-I、SOFA、院内死亡、生存时间和住院时长均未参与预处理、PCA、聚类或簇数选择；其余检验次数和距上次测量时间也被排除，以减少按医疗流程而非生理状态分组的风险。

## 方法

全部 {metadata['record_count']:,} 次住院先按 ICU 类型分层，划为 70% 聚类训练集和 30% 留出集。1% 和 99% 分位截尾、中位数填补与标准化都只从训练集估计。PCA 把相关变量旋转成互相正交的主成分，本次保留 {metadata['pca_components']} 个主成分，累计解释 {metadata['pca_explained_variance']:.1%} 的训练集方差。K-means 随后寻找 k 个中心，使每位患者到所属中心的平方距离之和尽量小，因此这里的“同一簇”表示标准化临床状态更接近，并不自动等同于一种疾病。

更直观地写，标准化先把不同单位的变量放到相近尺度；PCA 寻找能保留尽可能多总体变异的正交方向；K-means 最小化 $\sum_i\|z_i-\mu_{{c_i}}\|^2$，其中 $z_i$ 是患者的主成分坐标，$\mu_{{c_i}}$ 是所属簇中心。平方距离使极端值影响较大，因此截尾和仅用训练集估计预处理参数是聚类流程的一部分，而不是事后修饰。

候选 k 为 2–8。每个 k 都计算训练集和留出集轮廓系数，并在训练集上重复 {metadata['stability_repeats']} 次 80% 子样本拟合，用调整兰德指数（ARI）衡量分组能否复现。选择规则要求稳定性 ARI 中位数至少 0.75、最小簇至少占训练集 5%，再取留出集轮廓系数最高者；最终选择 k={metadata['selected_k']}。

{markdown_k_table(selection_metrics)}

最终 k={metadata['selected_k']} 的留出集轮廓系数为 {selection_metrics.loc[selection_metrics['k'] == metadata['selected_k'], 'holdout_silhouette'].iloc[0]:.3f}，稳定性 ARI 中位数为 {selection_metrics.loc[selection_metrics['k'] == metadata['selected_k'], 'stability_ari_median'].iloc[0]:.3f}。前者显示各簇仍有较多重叠，后者说明这种粗粒度划分在重复抽样下较容易复现；因此结果更适合解释为连续临床状态上的若干组，而不是边界清楚的疾病实体。

![聚类数选择](output/report_figures/k_selection.png)

## 簇后描述

下面的死亡率和住院时长只用于解释选定后的簇，不参与建簇或模型选择，因此不能把簇间差异解释为因果效应。簇编号按平均 PC1 从低到高重新排列，只为保持输出可读。

{markdown_cluster_table(characteristics)}

从标准化画像看，{profile_summary(standardized_profiles)}这些描述用于概括簇中心附近的相对差异，不能替代患者个体诊断。

![PCA 二维投影](output/report_figures/pca_scatter.png)

![标准化特征画像](output/report_figures/cluster_profile_heatmap.png)

聚类与 ICU 类型的标准化互信息为 {metadata['cluster_icu_type_nmi']:.3f}。数值接近 0 表示分组与 ICU 类型关联较弱，接近 1 则提示聚类可能主要重现科室分流。本实验还单独报告各簇的特征缺失率；即使缺失指示没有进入模型，系统性的测量缺失仍可能间接影响聚类。

这里尤其需要留意缺失率最高的簇。若一个簇同时表现为检验较少、未记录机械通气和较低死亡率，它可能部分代表照护与测量强度，而不完全是独立的生理表型；因此本结果保留原始簇编号和缺失率，不直接命名为“低风险型”或“健康型”。

## 解释边界与后续验证

无监督聚类没有“真实标签”，轮廓系数与 ARI 只能评价几何分离和重复抽样稳定性，不能证明簇具有临床意义。下一步应在不同填补方法、特征集合、时间窗和聚类算法下检查共识聚类，并由临床人员审阅特征画像；若能获得医院或时间信息，还应验证各簇在新中心和新时期是否仍存在。患者级簇标签保存在 Git 忽略的 `output/data/cluster_assignments.csv`，仓库中只保留汇总画像和复现代码。
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Experiments" / "shared" / "output" / "data")
    parser.add_argument("--random-state", type=int, default=20260906)
    parser.add_argument("--stability-repeats", type=int, default=20)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes = read_outcomes(args.data_dir)
    if args.max_records is not None:
        outcomes = outcomes.head(args.max_records).copy()
    record_ids = outcomes["RecordID"].to_numpy(np.int64)
    columns, matrices, quality, coverage, cache_hit = load_or_extract_features(
        data_dir=args.data_dir,
        record_ids=record_ids,
        horizons=[24],
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    feature_names, raw = build_phenotype_matrix(columns, matrices[24])
    icu_type = icu_type_from_matrix(columns, matrices[24])
    if np.any(icu_type == 0):
        raise ValueError("Every selected record must have a valid ICUType")

    all_indices = np.arange(len(raw))
    train_indices, holdout_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=args.random_state,
        stratify=icu_type,
    )
    train_clipped, holdout_clipped, winsor_low, winsor_high = winsorize(
        raw[train_indices], raw[holdout_indices]
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_clipped)
    holdout_imputed = imputer.transform(holdout_clipped)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    holdout_scaled = scaler.transform(holdout_imputed)
    pca = PCA(n_components=0.80, svd_solver="full")
    train_pc = pca.fit_transform(train_scaled)
    holdout_pc = pca.transform(holdout_scaled)

    k_values = list(range(2, min(8, len(train_indices) // 50) + 1))
    if len(k_values) < 2:
        raise ValueError("Too few records to evaluate clustering solutions")
    selection_metrics, models = evaluate_k_values(
        train_pc,
        holdout_pc,
        k_values=k_values,
        repeats=args.stability_repeats,
        random_state=args.random_state,
    )
    selected_k = choose_k(selection_metrics)
    selected_model = models[selected_k]
    labels_original = np.empty(len(raw), dtype=int)
    labels_original[train_indices] = selected_model.labels_
    labels_original[holdout_indices] = selected_model.predict(holdout_pc)
    all_pc = np.empty((len(raw), train_pc.shape[1]), dtype=float)
    all_pc[train_indices] = train_pc
    all_pc[holdout_indices] = holdout_pc
    labels, label_mapping = relabel_by_pc1(labels_original, all_pc)

    all_clipped = np.clip(raw, winsor_low, winsor_high)
    all_scaled = scaler.transform(imputer.transform(all_clipped))
    standardized_profiles = pd.DataFrame(all_scaled, columns=feature_names).assign(cluster=labels)
    profile_wide = standardized_profiles.groupby("cluster")[feature_names].mean().T

    profile_rows = []
    for cluster in sorted(np.unique(labels)):
        selected = labels == cluster
        for column_index, feature in enumerate(feature_names):
            values = raw[selected, column_index]
            observed = values[np.isfinite(values)]
            profile_rows.append(
                {
                    "cluster": cluster,
                    "feature": feature,
                    "observed_n": int(len(observed)),
                    "missing_rate": float(1 - len(observed) / selected.sum()),
                    "raw_median": float(np.median(observed)) if len(observed) else math.nan,
                    "raw_q25": float(np.quantile(observed, 0.25)) if len(observed) else math.nan,
                    "raw_q75": float(np.quantile(observed, 0.75)) if len(observed) else math.nan,
                    "standardized_mean": float(profile_wide.loc[feature, cluster]),
                }
            )
    profile_long = pd.DataFrame(profile_rows)

    characteristics_rows = []
    missing_by_patient = np.mean(~np.isfinite(raw), axis=1)
    death = outcomes["In-hospital_death"].to_numpy(float)
    los = outcomes["Length_of_stay"].to_numpy(float)
    valid_los = np.isfinite(los) & (los >= 2)
    for cluster in sorted(np.unique(labels)):
        selected = labels == cluster
        valid_death = selected & np.isin(death, [0, 1])
        cluster_los = los[selected & valid_los]
        characteristics_rows.append(
            {
                "cluster": cluster,
                "n": int(selected.sum()),
                "fraction": float(selected.mean()),
                "in_hospital_death_rate": float(np.mean(death[valid_death])),
                "death_outcome_n": int(valid_death.sum()),
                "los_n": int(len(cluster_los)),
                "los_median_days": float(np.median(cluster_los)),
                "los_q25_days": float(np.quantile(cluster_los, 0.25)),
                "los_q75_days": float(np.quantile(cluster_los, 0.75)),
                "mean_feature_missing_rate": float(np.mean(missing_by_patient[selected])),
            }
        )
    characteristics = pd.DataFrame(characteristics_rows)

    icu_rows = []
    for cluster in sorted(np.unique(labels)):
        for value in (1, 2, 3, 4):
            count = int(((labels == cluster) & (icu_type == value)).sum())
            icu_rows.append(
                {
                    "cluster": cluster,
                    "ICUType": value,
                    "n": count,
                    "within_cluster_fraction": count / int((labels == cluster).sum()),
                }
            )
    icu_distribution = pd.DataFrame(icu_rows)
    cluster_icu_nmi = float(normalized_mutual_info_score(icu_type, labels))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_output = args.output_dir / "data"
    figure_output = args.output_dir / "report_figures"
    data_output.mkdir(parents=True, exist_ok=True)
    figure_output.mkdir(parents=True, exist_ok=True)
    selection_metrics.to_csv(args.output_dir / "model_selection_metrics.csv", index=False)
    characteristics.to_csv(args.output_dir / "cluster_characteristics.csv", index=False)
    profile_long.to_csv(args.output_dir / "cluster_feature_profiles.csv", index=False)
    profile_wide.to_csv(args.output_dir / "cluster_standardized_profiles.csv")
    icu_distribution.to_csv(args.output_dir / "icu_type_distribution.csv", index=False)
    pd.DataFrame(
        {
            "feature": feature_names,
            "missing_rate": np.mean(~np.isfinite(raw), axis=0),
            "winsor_lower_1pct": winsor_low,
            "winsor_upper_99pct": winsor_high,
        }
    ).to_csv(args.output_dir / "feature_preprocessing_summary.csv", index=False)
    pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    ).to_csv(args.output_dir / "pca_loadings.csv")
    pd.DataFrame(
        {
            "component": np.arange(1, pca.n_components_ + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(args.output_dir / "pca_explained_variance.csv", index=False)
    split = np.full(len(raw), "holdout", dtype=object)
    split[train_indices] = "train"
    pd.DataFrame(
        {
            "RecordID": record_ids,
            "split": split,
            "cluster": labels,
            "ICUType": icu_type,
        }
    ).to_csv(data_output / "cluster_assignments.csv", index=False)

    draw_selection(selection_metrics, selected_k, figure_output / "k_selection.png")
    draw_pca_scatter(all_pc, labels, pca.explained_variance_ratio_, figure_output / "pca_scatter.png")
    draw_profile_heatmap(profile_wide, figure_output / "cluster_profile_heatmap.png")

    metadata: dict[str, object] = {
        "record_count": int(len(outcomes)),
        "horizon_hours": 24,
        "phenotype_feature_count": len(feature_names),
        "phenotype_features": feature_names,
        "excluded_from_clustering": [
            "RecordID",
            "SAPS-I",
            "SOFA",
            "Length_of_stay",
            "Survival",
            "In-hospital_death",
            "measurement counts except urine sum and documented ventilation construction",
            "hours since last measurement",
        ],
        "split_sizes": {"train": int(len(train_indices)), "holdout": int(len(holdout_indices))},
        "winsorization": "training 1st and 99th percentiles",
        "blood_pressure_cleaning": {
            "MAP_source_minimum_mmHg": 10,
            "systolic_BP_source_minimum_mmHg": 20,
            "rule": "values below the technical plausibility floor are set missing before invasive/non-invasive combination",
        },
        "imputation": "training median without missingness indicators",
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "k_candidates": k_values,
        "selected_k": selected_k,
        "selection_rule": "ARI median >= 0.75 and minimum cluster fraction >= 0.05, then maximum holdout silhouette",
        "stability_repeats": args.stability_repeats,
        "cluster_icu_type_nmi": cluster_icu_nmi,
        "label_mapping_original_to_reported": label_mapping,
        "random_state": args.random_state,
        "feature_cache_hit": cache_hit,
        "scikit_learn_version": sklearn.__version__,
        "quality_counters": quality,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(
        selection_metrics=selection_metrics,
        characteristics=characteristics,
        standardized_profiles=profile_wide,
        metadata=metadata,
        path=EXPERIMENT_DIR / "README.md",
    )
    print(selection_metrics.to_string(index=False))
    print(characteristics.to_string(index=False))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
