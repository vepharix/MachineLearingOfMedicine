#!/usr/bin/env python3
"""Course-aligned tuning, learning-curve and interpretation add-on for Experiment 03.

The fixed test set is evaluated once, after model family and hyperparameters have
been selected with training-fold CV and the validation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from Experiments.shared.icu_feature_cache import (  # noqa: E402
    default_data_dir,
    load_or_extract_features,
    read_outcomes,
)
import run_experiment as primary_experiment  # noqa: E402


BG, WHITE, INK, MUTED, GRID = "#F7F8FA", "#FFFFFF", "#17212B", "#5B6773", "#D9DEE5"
BLUE, ORANGE, GREEN = "#2474B5", "#D97824", "#37805B"


def font(size: int, bold: bool = False):
    paths = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fixed_splits(y: np.ndarray, random_state: int) -> dict[str, np.ndarray]:
    indices = np.arange(len(y))
    train, remainder = train_test_split(
        indices, test_size=0.30, random_state=random_state, stratify=y
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=random_state, stratify=y[remainder]
    )
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(y, probability)
    finite = np.isfinite(thresholds)
    score = true_positive[finite] - false_positive[finite]
    return float(thresholds[finite][int(np.argmax(score))])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": threshold,
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "mcc": float(matthews_corrcoef(y, prediction)),
    }


def candidate_searches(random_state: int) -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=random_state)),
        ]
    )
    boosting = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    min_samples_leaf=30,
                    early_stopping=True,
                    random_state=random_state,
                ),
            ),
        ]
    )
    common = dict(scoring="average_precision", cv=cv, n_jobs=-1, return_train_score=True)
    return {
        "logistic": GridSearchCV(
            logistic,
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
            **common,
        ),
        "hist_gradient_boosting": GridSearchCV(
            boosting,
            {
                "model__learning_rate": [0.03, 0.06],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            },
            **common,
        ),
    }


def draw_learning_curve(frame: pd.DataFrame, destination: Path) -> None:
    image = Image.new("RGB", (1300, 800), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "死亡预测学习曲线（训练集内部五折）", font=font(34, True), fill=INK)
    draw.text((55, 85), "AUPRC 越高越好；训练与交叉验证曲线的间距反映过拟合。", font=font(20), fill=MUTED)
    draw.rounded_rectangle((55, 135, 1245, 735), radius=18, fill=WHITE)
    x0, y0, x1, y1 = 145, 190, 1190, 650
    ymin = max(0.0, float(frame[["train_mean", "cv_mean"]].min().min()) - 0.05)
    ymax = min(1.0, float(frame[["train_mean", "cv_mean"]].max().max()) + 0.05)
    for value in np.linspace(ymin, ymax, 6):
        py = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
        draw.line((x0, py, x1, py), fill=GRID)
        draw.text((70, py - 10), f"{value:.2f}", font=font(16), fill=MUTED)
    sizes = frame["train_size"].to_numpy(float)
    for column, color, label in (("train_mean", BLUE, "训练"), ("cv_mean", ORANGE, "交叉验证")):
        points = []
        for size, value in zip(sizes, frame[column]):
            px = x0 + (size - sizes.min()) / (sizes.max() - sizes.min()) * (x1 - x0)
            py = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
            points.append((px, py))
        draw.line(points, fill=color, width=4)
        for px, py in points:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        draw.text((900, 160 if column == "train_mean" else 190), label, font=font(18), fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def draw_importance(frame: pd.DataFrame, destination: Path) -> None:
    top = frame.head(15).sort_values("importance_mean")
    image = Image.new("RGB", (1400, 900), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "验证集置换重要度", font=font(34, True), fill=INK)
    draw.text((55, 85), "逐列打乱后 AUPRC 的平均下降；表示预测关联，不表示因果作用。", font=font(20), fill=MUTED)
    draw.rounded_rectangle((45, 135, 1355, 840), radius=18, fill=WHITE)
    maximum = max(float(top["importance_mean"].max()), 1e-6)
    for row_number, row in enumerate(top.itertuples(index=False)):
        y = 790 - row_number * 41
        width = max(1, int(max(row.importance_mean, 0) / maximum * 650))
        draw.text((70, y - 11), str(row.feature)[:42], font=font(16), fill=INK)
        draw.rectangle((600, y - 10, 600 + width, y + 12), fill=GREEN)
        draw.text((610 + width, y - 11), f"{row.importance_mean:.4f}", font=font(15), fill=MUTED)
    image.save(destination)


def draw_pdp(frame: pd.DataFrame, destination: Path) -> None:
    features = list(frame["feature"].drop_duplicates())
    image = Image.new("RGB", (1500, 520 * len(features)), BG)
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), "部分依赖图（验证集范围）", font=font(34, True), fill=INK)
    draw.text((50, 78), "其余变量按验证集分布平均，仅用于观察模型形状。", font=font(20), fill=MUTED)
    colors = [BLUE, ORANGE, GREEN]
    for index, feature in enumerate(features):
        subset = frame[frame["feature"] == feature].sort_values("feature_value")
        x0, x1 = 150, 1400
        y0, y1 = 150 + index * 500, 520 + index * 500
        draw.rounded_rectangle((60, y0 - 45, 1440, y1 + 50), radius=16, fill=WHITE)
        draw.text((85, y0 - 28), feature, font=font(23, True), fill=INK)
        xs = subset["feature_value"].to_numpy(float)
        ys = subset["average_probability"].to_numpy(float)
        if np.ptp(xs) == 0 or np.ptp(ys) == 0:
            continue
        points = [
            (
                x0 + (x - xs.min()) / np.ptp(xs) * (x1 - x0),
                y1 - (y - ys.min()) / np.ptp(ys) * (y1 - y0),
            )
            for x, y in zip(xs, ys)
        ]
        draw.line(points, fill=colors[index % len(colors)], width=4)
        draw.text((85, y1 + 10), f"输入范围 {xs.min():.2f}—{xs.max():.2f}", font=font(16), fill=MUTED)
        draw.text((1050, y1 + 10), f"平均概率 {ys.min():.3f}—{ys.max():.3f}", font=font(16), fill=MUTED)
    image.save(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Experiments/shared/output/data")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output/course_extension")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes = read_outcomes(args.data_dir)
    if args.max_records is not None:
        outcomes = outcomes.head(args.max_records).copy()
    record_ids = outcomes["RecordID"].to_numpy(np.int64)
    columns, matrices, _, _, cache_hit = load_or_extract_features(
        data_dir=args.data_dir,
        record_ids=record_ids,
        horizons=[48],
        cache_dir=args.cache_dir,
        use_cache=True,
    )
    x = matrices[48]
    y = outcomes["In-hospital_death"].to_numpy(int)
    splits = fixed_splits(y, args.random_state)
    output = args.output_dir
    figures = output / "report_figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    cv_frames, validation_rows, fitted = [], [], {}
    for name, search in candidate_searches(args.random_state).items():
        search.fit(x[splits["train"]], y[splits["train"]])
        fitted[name] = search.best_estimator_
        frame = pd.DataFrame(search.cv_results_)
        frame.insert(0, "model", name)
        cv_frames.append(frame)
        probability = search.best_estimator_.predict_proba(x[splits["validation"]])[:, 1]
        threshold = choose_threshold(y[splits["validation"]], probability)
        row = {"model": name, "best_parameters": json.dumps(search.best_params_, ensure_ascii=False)}
        row.update(metrics(y[splits["validation"]], probability, threshold))
        validation_rows.append(row)

    cv_results = pd.concat(cv_frames, ignore_index=True)
    validation = pd.DataFrame(validation_rows).sort_values(
        ["auprc", "brier", "model"], ascending=[False, True, True]
    )
    selected_name = str(validation.iloc[0]["model"])
    selected = fitted[selected_name]
    selected_validation_probability = selected.predict_proba(x[splits["validation"]])[:, 1]
    threshold = choose_threshold(y[splits["validation"]], selected_validation_probability)

    development = np.sort(np.concatenate([splits["train"], splits["validation"]]))
    final_model = clone(selected).fit(x[development], y[development])
    test_probability = final_model.predict_proba(x[splits["test"]])[:, 1]
    final = pd.DataFrame(
        [{"model": selected_name, "fit_sample": "train_plus_validation", **metrics(y[splits["test"]], test_probability, threshold)}]
    )

    curve_sizes, curve_train, curve_cv = learning_curve(
        clone(selected),
        x[splits["train"]],
        y[splits["train"]],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state),
        scoring="average_precision",
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )
    curve = pd.DataFrame(
        {
            "train_size": curve_sizes,
            "train_mean": curve_train.mean(axis=1),
            "train_sd": curve_train.std(axis=1),
            "cv_mean": curve_cv.mean(axis=1),
            "cv_sd": curve_cv.std(axis=1),
        }
    )

    primary_configs = {config.name: config for config in primary_experiment.CONFIGS}
    primary_hgb = primary_experiment.build_model(
        "hist_gradient_boosting",
        primary_configs["raw_mean_indicator"],
        columns,
        args.random_state,
    ).fit(x[splits["train"]], y[splits["train"]])
    importance_result = permutation_importance(
        primary_hgb,
        x[splits["validation"]],
        y[splits["validation"]],
        scoring="average_precision",
        n_repeats=5,
        random_state=args.random_state,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": columns,
            "importance_mean": importance_result.importances_mean,
            "importance_sd": importance_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    pdp_rows = []
    # sklearn derives the PDP grid before the pipeline imputer is applied.
    # Use medians learned from the training partition so missing validation
    # values cannot turn the displayed grid into NaNs or leak information.
    training_medians = np.nanmedian(x[splits["train"]], axis=0)
    pdp_x = np.where(
        np.isfinite(x[splits["validation"]]),
        x[splits["validation"]],
        training_medians,
    )
    for feature in importance.head(3)["feature"]:
        feature_index = columns.index(feature)
        result = partial_dependence(
            primary_hgb,
            pdp_x,
            features=[feature_index],
            grid_resolution=20,
            kind="average",
        )
        for feature_value, average in zip(result["grid_values"][0], result["average"][0]):
            pdp_rows.append(
                {"feature": feature, "feature_value": float(feature_value), "average_probability": float(average)}
            )
    pdp = pd.DataFrame(pdp_rows)

    primary_logistic = primary_experiment.build_model(
        "logistic",
        primary_configs["raw_winsor_median_indicator"],
        columns,
        args.random_state,
    ).fit(x[splits["train"]], y[splits["train"]])
    logistic_feature_names = primary_logistic.named_steps["imputer"].get_feature_names_out(columns)
    logistic_coefficients = primary_logistic.named_steps["model"].coef_[0]
    logistic_interpretation = pd.DataFrame(
        {
            "feature": logistic_feature_names,
            "standardized_coefficient": logistic_coefficients,
            "odds_ratio_per_1sd": np.exp(logistic_coefficients),
            "absolute_coefficient": np.abs(logistic_coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    cv_results.to_csv(output / "tuning_cv_results.csv", index=False)
    validation.to_csv(output / "validation_model_comparison.csv", index=False)
    final.to_csv(output / "final_test_metrics.csv", index=False)
    curve.to_csv(output / "learning_curve.csv", index=False)
    importance.to_csv(output / "permutation_importance.csv", index=False)
    pdp.to_csv(output / "partial_dependence.csv", index=False)
    logistic_interpretation.to_csv(output / "logistic_coefficients_odds_ratios.csv", index=False)
    draw_learning_curve(curve, figures / "learning_curve.png")
    draw_importance(importance, figures / "permutation_importance.png")
    draw_pdp(pdp, figures / "partial_dependence.png")
    metadata = {
        "horizon_hours": 48,
        "split_sizes": {name: int(len(index)) for name, index in splits.items()},
        "selection": "5-fold training CV AUPRC tunes each family; validation AUPRC selects family; test opened once",
        "selected_model": selected_name,
        "selected_parameters": selected.get_params(deep=True),
        "interpretation_models": {
            "permutation_importance_and_partial_dependence": "primary hist_gradient_boosting with raw_mean_indicator",
            "coefficient_odds_ratio_table": "primary logistic with raw_winsor_median_indicator",
        },
        "permutation_importance": {
            "evaluation_split": "validation",
            "sample_fraction": 1.0,
            "repeats": 5,
            "scoring": "average_precision",
        },
        "class_imbalance": "class_weight None versus balanced is tuned for logistic regression",
        "cache_hit": cache_hit,
        "random_state": args.random_state,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(validation.to_string(index=False))
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
