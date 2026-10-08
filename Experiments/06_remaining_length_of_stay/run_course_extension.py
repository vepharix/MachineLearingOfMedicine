#!/usr/bin/env python3
"""Course-aligned tuning and OLS diagnostics for remaining length of stay."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image, ImageDraw, ImageFont
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, learning_curve


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import run_experiment as base  # noqa: E402
from Experiments.shared.icu_feature_cache import (  # noqa: E402
    default_data_dir,
    load_or_extract_features,
    read_outcomes,
)


BG, WHITE, INK, MUTED, GRID = "#F7F8FA", "#FFFFFF", "#17212B", "#5B6773", "#D9DEE5"
BLUE, ORANGE, GREEN = "#2474B5", "#D97824", "#37805B"

OLS_FEATURES = [
    "Age",
    "Gender",
    "InitialWeight",
    "ICUType_2",
    "ICUType_3",
    "ICUType_4",
    "Albumin__last",
    "BUN__last",
    "Creatinine__last",
    "GCS__min",
    "HR__mean",
    "Lactate__max",
    "MAP__min",
    "Na__last",
    "Platelets__last",
    "Temp__mean",
    "Urine__mean",
    "WBC__last",
]


def font(size: int, bold: bool = False):
    paths = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def negative_mae_days(estimator, x: np.ndarray, y_log: np.ndarray) -> float:
    observed = np.expm1(y_log)
    predicted = np.maximum(np.expm1(estimator.predict(x)), 0.0)
    return -float(mean_absolute_error(observed, predicted))


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "mae_days": float(mean_absolute_error(y, prediction)),
        "rmse_days": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
        "sum_bias_percent": float(100 * (prediction.sum() - y.sum()) / y.sum()),
    }


def searches(random_state: int) -> dict[str, GridSearchCV]:
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    common = dict(scoring=negative_mae_days, cv=cv, n_jobs=-1, return_train_score=True)
    return {
        "ridge": GridSearchCV(
            base.build_model("ridge", 1.0, random_state),
            {"ridge__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
            **common,
        ),
        "hist_gradient_boosting": GridSearchCV(
            base.build_model("hist_gradient_boosting", 1.0, random_state),
            {
                "histgradientboostingregressor__learning_rate": [0.03, 0.06],
                "histgradientboostingregressor__max_leaf_nodes": [15, 31],
                "histgradientboostingregressor__l2_regularization": [0.0, 1.0],
            },
            **common,
        ),
    }


def fit_ols(
    x: np.ndarray,
    y: np.ndarray,
    columns: list[str],
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], np.ndarray, np.ndarray]:
    indices = [columns.index(name) for name in OLS_FEATURES]
    selected = np.asarray(x[:, indices], dtype=float)
    medians = np.nanmedian(selected[train], axis=0)
    filled = np.where(np.isfinite(selected), selected, medians)
    means = filled[train].mean(axis=0)
    scales = filled[train].std(axis=0)
    scales[scales == 0] = 1.0
    standardized = (filled - means) / scales
    design = sm.add_constant(standardized, has_constant="add")
    model = sm.OLS(np.log1p(y[train]), design[train]).fit()
    coefficient_names = ["const"] + OLS_FEATURES
    confidence = model.conf_int(alpha=0.05)
    coefficients = pd.DataFrame(
        {
            "feature": coefficient_names,
            "coefficient_log1p": model.params,
            "standard_error": model.bse,
            "p_value": model.pvalues,
            "ci_2_5": confidence[:, 0],
            "ci_97_5": confidence[:, 1],
            "multiplicative_change_per_1sd": np.exp(model.params),
        }
    )
    metric_rows = []
    for split_name, split_index in (("validation", validation), ("test", test)):
        prediction = np.maximum(np.expm1(model.predict(design[split_index])), 0.0)
        metric_rows.append({"split": split_name, **regression_metrics(y[split_index], prediction)})
    fitted_log = model.predict(design[train])
    residual_log = np.log1p(y[train]) - fitted_log
    summary = {
        "n": int(model.nobs),
        "r2_log_scale": float(model.rsquared),
        "adjusted_r2_log_scale": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "durbin_watson": float(durbin_watson(residual_log)),
        "condition_number": float(model.condition_number),
        "target": "log1p(remaining_los_days)",
    }
    return coefficients, pd.DataFrame(metric_rows), summary, fitted_log, residual_log


def draw_learning_curve(frame: pd.DataFrame, destination: Path) -> None:
    image = Image.new("RGB", (1300, 800), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "剩余住院时间学习曲线（训练集内部五折）", font=font(34, True), fill=INK)
    draw.text((55, 85), "纵轴为 MAE（天，越低越好）；训练与交叉验证差距反映过拟合。", font=font(20), fill=MUTED)
    draw.rounded_rectangle((55, 135, 1245, 735), radius=18, fill=WHITE)
    x0, y0, x1, y1 = 145, 190, 1190, 650
    ymin = max(0.0, float(frame[["train_mae", "cv_mae"]].min().min()) - 0.3)
    ymax = float(frame[["train_mae", "cv_mae"]].max().max()) + 0.3
    for value in np.linspace(ymin, ymax, 6):
        py = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
        draw.line((x0, py, x1, py), fill=GRID)
        draw.text((75, py - 10), f"{value:.1f}", font=font(16), fill=MUTED)
    sizes = frame["train_size"].to_numpy(float)
    for column, color, label in (("train_mae", BLUE, "训练"), ("cv_mae", ORANGE, "交叉验证")):
        points = []
        for size, value in zip(sizes, frame[column]):
            px = x0 + (size - sizes.min()) / (sizes.max() - sizes.min()) * (x1 - x0)
            py = y1 - (value - ymin) / (ymax - ymin) * (y1 - y0)
            points.append((px, py))
        draw.line(points, fill=color, width=4)
        for px, py in points:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
        draw.text((930, 160 if column == "train_mae" else 190), label, font=font(18), fill=color)
    image.save(destination)


def draw_residual_diagnostics(fitted: np.ndarray, residual: np.ndarray, destination: Path) -> None:
    image = Image.new("RGB", (1800, 650), BG)
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), "OLS 残差诊断（训练集，log1p 尺度）", font=font(34, True), fill=INK)
    rng = np.random.default_rng(20260906)
    chosen = rng.choice(len(fitted), size=min(2500, len(fitted)), replace=False)
    fit_sample, residual_sample = fitted[chosen], residual[chosen]
    panels = [(50, 120, 570, 590), (640, 120, 1160, 590), (1230, 120, 1750, 590)]
    titles = ["残差 vs 拟合值", "正态 Q-Q", "残差分布"]
    for panel, title in zip(panels, titles):
        draw.rounded_rectangle(panel, radius=16, fill=WHITE)
        draw.text((panel[0] + 20, panel[1] + 15), title, font=font(21, True), fill=INK)

    def scatter(panel, xs, ys, color):
        left, top, right, bottom = panel[0] + 55, panel[1] + 70, panel[2] - 25, panel[3] - 45
        xlow, xhigh = np.quantile(xs, [0.01, 0.99])
        ylow, yhigh = np.quantile(ys, [0.01, 0.99])
        for xvalue, yvalue in zip(xs, ys):
            px = left + np.clip((xvalue - xlow) / max(xhigh - xlow, 1e-9), 0, 1) * (right - left)
            py = bottom - np.clip((yvalue - ylow) / max(yhigh - ylow, 1e-9), 0, 1) * (bottom - top)
            draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=color)
        return left, top, right, bottom

    left, top, right, bottom = scatter(panels[0], fit_sample, residual_sample, BLUE)
    zero_y = bottom - (0 - np.quantile(residual_sample, 0.01)) / max(np.quantile(residual_sample, 0.99) - np.quantile(residual_sample, 0.01), 1e-9) * (bottom - top)
    draw.line((left, zero_y, right, zero_y), fill=ORANGE, width=2)
    theoretical, ordered = stats.probplot(residual_sample, dist="norm", fit=False)
    scatter(panels[1], np.asarray(theoretical), np.asarray(ordered), GREEN)
    counts, edges = np.histogram(residual_sample, bins=35)
    panel = panels[2]
    left, top, right, bottom = panel[0] + 55, panel[1] + 70, panel[2] - 25, panel[3] - 45
    for index, count in enumerate(counts):
        x0 = left + index / len(counts) * (right - left)
        x1 = left + (index + 1) / len(counts) * (right - left)
        y0 = bottom - count / max(counts.max(), 1) * (bottom - top)
        draw.rectangle((x0, y0, x1, bottom), fill=ORANGE)
    image.save(destination)


def draw_importance(frame: pd.DataFrame, destination: Path) -> None:
    top = frame.head(15).sort_values("importance_mean")
    image = Image.new("RGB", (1400, 900), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), "剩余住院时间：验证集置换重要度", font=font(34, True), fill=INK)
    draw.text((55, 85), "打乱后 MAE 增量（天）；表示预测贡献，不表示因果作用。", font=font(20), fill=MUTED)
    draw.rounded_rectangle((45, 135, 1355, 840), radius=18, fill=WHITE)
    maximum = max(float(top["importance_mean"].max()), 1e-6)
    for row_number, row in enumerate(top.itertuples(index=False)):
        y = 790 - row_number * 41
        width = max(1, int(max(row.importance_mean, 0) / maximum * 650))
        draw.text((70, y - 11), str(row.feature)[:42], font=font(16), fill=INK)
        draw.rectangle((600, y - 10, 600 + width, y + 12), fill=GREEN)
        draw.text((610 + width, y - 11), f"{row.importance_mean:.3f}", font=font(15), fill=MUTED)
    image.save(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "Experiments/shared/output/data")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "output/course_extension")
    parser.add_argument("--random-state", type=int, default=20260906)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Refresh OLS tables and residual figure without repeating model tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outcomes_all = read_outcomes(args.data_dir)
    if args.max_records is not None:
        outcomes_all = outcomes_all.head(args.max_records).copy()
    record_ids = outcomes_all["RecordID"].to_numpy(np.int64)
    columns, matrices, _, _, cache_hit = load_or_extract_features(
        data_dir=args.data_dir,
        record_ids=record_ids,
        horizons=[48],
        cache_dir=args.cache_dir,
        use_cache=True,
    )
    los = outcomes_all["Length_of_stay"].to_numpy(float)
    valid = np.isfinite(los) & (los >= 2)
    outcomes = outcomes_all.loc[valid].reset_index(drop=True)
    x = matrices[48][valid]
    remaining = los[valid] - 2.0
    splits = base.make_quantile_splits(remaining, args.random_state)
    output, figures = args.output_dir, args.output_dir / "report_figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if args.diagnostics_only:
        coefficients, ols_metrics, ols_summary, fitted_log, residual_log = fit_ols(
            x,
            remaining,
            columns,
            splits["train"],
            splits["validation"],
            splits["test"],
        )
        coefficients.to_csv(output / "ols_coefficients.csv", index=False)
        ols_metrics.to_csv(output / "ols_validation_test_metrics.csv", index=False)
        (output / "ols_model_summary.json").write_text(
            json.dumps(ols_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        draw_residual_diagnostics(
            fitted_log, residual_log, figures / "ols_residual_diagnostics.png"
        )
        print(pd.Series(ols_summary).to_string())
        return

    cv_frames, validation_rows, fitted = [], [], {}
    y_log_train = np.log1p(remaining[splits["train"]])
    for name, search in searches(args.random_state).items():
        search.fit(x[splits["train"]], y_log_train)
        fitted[name] = search.best_estimator_
        frame = pd.DataFrame(search.cv_results_)
        frame.insert(0, "model", name)
        cv_frames.append(frame)
        prediction = base.predict_remaining(search.best_estimator_, x[splits["validation"]])
        validation_rows.append(
            {
                "model": name,
                "best_parameters": json.dumps(search.best_params_, ensure_ascii=False),
                **regression_metrics(remaining[splits["validation"]], prediction),
            }
        )
    cv_results = pd.concat(cv_frames, ignore_index=True)
    validation = pd.DataFrame(validation_rows).sort_values(["mae_days", "rmse_days", "model"])
    selected_name = str(validation.iloc[0]["model"])
    selected = fitted[selected_name]

    development = np.sort(np.concatenate([splits["train"], splits["validation"]]))
    final_model = clone(selected).fit(x[development], np.log1p(remaining[development]))
    final_prediction = base.predict_remaining(final_model, x[splits["test"]])
    final = pd.DataFrame(
        [{"model": selected_name, "fit_sample": "train_plus_validation", **regression_metrics(remaining[splits["test"]], final_prediction)}]
    )

    curve_sizes, curve_train, curve_cv = learning_curve(
        clone(selected),
        x[splits["train"]],
        y_log_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=args.random_state),
        scoring=negative_mae_days,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        n_jobs=-1,
    )
    curve = pd.DataFrame(
        {
            "train_size": curve_sizes,
            "train_mae": -curve_train.mean(axis=1),
            "train_sd": curve_train.std(axis=1),
            "cv_mae": -curve_cv.mean(axis=1),
            "cv_sd": curve_cv.std(axis=1),
        }
    )
    importance_result = permutation_importance(
        selected,
        x[splits["validation"]],
        np.log1p(remaining[splits["validation"]]),
        scoring=negative_mae_days,
        n_repeats=3,
        max_samples=0.70,
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

    coefficients, ols_metrics, ols_summary, fitted_log, residual_log = fit_ols(
        x,
        remaining,
        columns,
        splits["train"],
        splits["validation"],
        splits["test"],
    )
    cv_results.to_csv(output / "tuning_cv_results.csv", index=False)
    validation.to_csv(output / "validation_model_comparison.csv", index=False)
    final.to_csv(output / "final_test_metrics.csv", index=False)
    curve.to_csv(output / "learning_curve.csv", index=False)
    importance.to_csv(output / "permutation_importance.csv", index=False)
    coefficients.to_csv(output / "ols_coefficients.csv", index=False)
    ols_metrics.to_csv(output / "ols_validation_test_metrics.csv", index=False)
    (output / "ols_model_summary.json").write_text(
        json.dumps(ols_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_learning_curve(curve, figures / "learning_curve.png")
    draw_importance(importance, figures / "permutation_importance.png")
    draw_residual_diagnostics(fitted_log, residual_log, figures / "ols_residual_diagnostics.png")
    metadata = {
        "target": "remaining length of stay after the 48-hour landmark",
        "split_sizes": {name: int(len(index)) for name, index in splits.items()},
        "selection": "5-fold training CV MAE tunes each family; validation MAE selects family; test opened once",
        "selected_model": selected_name,
        "selected_parameters": selected.get_params(deep=True),
        "ols_features": OLS_FEATURES,
        "cache_hit": cache_hit,
        "random_state": args.random_state,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(validation.to_string(index=False))
    print(final.to_string(index=False))
    print(pd.Series(ols_summary).to_string())


if __name__ == "__main__":
    main()
