"""T2 Stage 3–4 的 fit 半边：剩余住院日回归。只看 train/val，不碰 test。
目标 = LOS_remaining = Length_of_stay − 2（医院床日，全体病人；t2_valid=1 即剔除 LOS=-1 与 LOS=1 的 172 例）。
口径：拟合 log1p(剩余)，预测后 expm1 反变换、按天报误差；ols_raw 是原始尺度的对照（用来展示为什么要取 log）。
- 五个模型在 val 上按 MAE（天）选超参：ols_raw / ols_log（无超参）→ ridge(alpha) → lasso(alpha) → rf_reg(min_samples_leaf)
- 冻结：模型（joblib）、val 报告；主线性模型 = ridge/lasso 中 val MAE 更低者
产出 output/models/t2_*.joblib、t2_fit_report.json
"""
import joblib
import numpy as np
import pandas as pd
from common import load_data, make_regressor, reg_metrics, to_days, smearing_factor, dump
from config import OUT, MODELS

X, lab, sp = load_data()
ok = lab["t2_valid"] == 1
tr = [i for i in sp["train"] if ok[i]]; va = [i for i in sp["val"] if ok[i]]
y = lab["LOS_remaining"]; ylog = np.log1p(y.where(ok))          # 无效行先置 NaN，免得 log1p(-1) 报除零
print(f"T2 有效：train {len(tr)}  val {len(va)}；剩余住院日 train 中位 {y[tr].median():.1f} 天、均值 {y[tr].mean():.2f} 天，"
      f"skew 原始 {y[tr].skew():.2f} → log1p {ylog[tr].skew():.2f}")

report = {"target": "LOS_remaining = Length_of_stay - 2（医院床日）", "transform": "log1p → expm1，clip ≥ 0",
          "n_train": len(tr), "n_val": len(va),
          "baseline_median_train_days": float(y[tr].median()), "baseline_mean_train_days": float(y[tr].mean())}

# 基线：永远预测 train 中位数（MAE 下的正确 naive 基线）；均值基线只为 RMSE 对照
report["baseline_median"] = {"val": reg_metrics(y[va], np.full(len(va), y[tr].median()))}
report["baseline_mean"] = {"val": reg_metrics(y[va], np.full(len(va), y[tr].mean()))}
print(f"基线（val）：中位数 MAE {report['baseline_median']['val']['mae_days']:.2f} 天；均值 MAE {report['baseline_mean']['val']['mae_days']:.2f} 天")

GRIDS = {
    "ols_raw": (None, [None]),
    "ols": (None, [None]),
    "ridge": ("alpha", [1, 10, 100, 300, 1000, 3000, 10000, 30000]),
    "lasso": ("alpha", [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]),
    "rf_reg": ("min_samples_leaf", [1, 2, 5, 10, 20, 50]),
}
for kind, (param, grid) in GRIDS.items():
    best = None
    for g in grid:
        kw = {param: g} if param else {}
        m = make_regressor("ols" if kind == "ols_raw" else kind, **kw)
        if kind == "ols_raw":
            m.fit(X.loc[tr], y[tr]); f = lambda Z: np.clip(m.predict(Z), 0, None)
        else:
            m.fit(X.loc[tr], ylog[tr]); f = lambda Z: to_days(m.predict(Z))
        r_tr = reg_metrics(y[tr], f(X.loc[tr])); r_va = reg_metrics(y[va], f(X.loc[va]))
        print(f"  {kind:7s} {param}={g}: train MAE {r_tr['mae_days']:.3f}  val MAE {r_va['mae_days']:.3f}  val Σ偏差 {r_va['sum_bias_pct']:+.1f}%")
        if best is None or r_va["mae_days"] < best[1]["mae_days"]:
            best = (g, r_va, m, r_tr)
    g, r_va, m, r_tr = best
    report[kind] = {"chosen": ({param: g} if param else {}), "val": r_va, "train_mae_days": r_tr["mae_days"],
                    "grid_edge": bool(param and (g == grid[0] or g == grid[-1]))}
    if kind == "lasso":
        coef = m.named_steps["reg"].coef_; report[kind]["n_nonzero_coef"] = int((coef != 0).sum()); report[kind]["n_features"] = int(len(coef))
    if kind != "ols_raw":
        # Duan smearing 因子只从 train 残差算；val 上同时报「中位数型」与「smearing 均值型」两种读数
        S = smearing_factor(ylog[tr].values - m.predict(X.loc[tr])); report[kind]["smearing_factor"] = S
        report[kind]["val_smeared"] = reg_metrics(y[va], to_days(m.predict(X.loc[va]), S))
        print(f"          smearing S={S:.3f}：val MAE {report[kind]['val_smeared']['mae_days']:.3f}  Σ偏差 {report[kind]['val_smeared']['sum_bias_pct']:+.1f}%")
    joblib.dump(m, MODELS / f"t2_{kind}.joblib")
    print(f"{kind}: 选 {param}={g}，val MAE {r_va['mae_days']:.3f} 天" + ("  ⚠️ 落在网格边界" if report[kind]["grid_edge"] else ""))

main = min(["ridge", "lasso"], key=lambda k: report[k]["val"]["mae_days"])
report["main_linear_model"] = main
dump(report, OUT / "t2_fit_report.json")
print(f"主线性模型 = {main}；模型已冻结 → {MODELS}，报告 → {OUT / 't2_fit_report.json'}")
