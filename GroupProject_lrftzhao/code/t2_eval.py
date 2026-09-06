"""T2 Stage 5–6 的 eval 半边：读冻结模型，test 只看这一次。
报：基线与五个模型的 MAE/RMSE/中位绝对误差（天）+ MAE bootstrap CI；Σŷ vs Σy（容量主张的主数，总体与按 ICU 类型）；
    残差诊断（LINE：原始尺度 vs log 尺度的残差形态、异方差检验）；误差按剩余住院日分段（尾巴主导）；
    幸存者子集敏感性（预登记为 null 结论）；系数表（Ridge/Lasso，每 SD）；成组置换重要度（val）。
产出 t2_metrics.json、t2_error_by_band.csv、t2_capacity_by_icu.csv、t2_coefficients.csv、t2_group_importance.csv、
     t2_test_predictions.csv、t2_residuals.png
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common import load_data, make_regressor, reg_metrics, to_days, bootstrap_mae_ci, variable_of, dump
from config import OUT, MODELS, LABEL_MAIN, RANDOM_STATE

X, lab, sp = load_data()
ok = lab["t2_valid"] == 1
tr = [i for i in sp["train"] if ok[i]]; va = [i for i in sp["val"] if ok[i]]; te = [i for i in sp["test"] if ok[i]]
y = lab["LOS_remaining"]; ylog = np.log1p(y.where(ok))
fit = json.load(open(OUT / "t2_fit_report.json")); main = fit["main_linear_model"]
results = {"n_test": len(te), "main_linear_model": main, "test_true_sum_days": float(y[te].sum()),
           "test_median_days": float(y[te].median()), "test_mean_days": float(y[te].mean())}

# ---- Stage 3 基线（test）
med = fit["baseline_median_train_days"]; mean_ = fit["baseline_mean_train_days"]
results["baseline_median"] = {**reg_metrics(y[te], np.full(len(te), med)), "predicts": med}
results["baseline_mean"] = {**reg_metrics(y[te], np.full(len(te), mean_)), "predicts": mean_}

# ---- Stage 5：五个模型 test 一次
preds = pd.DataFrame({"RecordID": te, "y_days": y[te].values, "died": lab.loc[te, LABEL_MAIN].values})
def predict_days(kind, m, Z):
    return np.clip(m.predict(Z), 0, None) if kind == "ols_raw" else to_days(m.predict(Z))
models = {}
for kind in ["ols_raw", "ols", "ridge", "lasso", "rf_reg"]:
    m = joblib.load(MODELS / f"t2_{kind}.joblib"); models[kind] = m
    p = predict_days(kind, m, X.loc[te]); preds[f"pred_{kind}"] = p
    results[kind] = {**reg_metrics(y[te], p), "mae_ci95": bootstrap_mae_ci(y[te], p), "chosen": fit[kind]["chosen"]}
    print(f"{kind:8s} test MAE {results[kind]['mae_days']:.2f} [{results[kind]['mae_ci95'][0]:.2f}, {results[kind]['mae_ci95'][1]:.2f}]  "
          f"RMSE {results[kind]['rmse_days']:.2f}  中位AE {results[kind]['median_ae_days']:.2f}  Σŷ/Σy 偏差 {results[kind]['sum_bias_pct']:+.1f}%")
    if kind != "ols_raw":   # 同一模型的第二种读数：Duan smearing 均值型（给容量 Σ）
        S = fit[kind]["smearing_factor"]; ps = to_days(m.predict(X.loc[te]), S); preds[f"pred_{kind}_smeared"] = ps
        results[f"{kind}_smeared"] = {**reg_metrics(y[te], ps), "mae_ci95": bootstrap_mae_ci(y[te], ps), "smearing_factor": S}
        print(f"{kind+'_smear':8s}  MAE {results[kind+'_smeared']['mae_days']:.2f}  Σŷ/Σy 偏差 {results[kind+'_smeared']['sum_bias_pct']:+.1f}%  (S={S:.3f})")
print(f"基线 中位数 MAE {results['baseline_median']['mae_days']:.2f}  Σ偏差 {results['baseline_median']['sum_bias_pct']:+.1f}%")
preds.to_csv(OUT / "t2_test_predictions.csv", index=False)
pm = preds[f"pred_{main}"].values; pms = preds[f"pred_{main}_smeared"].values

# ---- 容量主张：Σŷ vs Σy 按 ICU 类型（主线性模型）
icu = lab.loc[te, "ICUType"].map({1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}).values
cap = (pd.DataFrame({"icu": icu, "y": y[te].values, "pred_median_type": pm, "pred_smeared": pms, "pred_ols_raw": preds["pred_ols_raw"].values})
       .groupby("icu").agg(n=("y", "size"), true_days=("y", "sum"), pred_median_type=("pred_median_type", "sum"),
                           pred_smeared=("pred_smeared", "sum"), pred_ols_raw=("pred_ols_raw", "sum")))
cap.loc["ALL"] = cap.sum()
for c in ["pred_median_type", "pred_smeared", "pred_ols_raw"]:
    cap[f"bias_pct_{c[5:]}"] = 100 * (cap[c] - cap["true_days"]) / cap["true_days"]
cap.to_csv(OUT / "t2_capacity_by_icu.csv"); print(cap.round(1).to_string())

# ---- 误差按剩余住院日分段（尾巴主导）
band = pd.cut(y[te], [-1, 3, 7, 14, 24, 10_000], labels=["0-3", "4-7", "8-14", "15-24", "25+"])
eb = pd.DataFrame({"band": band.values, "y": y[te].values, "pred": pm, "pred_median": med})
rows = []
for b, g in eb.groupby("band", observed=True):
    rows.append({"band_days": b, "n": len(g), "share_patients": len(g) / len(eb), "share_days": g["y"].sum() / eb["y"].sum(),
                 "mae_main": np.mean(np.abs(g["pred"] - g["y"])), "bias_main": np.mean(g["pred"] - g["y"]),
                 "mae_median_baseline": np.mean(np.abs(g["pred_median"] - g["y"]))})
ebt = pd.DataFrame(rows); ebt.to_csv(OUT / "t2_error_by_band.csv", index=False); print(ebt.round(3).to_string(index=False))

# ---- 残差诊断（LINE）：原始尺度 OLS vs log 尺度主模型
def resid_diag(resid, fitted, label):
    resid = np.asarray(resid); fitted = np.asarray(fitted)
    rho, p = stats.spearmanr(np.abs(resid), fitted)
    return {"scale": label, "resid_mean": float(resid.mean()), "resid_skew": float(stats.skew(resid)),
            "resid_excess_kurtosis": float(stats.kurtosis(resid)),
            "heteroscedasticity_spearman_abs_resid_vs_fitted": float(rho), "heteroscedasticity_p": float(p),
            "shapiro_p_on_500_subsample": float(stats.shapiro(np.random.default_rng(RANDOM_STATE).choice(resid, min(500, len(resid)), replace=False)).pvalue)}
raw_fit = preds["pred_ols_raw"].values; raw_res = y[te].values - raw_fit
log_fit = models[main].predict(X.loc[te]); log_res = ylog[te].values - log_fit
results["residual_diagnostics"] = {"ols_raw_days": resid_diag(raw_res, raw_fit, "days (ols_raw)"),
                                   f"{main}_log1p": resid_diag(log_res, log_fit, f"log1p ({main})")}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
ax[0].scatter(raw_fit, raw_res, s=5, alpha=.3); ax[0].axhline(0, c="k", lw=1); ax[0].set(xlabel="fitted (days, OLS raw)", ylabel="residual (days)", title="Raw scale: funnel + long tail")
ax[1].scatter(log_fit, log_res, s=5, alpha=.3); ax[1].axhline(0, c="k", lw=1); ax[1].set(xlabel=f"fitted (log1p, {main})", ylabel="residual (log1p)", title="log1p scale: closer to LINE")
ax[2].scatter(y[te], pm, s=5, alpha=.3); lim = [0, np.percentile(y[te], 99)]; ax[2].plot(lim, lim, "k--", lw=1)
ax[2].set(xlim=lim, ylim=lim, xlabel="true remaining LOS (days)", ylabel=f"predicted ({main})", title="Predicted vs true (test)")
fig.tight_layout(); fig.savefig(OUT / "t2_residuals.png", dpi=150)

# ---- smearing 的前提检查：单一 S 假设 log 残差与拟合值无关（同方差）。异方差显著时用按拟合值三分位分层的 S 做敏感性
fit_tr = models[main].predict(X.loc[tr]); res_tr = ylog[tr].values - fit_tr
cuts = np.quantile(fit_tr, [1 / 3, 2 / 3]); S_strata = [float(np.mean(np.exp(res_tr[np.digitize(fit_tr, cuts) == k]))) for k in range(3)]
strata_te = np.digitize(log_fit, cuts); ps_strat = np.clip(np.exp(log_fit) * np.array(S_strata)[strata_te] - 1, 0, None)
results["smearing_stratified_sensitivity"] = {"fitted_tertile_cuts_log1p": [float(c) for c in cuts], "S_by_tertile": S_strata,
                                              "global_S": fit[main]["smearing_factor"], "test": reg_metrics(y[te], ps_strat),
                                              "note": "全局 S 的无偏性依赖 log 残差同方差；此处按 train 拟合值三分位分别算 S，看 Σ 偏差是否稳"}
print(f"分层 smearing 敏感性：S 按三分位 {np.round(S_strata, 3)}（全局 {fit[main]['smearing_factor']:.3f}）→ test Σ偏差 {results['smearing_stratified_sensitivity']['test']['sum_bias_pct']:+.1f}%（全局 S 时 {results[main + '_smeared']['sum_bias_pct']:+.1f}%）")

# ---- 幸存者子集敏感性（预登记为 null 结论）：全体训的模型在幸存者上 vs 只用幸存者重训
surv_tr = [i for i in tr if lab.loc[i, LABEL_MAIN] == 0]; surv_te = [i for i in te if lab.loc[i, LABEL_MAIN] == 0]
m_s = make_regressor(main, **fit[main]["chosen"]).fit(X.loc[surv_tr], ylog[surv_tr])
results["survivor_sensitivity"] = {
    "n_train_survivors": len(surv_tr), "n_test_survivors": len(surv_te),
    "all_patient_model__on_test_survivors": reg_metrics(y[surv_te], to_days(models[main].predict(X.loc[surv_te]))),
    "survivor_only_model__on_test_survivors": reg_metrics(y[surv_te], to_days(m_s.predict(X.loc[surv_te]))),
    "all_patient_model__on_test_deaths": reg_metrics(y[[i for i in te if i not in set(surv_te)]],
                                                     to_days(models[main].predict(X.loc[[i for i in te if i not in set(surv_te)]]))),
    "median_remaining_train": {"survivors": float(y[surv_tr].median()), "deaths": float(y[[i for i in tr if i not in set(surv_tr)]].median())}}
s = results["survivor_sensitivity"]
print(f"幸存者敏感性：全体模型在幸存者 MAE {s['all_patient_model__on_test_survivors']['mae_days']:.2f} vs 幸存者重训 {s['survivor_only_model__on_test_survivors']['mae_days']:.2f}；"
      f"全体模型在死亡者 MAE {s['all_patient_model__on_test_deaths']['mae_days']:.2f}")

# ---- Stage 6 可解释性：系数（每 SD，log1p 尺度）与成组置换重要度（val，MAE 上升）
rows = []
for kind in ["ridge", "lasso"]:
    coef = pd.Series(models[kind].named_steps["reg"].coef_, index=X.columns)
    for c, v in coef.items():
        # 模型拟合的是 log(1+y)：exp(v)−1 是「剩余+1」的百分变化；换算到剩余天数本身随基线 y 变，这里给 train 中位数处的近似值
        rows.append({"model": kind, "feature": c, "variable": variable_of(c), "coef_per_SD_log1p": v,
                     "pct_change_LOS_remaining_plus1_per_SD": 100 * (np.exp(v) - 1),
                     "approx_pct_change_LOS_remaining_at_train_median_per_SD": 100 * (np.exp(v) - 1) * (1 + med) / med,
                     "note": "二元/one-hot 列同样按每 1 SD 计，不是 0→1"})
coefs = pd.DataFrame(rows); coefs["abs"] = coefs["coef_per_SD_log1p"].abs()
coefs.sort_values(["model", "abs"], ascending=[True, False]).drop(columns="abs").to_csv(OUT / "t2_coefficients.csv", index=False)
print(f"Lasso 非零系数 {int((coefs[coefs.model == 'lasso'].coef_per_SD_log1p != 0).sum())} / {X.shape[1]}")
print(f"主模型 |系数| top10（每 SD：对 (剩余+1) 的 % 影响 / 在 train 中位数 {med:.0f} 天处对剩余天数的近似 %）：")
print(coefs[coefs.model == main].sort_values("abs", ascending=False).head(10)[["feature", "pct_change_LOS_remaining_plus1_per_SD", "approx_pct_change_LOS_remaining_at_train_median_per_SD"]].round(1).to_string(index=False))

rng = np.random.default_rng(RANDOM_STATE); Xv = X.loc[va]; groups = {}
for c in X.columns:
    groups.setdefault(variable_of(c), []).append(c)
imp_rows = []
for kind in [main, "rf_reg"]:
    m = models[kind]; base = np.mean(np.abs(to_days(m.predict(Xv)) - y[va].values))
    for var, cols in groups.items():
        rises = []
        for _ in range(3):
            Xp = Xv.copy(); Xp[cols] = Xp[cols].sample(frac=1, random_state=int(rng.integers(1e9))).values
            rises.append(np.mean(np.abs(to_days(m.predict(Xp)) - y[va].values)) - base)
        imp_rows.append({"model": kind, "variable": var, "n_cols": len(cols), "mae_rise_days": float(np.mean(rises))})
imp = pd.DataFrame(imp_rows).sort_values(["model", "mae_rise_days"], ascending=[True, False]); imp.to_csv(OUT / "t2_group_importance.csv", index=False)
print("成组置换重要度 top8（val 上 MAE 上升，天）："); print(imp.groupby("model").head(8).round(3).to_string(index=False))
dump(results, OUT / "t2_metrics.json")
