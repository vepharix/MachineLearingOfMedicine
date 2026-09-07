"""汇总：把 output/ 里散落的 JSON/CSV 收成一份 results.json——slide 上所有数字的唯一出处（方案 §四规则 4）。
只读不算；每个数字都能回溯到产出它的文件。产出 output/results.json 并打印一页摘要。"""
import json
import pandas as pd
from common import dump
from config import OUT

J = lambda n: json.load(open(OUT / n, encoding="utf-8"))
C = lambda n: pd.read_csv(OUT / n)
R = {"_source_files": {}, "_note": "所有数字来自 output/ 下的产物文件；test 集只评估过一次（t1_eval.py / t2_eval.py）"}

dq = J("data_quality.json"); R["data_quality"] = dq; R["_source_files"]["data_quality"] = "data_quality.json"
fit1 = J("t1_fit_report.json"); m1 = J("t1_metrics.json")
R["t1"] = {"label": "In-hospital_death", "n_test": m1["logreg"]["n"], "prevalence_test": m1["logreg"]["prevalence"],
           "hyperparameters_chosen_on_val": {k: v["chosen"] for k, v in fit1.items() if isinstance(v, dict) and "chosen" in v},
           "baselines_test": {"majority": 0.5, "SAPS-I_auc": m1["baseline_SAPS-I"]["auc_roc_imputed_full_test"], "SOFA_auc": m1["baseline_SOFA"]["auc_roc_imputed_full_test"]},
           "models_test": {k: {kk: m1[k][kk] for kk in ["auc_roc", "auc_roc_ci95", "auc_pr", "brier"]} for k in ["logreg", "tree", "rf", "xgb"]},
           "threshold_table_logreg": C("t1_threshold_table.csv").query("model=='logreg'").drop(columns="model").to_dict("records"),
           "calibration_top_bin": {k: g.iloc[-1][["mean_predicted", "observed_rate"]].round(3).to_dict() for k, g in C("t1_calibration.csv").groupby("model")},
           "subgroups": C("t1_subgroups.csv").round(3).to_dict("records"),
           "group_importance_top5": {k: g.head(5)[["variable", "auc_drop"]].round(3).to_dict("records") for k, g in C("t1_group_importance.csv").groupby("model")},
           "tree_depth_curve": C("t1_tree_depth_curve.csv").round(3).to_dict("records")}
R["_source_files"]["t1"] = ["t1_fit_report.json", "t1_metrics.json", "t1_threshold_table.csv", "t1_calibration.csv", "t1_subgroups.csv", "t1_group_importance.csv", "t1_tree_depth_curve.csv"]
R["C_leave_one_icu_out"] = C("C_leave_one_icu_out.csv").round(3).to_dict("records")
R["E_label_sensitivity"] = J("E_label_sensitivity.json")
R["T3_24h_ablation"] = J("T3_24h_ablation.json")
try:
    fit2 = J("t2_fit_report.json"); m2 = J("t2_metrics.json")
    R["t2"] = {"target": fit2["target"], "transform": fit2["transform"], "n_test": m2["n_test"], "main_linear_model": m2["main_linear_model"],
               "hyperparameters_chosen_on_val": {k: fit2[k]["chosen"] for k in ["ridge", "lasso", "rf_reg"]},
               "lasso_nonzero": fit2["lasso"].get("n_nonzero_coef"),
               "baseline_median_test": {k: m2["baseline_median"][k] for k in ["predicts", "mae_days", "rmse_days", "sum_bias_pct"]},
               "models_test": {k: {kk: m2[k][kk] for kk in ["mae_days", "mae_ci95", "rmse_days", "median_ae_days", "r2_days", "sum_bias_pct"]} for k in ["ols_raw", "ols", "ridge", "lasso", "rf_reg"]},
               "models_test_smeared": {k: {kk: m2[f"{k}_smeared"][kk] for kk in ["mae_days", "mae_ci95", "rmse_days", "sum_bias_pct", "smearing_factor"]} for k in ["ols", "ridge", "lasso", "rf_reg"]},
               "capacity_by_icu": C("t2_capacity_by_icu.csv").round(1).to_dict("records"),
               "error_by_band": C("t2_error_by_band.csv").round(3).to_dict("records"),
               "residual_diagnostics": m2["residual_diagnostics"], "survivor_sensitivity": m2["survivor_sensitivity"],
               "smearing_stratified_sensitivity": m2.get("smearing_stratified_sensitivity"),
               "coefficients_top15": {k: g.head(15).drop(columns=["model", "note"], errors="ignore").round(4).to_dict("records") for k, g in C("t2_coefficients.csv").groupby("model")},
               "group_importance_top5": {k: g.head(5)[["variable", "mae_rise_days"]].round(3).to_dict("records") for k, g in C("t2_group_importance.csv").groupby("model")}}
    R["_source_files"]["t2"] = ["t2_fit_report.json", "t2_metrics.json", "t2_capacity_by_icu.csv", "t2_error_by_band.csv", "t2_coefficients.csv", "t2_group_importance.csv"]
except FileNotFoundError as e:
    R["t2"] = f"未跑：{e}"
try:
    R["D_fn_autopsy"] = J("D_fn_autopsy.json")
except FileNotFoundError:
    R["D_fn_autopsy"] = "未跑"
try:
    R["A_missingness_only"] = J("A_missingness_only.json")
except FileNotFoundError:
    R["A_missingness_only"] = "未跑"
try:
    R["B_time_value"] = J("B_time_value.json")
except FileNotFoundError:
    R["B_time_value"] = "未跑"
dump(R, OUT / "results.json")

t = R["t1"]
print(f"T1 test n={t['n_test']} 死亡率 {t['prevalence_test']:.1%} | SAPS-I {t['baselines_test']['SAPS-I_auc']:.3f} SOFA {t['baselines_test']['SOFA_auc']:.3f} | " +
      " ".join(f"{k} {v['auc_roc']:.3f}" for k, v in t["models_test"].items()))
if isinstance(R["t2"], dict):
    t2 = R["t2"]
    print(f"T2 test n={t2['n_test']} | 中位数基线 MAE {t2['baseline_median_test']['mae_days']:.2f} | " +
          " ".join(f"{k} {v['mae_days']:.2f}" for k, v in t2["models_test"].items()) + f" | 主模型 {t2['main_linear_model']}：中位数型 Σ偏差 {t2['models_test'][t2['main_linear_model']]['sum_bias_pct']:+.1f}% / smearing 均值型 {t2['models_test_smeared'][t2['main_linear_model']]['sum_bias_pct']:+.1f}%")
if isinstance(R["D_fn_autopsy"], dict):
    d = R["D_fn_autopsy"]; print(f"D: 死亡 {d['n_deaths']}，FN {d['FN']} / TP {d['TP']}；{d['n_vars_FN_closer_to_TN']}/{d['n_vars']} 个变量上 FN 更像存活者")
if isinstance(R["A_missingness_only"], dict):
    a = R["A_missingness_only"]; print(f"A: 只看是否测过 AUC {a['A1_measured_only']['test']['auc_roc']:.3f}；加测量次数 {a['A2_measured_plus_counts']['test']['auc_roc']:.3f}（T1 全特征 {a['reference_test_auc']['T1_logreg_full_features']:.3f}）")
if isinstance(R["B_time_value"], dict):
    b = R["B_time_value"]; print("B: " + "  ".join(f"{r['window_h']}h {r['auc_logreg']:.3f}" for r in b["rows"]) + "  (logreg test AUC)")
print("→", OUT / "results.json")
