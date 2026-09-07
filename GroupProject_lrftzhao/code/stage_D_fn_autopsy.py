"""打法 D：假阴性尸检。工作点 = 冻结的 LogReg sens=0.90 阈值（t1_operating_points.json），在 test 上：
  FN = 死亡但未告警；TP = 死亡且告警；TN = 存活且未告警。
问：漏掉的人长什么样？假设「前 48h 平稳、之后才恶化」——用 48h 内的最后值/趋势、住院天数（死亡者的 LOS ≈ 入 ICU 到死亡的天数）、
   基线评分来检验。只做描述统计 + Mann-Whitney，不重训任何模型；Length_of_stay 只用于尸检描述，不进特征。
产出 D_fn_autopsy.csv（FN/TP/TN 三组中位数与 FN-vs-TP 检验）、D_fn_autopsy.json（摘要）
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from common import load_data, dump
from config import OUT, LABEL_MAIN

X, lab, sp = load_data()
te = sp["test"]
preds = pd.read_csv(OUT / "t1_test_predictions.csv").set_index("RecordID").loc[te]
thr = json.load(open(OUT / "t1_operating_points.json"))["logreg"]
t90 = thr["0.9"]; t95 = thr["0.95"]
y = lab.loc[te, LABEL_MAIN].values; p = preds["p_logreg"].values
flag = p >= t90
grp = np.where((y == 1) & ~flag, "FN", np.where((y == 1) & flag, "TP", np.where((y == 0) & ~flag, "TN", "FP")))
df = X.loc[te].copy(); df["group"] = grp; df["p_logreg"] = p
df["Length_of_stay"] = lab.loc[te, "Length_of_stay"].values; df["SAPS-I"] = lab.loc[te, "SAPS-I"].values; df["SOFA"] = lab.loc[te, "SOFA"].values
df["ICUType"] = lab.loc[te, "ICUType"].map({1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}).values
for v in ["GCS", "HR", "SysABP", "NISysABP", "Creatinine", "BUN", "Lactate", "Platelets", "Temp", "RespRate"]:
    if f"{v}_last" in df and f"{v}_first" in df:
        df[f"{v}_delta"] = df[f"{v}_last"] - df[f"{v}_first"]

summary = {"threshold_sens90": t90, "n_test": len(te), "n_deaths": int(y.sum()),
           "FN": int((grp == "FN").sum()), "TP": int((grp == "TP").sum()), "FP": int((grp == "FP").sum()), "TN": int((grp == "TN").sum())}
fn = df[df.group == "FN"]; tp = df[df.group == "TP"]; tn = df[df.group == "TN"]
summary["FN_p_logreg"] = {"median": float(fn.p_logreg.median()), "q25": float(fn.p_logreg.quantile(.25)), "q75": float(fn.p_logreg.quantile(.75)),
                          "would_be_caught_at_sens95": int((fn.p_logreg >= t95).sum()), "below_half_threshold": int((fn.p_logreg < t90 / 2).sum())}
summary["FN_icu_mix"] = fn.ICUType.value_counts(normalize=True).round(3).to_dict()
summary["TP_icu_mix"] = tp.ICUType.value_counts(normalize=True).round(3).to_dict()
summary["measured_share_FN_vs_TP"] = {v: [float(fn[f"measured_{v}"].mean()), float(tp[f"measured_{v}"].mean())]
                                      for v in ["Lactate", "TroponinT", "Albumin", "Bilirubin", "pH"] if f"measured_{v}" in df}

VARS = ["Age", "SAPS-I", "SOFA", "Length_of_stay", "GCS_first", "GCS_last", "GCS_min", "GCS_delta", "HR_last", "HR_max", "HR_delta",
        "SysABP_min", "NISysABP_min", "NISysABP_delta", "Temp_max", "RespRate_max", "Lactate_last", "Lactate_max", "Creatinine_last", "BUN_last",
        "Platelets_min", "Urine_sum", "MechVent_any", "measured_Lactate", "Glucose_last", "HCO3_min", "pH_min", "PaO2_min", "FiO2_max"]
VARS = [v for v in VARS if v in df]
rows = []
for v in VARS:
    a = fn[v].dropna(); b = tp[v].dropna(); c = tn[v].dropna()
    pv = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) > 2 and len(b) > 2 else np.nan
    rows.append({"variable": v, "FN_median": a.median(), "TP_median": b.median(), "TN_median": c.median(),
                 "FN_n": len(a), "TP_n": len(b), "FN_vs_TP_mannwhitney_p": pv,
                 "FN_closer_to": "TN" if abs(a.median() - c.median()) < abs(a.median() - b.median()) else "TP"})
tab = pd.DataFrame(rows)
# Holm 校正（29 个变量的多重比较）：原始 p 与校正 p 都报，名单按校正后给
pv = tab["FN_vs_TP_mannwhitney_p"].values; order = np.argsort(pv); m_ = np.isfinite(pv).sum(); holm = np.full(len(pv), np.nan)
running = 0.0
for rank, idx in enumerate(order):
    if not np.isfinite(pv[idx]):
        continue
    running = max(running, (m_ - rank) * pv[idx]); holm[idx] = min(1.0, running)
tab["FN_vs_TP_p_holm"] = holm
tab.to_csv(OUT / "D_fn_autopsy.csv", index=False)
summary["n_vars_FN_closer_to_TN"] = int((tab.FN_closer_to == "TN").sum()); summary["n_vars"] = len(tab)
summary["FN_vs_TP_p_uncorrected<0.05"] = tab[tab.FN_vs_TP_mannwhitney_p < 0.05].variable.tolist()
summary["FN_vs_TP_p_holm<0.05"] = tab[tab.FN_vs_TP_p_holm < 0.05].variable.tolist()
dump(summary, OUT / "D_fn_autopsy.json")
print(f"test 死亡 {summary['n_deaths']}：TP {summary['TP']}，FN {summary['FN']}（sens=0.90 阈值 {t90:.3f}）；FP {summary['FP']}，TN {summary['TN']}")
print(f"FN 的 p 中位 {summary['FN_p_logreg']['median']:.3f}；把 sens 提到 0.95 能再捞回 {summary['FN_p_logreg']['would_be_caught_at_sens95']} 个")
print(f"{summary['n_vars_FN_closer_to_TN']}/{summary['n_vars']} 个变量上 FN 的中位数更像存活者(TN)而不是被抓住的死亡者(TP)")
print(f"FN vs TP：未校正 p<0.05 共 {len(summary['FN_vs_TP_p_uncorrected<0.05'])} 个，Holm 校正后 {len(summary['FN_vs_TP_p_holm<0.05'])} 个：{summary['FN_vs_TP_p_holm<0.05']}")
print(tab.round(3).to_string(index=False))
