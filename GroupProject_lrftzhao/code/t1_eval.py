"""T1 Stage 5–6 的 eval 半边：读冻结模型与阈值，test 只看这一次。此后为提高 test 数字而改模型是禁止的。
产出 t1_metrics.json（含 bootstrap 95% CI）、t1_threshold_table.csv、t1_calibration.{csv,png}、
     t1_subgroups.csv、t1_logreg_odds_ratios.csv、t1_group_importance.csv、t1_test_predictions.csv
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import load_data, metrics, bootstrap_auc_ci, calibration_table, threshold_table, variable_of, dump
from config import OUT, MODELS, LABEL_MAIN, RANDOM_STATE

X, lab, sp = load_data()
y = lab[LABEL_MAIN]; tr, va, te = sp["train"], sp["val"], sp["test"]
thresholds = json.load(open(OUT / "t1_operating_points.json"))
results = {}

# ---- Stage 3 基线（test）
results["baseline_majority"] = {"auc_roc": 0.5, "note": "无信息参照；常数预测的 AUC-PR = 患病率，不报"}
for score in ["SAPS-I", "SOFA"]:
    s = lab[score]
    has = s[te].notna()
    r = {"auc_roc_complete_subset": float(roc_auc_score(y[te][has], s[te][has])), "n_complete": int(has.sum())}
    # 缺失填 train 中位数后的全 test AUC（与模型同一 n）
    s_imp = s.fillna(s[tr].median())
    r["auc_roc_imputed_full_test"] = float(roc_auc_score(y[te], s_imp[te]))
    # 概率基线：train 上单变量 logistic，才能报 Brier / 校准
    uni = LogisticRegression().fit(s_imp[tr].values.reshape(-1, 1), y[tr])
    p = uni.predict_proba(s_imp[te].values.reshape(-1, 1))[:, 1]
    r["univariate_logistic"] = metrics(y[te], p)
    results[f"baseline_{score}"] = r
print("基线 AUC（test）：", {k: round(v.get("auc_roc_imputed_full_test", v.get("auc_roc", float("nan"))), 3) for k, v in results.items()})

# ---- Stage 5：四个模型 test 一次
preds = pd.DataFrame({"RecordID": te, "y": y[te].values})
thr_tabs, cal_tabs = [], []
for kind in ["logreg", "tree", "rf", "xgb"]:
    m = joblib.load(MODELS / f"t1_{kind}.joblib")
    p = m.predict_proba(X.loc[te])[:, 1]; preds[f"p_{kind}"] = p
    results[kind] = {**metrics(y[te], p), "auc_roc_ci95": bootstrap_auc_ci(y[te], p)}
    t = threshold_table(y[te], p, thresholds[kind]); t.insert(0, "model", kind); thr_tabs.append(t)
    c = calibration_table(y[te], p); c.insert(0, "model", kind); cal_tabs.append(c)
    print(f"{kind:6s} test AUC {results[kind]['auc_roc']:.3f} [{results[kind]['auc_roc_ci95'][0]:.3f}, {results[kind]['auc_roc_ci95'][1]:.3f}]  "
          f"PR {results[kind]['auc_pr']:.3f}  Brier {results[kind]['brier']:.3f}")
pd.concat(thr_tabs).to_csv(OUT / "t1_threshold_table.csv", index=False)
cal = pd.concat(cal_tabs); cal.to_csv(OUT / "t1_calibration.csv", index=False)
preds.to_csv(OUT / "t1_test_predictions.csv", index=False)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
for name, c in cal.groupby("model"):
    ax.plot(c["mean_predicted"], c["observed_rate"], marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Mean predicted risk"); ax.set_ylabel("Observed death rate")
ax.set_title("T1 calibration (test, quantile bins)"); ax.legend(); fig.tight_layout(); fig.savefig(OUT / "t1_calibration.png", dpi=150)

# ---- 亚组：ICU 类型 / 年龄段 / 性别（test 上，LogReg 与 RF）
sub = lab.loc[te].copy(); sub["p_logreg"] = preds["p_logreg"].values; sub["p_rf"] = preds["p_rf"].values
sub["age_band"] = pd.cut(sub["Age"], [0, 45, 65, 80, 200], labels=["<45", "45-64", "65-79", "80+"])
sub["ICUType"] = sub["ICUType"].map({1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"})
sub["Gender"] = sub["Gender"].map({0: "F", 1: "M"})
rows = []
for var in ["ICUType", "age_band", "Gender"]:
    for level, g in sub.groupby(var, observed=True):
        row = {"variable": var, "level": level, "n": len(g), "prevalence": g[LABEL_MAIN].mean()}
        for kind in ["logreg", "rf"]:
            row[f"auc_{kind}"] = roc_auc_score(g[LABEL_MAIN], g[f"p_{kind}"]) if g[LABEL_MAIN].nunique() == 2 else np.nan
            row[f"mean_pred_{kind}"] = g[f"p_{kind}"].mean()
        rows.append(row)
subgroups = pd.DataFrame(rows); subgroups.to_csv(OUT / "t1_subgroups.csv", index=False)
print(subgroups.round(3).to_string(index=False))

# ---- Stage 6 可解释性
lr = joblib.load(MODELS / "t1_logreg.joblib")
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=X.columns)
orr = pd.DataFrame({"variable": [variable_of(c) for c in coef.index], "coef_per_SD": coef, "odds_ratio_per_SD": np.exp(coef)})
orr.sort_values("coef_per_SD", key=abs, ascending=False).to_csv(OUT / "t1_logreg_odds_ratios.csv")
# 成组置换重要度（在 val 上，对 LogReg 与 RF）：同一变量的全部列一起打乱，避免共线列瓜分重要度
rng = np.random.default_rng(RANDOM_STATE)
Xv = X.loc[va]; groups = {}
for c in X.columns:
    groups.setdefault(variable_of(c), []).append(c)
imp_rows = []
for kind in ["logreg", "rf"]:
    m = joblib.load(MODELS / f"t1_{kind}.joblib")
    base = roc_auc_score(y[va], m.predict_proba(Xv)[:, 1])
    for var, cols in groups.items():
        drops = []
        for _ in range(3):
            Xp = Xv.copy(); Xp[cols] = Xp[cols].sample(frac=1, random_state=int(rng.integers(1e9))).values
            drops.append(base - roc_auc_score(y[va], m.predict_proba(Xp)[:, 1]))
        imp_rows.append({"model": kind, "variable": var, "n_cols": len(cols), "auc_drop": float(np.mean(drops))})
imp = pd.DataFrame(imp_rows).sort_values(["model", "auc_drop"], ascending=[True, False])
imp.to_csv(OUT / "t1_group_importance.csv", index=False)
print("成组置换重要度 top8（val 上 AUC 下降）：")
print(imp.groupby("model").head(8).round(4).to_string(index=False))
dump(results, OUT / "t1_metrics.json")
