"""打法 A：纯缺失模型。模型不看任何数值，只看 48h 内「开了哪些检查 / 测了多少次」，AUC 能到多少。
两个变体（都不含年龄/性别/ICUType，避免把病房流程混进来）：
  A1 measured_* 指示列（是否测过，0/1）
  A2 A1 + *_count（测了几次）+ MechVent_any
纪律与 T1 相同：LogReg 的 C 在 val 上选，test 只评一次；同一切分。
解读口径：48h 内开单模式是「临床关注度」的代理，不是生理信号——濒死才查乳酸的循环性是这个实验的一部分，不是泄漏（48h 时点已知）。
产出 A_missingness_only.json、A_missingness_odds_ratios.csv
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import load_data, make_model, metrics, bootstrap_auc_ci, dump
from config import OUT, LABEL_MAIN

X, lab, sp = load_data()
y = lab[LABEL_MAIN]; tr, va, te = sp["train"], sp["val"], sp["test"]
meas = [c for c in X.columns if c.startswith("measured_")]
cnt = [c for c in X.columns if c.endswith("_count")]
variants = {"A1_measured_only": meas, "A2_measured_plus_counts": meas + cnt + (["MechVent_any"] if "MechVent_any" in X else [])}
t1 = json.load(open(OUT / "t1_metrics.json"))
out = {"reference_test_auc": {"T1_logreg_full_features": t1["logreg"]["auc_roc"], "SAPS-I": t1["baseline_SAPS-I"]["auc_roc_imputed_full_test"]},
       "n_measured_indicators": len(meas), "n_count_columns": len(cnt),
       "caveat": "48h 内的开单模式部分反映临床团队已察觉的恶化（濒死才查乳酸），是临床关注度的代理而非生理信号；48h 时点已知，非泄漏"}
preds = pd.read_csv(OUT / "t1_test_predictions.csv").set_index("RecordID").loc[te]
thr90 = json.load(open(OUT / "t1_operating_points.json"))["logreg"]["0.9"]
fn_ids = [i for i in te if y[i] == 1 and preds.loc[i, "p_logreg"] < thr90]
tp_ids = [i for i in te if y[i] == 1 and preds.loc[i, "p_logreg"] >= thr90]

for name, cols in variants.items():
    best = None
    for C in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]:
        m = make_model("logreg", C=C).fit(X.loc[tr, cols], y[tr])
        auc_va = roc_auc_score(y[va], m.predict_proba(X.loc[va, cols])[:, 1])
        if best is None or auc_va > best[1]:
            best = (C, auc_va, m)
    C, auc_va, m = best
    p = m.predict_proba(X.loc[te, cols])[:, 1]
    r = {"n_features": len(cols), "chosen_C": C, "val_auc": float(auc_va), "test": {**metrics(y[te], p), "auc_roc_ci95": bootstrap_auc_ci(y[te], p)}}
    # 与 D 的衔接：T1 漏掉的 44 人在纯缺失模型上的分数
    pa = pd.Series(p, index=te)
    r["link_to_D"] = {"FN_of_T1_median_score": float(pa[fn_ids].median()), "TP_of_T1_median_score": float(pa[tp_ids].median()),
                      "survivors_median_score": float(pa[[i for i in te if y[i] == 0]].median())}
    out[name] = r
    print(f"{name}: {len(cols)} 列, C={C}, val AUC {auc_va:.3f}, test AUC {r['test']['auc_roc']:.3f} [{r['test']['auc_roc_ci95'][0]:.3f}, {r['test']['auc_roc_ci95'][1]:.3f}]  "
          f"PR {r['test']['auc_pr']:.3f}  | T1 漏掉者在此模型的中位分 {r['link_to_D']['FN_of_T1_median_score']:.3f} vs 抓住者 {r['link_to_D']['TP_of_T1_median_score']:.3f} vs 存活者 {r['link_to_D']['survivors_median_score']:.3f}")
    if name == "A1_measured_only":
        coef = pd.Series(m.named_steps["clf"].coef_[0], index=cols)
        base_rate = X.loc[tr, cols].mean()
        orr = pd.DataFrame({"feature": cols, "share_measured_train": base_rate.values, "coef_per_SD": coef.values, "odds_ratio_per_SD": np.exp(coef.values)})
        orr.sort_values("coef_per_SD", key=abs, ascending=False).to_csv(OUT / "A_missingness_odds_ratios.csv", index=False)
        print("A1 |系数| top8（每 SD 的 odds ratio；>1 = 测过这项→更可能死亡）：")
        print(orr.sort_values("coef_per_SD", key=abs, ascending=False).head(8).round(3).to_string(index=False))
print(f"参照：T1 全特征 LogReg test AUC {out['reference_test_auc']['T1_logreg_full_features']:.3f}；SAPS-I {out['reference_test_auc']['SAPS-I']:.3f}")
dump(out, OUT / "A_missingness_only.json")
