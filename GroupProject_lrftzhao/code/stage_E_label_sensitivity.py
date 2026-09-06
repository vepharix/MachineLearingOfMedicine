"""打法 E：标签边界敏感性分析。
院内模型 = 冻结的 T1（读 t1_test_predictions.csv，不重训）；只有「30 天标签」那一侧重新拟合（同一超参、同一切分）。
工作点 = t1_operating_points.json 里 sens=0.90 的冻结阈值。
产出 E_label_sensitivity.json、E_fp_to_tp.csv
"""
import json
import joblib
import numpy as np
import pandas as pd
from common import load_data, make_model, metrics, dump
from config import OUT, MODELS, LABEL_MAIN, LABEL_E

X, lab, sp = load_data()
tr, va, te = sp["train"], sp["val"], sp["test"]
preds = pd.read_csv(OUT / "t1_test_predictions.csv").set_index("RecordID")
fit_report = json.load(open(OUT / "t1_fit_report.json"))
thresholds = json.load(open(OUT / "t1_operating_points.json"))

ok = lab[LABEL_E].notna()
tr_ok = [i for i in tr if ok[i]]; te_ok = [i for i in te if ok[i]]
y_in = lab[LABEL_MAIN]; y_e = lab[LABEL_E]
n_post = int(((y_e == 1) & (y_in == 0)).loc[te_ok].sum())
out = {"label_E_definition": "院内死亡 ∪ 出院后≤30天死亡（不是入院30天死亡率）",
       "n_test_judgeable": len(te_ok), "prevalence_inhospital_test": float(y_in[te_ok].mean()),
       "prevalence_E_test": float(y_e[te_ok].mean()), "post_discharge_30d_deaths_in_test": n_post,
       "post_discharge_30d_rate_among_survivors_test": float(n_post / (y_in[te_ok] == 0).sum())}

for kind in ["logreg", "rf"]:
    p_in = preds.loc[te_ok, f"p_{kind}"].values                      # 冻结的院内模型
    m_e = make_model(kind, **fit_report[kind]["chosen"]).fit(X.loc[tr_ok], y_e[tr_ok].astype(int))
    p_e = m_e.predict_proba(X.loc[te_ok])[:, 1]
    thr = thresholds[kind]["0.9"]
    flag = p_in >= thr
    fp = flag & (y_in[te_ok].values == 0); fp_died = fp & (y_e[te_ok].values == 1)
    out[kind] = {
        "frozen_inhospital_model__eval_inhospital": metrics(y_in[te_ok], p_in),
        "frozen_inhospital_model__eval_E": metrics(y_e[te_ok], p_in),
        "refit_E_model__eval_E": metrics(y_e[te_ok], p_e),
        "refit_E_model__eval_inhospital": metrics(y_in[te_ok], p_e),
        "working_point_sens90_frozen": {"threshold": thr, "flagged": int(flag.sum()), "false_positives": int(fp.sum()),
                                        "fp_that_died_within_30d_post": int(fp_died.sum()),
                                        "share_of_fp": float(fp_died.sum() / max(fp.sum(), 1)),
                                        "share_of_post_discharge_deaths_caught": float(fp_died.sum() / max(n_post, 1))},
    }
    if kind == "logreg":
        pd.DataFrame({"RecordID": te_ok, "p_inhospital_frozen": p_in, "y_inhospital": y_in[te_ok].values,
                      "y_E": y_e[te_ok].values.astype(int), "flagged_sens90": flag}).to_csv(OUT / "E_fp_to_tp.csv", index=False)
dump(out, OUT / "E_label_sensitivity.json")
print(f"test 可判定 {out['n_test_judgeable']}；阳性率 院内 {out['prevalence_inhospital_test']:.1%} → E {out['prevalence_E_test']:.1%}；"
      f"出院后 30 天内死亡 {n_post} 人（幸存者的 {out['post_discharge_30d_rate_among_survivors_test']:.1%}）")
for kind in ["logreg", "rf"]:
    r = out[kind]; w = r["working_point_sens90_frozen"]
    print(f"[{kind}] AUC 冻结院内模型：院内评 {r['frozen_inhospital_model__eval_inhospital']['auc_roc']:.3f} / E评 {r['frozen_inhospital_model__eval_E']['auc_roc']:.3f}；"
          f"E标签重训：E评 {r['refit_E_model__eval_E']['auc_roc']:.3f}")
    print(f"        冻结 sens=0.90 阈值 {w['threshold']:.3f}：告警 {w['flagged']}，FP {w['false_positives']}，其中出院后30天内死亡 {w['fp_that_died_within_30d_post']}"
          f"（占 FP {w['share_of_fp']:.1%}；占该类死亡 {w['share_of_post_discharge_deaths_caught']:.0%}）")
