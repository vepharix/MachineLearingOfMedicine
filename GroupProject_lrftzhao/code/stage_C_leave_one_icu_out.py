"""打法 C：换病房测试（leave-one-ICU-type-out）。
去掉 ICUType one-hot 后，在三类 ICU 的全部病人上训练、第四类上测。看两层：AUC（排序）与 校准（mean predicted vs observed）。
超参沿用 t1_fit_report.json。此实验独立于主切分，是对 0.846 的证伪实验，不替代主报告的随机 test。
产出 C_leave_one_icu_out.csv
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from common import load_data, make_model
from config import OUT, LABEL_MAIN

X, lab, _ = load_data()
fit_report = json.load(open(OUT / "t1_fit_report.json"))
icu_cols = [c for c in X.columns if c.startswith("ICUType_")]
Xn = X.drop(columns=icu_cols)
y = lab[LABEL_MAIN]; icu = lab["ICUType"].map({1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"})
rows = []
for held in ["CCU", "CSRU", "MICU", "SICU"]:
    tr = icu.index[icu != held]; te = icu.index[icu == held]
    for kind in ["logreg", "rf"]:
        m = make_model(kind, **fit_report[kind]["chosen"]).fit(Xn.loc[tr], y[tr])
        p = m.predict_proba(Xn.loc[te])[:, 1]
        rows.append({"held_out_icu": held, "model": kind, "n_test": len(te), "prevalence": y[te].mean(),
                     "mean_predicted": p.mean(), "obs_over_pred": y[te].mean() / p.mean(),
                     "auc_roc": roc_auc_score(y[te], p), "brier": brier_score_loss(y[te], p)})
res = pd.DataFrame(rows); res.to_csv(OUT / "C_leave_one_icu_out.csv", index=False)
print(res.round(3).to_string(index=False))
