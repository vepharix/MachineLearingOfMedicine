"""打法 B：时间价值曲线。同一 pipeline、同一切分、同一超参，观察窗取 6/12/24/36/48h，看 test AUC 随时间怎么爬。
T3 是它的两点版（24 vs 48）。缺的特征表用 stage0_1_build_features.py --window 现建。
⚠️ 条件存活偏倚：数据集只含住满 48h 的人，6h 时点的 AUC 是「最终住满 48h 的人里前 6h 的信息量」，不是「入院 6 小时就能预测」。
产出 B_time_value.json、B_time_value.csv、B_time_value.png
"""
import json
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_data, make_model, metrics, bootstrap_auc_ci, dump
from config import OUT, FEATURES, LABEL_MAIN

WINDOWS = [6, 12, 24, 36, 48]
fit_report = json.load(open(OUT / "t1_fit_report.json"))
rows = []
for w in WINDOWS:
    path = FEATURES if w == 48 else OUT / f"features_{w}h.csv"
    if not path.exists():
        print(f"建 {w}h 特征表 ...", flush=True)
        subprocess.run([sys.executable, "stage0_1_build_features.py", "--window", str(w)], check=True)
    X, lab, sp = load_data(path); y = lab[LABEL_MAIN]; tr, te = sp["train"], sp["test"]
    meas = [c for c in X.columns if c.startswith("measured_")]
    row = {"window_h": w, "n_features": X.shape[1], "mean_variables_measured_per_patient": float(X[meas].sum(axis=1).mean()),
           "share_with_lactate": float(X["measured_Lactate"].mean()) if "measured_Lactate" in X else np.nan,
           "share_with_any_lab": float((X[[c for c in meas if c.split("_", 1)[1] in ("BUN", "Creatinine", "Na", "K", "HCT", "Platelets", "WBC")]].sum(axis=1) > 0).mean())}
    for kind in ["logreg", "rf"]:
        m = make_model(kind, **fit_report[kind]["chosen"]).fit(X.loc[tr], y[tr])
        p = m.predict_proba(X.loc[te])[:, 1]; r = metrics(y[te], p); ci = bootstrap_auc_ci(y[te], p)
        row[f"auc_{kind}"] = r["auc_roc"]; row[f"auc_{kind}_lo"] = ci[0]; row[f"auc_{kind}_hi"] = ci[1]; row[f"aucpr_{kind}"] = r["auc_pr"]
    rows.append(row)
    print(f"{w:2d}h: 每人平均测过 {row['mean_variables_measured_per_patient']:.1f} 个变量，有乳酸 {row['share_with_lactate']:.0%}；"
          f"logreg AUC {row['auc_logreg']:.3f} [{row['auc_logreg_lo']:.3f}, {row['auc_logreg_hi']:.3f}]  rf {row['auc_rf']:.3f}", flush=True)
tab = pd.DataFrame(rows); tab.to_csv(OUT / "B_time_value.csv", index=False)
gain = {f"{a}h→{b}h_logreg": float(tab.loc[tab.window_h == b, "auc_logreg"].iloc[0] - tab.loc[tab.window_h == a, "auc_logreg"].iloc[0])
        for a, b in zip(WINDOWS[:-1], WINDOWS[1:])}
out = {"windows_h": WINDOWS, "rows": rows, "auc_gain_per_step_logreg": gain,
       "caveat": "数据集只含住满 48h 的病人：早时点 AUC 是条件于「最终住满 48h」的人群，不等于入院早期即可部署的表现",
       "note": "同一切分、同一超参（t1_fit_report.json），只改观察窗；T3 是本曲线的 24/48 两点"}
dump(out, OUT / "B_time_value.json")
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for kind, mk in [("logreg", "o-"), ("rf", "s--")]:
    ax.errorbar(tab.window_h, tab[f"auc_{kind}"], yerr=[tab[f"auc_{kind}"] - tab[f"auc_{kind}_lo"], tab[f"auc_{kind}_hi"] - tab[f"auc_{kind}"]],
                fmt=mk, capsize=3, label=kind)
ax.axhline(json.load(open(OUT / "t1_metrics.json"))["baseline_SAPS-I"]["auc_roc_imputed_full_test"], color="grey", ls=":", label="SAPS-I (24h score)")
ax.set(xlabel="observation window (hours after ICU admission)", ylabel="test AUC-ROC", xticks=WINDOWS,
       title="B: what each extra 6–12 h of data buys (cohort = stays ≥48 h)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "B_time_value.png", dpi=150)
print("逐段增益（logreg）：", {k: round(v, 3) for k, v in gain.items()})
