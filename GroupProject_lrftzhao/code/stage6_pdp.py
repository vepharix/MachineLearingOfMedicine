"""Stage 6 补：Partial Dependence（树模型）。对 T1 的 RF 与 XGBoost，在 val 上画成组重要度前 6 个变量的 PDP。
每个变量取代表列：优先 _sum（Urine 48h 总量）、_last，其次 _min/_max/_mean，再次原名（Age）。PDP 假设特征独立，共线变量的曲线只能定性读。
产出 t1_pdp.png、t1_pdp_values.csv
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence
from common import load_data
from config import OUT, MODELS

X, lab, sp = load_data(); Xv = X.loc[sp["val"]]
imp = pd.read_csv(OUT / "t1_group_importance.csv")
top = imp[imp.model == "rf"].sort_values("auc_drop", ascending=False).variable.tolist()
def rep_col(var):
    for suf in ["_sum", "_last", "_min", "_max", "_mean", ""]:
        if f"{var}{suf}" in X.columns and not var.startswith("ICUType"):
            return f"{var}{suf}"
    return None
cols = [c for c in (rep_col(v) for v in top) if c][:6]
rows = []
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5)); axes = axes.ravel()
for ax, c in zip(axes, cols):
    for kind, ls in [("rf", "-"), ("xgb", "--")]:
        m = joblib.load(MODELS / f"t1_{kind}.joblib")
        lo, hi = np.nanpercentile(Xv[c], [2, 98])          # 列里有 NaN 时 sklearn 自动取分位数会失败，网格手动给
        grid = np.linspace(lo, hi, 25)
        pd_ = partial_dependence(m, Xv, [c], custom_values={c: grid}, kind="average")
        ax.plot(pd_["grid_values"][0], pd_["average"][0], ls, label=kind)
        for g, v in zip(pd_["grid_values"][0], pd_["average"][0]):
            rows.append({"model": kind, "feature": c, "value": g, "partial_dependence": v})
    ax.set_title(c); ax.set_ylabel("P(in-hospital death)"); ax.legend()
fig.suptitle("T1 partial dependence (val, top-6 variables by RF group importance)"); fig.tight_layout(); fig.savefig(OUT / "t1_pdp.png", dpi=150)
pd.DataFrame(rows).to_csv(OUT / "t1_pdp_values.csv", index=False)
print("PDP 变量：", cols, "→", OUT / "t1_pdp.png")
