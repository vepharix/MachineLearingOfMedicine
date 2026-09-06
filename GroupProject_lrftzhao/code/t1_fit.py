"""T1 Stage 3–4 的 fit 半边：只看 train/val，不碰 test。
- 四个模型在 val 上选超参：LogReg(C) → 决策树(max_depth) → RF(min_samples_leaf) → XGBoost(max_depth)
- 冻结：模型（joblib）、val 上按目标 sensitivity 反推的阈值、val 报告、决策树深度-过拟合曲线
产出 output/models/*.joblib、t1_operating_points.json、t1_fit_report.json、t1_tree_depth_curve.csv
"""
import joblib
import numpy as np
import pandas as pd
from common import load_data, make_model, metrics, pick_thresholds, dump, boosting_backend
from config import OUT, MODELS, LABEL_MAIN, SENS_TARGETS

X, lab, sp = load_data()
y = lab[LABEL_MAIN]; tr, va = sp["train"], sp["val"]
report = {"_boosting_backend": boosting_backend()}; thresholds = {}
print("boosting 后端：", report["_boosting_backend"])

GRIDS = {
    "logreg": ("C", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]),
    "tree": ("max_depth", [2, 3, 4, 5, 6, 8, 12, None]),
    "rf": ("min_samples_leaf", [1, 5, 20, 50]),
    "xgb": ("max_depth", [2, 3, 4, 6]),
}
depth_curve = []
for kind, (param, grid) in GRIDS.items():
    best = None
    for g in grid:
        m = make_model(kind, **{param: g}).fit(X.loc[tr], y[tr])
        auc_tr = metrics(y[tr], m.predict_proba(X.loc[tr])[:, 1])["auc_roc"]
        auc_va = metrics(y[va], m.predict_proba(X.loc[va])[:, 1])["auc_roc"]
        print(f"  {kind:6s} {param}={g}: train AUC {auc_tr:.4f}  val AUC {auc_va:.4f}")
        if kind == "tree":
            depth_curve.append({"max_depth": g, "train_auc": auc_tr, "val_auc": auc_va})
        if best is None or auc_va > best[1]:
            best = (g, auc_va, m)
    g, auc_va, m = best
    p_va = m.predict_proba(X.loc[va])[:, 1]
    report[kind] = {"chosen": {param: g}, "val": metrics(y[va], p_va),
                    "grid_edge": bool(g == grid[0] or g == grid[-1])}
    thresholds[kind] = pick_thresholds(y[va], p_va, SENS_TARGETS)
    joblib.dump(m, MODELS / f"t1_{kind}.joblib")
    print(f"{kind}: 选 {param}={g}，val AUC {auc_va:.4f}" + ("  ⚠️ 落在网格边界" if report[kind]["grid_edge"] else ""))

pd.DataFrame(depth_curve).to_csv(OUT / "t1_tree_depth_curve.csv", index=False)
dump(report, OUT / "t1_fit_report.json"); dump(thresholds, OUT / "t1_operating_points.json")
print("模型与工作点已冻结 →", MODELS, OUT / "t1_operating_points.json")
