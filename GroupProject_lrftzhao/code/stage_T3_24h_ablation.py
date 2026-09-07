"""T3：时间窗消融。同一 pipeline、同一切分、同一超参，只把观察窗从 48h 缩到 24h（features_24h.csv），比 test AUC。
回答「第二天的数据买来多少」，也是与 SAPS-I（24h 评分）比较的公平条件。产出 T3_24h_ablation.json"""
import json
from common import load_data, make_model, metrics, bootstrap_auc_ci, dump
from config import OUT, FEATURES, FEATURES_24H, LABEL_MAIN

fit_report = json.load(open(OUT / "t1_fit_report.json"))
out = {}
for name, path in [("48h", FEATURES), ("24h", FEATURES_24H)]:
    X, lab, sp = load_data(path); y = lab[LABEL_MAIN]; tr, te = sp["train"], sp["test"]
    for kind in ["logreg", "rf"]:
        m = make_model(kind, **fit_report[kind]["chosen"]).fit(X.loc[tr], y[tr])
        p = m.predict_proba(X.loc[te])[:, 1]
        out[f"{name}_{kind}"] = {**metrics(y[te], p), "auc_roc_ci95": bootstrap_auc_ci(y[te], p), "n_features": X.shape[1]}
        print(f"{name} {kind:6s} test AUC {out[f'{name}_{kind}']['auc_roc']:.3f}  PR {out[f'{name}_{kind}']['auc_pr']:.3f}")
dump(out, OUT / "T3_24h_ablation.json")
