"""评估与建模的公共件：读数据、模型管道、指标、阈值、bootstrap 区间。"""
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve, mean_absolute_error, r2_score
from sklearn.calibration import calibration_curve
from config import FEATURES, LABELS, SPLITS, RANDOM_STATE


def load_data(features_path=FEATURES):
    X = pd.read_csv(features_path).set_index("RecordID")
    lab = pd.read_csv(LABELS).set_index("RecordID")
    splits = {k: pd.read_csv(SPLITS / f"{k}_ids.csv")["RecordID"].values for k in ["train", "val", "test"]}
    return X, lab, splits


def make_model(kind: str, **kw):
    """插补与标准化只在 train 上 fit（Pipeline 保证）。二元/one-hot 列用 median 插补等价于众数。"""
    imp = ("impute", SimpleImputer(strategy="median"))
    if kind == "logreg":
        return Pipeline([imp, ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=kw.get("C", 0.01), max_iter=3000))])
    if kind == "tree":
        return Pipeline([imp, ("clf", DecisionTreeClassifier(max_depth=kw.get("max_depth", 5), min_samples_leaf=20, random_state=RANDOM_STATE))])
    if kind == "rf":
        return Pipeline([imp, ("clf", RandomForestClassifier(n_estimators=kw.get("n_estimators", 400), min_samples_leaf=kw.get("min_samples_leaf", 5),
                                                             n_jobs=-1, random_state=RANDOM_STATE))])
    if kind == "xgb":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(n_estimators=kw.get("n_estimators", 300), max_depth=kw.get("max_depth", 3),
                                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
        except Exception:
            # xgboost 需要 libomp；装不上时退到 sklearn 自带的直方图梯度提升（同类算法，原生处理 NaN）
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(max_iter=kw.get("n_estimators", 300), max_depth=kw.get("max_depth", 3),
                                                 learning_rate=0.05, random_state=RANDOM_STATE)
        return Pipeline([imp, ("clf", clf)])
    raise ValueError(kind)


def boosting_backend() -> str:
    try:
        import xgboost  # noqa
        return "xgboost"
    except Exception:
        return "sklearn HistGradientBoosting (xgboost 未能加载 libomp)"


def metrics(y, p) -> dict:
    return {"auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "prevalence": float(np.mean(y)), "n": int(len(y))}


def bootstrap_auc_ci(y, p, n_boot=1000, seed=RANDOM_STATE):
    y = np.asarray(y); p = np.asarray(p); rng = np.random.default_rng(seed); aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    frac, mean_pred = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac})


def pick_thresholds(y_val, p_val, sens_targets) -> dict:
    """在 val 上按目标 sensitivity 反推阈值并冻结。工作点只许在 val 上定。"""
    fpr, tpr, thr = roc_curve(y_val, p_val)
    return {str(s): float(thr[int(np.argmax(tpr >= s))]) for s in sens_targets}


def threshold_table(y, p, thresholds: dict) -> pd.DataFrame:
    """用冻结阈值在 test 上报：sensitivity、每 100 人告警数、PPV、漏掉的死亡数。"""
    y = np.asarray(y); p = np.asarray(p); rows = []
    for s, t in thresholds.items():
        flag = p >= t
        tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum()); fn = int((~flag & (y == 1)).sum())
        rows.append({"target_sens_on_val": float(s), "threshold": t, "test_sensitivity": tp / max(tp + fn, 1),
                     "alerts_per_100": 100 * flag.mean(), "ppv": tp / max(tp + fp, 1), "missed_deaths": fn, "false_alarms": fp})
    return pd.DataFrame(rows)


def variable_of(col: str) -> str:
    """特征列 → 所属变量（成组解读用）。GCS_last → GCS；measured_GCS → GCS；ICUType_MICU → ICUType。"""
    if col.startswith("measured_"):
        return col[len("measured_"):]
    if col.startswith("ICUType_"):
        return "ICUType"
    for suf in ["_first", "_last", "_min", "_max", "_mean", "_count", "_sum", "_any"]:
        if col.endswith(suf):
            return col[: -len(suf)]
    return col


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------- T2 回归公共件 ----------------
def make_regressor(kind: str, **kw):
    """T2 的模型管道。插补/标准化只在 train 上 fit。目标是 log1p(剩余住院日)，由调用方负责变换与反变换。"""
    imp = ("impute", SimpleImputer(strategy="median")); sc = ("scale", StandardScaler())
    if kind == "ols":
        return Pipeline([imp, sc, ("reg", LinearRegression())])
    if kind == "ridge":
        return Pipeline([imp, sc, ("reg", Ridge(alpha=kw.get("alpha", 10.0)))])
    if kind == "lasso":
        return Pipeline([imp, sc, ("reg", Lasso(alpha=kw.get("alpha", 0.01), max_iter=50000))])
    if kind == "rf_reg":
        return Pipeline([imp, ("reg", RandomForestRegressor(n_estimators=400, min_samples_leaf=kw.get("min_samples_leaf", 5),
                                                            n_jobs=-1, random_state=RANDOM_STATE))])
    raise ValueError(kind)


def to_days(pred_log, smear: float = 1.0):
    """log1p 尺度的预测 → 天，clip 到 ≥0。
    smear=1：直接 expm1，得到的是条件几何均值型预测（log 残差对称时≈条件中位数；本例 log 残差偏度 −0.04），给个体读数。
    smear=S：Duan smearing，S = train 残差的 mean(exp(r))，把几何均值抬回算术均值（给 Σŷ 容量主张）。
    log1p(y) = f + r ⇒ E[y+1] = exp(f)·E[exp(r)] ⇒ ŷ = exp(f)·S − 1。"""
    return np.clip(np.exp(np.asarray(pred_log)) * smear - 1, 0, None)


def smearing_factor(resid_log) -> float:
    """Duan (1983) smearing：对数尺度训练残差的 mean(exp(r))。"""
    return float(np.mean(np.exp(np.asarray(resid_log))))


def reg_metrics(y_days, yhat_days) -> dict:
    """T2 指标一律在天的尺度上报。sum_bias_pct = (Σŷ − Σy)/Σy，是容量主张的主数。"""
    y = np.asarray(y_days, float); yh = np.asarray(yhat_days, float); e = yh - y
    return {"mae_days": float(np.mean(np.abs(e))), "rmse_days": float(np.sqrt(np.mean(e ** 2))),
            "median_ae_days": float(np.median(np.abs(e))), "mean_bias_days": float(np.mean(e)),
            "r2_days": float(r2_score(y, yh)), "sum_true_days": float(y.sum()), "sum_pred_days": float(yh.sum()),
            "sum_bias_pct": float(100 * (yh.sum() - y.sum()) / y.sum()), "n": int(len(y))}


def bootstrap_mae_ci(y_days, yhat_days, n_boot=1000, seed=RANDOM_STATE):
    y = np.asarray(y_days, float); yh = np.asarray(yhat_days, float); rng = np.random.default_rng(seed)
    maes = [np.mean(np.abs(yh[i] - y[i])) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))]
