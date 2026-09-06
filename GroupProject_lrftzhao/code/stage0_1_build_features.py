"""Stage 0 读取与清洗 + Stage 1 特征工程。

产出：
  output/features.csv      每人一行；Age/Gender/Height/Weight + ICUType one-hot
                           + 每个时变量的 first/last/min/max/mean/count + measured_<var> 指示列。
                           不含 Survival、SAPS-I、SOFA 及任何派生。
  output/labels.csv        In-hospital_death, death_inhosp_or_30d_post, Length_of_stay, LOS_remaining,
                           t2_valid, SAPS-I, SOFA, ICUType, Age, Gender（后三列只做分层/打法 C）
  output/data_quality.json 清洗计数 + 标签侧数据质量清单（limitations 素材）
纪律：replace(-1, NaN) 与不可能值处理都在聚合之前；观察窗 ≤ WINDOW_HOURS 有断言。
用法：python stage0_1_build_features.py [--window 24]
"""
import argparse
import json
import numpy as np
import pandas as pd
from config import (RECORDS, OUTCOMES, FEATURES, FEATURES_24H, LABELS, OUT, DESCRIPTORS,
                    POST_DISCHARGE_WINDOW_DAYS, WINDOW_HOURS, LABEL_MAIN, LABEL_E)


def load_long() -> pd.DataFrame:
    files = sorted(RECORDS.glob("*.csv"))
    print(f"读取 {len(files)} 份记录 ...", flush=True)
    parts = []
    for i, f in enumerate(files, 1):
        d = pd.read_csv(f, dtype={"Time": str, "Parameter": str, "Value": float})
        d["RecordID"] = int(f.stem)
        parts.append(d)
        if i % 3000 == 0:
            print(f"  {i}", flush=True)
    long = pd.concat(parts, ignore_index=True)
    long = long[long["Parameter"] != "RecordID"]
    long["minutes"] = long["Time"].str.slice(0, 2).astype(int) * 60 + long["Time"].str.slice(3, 5).astype(int)
    return long


def clean(long: pd.DataFrame, window_hours: int) -> tuple[pd.DataFrame, dict]:
    """纪律 1：-1 → NaN 在一切聚合之前；随后处理数据集速览 §五 列出的不可能值；最后锁观察窗。"""
    v = long["Value"].replace(-1, np.nan)
    p = long["Parameter"]
    log = {}

    def fix(mask, newval, name):
        log[name] = int(mask.sum()); v[mask] = newval

    for bp in ["SysABP", "DiasABP", "MAP", "NISysABP", "NIDiasABP", "NIMAP"]:   # 血压 0 = 导管掉线/冲管
        fix((p == bp) & (v <= 0), np.nan, f"{bp}<=0")
    fix((p == "Weight") & (v <= 0), np.nan, "Weight<=0")
    fix((p == "Height") & (v <= 0), np.nan, "Height<=0")
    fix((p == "Temp") & ((v < 25) | (v > 45)), np.nan, "Temp<25或>45")
    fix((p == "HR") & (v <= 0), np.nan, "HR<=0")
    fix((p == "GCS") & ((v < 3) | (v > 15)), np.nan, "GCS越界")
    m = (p == "Height") & (v < 10) & (v > 0); v[m] = v[m] * 100; log["Height米→cm"] = int(m.sum())
    fix((p == "Height") & (((v >= 10) & (v < 100)) | (v > 230)), np.nan, "Height不可解释")
    m = (p == "pH") & (v > 100); v[m] = v[m] / 100; log["pH/100"] = int(m.sum())
    fix((p == "pH") & ((v < 6.5) | (v > 7.8)), np.nan, "pH<6.5或>7.8")
    fix((p == "K") & (v > 15), np.nan, "K>15")

    long = long.copy(); long["Value"] = v
    before = len(long); long = long.dropna(subset=["Value"]); log["丢弃NaN行"] = before - len(long)
    # 观察窗：这是时间泄漏的最后一道闸，不靠「release 应该已经截过了」
    before = len(long); long = long[long["minutes"] <= window_hours * 60]
    log[f"超出{window_hours}h被丢弃行"] = before - len(long)
    assert long["minutes"].max() <= window_hours * 60
    log["最大时间戳(分钟)"] = int(long["minutes"].max())
    print("清洗记录：", log)
    return long, log


def build_features(long: pd.DataFrame) -> pd.DataFrame:
    desc = (long[long["Parameter"].isin(DESCRIPTORS)].sort_values(["RecordID", "minutes"])
            .groupby(["RecordID", "Parameter"])["Value"].first().unstack())
    # ICUType 无序 → one-hot；原始列不进特征表（留在 labels.csv 做分层与打法 C）
    for k, name in {1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}.items():
        desc[f"ICUType_{name}"] = (desc["ICUType"] == k).astype(int)
    desc = desc.drop(columns=["ICUType"])

    tv = long[~long["Parameter"].isin(["Age", "Gender", "Height", "ICUType"])].sort_values(["RecordID", "minutes"])
    g = tv.groupby(["RecordID", "Parameter"])["Value"]
    agg = pd.concat({"first": g.first(), "last": g.last(), "min": g.min(), "max": g.max(), "mean": g.mean(), "count": g.size()}, axis=1)
    wide = agg.unstack("Parameter")
    wide.columns = [f"{param}_{stat}" for stat, param in wide.columns]
    count_cols = [c for c in wide.columns if c.endswith("_count")]
    wide[count_cols] = wide[count_cols].fillna(0).astype(int)
    for c in count_cols:
        wide[f"measured_{c[:-6]}"] = (wide[c] > 0).astype(int)
    # 没测过的保持 NaN，缺失交给 measured_* 指示列——不许把「没测」写成「无尿」
    if "Urine_count" in wide:
        wide["Urine_sum"] = tv[tv["Parameter"] == "Urine"].groupby("RecordID")["Value"].sum().reindex(wide.index)
        wide.loc[wide["Urine_count"] == 0, "Urine_sum"] = np.nan
    if "MechVent_count" in wide:
        wide["MechVent_any"] = (wide["MechVent_count"] > 0).astype(int)
    feats = desc.join(wide, how="outer"); feats.index.name = "RecordID"
    return feats.sort_index()


def build_labels(long: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    o = pd.read_csv(OUTCOMES).set_index("RecordID").sort_index()
    lab = pd.DataFrame(index=o.index)
    lab[LABEL_MAIN] = o["In-hospital_death"].astype(int)
    los = o["Length_of_stay"].where(o["Length_of_stay"] > 0)
    lab["Length_of_stay"] = los
    lab["LOS_remaining"] = los - 2
    lab["t2_valid"] = (o["Length_of_stay"] > 1).astype(int)             # 剔 LOS=-1 与 LOS=1
    surv = o["Survival"]; post = surv - los
    d = pd.Series(np.nan, index=o.index)
    d[surv <= 0] = 0                                                      # 随访期内未死（Survival=-23 那 1 例无效，按未死）
    ok = (surv > 0) & los.notna()
    d[ok] = (post[ok] <= POST_DISCHARGE_WINDOW_DAYS).astype(float)
    d[lab[LABEL_MAIN] == 1] = 1
    lab[LABEL_E] = d                                                      # NaN = 幸存者但 LOS 缺失且随访期死亡，无法判定
    lab["SAPS-I"] = o["SAPS-I"].where(o["SAPS-I"] >= 0)
    lab["SOFA"] = o["SOFA"].where(o["SOFA"] >= 0)
    desc = long[long["Parameter"].isin(["ICUType", "Age", "Gender"])].groupby(["RecordID", "Parameter"])["Value"].first().unstack()
    lab = lab.join(desc[["ICUType", "Age", "Gender"]])
    dead = o["In-hospital_death"] == 1
    dq = {
        "LOS=-1(缺失)": int((o["Length_of_stay"] == -1).sum()),
        "LOS=1(与48h窗矛盾)": int((o["Length_of_stay"] == 1).sum()),
        "死亡者Survival=1": int((dead & (surv == 1)).sum()),
        "死亡者LOS-Survival>2(死亡日与出院日错位)": int((dead & ((o["Length_of_stay"] - surv) > 2) & (o["Length_of_stay"] > 0)).sum()),
        "存活但Survival<=LOS或Survival<0(硬矛盾)": [int(i) for i in o.index[(~dead) & (((surv > 0) & (surv <= o["Length_of_stay"])) | (surv < -1))]],
        "SAPS-I缺失": int(lab["SAPS-I"].isna().sum()), "SOFA缺失": int(lab["SOFA"].isna().sum()),
        f"{LABEL_E}无法判定": int(lab[LABEL_E].isna().sum()),
    }
    return lab, dq


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--window", type=int, default=WINDOW_HOURS); a = ap.parse_args()
    long, clean_log = clean(load_long(), a.window)
    feats = build_features(long)
    assert not any(("Survival" in c) or ("SAPS" in c) or ("SOFA" in c) for c in feats.columns)
    if a.window == WINDOW_HOURS:
        lab, dq = build_labels(long)
        assert set(feats.index) == set(lab.index)
        feats.to_csv(FEATURES); lab.to_csv(LABELS)
        json.dump({"清洗": clean_log, "标签侧": dq}, open(OUT / "data_quality.json", "w"), ensure_ascii=False, indent=2)
        print(f"features.csv: {feats.shape} ；labels.csv: {lab.shape}")
        print("标签阳性率：院内死亡 %.1f%%；%s %.1f%%（可判定 %d 人）" % (
            lab[LABEL_MAIN].mean() * 100, LABEL_E, lab[LABEL_E].mean() * 100, lab[LABEL_E].notna().sum()))
        print("标签侧数据质量：", dq)
    else:
        path = OUT / f"features_{a.window}h.csv"; feats.to_csv(path); print(f"{path}: {feats.shape}")
