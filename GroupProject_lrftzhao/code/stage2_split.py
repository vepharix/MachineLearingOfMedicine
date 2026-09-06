"""Stage 2：唯一的一次切分。train/val/test = 60/20/20，stratify=In-hospital_death，random_state=2026。
产出 output/splits/{train,val,test}_ids.csv。T2 的有效子集 = 这三份 ∩ t2_valid。"""
import pandas as pd
from sklearn.model_selection import train_test_split
from config import LABELS, SPLITS, RANDOM_STATE, TEST_SIZE, VAL_SIZE

lab = pd.read_csv(LABELS).set_index("RecordID")
y = lab["In-hospital_death"]
ids = lab.index.to_series()
trainval, test = train_test_split(ids, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
train, val = train_test_split(trainval, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y[trainval], random_state=RANDOM_STATE)
SPLITS.mkdir(exist_ok=True)
for name, s in [("train", train), ("val", val), ("test", test)]:
    pd.Series(sorted(s), name="RecordID").to_csv(SPLITS / f"{name}_ids.csv", index=False)
    print(f"{name:5s} n={len(s):5d}  院内死亡率 {y[s].mean():.3%}  T2有效 {lab.loc[s, 't2_valid'].sum()}")
