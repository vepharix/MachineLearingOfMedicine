"""全局配置：路径与常量。所有脚本从这里取，不各自硬编码。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # group project/
RELEASE = ROOT / "release"
RECORDS = RELEASE / "icu_records"
OUTCOMES = RELEASE / "outcomes.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
MODELS = OUT / "models"
MODELS.mkdir(exist_ok=True)

FEATURES = OUT / "features.csv"      # Stage 1 冻结产物，每人一行（48h）
FEATURES_24H = OUT / "features_24h.csv"  # T3 消融用，同一代码只改观察窗
LABELS = OUT / "labels.csv"          # 标签 + 分层变量 + 基线评分（SAPS-I/SOFA 只许做基线，不许进特征）
SPLITS = OUT / "splits"              # Stage 2 RecordID 列表

RANDOM_STATE = 2026
TEST_SIZE = 0.20
VAL_SIZE = 0.20                      # 占全体的比例；train = 0.60

WINDOW_HOURS = 48                    # 观察窗；T3 用 24
DESCRIPTORS = ["Age", "Gender", "Height", "ICUType", "Weight"]

# 打法 E：出院后多少天内死亡算「放宽后的阳性」
POST_DISCHARGE_WINDOW_DAYS = 30
LABEL_MAIN = "In-hospital_death"
LABEL_E = "death_inhosp_or_30d_post"   # 院内死亡 ∪ 出院后 ≤30 天死亡；不是「入院 30 天死亡率」

SENS_TARGETS = (0.70, 0.80, 0.90, 0.95)
