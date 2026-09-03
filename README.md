# ICU 前 48 小时数据探索

这个项目分析 ICU 患者入院后前 48 小时的不规则临床时序数据，当前包含数据质量检查、描述性统计、缺失与覆盖率分析，以及 SAPS-I、SOFA 与患者结局的相关性分析。

## 内容边界

- `EDA/` 和后续的 `Experiments/` 属于个人项目，包括分析代码、实验记录和可复现结果。
- `CourseMaterials/` 只用于保存课程教案、讲义或任务要求，不存放项目实现。
- `release/README.md` 是数据字段说明，不是课程教案；原始数据文件不会上传到 GitHub。

项目结论以本仓库中的代码和实验结果为准，不把课程材料中的示例直接作为项目实验结果。

## 数据

原始数据来自 [PhysioNet/Computing in Cardiology Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)。数据包含 12,000 次成人 ICU 住院记录，每次记录最多包含 42 类静态信息和时序测量。

原始患者记录和结局文件不进入 Git 仓库。运行分析前，请按以下结构准备数据：

```text
release/
  README.md
  outcomes.csv
  icu_records/
    <RecordID>.csv
```

字段、单位和缺失值约定见 [`release/README.md`](release/README.md)。请同时遵守原始数据集的许可和使用要求。

## 已有分析

- [`EDA/REPORT.md`](EDA/REPORT.md)：数据完整性、缺失情况、变量覆盖率、分布和院内死亡组间比较。
- [`EDA/CORRELATION_REPORT.md`](EDA/CORRELATION_REPORT.md)：SAPS-I、SOFA、死亡时间、住院时间和院内死亡的 Pearson/Spearman 相关性。
- `EDA/output/Distribution/`：EDA 汇总表和图像。
- `EDA/output/correlation/`：相关性矩阵、六组散点图和数值结果。

分析结果仅用于数据研究，不构成临床判断或医疗建议。

## 运行

推荐使用 Python 3.11 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python EDA/eda.py
python EDA/correlation_analysis.py
```

脚本只读取 `release/` 中的原始数据，不会修改它们。
