# MachineLearingOfMedicine

## ICU 前 48 小时数据探索

这个项目分析 ICU 患者入院后前 48 小时的不规则临床时序数据，包含数据质量检查与 EDA、院内死亡预测及稳健性验证、住院时长与剩余床日回归，以及 24 小时患者表型聚类。

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

- [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md)：从全量 EDA、缺失与异常处理到多时间窗死亡预测训练、指标解释、局限性和后续实验方案的完整研究报告。
- [`PRESENTATION_PLAN.md`](PRESENTATION_PLAN.md)：结合课程评分要求和当前实验结果整理的汇报主题、八页页序、时间取舍与答辩问题。
- [`EDA/REPORT.md`](EDA/REPORT.md)：数据完整性、缺失情况、变量覆盖率、分布和院内死亡组间比较。
- [`EDA/CORRELATION_REPORT.md`](EDA/CORRELATION_REPORT.md)：SAPS-I、SOFA、死亡时间、住院时间和院内死亡的 Pearson/Spearman 相关性。
- `EDA/output/Distribution/`：EDA 汇总表和图像。
- `EDA/output/correlation/`：相关性矩阵、六组散点图和数值结果。
- [`Experiments/01_length_of_stay_by_mortality/`](Experiments/01_length_of_stay_by_mortality/)：第一步实验，按是否院内死亡拆分，比较 SAPS-I、SOFA 与住院时间的关系。
- [`Experiments/02_mortality_prediction_baseline/`](Experiments/02_mortality_prediction_baseline/)：使用前 6、12、24、48 小时原始记录建立院内死亡预测基线，并与静态信息和 SAPS-I/SOFA 参照模型比较。
- [`Experiments/03_mortality_data_quality/`](Experiments/03_mortality_data_quality/)：在固定死亡标签和住院记录划分下，系统比较缺失填补、异常值清洗、标签敏感性、校准和亚组稳定性，并补充目标灵敏度工作点、再校准与留一 ICU 压力测试。
- [`Experiments/04_length_of_stay_regression/`](Experiments/04_length_of_stay_regression/)：使用前 6、12、24、48 小时记录预测完整住院时长，比较岭回归和梯度提升，并保留固定测试集与 bootstrap 置信区间。
- [`Experiments/05_patient_phenotype_clustering/`](Experiments/05_patient_phenotype_clustering/)：使用前 24 小时生理状态进行 PCA 与 K-means 分型，并用留出集轮廓系数和子样本 ARI 检查分离度与稳定性。
- [`Experiments/06_remaining_length_of_stay/`](Experiments/06_remaining_length_of_stay/)：预测 48 小时后的剩余住院时间，分别选择个体 MAE 模型和经 smearing 校正的队列总床日模型。

分析结果仅用于数据研究，不构成临床判断或医疗建议。

## 运行

推荐使用 Python 3.11 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python EDA/eda.py
python EDA/correlation_analysis.py
python Experiments/01_length_of_stay_by_mortality/run_experiment.py
python Experiments/02_mortality_prediction_baseline/run_experiment.py
python Experiments/03_mortality_data_quality/run_experiment.py
python Experiments/03_mortality_data_quality/analyze_selected_models.py
python Experiments/03_mortality_data_quality/analyze_operating_points.py
python Experiments/03_mortality_data_quality/plot_results.py
python Experiments/04_length_of_stay_regression/run_experiment.py
python Experiments/05_patient_phenotype_clustering/run_experiment.py
python Experiments/06_remaining_length_of_stay/run_experiment.py
```

脚本只读取 `release/` 中的原始数据，不会修改它们。
