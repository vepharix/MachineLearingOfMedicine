# 实验路线

本目录把同一份 ICU 前 48 小时数据拆成不同研究问题。实验 02 和后续的数据质量分支属于院内死亡二分类：目标是预测一个已定义的二元结局，并重点比较清洗、填补、时间窗和模型。实验 04 的住院时长回归使用连续结局，回答“预计住院多少天”；实验 05 的聚类没有监督标签，回答“早期临床状态是否形成可复现的患者亚群”。因此，回归和聚类不是死亡二分类后续方案中的另一种填补方法，而是使用同一数据源开展的其他实验。

编号 03 保留给死亡数据质量、填补与稳健性研究；04—06 分别承载总住院时长回归、患者表型聚类和 48 小时后的剩余住院时间，避免把不同目标的模型结果混在同一实验中。

## 已实现实验

- [`01_length_of_stay_by_mortality/`](01_length_of_stay_by_mortality/)：按院内死亡分组，描述严重程度评分与住院时长的关系。
- [`02_mortality_prediction_baseline/`](02_mortality_prediction_baseline/)：多时间窗院内死亡二分类基线。
- [`03_mortality_data_quality/`](03_mortality_data_quality/)：死亡模型的清洗、填补、校准、调参、解释和跨 ICU 压力测试。
- [`04_length_of_stay_regression/`](04_length_of_stay_regression/)：多时间窗住院时长回归。
- [`05_patient_phenotype_clustering/`](05_patient_phenotype_clustering/)：24 小时临床状态的无监督患者分型。
- [`06_remaining_length_of_stay/`](06_remaining_length_of_stay/)：48 小时后的剩余住院时间、队列床日估计与线性/非线性诊断。

患者级特征、数据划分、预测和簇标签位于各实验的 `output/data/`，由 Git 忽略。仓库只提交可公开复核的代码、聚合指标、图像和运行元数据；原始数据始终只读。
