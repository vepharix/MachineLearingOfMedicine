# 实验 02：多时间窗院内死亡预测基线

## 目的

使用入 ICU 后前 6、12、24、48 小时的原始静态信息、生命体征和实验室检查预测院内死亡。所有时间特征都严格使用预测时点之前的观测，`RecordID` 不进入模型。SAPS-I 与 SOFA 只组成一个不与早期时间窗严格匹配的参照模型，不混入原始时序模型。

## 方法

每个时序变量在各时间窗内生成首次值、末次值、最小值、最大值、均值、标准差、有效时间点数、每小时线性趋势和距末次测量时间。同一分钟的重复测量先取中位数；`-1` 按缺失处理；空参数和未知参数排除；48:00 的记录不进入“前48小时”特征。

数据按 ICU 住院记录固定分为训练集 8,400 例、验证集 1,800 例和测试集 1,800 例，并按院内死亡分层。两类模型都使用仅从训练集学习的中位数填补并保留缺失指示；逻辑回归额外进行标准化。分类阈值仅在验证集上按 Youden J 选择，随后原样应用到测试集。

![完整训练流程](output/report_figures/training_pipeline.png)

逻辑回归先计算加权分数 `z=b₀+Σbᵢxᵢ`，再通过 Sigmoid 函数转成死亡概率，主要表达各特征对风险的加性推动。梯度提升按照 `Fₘ(x)=Fₘ₋₁(x)+ηhₘ(x)` 依次累加小树，每棵树修正前一步剩余的误差，因此可以表达阈值、非线性和特征交互。

![两个模型的数学直观](output/report_figures/model_intuition.png)

## 测试集结果

| 特征集 | 模型 | AUROC | AUPRC | Brier | 灵敏度 | 特异度 |
|---|---|---:|---:|---:|---:|---:|
| 静态信息 | LogisticRegression | 0.711 | 0.261 | 0.114 | 0.551 | 0.744 |
| SAPS-I+SOFA参照（非时间匹配） | LogisticRegression | 0.663 | 0.267 | 0.116 | 0.844 | 0.366 |
| 前6小时原始记录 | LogisticRegression | 0.773 | 0.362 | 0.109 | 0.785 | 0.636 |
| 前6小时原始记录 | HistGradientBoostingClassifier | 0.806 | 0.387 | 0.104 | 0.699 | 0.755 |
| 前12小时原始记录 | LogisticRegression | 0.790 | 0.364 | 0.108 | 0.719 | 0.716 |
| 前12小时原始记录 | HistGradientBoostingClassifier | 0.830 | 0.418 | 0.100 | 0.789 | 0.704 |
| 前24小时原始记录 | LogisticRegression | 0.825 | 0.426 | 0.102 | 0.785 | 0.729 |
| 前24小时原始记录 | HistGradientBoostingClassifier | 0.856 | 0.520 | 0.092 | 0.777 | 0.769 |
| 前48小时原始记录 | LogisticRegression | 0.862 | 0.536 | 0.091 | 0.789 | 0.762 |
| 前48小时原始记录 | HistGradientBoostingClassifier | 0.882 | 0.592 | 0.084 | 0.848 | 0.730 |

![多时间窗模型结果](output/report_figures/model_performance_by_horizon.png)

![最佳模型的混淆矩阵和初步校准](output/report_figures/best_model_diagnostics.png)

这里的单次固定划分用于建立可复现基线，并不等同于外部验证。AUPRC 应结合测试集死亡率理解；Brier 分数越低越好。灵敏度和特异度依赖验证集选出的阈值，不能直接解释为临床决策阈值。

## 边界与后续工作

数据采用仓库文档规定的 `release/outcomes.csv` 与 `release/icu_records/` 布局。数据没有患者级身份、医院或入院日期，因此无法排除同一患者多次住院，也不能执行医院外或时间外验证。后续应增加重复分层交叉验证、置信区间、校准曲线与 ICU 类型/年龄/性别亚组评估，再考虑更复杂的非规则时间序列模型。

汇总结果保存在 `output/metrics.csv`、`output/feature_coverage.csv` 和 `output/run_metadata.json`；患者级划分与预测位于被 Git 忽略的 `output/data/`。

运行 `python plot_report_figures.py` 可以从上述结果重新生成报告图表。
