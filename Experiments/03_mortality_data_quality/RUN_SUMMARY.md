# 实验 03：院内死亡模型的数据填补与清洗稳健性

## 研究问题

本实验不改变院内死亡二分类目标、患者划分、时间窗或模型家族，只比较缺失填补、缺失指示、宽松异常值清洗和训练集分位数截断。24 小时训练集五折交叉验证用于筛选，验证集确认候选，测试集在方案冻结后评价。

## 设计

- 原始基线：只处理 `-1`、空参数、同分钟重复和时间边界。
- 宽松清洗：只把明显违反单位或技术范围的值改为缺失，不把临床异常值当作错误。
- 填补：中位数、均值、是否加入缺失指示，以及梯度提升的原生缺失分支。
- Winsorization：只由当前训练折估计 0.5% 和 99.5% 分位数。
- 主要筛选指标：AUPRC；Brier 和 AUROC 作为共同约束。

## 24 小时训练集五折结果

| model | preprocess | auroc_mean | auprc_mean | brier_mean |
| --- | --- | --- | --- | --- |
| hist_gradient_boosting | raw_mean_indicator | 0.8370 | 0.4697 | 0.0961 |
| hist_gradient_boosting | clean_median_indicator | 0.8362 | 0.4675 | 0.0964 |
| hist_gradient_boosting | raw_winsor_median_indicator | 0.8362 | 0.4673 | 0.0962 |
| hist_gradient_boosting | raw_median | 0.8354 | 0.4672 | 0.0964 |
| hist_gradient_boosting | raw_median_indicator | 0.8354 | 0.4672 | 0.0964 |
| hist_gradient_boosting | raw_native_missing | 0.8368 | 0.4668 | 0.0962 |
| hist_gradient_boosting | clean_winsor_median_indicator | 0.8345 | 0.4650 | 0.0966 |
| logistic | clean_winsor_median_indicator | 0.8099 | 0.4282 | 0.1034 |
| logistic | raw_winsor_median_indicator | 0.8090 | 0.4276 | 0.1036 |
| logistic | raw_median | 0.8063 | 0.4211 | 0.1031 |
| logistic | clean_median_indicator | 0.8079 | 0.4175 | 0.1040 |
| logistic | raw_median_indicator | 0.8065 | 0.4141 | 0.1044 |
| logistic | raw_mean_indicator | 0.8064 | 0.4141 | 0.1044 |

## 验证集选定方案

- 逻辑回归：`raw_winsor_median_indicator`
- 梯度提升：`raw_mean_indicator`

候选方案在 6、12、24、48 小时验证集上的完整结果见 `output/validation_candidates.csv`。

## 冻结后的测试集结果

| model | preprocess | horizon_hours | auroc | auprc | brier | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | raw_winsor_median_indicator | 6 | 0.7694 | 0.3604 | 0.1088 | 0.6836 | 0.6930 |
| logistic | raw_winsor_median_indicator | 12 | 0.7952 | 0.3833 | 0.1057 | 0.7148 | 0.7545 |
| logistic | raw_winsor_median_indicator | 24 | 0.8332 | 0.4514 | 0.0998 | 0.8281 | 0.7118 |
| logistic | raw_winsor_median_indicator | 48 | 0.8575 | 0.5423 | 0.0916 | 0.8164 | 0.7494 |
| hist_gradient_boosting | raw_mean_indicator | 6 | 0.8039 | 0.4070 | 0.1030 | 0.8477 | 0.6088 |
| hist_gradient_boosting | raw_mean_indicator | 12 | 0.8253 | 0.4234 | 0.1005 | 0.7148 | 0.7746 |
| hist_gradient_boosting | raw_mean_indicator | 24 | 0.8463 | 0.4648 | 0.0966 | 0.7695 | 0.7623 |
| hist_gradient_boosting | raw_mean_indicator | 48 | 0.8680 | 0.5674 | 0.0868 | 0.8047 | 0.7843 |

测试集已经在实验 02 中用于建立基线，因此这里称为“基线后重新锁定的确认集”，不把它描述成研究全过程中从未查看过的盲测集。置信区间、标签敏感性和清洗数量分别保存在 `output/test_metrics.csv`、`output/label_sensitivity_metrics.csv` 和 `output/cleaning_counts.csv`。
