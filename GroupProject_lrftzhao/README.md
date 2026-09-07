# GroupProject_lrftzhao：小组项目 pipeline 与叙事材料

> 与仓库根目录的 `EDA/`、`Experiments/` 平行，是 lrftzhao 在同一份 PhysioNet 2012 数据上独立做的一条线：T1 院内死亡分类 + T2 剩余住院日回归 + T3 时间窗消融，外加差异化打法 A–E。2026-09-06 整体搬入本仓库，内容与本地 2026-09-04 状态一致。

## 先看哪几个文件

| 文件 | 内容 |
|---|---|
| [`成果汇报_20260904.md`](成果汇报_20260904.md) | 一页纸总览：做了什么、关键数字、还缺什么 |
| [`code/README.md`](code/README.md) | pipeline 说明 + **全部结果数字**（T1 四模型、阈值表、校准、亚组、打法 A–E、T2 回归） |
| [`分工与方案建议.md`](分工与方案建议.md) | 方案 v3.4：选题、pipeline 架构、评估纪律、差异化打法的取舍与版本历史 |
| [`deck骨架_v0.md`](deck骨架_v0.md) | 8 页展示骨架，每页一句主张 + 证据 + 预期提问 |
| [`problem_statement.md`](problem_statement.md) / [`limitations_初稿.md`](limitations_初稿.md) | 临床问题陈述与局限性 |
| [`红队问题库.md`](红队问题库.md) / [`咨询简报_0921.md`](咨询简报_0921.md) | Q&A 备答 40 题；9/21 第二次咨询要带的问题 |
| [`数据集速览.md`](数据集速览.md) / [`作业要求.md`](作业要求.md) | 数据字段与不可能值清单；作业硬性要求 |
| `grok*.md` / [`GAI申报日志.md`](GAI申报日志.md) | 外部模型评审记录与生成式 AI 使用申报 |

## 目录

```
code/      Stage 0–6 + 打法 A–E + T2 + PDP + make_results.py，./run_all.sh 一键跑通
output/    汇总结果：results.json（slide 取数的唯一出处）、各 *_metrics.json、聚合 CSV、图
```

## 与仓库根目录 `Experiments/02` 的关系

同一数据、同一主任务（院内死亡），两条线独立完成，可互相印证：

| | 本文件夹 | `Experiments/02` |
|---|---|---|
| 切分 | 60/20/20，random_state=2026 | 70/15/15 |
| 特征 | 269 列（first/last/min/max/mean/count + measured_* + ICUType） | 341 列（另有斜率、距末次测量小时数） |
| 模型 | LogReg / 决策树 / RF / XGBoost，bootstrap 95% CI | LogReg / HistGradientBoosting |
| 48h test AUC | XGB 0.857、LogReg 0.847（CI 0.825–0.867） | HGB 0.882、LogReg 0.862 |
| 额外 | 清洗不可能值、校准、亚组、留一 ICU、标签敏感性、FN 尸检、T2 回归 | 6/12/24/48h 多窗口 |

两边的 AUC 不能直接比（切分不同、test 大小不同、单侧无 CI）。

## 复现

1. 把仓库根目录的 `release/`（`outcomes.csv` + `icu_records/`）复制或软链接到本文件夹下，成为 `GroupProject_lrftzhao/release/`（`code/config.py` 以本文件夹为根找数据）。
2. 依赖：pandas、scikit-learn、xgboost、scipy、matplotlib、joblib。
3. 运行：

```bash
cd GroupProject_lrftzhao/code
PY="py -3" ./run_all.sh      # Windows Git Bash；macOS 用 PY=python3
```

约 8 分钟。会重新生成被 `.gitignore` 排除的病人级文件（`features*.csv`、`labels.csv`、`splits/`、`models/`、`*_test_predictions.csv`），这些文件因含 RecordID 级数据不进 Git。
