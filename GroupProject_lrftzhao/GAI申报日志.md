# GAI 使用申报日志

> 作业要求：交作业时的 Declaration 直接从这里抄。每次用生成式 AI 做了什么、产出落在哪、人工核过什么，随手记一行。

| 日期 | 工具 | 用途 | 产出 | 人工核对 |
|---|---|---|---|---|
| 2026-08-26 | Claude | 数据集探查、方案 v1 | `数据集速览.md`、`分工与方案建议_v1_按工序分工.md` | 数字对 release/ 复算 |
| 2026-08-28～29 | Claude + Grok CLI | 方案 v2/v3 及两轮策略评审 | `分工与方案建议.md`、`grok策略评审*.md` | Grok 全部数字由 Claude 对 release/ 复算 |
| 2026-09-03 | Claude | 量化「出院后死亡」问题，方案升 v3.2；搭 pipeline 代码（Stage 0–2、T1 基线/LogReg/RF、打法 E） | `code/`、`output/` | 清洗计数与 `数据集速览.md` §五逐项核对一致；切分分层率、标签阳性率人工核 |
| 2026-09-03 | Grok CLI（headless） | 审阅 pipeline 代码：泄漏、bug、与方案规范不符处 | 评审记录见 `grok代码评审_20260903.md` | 事实断言逐条对 release/ 复算（见该文件首节表格）；采纳 10 项改进后重跑为第二版 |
| 2026-09-03 | Claude | 按 Grok 评审改第二版：拆 fit/eval、val 锁阈值、ICUType one-hot、SAPS 单变量 logistic 基线、E 读冻结预测；新增决策树/XGBoost、打法 C、T3、亚组、bootstrap CI | `code/`、`output/`、`code/README.md` | `run_all.sh` 全链复跑验证可复现 |
| 2026-09-04 | Claude | Windows 上装包复跑全 pipeline 验证可复现；新写 T2 回归线（`t2_fit.py`/`t2_eval.py`，含 Duan smearing 双读数、Σŷ vs Σy 按 ICU、分段误差、LINE 残差诊断、幸存者敏感性、系数与成组重要度）、打法 D（`stage_D_fn_autopsy.py`）、PDP（`stage6_pdp.py`）、`make_results.py`；更新 `run_all.sh` 跨平台 | `code/`、`output/`、`code/README.md` T2/D/PDP 段 | 特征表与切分与 Mac 产出逐字节比对；T2 超参只在 val 选、test 只评一次；Σ 偏差用 test 真值复核 |
| 2026-09-04 | Grok CLI（headless，只读） | 审阅 T2/D/PDP 新脚本：泄漏、统计方法（smearing、残差诊断、bootstrap、Mann-Whitney）、pandas 3 兼容 | 评审记录见 `grok代码评审_20260904.md`（第一次派活递归被终止，第二次限时 8 分钟截断） | 5 条意见逐条核对全部成立并采纳：系数换算改为 (剩余+1)、smearing 加分层敏感性与前提说明、「中位数型」措辞改几何均值型、Mann-Whitney 加 Holm、results.json 补系数表 |
| 2026-09-04 | Claude + Grok CLI | 「A 与 D 哪个更出彩、B 值不值得做」双线并行讨论（Claude 先独立写、Grok 再跑、对撞合成）；随后写 `stage_A_missingness_only.py` 并跑出 A1/A2 | `code/README.md` 打法 A 段、`分工与方案建议.md` v3.4 注 | A 的 C 只在 val 选、test 一次；解读口径（临床关注度代理、非泄漏）由 Claude 定 |
| 2026-09-04 | Claude | 写 `stage_B_time_value.py`，建 6/12/36h 特征表并跑时间价值曲线 | `code/README.md` 打法 B 段、`output/B_time_value.*` | 同一切分与超参；增益与 CI 核对；偏倚说明由 Claude 写 |
| 2026-09-04 | Claude | 叙事交付物 v0：deck 骨架、problem statement、limitations 初稿、9/21 咨询简报、红队问题库 40 题、成果汇报一页 | `deck骨架_v0.md`、`problem_statement.md`、`limitations_初稿.md`、`咨询简报_0921.md`、`红队问题库.md`、`成果汇报_20260904.md` | 所有数字对 `output/results.json` 核；行动口径与三个 blocker 的倾向由用户在 9/21 后定 |
| 2026-09-03 | Claude | 采纳 Grok 评审第一、二部分：val 冻结阈值、48h 断言、Urine_sum 不填 0、ICUType one-hot、SAPS 单变量 logistic 基线、拆 fit/eval、E 读冻结预测、补决策树/XGBoost、亚组、打法 C、T3 | `code/`、`output/`、`code/README.md` | 评审的事实性断言逐条对 release/ 复算，见 `grok代码评审_20260903.md` 顶部表 |
