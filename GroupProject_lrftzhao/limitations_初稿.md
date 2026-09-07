# Limitations & Recommendations（初稿 v0 · 2026-09-04）

> 对应 deck 第 8 页。英文是 slide/报告用的正文（每条一句主张 + 一句证据，能直接上页）；中文注是讲的时候怎么展开、TA 追问怎么接。数字来自 `output/results.json`。
> 结构：先建议（能做什么），再局限（三条必写 + 两条数据侧 + 一条验证侧），最后一句收口。

---

## Recommendations — what this work supports

1. **Pilot as a ranking tool for senior review at the 48-hour mark.** Regularised logistic regression, test AUC 0.847 (95% CI 0.825–0.867), calibration on the diagonal within this cohort. The operating point is a clinical choice: sensitivity 0.90 flags 43 of 100 admissions; 0.80 flags 28. We recommend the team choose the alert budget first and read the sensitivity off the table, not the other way round.
2. **Report remaining hospital bed-days at cohort level only.** The Lasso model's mean-type read-out sums to within −0.2% of true bed-days on the test cohort (+1.7% under stratified smearing); individual MAE is 6.6 days and the longest-staying 10% of patients hold 36% of bed-days and are under-predicted by 28 days on average. **No individual discharge date should be promised.**
3. **If earlier warning is wanted, 36 hours costs almost nothing.** AUC rises from 0.776 at 6 h to 0.840 at 36 h and 0.847 at 48 h; the last 12 hours buy 0.007. (Caveat: the cohort only contains stays ≥ 48 h.)
4. **Before use in any other unit, re-calibrate locally.** See limitation 5.

## Limitations — what this work cannot support

**1. The self-fulfilling paradox.** If a high-risk flag triggers review and the review saves the patient, the model was "wrong". Any prospective evaluation must be designed around this; retrospective AUC cannot capture it.
> 中文注：这是所有院内预警模型共有的悖论，说出来是加分不是减分。追问"那怎么评估"——答：前瞻性评估看的是流程指标（复核是否发生、干预时间是否提前），不是 AUC。

**2. Confounding by indication — part of what the model learns is the care process, not physiology.** A model that sees no measured values, only which tests were ordered and how often, reaches AUC 0.74 (vs SAPS-I 0.64 and the full model 0.85). Ventilation within 48 h predicts death partly because only the sickest are ventilated. We quantify the size of this component rather than claim the model reads physiology alone.
> 中文注：打法 A。TA 问"开了检查的人不就是更重吗"——对，这一条就是把它变成数字。0.74→0.85 的 0.11 才是化验值本身贡献的。**不要说"检查致死"，不要说"模型只学了流程"**。

**3. The label "survived" means "left hospital alive", not "survived".** Of 10,293 in-hospital survivors, 389 (3.8%) die within 30 days of discharge and 796 (7.7%) within 90 days. Relaxing the label to "alive 30 days after discharge" raises prevalence from 14.2% to 17.1% and leaves the ranking unchanged (AUC 0.847 → 0.843); among the 729 false positives at sensitivity 0.90, 55 (7.5%) died within 30 days of discharge — twice the survivor base rate, but 93% remain true false positives. We keep the in-hospital label and report this as label noise concentrated in the high-risk region.
> 中文注：打法 E。数字要说对：**不是"四分之一标签错了"**——2,848 个有随访死亡日期的人出院后存活中位数 328 天，那是 ICU 人群的长期死亡率；真正"出院后几天就死"的是一两百人。追问"为什么不直接预测 30 天死亡"——作业允许，但口径与院内行动不匹配，我们做成敏感性分析而不开新任务。

**4. The 48-hour window has an intrinsic blind spot.** At sensitivity 0.90 the model misses 44 of 341 deaths. These patients look like survivors in the first 48 hours — median GCS 15, lower SAPS-I/SOFA, no lactate ordered, normal bicarbonate (7 of 29 variables differ from caught deaths after Holm correction) — and they do not die later than the caught deaths (median LOS 9 vs 9). Raising sensitivity to 0.95 recovers 17 of them at the cost of flagging 50 of 100 admissions.
> 中文注：打法 D。结论是"这些死亡不在 48h 数据里"，不是"阈值太高"。"不比抓住的人死得晚"用的是死亡者的住院天数当到死亡天数的代理，证据偏弱，别说成独立表型。

**5. Internal validation only; calibration does not travel.** Train on three ICU types and test on the fourth: AUC holds (0.82–0.86) but predicted risk over-estimates observed mortality by 84% (logistic) to 150% (random forest) in the cardiac-surgery unit, where mortality is 4.9% against 19.8% in the medical ICU. The same pattern appears in T2 (bed-days over-estimated by 20% in CSRU, under-estimated by 5% in MICU/SICU). There is no external hospital, no temporal hold-out and no prospective validation in this work.
> 中文注：打法 C。这是要求 03"未见数据上的表现"的答案，也是"为什么不报概率"的理由。追问"外部验证呢"——这是内部版的外部验证（Topic 07 验证阶梯第三级），真正的外部验证需要另一家医院的数据。

**6. Data quality is imperfect and documented.** 167 stays have no length of stay; 5 have LOS = 1 day despite a 48-hour window; 403 deaths have death-date and discharge-date misaligned by more than 2 days (an extraction artefact, not "occupying a bed after death"); SAPS-I is missing for 518 and SOFA for 403 patients. All counts are produced by the pipeline (`data_quality.json`) and excluded or handled as stated in the preprocessing page.
> 中文注：403 例按 PhysioNet 官方定义是抽取错位，方案 v3.1 已勘误，别说成"死后占床"。

**7. Model choice is a judgement, not a measurement.** XGBoost reaches 0.855 against 0.847 for logistic regression; the confidence intervals overlap almost entirely. We report the simpler, better-calibrated model as primary and the difference as within noise — not as a finding.
> 中文注：Topic 07 原话"追逐比噪声还小的改进"。

## Closing line

*The model beats the clinical scores; part of that margin is the care process rather than physiology, and we have measured how much. Ranking and probability are different claims, and only the first travels across units. The blind spots of a 48-hour window are in the data, not the algorithm — we found them and described them. The model proposes; the clinician disposes.*

---

### 中文注：这页怎么讲（1 分钟）

先说三条建议（正面），再说局限，局限按"我们量过"的顺序讲：第 2 条有 0.74，第 3 条有 3.8%，第 4 条有 44 人，第 5 条有 84%。每条局限都带一个数字，是这一页和别组的区别。收口那四句话背下来。
