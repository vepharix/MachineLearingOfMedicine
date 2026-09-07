# Problem Statement（一页，v0 · 2026-09-04）

> 英文是交付版（咨询、deck 第 1 页、报告开头直接用）；中文注是给自己的。数字来自 `output/results.json`。

---

## At 48 hours after ICU admission: what happens to this patient next?

**Setting.** Adult ICU stays of at least 48 hours (PhysioNet/CinC 2012; 12,000 stays across cardiac, cardiac-surgery, medical and surgical ICUs). Every record covers the first 48 hours only. In-hospital mortality is 14.2%.

**The decision point.** The end of the second ICU day is when the admitting picture is complete, the team has a working diagnosis, and the question changes from *what is wrong* to *how is this going to go*. We ask one question from three sides, all at that single time point:

| | Question | Task | Who acts, and how |
|---|---|---|---|
| **T1** | Will this patient die in hospital? | Binary classification | **Senior clinician review is triggered for patients the model ranks high-risk but the team has not flagged.** This is the only action we claim. |
| **T2** | How many more hospital bed-days will this admission consume? | Regression (remaining LOS = LOS − 2, all patients, survivors and non-survivors) | Bed management: a cohort-level estimate of remaining bed-days for the patients who have just passed 48 h. **No individual discharge date is promised.** |
| **T3** | Could this judgement have been made on day one? | Ablation (24 h vs 48 h window) | Tells us what the second day of data buys, and makes the comparison with SAPS-I (a 24 h score) fair. |

**Why T1's action is deliberately cheap.** A review costs a few minutes of a senior clinician's time; a missed deterioration costs a life. So false negatives are more expensive than false positives, the operating point is set for sensitivity, and we report the price of that choice explicitly: at sensitivity 0.87 on the test set, **43 of every 100 admissions are flagged for review** (PPV 0.29). Whether that alert burden is acceptable is a clinical decision, not a modelling one; the 0.80 target (28 per 100 flagged) is offered as the alternative. Actions we considered and rejected for the main claim — family communication, step-down decisions — are more expensive per false positive and would require a calibrated probability rather than a ranking.

**What "better than current practice" means here.** The bedside team already has SAPS-I and SOFA. Beating those scores (test AUC 0.642 / 0.622 vs 0.847 for regularised logistic regression) is the entry ticket, not the result. The model's value lies in **discordance**: patients it ranks high whom the team has not flagged. We do not claim to replace clinical judgement.

**What the data cannot support, stated up front.**
- *Survival* (days to death, including post-discharge follow-up) is a perfect leak and is never used as a feature. It is used once, to build a sensitivity-analysis label.
- "Survived" means *left hospital alive*: 3.8% of in-hospital survivors die within 30 days of discharge. Our main label keeps the in-hospital definition; the sensitivity analysis shows ranking is unchanged (AUC 0.847 → 0.843).
- There is no ICU discharge time, so T2 speaks of *hospital* bed-days, not ICU beds.
- Part of what any model learns from this record is the *care process* (which tests were ordered), not physiology: a model that sees only which tests were ordered reaches AUC 0.74. We quantify this rather than hide it.

**Primary metrics, chosen before modelling.** T1: AUC-ROC for ranking (the review queue is a ranking problem), with AUC-PR and a threshold–alert-burden table to show why a default 0.5 cut-off is unusable at 14% prevalence, and calibration curves because a probability must not be spoken aloud unless it is calibrated. T2: MAE in days for individual error (against a predict-the-median baseline), and the cohort sum Σŷ vs Σy for the capacity claim — two different read-outs of the same model, never mixed.

**Validation.** One stratified 60/20/20 split fixed before any modelling; hyper-parameters chosen on validation only; the test set evaluated once per task. Unseen-data performance is probed by leave-one-ICU-type-out (train on three unit types, test on the fourth): ranking holds (AUC 0.82–0.86) but calibration breaks (probabilities over-estimated by 84% in the cardiac-surgery unit) — which is exactly why we do not report probabilities.

**One sentence.** *At 48 hours we rank ICU patients for senior review, estimate the cohort's remaining hospital bed-days, and show what the second day of data was worth — with the alert burden, the calibration failure across units, and the limits of the label stated on the same page as the AUC.*

---

### 中文注（不交）

- 行动只留"触发复核"一个，是方案 v3.1 定的：行动越贵，误报代价越高，阈值故事越难守。家属沟通、转普通病房留 Q&A。
- "0.80 档作为替代"是 09-04 写 deck 骨架时新加的：sens 0.90 要复核 43% 的人，"廉价第二意见"这个词撑不住；**9/21 咨询直接问老师哪个预算更像现实**。
- "discordance"是回答"凭什么比床旁医生强"的关键词，第 1 页和 Q&A 第 1 条共用。
- "care process 0.74"那句是打法 A，主动放进问题陈述是为了收口，不给 TA 先问出来的机会。
