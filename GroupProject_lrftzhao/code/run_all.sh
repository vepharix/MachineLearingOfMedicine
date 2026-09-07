#!/usr/bin/env bash
# 一键跑通。顺序：Stage 0–1（48h + 24h）→ 2 → T1 fit → T1 eval（test 只看一次）→ E → C → T3 → T2 fit → T2 eval → D → PDP → results.json
# Mac：PY=/opt/homebrew/Caskroom/miniforge/base/bin/python3 ./run_all.sh   Windows(Git Bash)：PY="py -3" ./run_all.sh
set -e
PY="${PY:-python3}"
export PYTHONUTF8=1
cd "$(dirname "$0")"
$PY stage0_1_build_features.py
$PY stage0_1_build_features.py --window 24
$PY stage2_split.py
$PY t1_fit.py
$PY t1_eval.py
$PY stage_E_label_sensitivity.py
$PY stage_C_leave_one_icu_out.py
$PY stage_T3_24h_ablation.py
$PY t2_fit.py
$PY t2_eval.py
$PY stage_D_fn_autopsy.py
$PY stage_A_missingness_only.py
$PY stage_B_time_value.py
$PY stage6_pdp.py
$PY make_results.py
