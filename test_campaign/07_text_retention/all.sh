PY=/home/zera/laya-vision/.venv/bin/python
R=/home/zera/laya-vision/runs
$PY run_preds.py final_c085cal vision $R/stage2_c/wise085_calibrated
$PY run_preds.py layaagent_c085cal layaagent $R/stage2_c/wise085_calibrated
$PY run_preds.py b_wise080cal vision $R/stage2_b/wise080_calibrated
$PY run_preds.py b_unmerged_cal vision $R/stage2_b/calibrated
$PY run_preds.py a_cal vision $R/stage2_a/calibrated
$PY run_preds.py c_sweep wise $R/stage2_c/final 0.00 0.25 0.50 0.70 0.85 1.00
