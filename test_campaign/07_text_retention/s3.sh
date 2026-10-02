export OMP_NUM_THREADS=2
PY=/home/zera/laya-vision/.venv/bin/python; R=/home/zera/laya-vision/runs
$PY -u run_preds.py b_wise080cal vision $R/stage2_b/wise080_calibrated > log_b080.txt 2>&1
$PY -u run_preds.py b_unmerged_cal vision $R/stage2_b/calibrated > log_bunm.txt 2>&1
$PY -u run_preds.py a_cal vision $R/stage2_a/calibrated > log_a.txt 2>&1
