export OMP_NUM_THREADS=2
PY=/home/zera/laya-vision/.venv/bin/python; R=/home/zera/laya-vision/runs
$PY -u run_preds.py stock stock x > log_stock.txt 2>&1
$PY -u run_preds.py final_c085cal vision $R/stage2_c/wise085_calibrated > log_c085.txt 2>&1
