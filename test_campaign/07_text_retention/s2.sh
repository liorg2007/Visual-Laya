export OMP_NUM_THREADS=2
PY=/home/zera/laya-vision/.venv/bin/python; R=/home/zera/laya-vision/runs
$PY -u run_preds.py c_sweep wise $R/stage2_c/final 0.00 0.50 0.70 0.85 1.00 > log_sweep.txt 2>&1
