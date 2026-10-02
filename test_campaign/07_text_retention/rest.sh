export OMP_NUM_THREADS=2
cd /home/zera/laya-vision/test_campaign/07_text_retention
PY=/home/zera/laya-vision/.venv/bin/python; R=/home/zera/laya-vision/runs
$PY -u run_preds.py stock stock x > log_stock.txt 2>&1
$PY -u run_preds.py b_wise080cal vision $R/stage2_b/wise080_calibrated > log_b080.txt 2>&1
$PY -u run_preds.py a_cal vision $R/stage2_a/calibrated > log_a.txt 2>&1
$PY -u run_preds.py layaagent_c085cal layaagent $R/stage2_c/wise085_calibrated > log_lay.txt 2>&1
echo ALLDONE > rest.done
