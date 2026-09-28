#!/usr/bin/env bash
# End-to-end CPU smoke run (< 2 min): stage 1 -> stage 2 -> calibrate -> VisionAgent.predict,
# on a random tiny Laya + tiny SigLIP tower and a generated toy dataset (configs/smoke.yaml).
# Usage: bash scripts/smoke_train.sh [output_root]      (PYTHON=... to pick the interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
OUT=${1:-runs/smoke}
rm -rf "$OUT"
DATA="$OUT/data"

echo "== stage 1 (projector only) =="
$PY -m laya_vision.train.stage1 --config configs/smoke.yaml \
  -o output_dir="$OUT/stage1" data.synthetic.dir="$DATA"

echo "== stage 2 (projector + encoder + head, last tower block) =="
$PY -m laya_vision.train.stage2 --config configs/smoke.yaml \
  -o output_dir="$OUT/stage2" init_from="$OUT/stage1/final" data.synthetic.dir="$DATA" \
     'train.lr={projector: 1.0e-4, encoder: 2.5e-5, head: 1.0e-4, vision: 1.0e-6}' train.unfreeze_vision_blocks=1

echo "== calibrate =="
$PY -m laya_vision.train.calibrate --checkpoint "$OUT/stage2/final" --out "$OUT/calibrated" \
  --data "$DATA/calib.jsonl" --image-root "$DATA" --min-bucket 10 --refit-text --device cpu --num-workers 0

echo "== load with VisionAgent and predict =="
$PY - "$OUT/calibrated" "$DATA" <<'PYEOF'
import json, os, sys
import laya_vision
ckpt, data = sys.argv[1], sys.argv[2]
agent = laya_vision.load(ckpt, device="cpu")
qs = {"colour": {"type": "choice", "instructions": "Which colour dominates?",
                 "criteria": {"red": "mostly red", "green": "mostly green", "blue": "mostly blue"}},
      "red": {"type": "noul", "instructions": "Is the image red?"}}
img = os.path.join(data, "images", "calib_0000.png")
print("image:", json.dumps(agent.predict({"image": img}, qs)["answers"], indent=None)[:400])
print("text: ", json.dumps(agent.predict("A red picture.", qs)["answers"], indent=None)[:400])
print("SMOKE OK")
PYEOF
