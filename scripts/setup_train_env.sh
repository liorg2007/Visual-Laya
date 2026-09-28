#!/usr/bin/env bash
# Training environment on a Linux + NVIDIA box.
# Usage: bash scripts/setup_train_env.sh [venv_dir]
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124  (default; use cu121 / cu126 / cu128 to match
#                                                        the driver: `nvidia-smi` shows "CUDA Version")
#   PYTHON=python3.11                                    (any 3.10-3.12)
#   SKIP_DOWNLOAD=1                                      (don't pre-fetch Laya + SigLIP)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${1:-.venv}
PY=${PYTHON:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}

command -v nvidia-smi >/dev/null && nvidia-smi || echo "WARNING: nvidia-smi not found (no NVIDIA driver?)"

$PY -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip wheel
# Modern CUDA torch first, so the package install does not pull a CPU wheel.
python -m pip install "torch>=2.5" --index-url "$TORCH_INDEX"
python -m pip install -e ".[train,eval,serve,dev]"
# transformers 5.x on the training box (pyproject allows 4.48-5.x)
python -m pip install -U "transformers>=5,<6" || echo "NOTE: staying on transformers 4.x"
python -m pip install wandb || true  # optional: enable with wandb.enabled=true in a config

python - <<'PYEOF'
import torch, transformers, laya
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| devices:", torch.cuda.device_count(),
      "| bf16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
print("transformers", transformers.__version__, "| laya", laya.__version__)
assert torch.cuda.is_available(), "CUDA not available: check the driver and TORCH_INDEX"
PYEOF

# Hub auth: the public checkpoints need no token, but a token avoids rate limits and is required
# for gated datasets / pushing checkpoints:  huggingface-cli login   (or export HF_TOKEN=...)
if [ -z "${SKIP_DOWNLOAD:-}" ]; then
  python - <<'PYEOF'
from huggingface_hub import snapshot_download
for rid, pats in [("convaiinnovations/laya", None),
                  ("google/siglip-base-patch16-224", ["*.json", "*.safetensors", "*.txt", "*.model"])]:
    print("downloading", rid, "->", snapshot_download(rid, allow_patterns=pats))
PYEOF
fi

echo "Environment ready. Next: source $VENV/bin/activate && bash scripts/smoke_train.sh"
