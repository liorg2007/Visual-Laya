"""Weight-space interpolation toward stock Laya (WiSE-FT) to undo text regression after stage 2.

    python -m laya_vision.train.interpolate --checkpoint runs/stage2_b/best --alpha 0.8 \
        --out runs/stage2_b/wise --base convaiinnovations/laya --base-revision <sha>

Laya's weights become ``base + alpha * (checkpoint - base)``; the vision side (tower, pooler,
projector, modality_emb) and the config are copied unchanged. ``alpha = 1`` is the checkpoint,
``alpha = 0`` stock Laya. Run ``calibrate`` on the result: the fitted temperatures belong to the
un-interpolated weights. Choose ``alpha`` on held-out data (e.g. a BoolQ train-split slice and the
image calib splits), never on the eval sets.
"""
import argparse
import json
import os
import shutil
from typing import Dict, Optional

import torch


def interpolate(ft: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    if set(ft) != set(base):
        raise ValueError("checkpoint and base Laya weights differ: %s"
                         % sorted(set(ft) ^ set(base))[:5])
    out = {}
    for k, v in ft.items():
        if v.is_floating_point():
            b = base[k].float()
            out[k] = (b + alpha * (v.float() - b)).to(v.dtype)
        else:
            out[k] = v
    return out


def main(argv: Optional[list] = None) -> str:
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="laya_vision checkpoint dir (after stage 2)")
    ap.add_argument("--alpha", type=float, required=True, help="0 = stock Laya, 1 = checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="stock Laya id/dir (default: the checkpoint's vision.laya_base)")
    ap.add_argument("--base-revision", dest="base_revision", default=None)
    a = ap.parse_args(argv)
    if not 0.0 <= a.alpha <= 1.0:
        ap.error("--alpha must be in [0, 1]")

    with open(os.path.join(a.checkpoint, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    base, rev = a.base, a.base_revision
    if base is None:  # vision.laya_base is "<id>@<sha>" when the base came from the Hub
        base, _, sha = (cfg.get("vision", {}).get("laya_base") or "").partition("@")
        rev = rev or (sha or None)
        if not base:
            ap.error("--base is required: the checkpoint does not record its Laya base")
    base_dir = base if os.path.isdir(base) else snapshot_download(base, revision=rev)

    if os.path.abspath(a.out) != os.path.abspath(a.checkpoint):
        if os.path.exists(a.out):
            shutil.rmtree(a.out)
        shutil.copytree(a.checkpoint, a.out, ignore=shutil.ignore_patterns("trainer_state.pt"))
    ft = load_file(os.path.join(a.checkpoint, "model.safetensors"))
    mixed = interpolate(ft, load_file(os.path.join(base_dir, "model.safetensors")), a.alpha)
    save_file(mixed, os.path.join(a.out, "model.safetensors"))
    cfg.setdefault("vision", {})["wise_alpha"] = a.alpha
    with open(os.path.join(a.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("[interpolate] alpha=%.3f base=%s%s -> %s" % (a.alpha, base, "@" + rev if rev else "", a.out))
    return a.out


if __name__ == "__main__":
    main()
