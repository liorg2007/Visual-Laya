"""Shared helpers for the core-model tests.

The tiny random Laya gives near-identical hidden states at every [MASK] (all logits equal, noul
p=0.5), so parity tests on it would pass even if the code were wrong. ``perturb`` adds seeded
noise to every Laya weight so that options, states and questions produce distinct logits.
"""
import io
import os
import shutil

import numpy as np
import torch

QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What does the user want?",
               "criteria": {"refund": "asks for money back", "cancel": "wants to cancel", "other": None}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat urgent", "very urgent"]},
    "angry": {"type": "noul", "instructions": "Is the user angry?"},
}


def perturb(module: torch.nn.Module, seed: int = 0, std: float = 0.3) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.is_floating_point():
                p.add_(torch.randn(p.shape, generator=g) * std)
    return module


def perturbed_laya_dir(src: str, dst: str, seed: int = 0) -> str:
    """Copy of the tiny Laya checkpoint ``src`` with perturbed weights (loadable by laya.Agent)."""
    from safetensors.torch import load_file, save_file
    from laya.common import build_model
    import json

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    with open(os.path.join(dst, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    m = build_model(cfg, encoder_dir=os.path.join(dst, "encoder"), pretrained=False)
    m.load_state_dict(load_file(os.path.join(dst, "model.safetensors")), strict=True)
    perturb(m, seed)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, os.path.join(dst, "model.safetensors"))
    return dst


def png_bytes(color=(200, 30, 30), size=(40, 30), mode="RGB", fmt="PNG") -> bytes:
    from PIL import Image

    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def noise_image(seed: int = 0, size=(48, 40)):
    from PIL import Image

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB")


_CACHE = {}


def perturbed_laya(tiny_laya_dir: str) -> str:
    """Session-cached perturbed copy of the tiny Laya checkpoint."""
    key = ("laya", tiny_laya_dir)
    if key not in _CACHE:
        _CACHE[key] = perturbed_laya_dir(tiny_laya_dir, tiny_laya_dir.rstrip("/") + "_perturbed")
    return _CACHE[key]


def vision_ckpt(tiny_laya_dir: str, pool_mode: str = "avg", pool_k: int = 2, temps=None) -> str:
    """Session-cached Laya-Vision checkpoint: perturbed tiny Laya + tiny random SigLIP + perturbed projector."""
    key = ("vision", tiny_laya_dir, pool_mode, pool_k, str(temps))
    if key not in _CACHE:
        from laya_vision.checkpoint import init_from_laya, save_checkpoint
        from laya_vision.config import VisionConfig
        from laya_vision.testing import tiny_siglip_config

        torch.manual_seed(0)
        vcfg = VisionConfig(tower="tiny-random-siglip", pool_mode=pool_mode, pool_k=pool_k)
        if temps is not None:
            vcfg.temperature, vcfg.temperature_by_options = temps
        model, tok, cfg = init_from_laya(perturbed_laya(tiny_laya_dir), vcfg, tower_config=tiny_siglip_config())
        perturb(model.projector, seed=1, std=0.5)
        with torch.no_grad():
            model.modality_emb.normal_(0, 0.5, generator=torch.Generator().manual_seed(2))
        out = "%s_vision_%s_%d_%d" % (tiny_laya_dir.rstrip("/"), pool_mode, pool_k, len(_CACHE))
        save_checkpoint(model, tok, cfg, vcfg, out, dtype=torch.float32)
        _CACHE[key] = out
    return _CACHE[key]
