"""Tiny, randomly initialised stand-ins for Laya and SigLIP, for CPU tests and smoke runs.

``make_tiny_laya_dir`` writes a directory in the real Laya checkpoint layout
(``rl_agent_config.json``, ``model.safetensors``, ``encoder/``, ``tokenizer/``) that stock
``laya.Agent`` loads. It uses the real Laya tokenizer (downloaded once, a few MB), so sequences
built in tests are the same as production sequences; only the network is tiny.
"""
import json
import os
import shutil
from typing import Optional

import torch

TINY_HIDDEN = 128


def laya_tokenizer_dir(model_id: str = "convaiinnovations/laya", revision: Optional[str] = None) -> str:
    """Local path of the Laya tokenizer directory (downloads only ``tokenizer/*`` if needed)."""
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config

    d = snapshot_download(model_id, revision=revision,
                          allow_patterns=["tokenizer/*", "rl_agent_config.json"])
    _fix_tokenizer_config(d)
    return os.path.join(d, "tokenizer")


def tiny_encoder_config(vocab_size: int = 50368, pad_token_id: int = 50283):
    from transformers import ModernBertConfig

    return ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=TINY_HIDDEN,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=2,
        global_attn_every_n_layers=3,
        local_attention=128,
        max_position_embeddings=2048,
        pad_token_id=pad_token_id,
        bos_token_id=50281,
        eos_token_id=50282,
        cls_token_id=50281,
        sep_token_id=50282,
        reference_compile=False,
    )


def tiny_siglip_config(image_size: int = 32, patch_size: int = 8):
    """4x4 patch grid, width 32. With pool_k=2 this gives 4 image tokens."""
    from transformers import SiglipVisionConfig

    return SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
    )


def make_tiny_laya_dir(out_dir: str, seed: int = 0, max_len: int = 512, head_max_len: int = 192) -> str:
    """Write a random tiny Laya checkpoint that stock ``laya.Agent(out_dir)`` can load."""
    from safetensors.torch import save_file
    from transformers import AutoModel
    from laya.common import DecisionModel

    os.makedirs(out_dir, exist_ok=True)
    tok_src = laya_tokenizer_dir()
    tok_dst = os.path.join(out_dir, "tokenizer")
    if not os.path.isdir(tok_dst):
        shutil.copytree(tok_src, tok_dst)

    ecfg = tiny_encoder_config()
    ecfg.save_pretrained(os.path.join(out_dir, "encoder"))
    torch.manual_seed(seed)
    enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    model = DecisionModel(enc, head_layers=2, n_act=2)
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))

    cfg = {
        "encoder": "answerdotai/ModernBERT-large",
        "head_layers": 2,
        "max_len": max_len,
        "head_max_len": head_max_len,
        "act_costs": {"escalate": 0.5},
        "amp_dtype": "bf16",
        "model_name": "tiny-test",
        "temperature": [1.0, 1.0, 1.0],
        "temperature_by_options": {},
    }
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    return out_dir
