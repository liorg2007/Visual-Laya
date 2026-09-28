"""LoRA on Laya's ModernBERT encoder (stage-2 variant), merged before every save.

Adapters are injected in place (``peft.inject_adapter_in_model``), so ``model.laya.encoder`` stays a
``ModernBertModel`` and the forward code is unchanged. ``merged_copy`` returns a deep copy with every
adapter folded into its base Linear, so saved checkpoints are plain Laya state dicts.
"""
import copy
from typing import Iterable, List

import torch.nn as nn

# ModernBERT: attn.Wqkv, attn.Wo, mlp.Wi, mlp.Wo ("Wo" matches both).
DEFAULT_TARGETS = ("Wqkv", "Wo", "Wi")


def apply_lora(encoder: nn.Module, r: int = 16, alpha: int = 32, dropout: float = 0.05,
               targets: Iterable[str] = DEFAULT_TARGETS) -> List[nn.Parameter]:
    """Inject LoRA adapters into ``encoder``; freeze its base weights; return the adapter params."""
    from peft import LoraConfig, inject_adapter_in_model

    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets), bias="none")
    inject_adapter_in_model(cfg, encoder)
    params = []
    for n, p in encoder.named_parameters():
        is_lora = "lora_" in n
        p.requires_grad_(is_lora)
        if is_lora:
            params.append(p)
    if not params:
        raise ValueError("LoRA matched no modules for targets %s" % list(targets))
    return params


def has_lora(module: nn.Module) -> bool:
    return any("lora_" in n for n, _ in module.named_parameters())


def merge_lora_(module: nn.Module) -> nn.Module:
    """Fold every LoRA layer inside ``module`` into its base layer and remove the wrapper (in place)."""
    from peft.tuners.lora import LoraLayer

    targets = [(name, m) for name, m in module.named_modules() if isinstance(m, LoraLayer)]
    for name, m in targets:
        m.merge()
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, m.get_base_layer())
    return module


def merged_copy(model: nn.Module) -> nn.Module:
    """A deep copy of ``model`` with LoRA merged (``model`` itself is untouched)."""
    return merge_lora_(copy.deepcopy(model)) if has_lora(model) else model
