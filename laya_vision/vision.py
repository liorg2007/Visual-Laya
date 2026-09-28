"""SigLIP vision tower returning patch features from a chosen layer (plan.md §2.2)."""
from typing import Optional

import torch.nn as nn


class VisionEncoder(nn.Module):
    """``pixel_values [M,3,S,S] -> [M, G*G, C_v]``.

    ``feature_layer=-1`` is ``last_hidden_state`` (post-LayerNorm in HF SigLIP); any other value
    indexes ``hidden_states`` (``-2`` = second-to-last layer output, the LLaVA convention).
    """

    def __init__(self, tower: nn.Module, feature_layer: int = -2):
        super().__init__()
        self.tower = tower
        self.feature_layer = int(feature_layer)
        cfg = tower.config
        self.image_size, self.patch_size, self.width = cfg.image_size, cfg.patch_size, cfg.hidden_size
        self.grid = self.image_size // self.patch_size

    @property
    def config(self):
        return self.tower.config

    @classmethod
    def from_pretrained(cls, name: str, revision: Optional[str] = None, feature_layer: int = -2,
                        **kw) -> "VisionEncoder":
        from transformers import SiglipVisionModel

        if revision:
            kw["revision"] = revision
        kw.setdefault("attn_implementation", "sdpa")
        return cls(SiglipVisionModel.from_pretrained(name, **kw), feature_layer)

    @classmethod
    def from_config(cls, config, feature_layer: int = -2) -> "VisionEncoder":
        from transformers import SiglipVisionModel

        if hasattr(config, "vision_config"):  # a full SiglipConfig
            config = config.vision_config
        try:
            tower = SiglipVisionModel._from_config(config, attn_implementation="sdpa")
        except (AttributeError, TypeError):  # private helper; fall back to the plain constructor
            tower = SiglipVisionModel(config)
        return cls(tower, feature_layer)

    def forward(self, pixel_values):
        dtype = next(self.tower.parameters()).dtype
        if self.feature_layer == -1:
            return self.tower(pixel_values=pixel_values.to(dtype)).last_hidden_state
        out = self.tower(pixel_values=pixel_values.to(dtype), output_hidden_states=True)
        return out.hidden_states[self.feature_layer]
