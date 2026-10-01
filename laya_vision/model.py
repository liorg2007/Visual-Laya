"""Laya's DecisionModel with image tokens injected into the state region (plan.md §2.5)."""
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .projector import Pooler, Projector
from .vision import VisionEncoder


def decision_head(laya_model, h, attention_mask, marker_pos, marker_mask, qtype, detach_encoder: bool = False,
                  return_hidden: bool = False):
    """Everything in ``laya.common.DecisionModel.forward`` after the encoder call.

    ``return_hidden`` also returns the head's hidden states at the option markers ``[B, K, d]``
    (the scorer input), used by caption-teacher distillation.

    Copied verbatim from laya 0.3.21 (``laya/common.py``, ``DecisionModel.forward``), with
    ``self`` renamed to ``laya_model``. ``tests/test_text_parity.py`` fails if Laya's forward
    drifts from this copy.
    """
    self = laya_model
    if detach_encoder:
        h = h.detach()
    h = h + self.type_emb(qtype)[:, None, :]
    if self.head is not None:
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            if self.head_checkpointing and self.training and torch.is_grad_enabled():
                # Non-reentrant checkpointing also trains the head when its input
                # is frozen. Default RNG preservation keeps dropout consistent.
                h = checkpoint(layer, h, src_key_padding_mask=pad, use_reentrant=False)
            else:
                h = layer(h, src_key_padding_mask=pad)
    idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
    m = torch.gather(h, 1, idx)
    logits = self.scorer(m).squeeze(-1).float()
    logits = logits.masked_fill(~marker_mask, -1e4)

    p = torch.softmax(logits.detach(), -1)
    k = marker_mask.sum(-1).clamp(min=2).float()
    ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
    if p.size(-1) >= 2:
        top2 = p.topk(2, -1).values
    else:
        # Single-option question: pad the missing second entry with 0.0 (see laya).
        top1 = p.topk(1, -1).values
        top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)
    feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
    pooled = h[:, 0].float()
    act_logits = self.act_head(torch.cat([pooled, feats], -1))
    if return_hidden:
        return logits, act_logits, m
    return logits, act_logits


class LayaVisionModel(nn.Module):
    """``laya.*`` is an unchanged ``laya.common.DecisionModel``; vision adds ``vision``, ``pooler``,
    ``projector`` and ``modality_emb``. Without images the output equals ``DecisionModel``'s."""

    def __init__(self, laya: nn.Module, vision: VisionEncoder, pooler: Pooler, projector: Projector):
        super().__init__()
        self.laya = laya
        self.vision, self.pooler, self.projector = vision, pooler, projector
        d = laya.encoder.config.hidden_size
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, d))
        # Laya disables it too: torch.compile of the embeddings is a loss (and hangs on some hosts).
        try:
            laya.encoder.config.reference_compile = False
        except Exception:
            pass

    @property
    def n_image_tokens(self) -> int:
        return self.pooler.n_tokens

    def encode_images(self, pixel_values) -> torch.Tensor:
        """``[M,3,S,S] -> [M, N, d]``."""
        return self.projector(self.pooler(self.vision(pixel_values))) + self.modality_emb

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                image_tokens: Optional[torch.Tensor] = None, image_index: Optional[torch.Tensor] = None,
                image_start: Optional[torch.Tensor] = None, pixel_values: Optional[torch.Tensor] = None,
                detach_encoder: bool = False, return_hidden: bool = False):
        if image_tokens is None and pixel_values is not None:
            image_tokens = self.encode_images(pixel_values)
        enc = self.laya.encoder
        # Pre-LayerNorm token embeddings: the encoder applies its own embedding norm + dropout to
        # inputs_embeds, so image and text positions are normalised by the same (shared) LN.
        emb = enc.embeddings.tok_embeddings(input_ids)
        if image_tokens is not None:
            if image_index is None or image_start is None:
                raise ValueError("image_tokens need image_index and image_start")
            image_index, image_start = image_index.to(emb.device), image_start.to(emb.device)
            rows = (image_index >= 0).nonzero(as_tuple=True)[0]
            if rows.numel():
                n = image_tokens.shape[1]
                pos = image_start[rows, None] + torch.arange(n, device=emb.device)
                vals = image_tokens[image_index[rows]].to(emb.dtype)
                # Out of place: keeps autograd right when the embedding matrix is frozen.
                emb = emb.index_put((rows[:, None], pos), vals)
        h = enc(inputs_embeds=emb, attention_mask=attention_mask).last_hidden_state
        return decision_head(self.laya, h, attention_mask, marker_pos, marker_mask, qtype, detach_encoder,
                             return_hidden)

    def trainable_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Parameters by training group: vision / projector / encoder / head."""
        enc_ids = {id(p) for p in self.laya.encoder.parameters()}
        return {
            "vision": list(self.vision.parameters()),
            "projector": list(self.pooler.parameters()) + list(self.projector.parameters()) + [self.modality_emb],
            "encoder": list(self.laya.encoder.parameters()),
            "head": [p for p in self.laya.parameters() if id(p) not in enc_ids],
        }


def build_vision_side(vision: VisionEncoder, vcfg, d: int):
    """Pooler + projector for ``vision`` under ``vcfg``; fills the tower-derived fields of ``vcfg``."""
    vcfg.image_size, vcfg.patch_size, vcfg.vision_width = vision.image_size, vision.patch_size, vision.width
    pooler = Pooler(vcfg.pool_mode, vcfg.pool_k, vision.grid, vision.width)
    if pooler.n_tokens != vcfg.n_tokens or pooler.out_width != vcfg.pooled_width:
        raise AssertionError("Pooler and VisionConfig disagree on the image token shape")
    return pooler, Projector(pooler.out_width, d, in_norm=vcfg.proj_in_norm, standardize=vcfg.proj_standardize)
