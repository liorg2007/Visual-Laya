"""RLCD loss, ported line for line from Laya's fine-tuning notebook (``train_ddp.py``).

Per row: sample ``G`` zero-mean Gaussian perturbations of the logits inside the option mask,
score each perturbed distribution with Laya's strictly proper reward (log + w_sph * spherical
- w_rps * RPS for ``score``), normalise the group-centred advantage by its std, and weight the
Gaussian log-density of the sample. A soft cross-entropy term towards the target is added with
``ce_weight`` (1.0 in the notebook). Gradient accumulation scaling and the ``0 * act.sum()`` DDP
term live in the trainer.
"""
from typing import Dict, Optional, Tuple

import torch
from laya.common import proper_reward


def sigma_at(progress: float, start: float = 0.4, end: float = 0.1) -> float:
    """Exploration sigma, linear in training progress in [0, 1] (notebook: 0.4 -> 0.1)."""
    progress = min(1.0, max(0.0, float(progress)))
    return start + (end - start) * progress


def rlcd_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
    sigma: float,
    group_size: int = 4,
    w_sph: float = 0.75,
    w_rps: float = 1.0,
    ce_weight: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``(loss, stats)`` for one micro-batch. ``logits`` [B,K] (any dtype; cast to float32),
    ``target`` [B,K] distributions, ``marker_mask`` [B,K] bool, ``qtype`` [B] long."""
    logits = logits.float()
    mask = marker_mask.bool()
    k = mask.sum(-1, keepdim=True).float()
    target = target.float()

    # 1. G noisy logit vectors, projected to zero mean over the valid options
    eps = torch.randn((group_size,) + tuple(logits.shape), device=logits.device, dtype=logits.dtype,
                      generator=generator) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    # 2. strictly proper reward, group-centred advantage normalised by its std
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    # 3. Gaussian policy log-density + soft cross-entropy guidance
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    loss = loss_rl + ce_weight * loss_ce
    stats = {"loss": float(loss.detach()), "loss_rl": float(loss_rl.detach()),
             "loss_ce": float(loss_ce.detach()), "reward": float(r.mean()), "sigma": float(sigma)}
    return loss, stats
