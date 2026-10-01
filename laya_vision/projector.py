"""Patch-grid pooling and the MLP projector into Laya's embedding space (plan.md §2.3)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import POOL_MODES, pooled_grid


class Pooler(nn.Module):
    """``[M, G*G, C] -> [M, N, C']``, tokens flattened row-major (top-left to bottom-right).

    ``avg``: k x k average pooling with ``ceil_mode=True`` (C' = C).
    ``shuffle``: zero-pad the grid to a multiple of k, then space-to-depth (C' = C * k^2).
    """

    def __init__(self, mode: str, k: int, grid: int, width: int):
        super().__init__()
        if mode not in POOL_MODES:
            raise ValueError("pool mode must be one of %s, got %r" % (POOL_MODES, mode))
        self.mode, self.k, self.grid, self.width = mode, int(k), int(grid), int(width)
        self.out_grid = pooled_grid(self.grid, self.k, mode)
        self.n_tokens = self.out_grid ** 2
        self.out_width = self.width * (self.k ** 2 if mode == "shuffle" else 1)

    def forward(self, x):
        m, n, c = x.shape
        g, k = self.grid, self.k
        if n != g * g or c != self.width:
            raise ValueError("Pooler expected [M, %d, %d], got %s" % (g * g, self.width, tuple(x.shape)))
        x = x.transpose(1, 2).reshape(m, c, g, g)
        if self.mode == "avg":
            # count_include_pad is irrelevant without padding; ceil_mode averages the partial
            # edge windows over their real cells only.
            x = F.avg_pool2d(x, k, stride=k, ceil_mode=True)
        else:
            pad = self.out_grid * k - g
            if pad:
                x = F.pad(x, (0, pad, 0, pad))
            x = F.pixel_unshuffle(x, k)
        return x.flatten(2).transpose(1, 2)


class RunningStandardize(nn.Module):
    """``(x - running_mean) / running_std * scale``; the statistics are never back-propagated.

    Guards against modality collapse. Laya's embedding LayerNorm makes the projector scale-free,
    and stage-1 training pushed every image token onto one shared direction (cos between images
    0.9995). Subtracting the running mean over images removes the shared component, and dividing
    by the running std keeps the image-specific part at a fixed size, so it cannot be trained
    away. ``LayaVisionModel.modality_emb`` carries the shared component. Statistics update in
    train mode only; per-batch statistics are not used because a stage-2 micro-batch can hold a
    single image.
    """

    def __init__(self, d: int, momentum: float = 0.01, eps: float = 1e-5, target_norm: float = 2.0):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(()))  # scalar: mean per-dim variance
        self.register_buffer("initialized", torch.zeros((), dtype=torch.long))
        self.scale = target_norm / math.sqrt(d)  # unit-variance dims -> token norm ~ target_norm (text ~2.1)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                flat = x.detach().float().reshape(-1, x.shape[-1])
                mu = flat.mean(0)
                var = (flat - mu).pow(2).mean()
                if not bool(self.initialized):
                    self.running_mean.copy_(mu)
                    self.running_var.copy_(var)
                    self.initialized.fill_(1)
                else:
                    self.running_mean.lerp_(mu, self.momentum)
                    self.running_var.lerp_(var, self.momentum)
        z = (x - self.running_mean.to(x.dtype)) * torch.rsqrt(self.running_var + self.eps).to(x.dtype)
        return z * self.scale


class Projector(nn.Module):
    """[LayerNorm ->] Linear -> GELU -> Linear [-> RunningStandardize].

    ``in_norm`` / ``standardize`` (``VisionConfig.proj_in_norm`` / ``proj_standardize``) are off
    for checkpoints written before they existed, so those load unchanged. The last Linear starts
    small; ModernBERT's embedding LN rescales.
    """

    def __init__(self, in_width: int, d: int, in_norm: bool = False, standardize: bool = False):
        super().__init__()
        self.in_norm = nn.LayerNorm(in_width) if in_norm else None
        self.fc1 = nn.Linear(in_width, d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d, d)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)
        self.std = RunningStandardize(d) if standardize else None

    def forward(self, x):
        if self.in_norm is not None:
            x = self.in_norm(x)
        x = self.fc2(self.act(self.fc1(x)))
        if self.std is not None:
            x = self.std(x)
        return x
