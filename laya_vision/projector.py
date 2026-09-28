"""Patch-grid pooling and the MLP projector into Laya's embedding space (plan.md §2.3)."""
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


class Projector(nn.Module):
    """Linear -> GELU -> Linear. The last layer starts small; ModernBERT's embedding LN rescales."""

    def __init__(self, in_width: int, d: int):
        super().__init__()
        self.fc1 = nn.Linear(in_width, d)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d, d)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
