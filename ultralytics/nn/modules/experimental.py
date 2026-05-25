# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv


class DWConvBlock(nn.Module):
    """Depthwise-separable convolution block."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: bool = True):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1, act=act)
        self.pw = Conv(c1, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class CoordAtt(nn.Module):
    """Coordinate Attention module."""

    def __init__(self, c1: int, c2: int | None = None, reduction: int = 32):
        super().__init__()
        c2 = c2 or c1
        mip = max(8, c1 // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c1, mip, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, c2, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c2, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_h * a_w


class BiFormerBlock(nn.Module):
    """Lightweight BiFormer-like global attention block."""

    def __init__(self, c1: int, c2: int | None = None, heads: int = 8):
        super().__init__()
        c2 = c2 or c1
        self.proj = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1)
        self.norm1 = nn.LayerNorm(c2)
        self.attn = nn.MultiheadAttention(c2, num_heads=max(1, min(heads, c2 // 8)), batch_first=True)
        self.norm2 = nn.LayerNorm(c2)
        hidden = max(32, c2 * 2)
        self.mlp = nn.Sequential(nn.Linear(c2, hidden), nn.GELU(), nn.Linear(hidden, c2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self.attn(self.norm1(t), self.norm1(t), self.norm1(t), need_weights=False)[0]
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class BiFANFusion(nn.Module):
    """BiFAN weighted feature fusion (2-input)."""

    def __init__(self, c1: int | list[int] | tuple[int, ...], c2: int | None = None):
        super().__init__()
        in_channels = list(c1) if isinstance(c1, (list, tuple)) else [c1, c1]
        c2 = c2 or in_channels[0]
        self.w = nn.Parameter(torch.ones(len(in_channels), dtype=torch.float32))
        self.eps = 1e-4
        self.proj = nn.ModuleList(Conv(ci, c2, 1, 1) if ci != c2 else nn.Identity() for ci in in_channels)
        self.conv = Conv(c2, c2, 3, 1)
    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c2 = c2 or c1
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = 1e-4
        self.conv = Conv(c1, c2, 3, 1)

    def forward(self, xs: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        w = torch.relu(self.w)
        w = w / (w.sum() + self.eps)
        x = 0
        for i, t in enumerate(xs):
            x = x + w[i] * self.proj[i](t)
        x = w[0] * xs[0] + w[1] * xs[1]
        return self.conv(x)
