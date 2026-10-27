"""Rectified Flow (a Flow Matching method) for 2D point clouds.

Training path:  x_t = (1 - t) * x0 + t * x1,   x0 ~ N(0, I), x1 ~ data
Target:         v = x1 - x0
Sampling:       Euler integration of dx/dt = v_theta(x, t), t: 0 -> 1
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        freqs = 2.0 ** torch.linspace(0.0, 6.0, dim // 2)
        self.register_buffer("freqs", freqs * math.pi)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs
        return torch.cat([torch.sin(args), torch.cos(args), t], dim=-1)


class VelocityMLP(nn.Module):
    """Per-particle MLP: (x, y, t_embed) -> velocity (vx, vy)."""

    def __init__(self, hidden: int = 320, t_dim: int = 64, depth: int = 5):
        super().__init__()
        self.time_emb = TimeEmbedding(t_dim)
        in_dim = 2 + t_dim + 1
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x, self.time_emb(t)], dim=-1)
        return self.out(self.net(h))


@torch.no_grad()
def euler_sample(model: nn.Module, x0: torch.Tensor, steps: int, record_frames: int | None = None):
    """Integrate t: 0 -> 1. Returns (frames, final), frames are CPU tensors."""
    model.eval()
    n = x0.shape[0]
    ts = torch.linspace(0.0, 1.0, steps + 1, device=x0.device)
    x = x0
    frames = [x.detach().cpu()]
    for k in range(steps):
        t = ts[k].expand(n, 1)
        v = model(x, t)
        x = x + (ts[k + 1] - ts[k]) * v
        frames.append(x.detach().cpu())

    if record_frames is not None and record_frames < len(frames):
        idx = torch.linspace(0, len(frames) - 1, record_frames).round().long().tolist()
        frames = [frames[i] for i in idx]
    return frames, x.detach().cpu()
