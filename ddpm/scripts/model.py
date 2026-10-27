"""Small 2D point-cloud diffusion model (DDPM training, DDIM sampling, v-prediction)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# noise schedule
# --------------------------------------------------------------------------- #
class Schedule:
    """Linear beta schedule; continuous-time queries by interpolation."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        self.ab = torch.cumprod(alphas, dim=0).float()  # alpha_bar at discrete t=1..T

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """t in [0, 1] -> alpha_bar, shape like t."""
        ab = self.ab.to(t.device)
        idx = (t.clamp(0, 1) * (self.T - 1)).clamp(0, self.T - 1)
        i0 = idx.floor().long()
        i1 = (i0 + 1).clamp(max=self.T - 1)
        frac = (idx - i0).to(ab.dtype)
        return ab[i0] * (1 - frac) + ab[i1] * frac


# --------------------------------------------------------------------------- #
# network
# --------------------------------------------------------------------------- #
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        # geometric frequencies spanning ~1..64 cycles across t in [0, 1]
        freqs = 2.0 ** torch.linspace(0.0, 6.0, dim // 2)
        self.register_buffer("freqs", freqs * math.pi)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs  # [B, dim/2]
        return torch.cat([torch.sin(args), torch.cos(args), t], dim=-1)


class VelocityMLP(nn.Module):
    """Per-point MLP: (x, y, t_embed) -> predicted velocity (vx, vy)."""

    def __init__(self, hidden: int = 256, t_dim: int = 64, depth: int = 4):
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
        """x: [B,2], t: [B,1] in [0,1] -> v: [B,2]."""
        h = torch.cat([x, self.time_emb(t)], dim=-1)
        return self.out(self.net(h))


# --------------------------------------------------------------------------- #
# v-prediction helpers
# --------------------------------------------------------------------------- #
def make_v(x0: torch.Tensor, eps: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    ab = ab.view(-1, 1)
    return ab.sqrt() * eps - (1 - ab).sqrt() * x0


def predict_x0_eps(x_t: torch.Tensor, v: torch.Tensor, ab: torch.Tensor):
    ab = ab.view(-1, 1)
    x0 = ab.sqrt() * x_t - (1 - ab).sqrt() * v
    eps = ab.sqrt() * v + (1 - ab).sqrt() * x_t
    return x0, eps


# --------------------------------------------------------------------------- #
# DDIM sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    schedule: Schedule,
    x_T: torch.Tensor,
    steps: int,
    clamp: float = 1.6,
    record_frames: int | None = None,
):
    """Deterministic DDIM (eta=0). Returns (frames, final).

    frames has record_frames evenly spaced snapshots including x_T and x_0.
    """
    model.eval()
    device = x_T.device
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    x = x_T
    snapshots: list[torch.Tensor] = [x.detach().cpu()]

    for i in range(steps):
        t = ts[i].expand(x.shape[0], 1)
        t_prev = ts[i + 1].expand(x.shape[0], 1)
        ab_t = schedule.alpha_bar(t)
        ab_prev = schedule.alpha_bar(t_prev)
        v = model(x, t)
        x0, eps = predict_x0_eps(x, v, ab_t)
        x0 = x0.clamp(-clamp, clamp)
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        snapshots.append(x.detach().cpu())

    frames = snapshots
    if record_frames is not None and record_frames < len(snapshots):
        idx = torch.linspace(0, len(snapshots) - 1, record_frames).round().long().tolist()
        frames = [snapshots[i] for i in idx]
    return frames, x.detach().cpu()
