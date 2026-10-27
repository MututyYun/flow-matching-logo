"""Train a tiny 2D point-cloud DDPM on the sampled logo shape.

Usage:  python scripts/train.py [--steps 5000] [--batch 8192]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import Schedule, VelocityMLP, ddim_sample, make_v

ROOT = Path(__file__).resolve().parents[1]
SEED = 1234


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lr_at(step: int, total: int, base: float, warmup: int = 100) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = np.load(ROOT / "data" / "target_points.npz")
    xy = torch.from_numpy(data["xy"]).float().to(device)
    n = xy.shape[0]
    print(f"target points: {n}, range {xy.amin(0).tolist()} .. {xy.amax(0).tolist()}")

    schedule = Schedule()
    model = VelocityMLP(hidden=args.hidden, depth=args.depth).to(device)
    ema = copy.deepcopy(model).to(device)
    for p in ema.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.99))
    gen = torch.Generator(device=device).manual_seed(SEED)

    history = []
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        lr = lr_at(step - 1, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        # sample a fresh mini-set of points from the empirical target cloud
        idx = torch.randint(0, n, (args.batch,), device=device, generator=gen)
        x0 = xy[idx]

        t = torch.rand(args.batch, 1, device=device, generator=gen)
        eps = torch.randn_like(x0, generator=gen)
        ab = schedule.alpha_bar(t)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        v_target = make_v(x0, eps, ab)

        v_pred = model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            for p, ep in zip(model.parameters(), ema.parameters()):
                ep.mul_(args.ema_decay).add_(p, alpha=1 - args.ema_decay)

        if step % args.log_every == 0 or step == 1:
            entry = {
                "step": step,
                "loss": float(loss.detach()),
                "lr": lr,
                "elapsed": round(time.time() - t0, 2),
            }
            history.append(entry)
            print(f"step {step:5d}/{args.steps}  loss {entry['loss']:.4f}  lr {lr:.2e}  {entry['elapsed']:.1f}s")

    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s")

    ckpt = {
        "ema": ema.state_dict(),
        "raw": model.state_dict(),
        "config": {
            "hidden": args.hidden,
            "depth": args.depth,
            "steps": args.steps,
            "batch": args.batch,
            "n_points": n,
            "n_params": n_params,
            "T": schedule.T,
            "seed": SEED,
            "train_seconds": round(elapsed, 2),
        },
        "history": history,
    }
    torch.save(ckpt, ROOT / "model.pt")
    print("saved model.pt")

    with open(ROOT / "data" / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"config": ckpt["config"], "history": history}, f, ensure_ascii=False, indent=2)

    # quick preview of EMA samples at several step counts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ema.eval()
    g = torch.Generator(device=device).manual_seed(20261026)
    x_T = torch.randn(n, 2, device=device, generator=g)

    step_counts = [1, 2, 5, 10, 25, 50, 100]
    fig, axes = plt.subplots(1, len(step_counts), figsize=(3.0 * len(step_counts), 3.1), dpi=110)
    for ax, steps in zip(axes, step_counts):
        _, final = ddim_sample(ema, schedule, x_T, steps=steps)
        p = final.numpy()
        ax.set_facecolor("#0b0d13")
        ax.scatter(p[:, 0], p[:, 1], s=1.0, c="#ff5a5a", linewidths=0, alpha=0.85)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_title(f"{steps} step(s)", color="white", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#333844")
    fig.patch.set_facecolor("#0b0d13")
    fig.tight_layout()
    out = ROOT / "assets" / "train_steps_preview.png"
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
