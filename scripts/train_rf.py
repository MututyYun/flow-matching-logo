"""Train 1-Rectified Flow, then one Reflow round (2-RF).

Stage 1: random coupling (x0 ~ N(0,I), x1 ~ target cloud) -> 1-RF.
Stage 2: fixed coupling (x0, x1_pred = 1-RF(x0))                -> 2-RF.

The second stage is what "straightens" the trajectories: the training lines
no longer conflict randomly at crossings, so the conditional mean velocity
is closer to a single direction.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rf_model import VelocityMLP, euler_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 1234
SAMPLE_SEED = 20261026
N_VIS = 8000
N_PAIRS = 32000
STEP_COUNTS = [1, 2, 4, 10, 50]
MIN_ENDPOINT_DIST = 0.01


def lr_at(step: int, total: int, base: float, warmup: int = 100) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))


def ema_update(model, ema, decay: float) -> None:
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema.parameters()):
            ep.mul_(decay).add_(p, alpha=1 - decay)


def train_flow(model, ema, sampler, steps, batch, lr, device, tag, log_every=200):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.99))
    gen = torch.Generator(device=device).manual_seed(SEED + (0 if tag == "1rf" else 1))
    history = []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        cur_lr = lr_at(step - 1, steps, lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x0, x1 = sampler(batch, gen)
        t = torch.rand(batch, 1, device=device, generator=gen)
        x_t = (1 - t) * x0 + t * x1
        target = x1 - x0
        pred = model(x_t, t)
        loss = F.mse_loss(pred, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(model, ema, 0.999)

        if step % log_every == 0 or step == 1:
            e = {"step": step, "loss": float(loss.detach()), "lr": cur_lr, "elapsed": round(time.time() - t0, 2)}
            history.append(e)
            print(f"[{tag}] step {step:5d}/{steps}  loss {e['loss']:.4f}  lr {cur_lr:.2e}  {e['elapsed']:.1f}s")
    print(f"[{tag}] done in {time.time() - t0:.1f}s")
    return history


def make_random_sampler(xy, device):
    n = xy.shape[0]

    def f(batch, gen):
        idx = torch.randint(0, n, (batch,), device=device, generator=gen)
        x1 = xy[idx]
        x0 = torch.randn(batch, 2, device=device, generator=gen)
        return x0, x1

    return f


def make_pair_sampler(x0_all, x1_all):
    n = x0_all.shape[0]

    def f(batch, gen):
        idx = torch.randint(0, n, (batch,), device=x0_all.device, generator=gen)
        return x0_all[idx], x1_all[idx]

    return f


def crossing_rate(a, b, sample: int = 300) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(a), min(sample, len(a)), replace=False)
    a, b = a[sel], b[sel]

    def orient(P, Q, R):
        return (Q[..., 0] - P[..., 0]) * (R[..., 1] - P[..., 1]) - (Q[..., 1] - P[..., 1]) * (R[..., 0] - P[..., 0])

    A, B = a[:, None, :], b[:, None, :]
    C, D = a[None, :, :], b[None, :, :]
    o1, o2 = orient(A, B, C), orient(A, B, D)
    o3, o4 = orient(C, D, A), orient(C, D, B)
    proper = (o1 * o2 < 0) & (o3 * o4 < 0)

    near = np.zeros_like(proper)
    for P, Q in ((A, C), (A, D), (B, C), (B, D)):
        near |= np.linalg.norm(P - Q, axis=-1) < MIN_ENDPOINT_DIST
    proper &= ~near

    iu = np.triu_indices(len(sel), 1)
    return float(proper[iu].sum() / len(iu[0]))


def curvature(frames) -> dict:
    p = torch.stack(frames).float()
    seg = (p[1:] - p[:-1]).norm(dim=-1).sum(0)
    chord = (p[-1] - p[0]).norm(dim=-1).clamp_min(1e-6)
    ratio = seg / chord
    return {
        "mean_extra_path": float((ratio - 1).mean()),
        "median_ratio": float(ratio.median()),
        "max_extra_path": float((ratio - 1).max()),
    }


def chamfer(p, q, tree_q) -> float:
    return float(cKDTree(p).query(q)[0].mean() + tree_q.query(p)[0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps1", type=int, default=12000)
    ap.add_argument("--steps2", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=320)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reflow-pairs", type=int, default=N_PAIRS)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data = np.load(ROOT / "data" / "target_points.npz")
    target_xy = data["xy"]
    xy = torch.from_numpy(target_xy).float().to(device)
    n = xy.shape[0]
    print("target points:", n, "range", xy.amin(0).tolist(), "..", xy.amax(0).tolist())
    tree_q = cKDTree(target_xy)

    # ---------------- stage 1: 1-RF ----------------
    model1 = VelocityMLP(hidden=args.hidden, depth=args.depth).to(device)
    ema1 = copy.deepcopy(model1).to(device)
    for p in ema1.parameters():
        p.requires_grad_(False)
    print("params:", sum(p.numel() for p in model1.parameters()))
    hist1 = train_flow(model1, ema1, make_random_sampler(xy, device), args.steps1, args.batch, args.lr, device, "1rf")

    (ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "ema": ema1.state_dict(),
            "raw": model1.state_dict(),
            "config": {"hidden": args.hidden, "depth": args.depth, "steps": args.steps1,
                       "batch": args.batch, "n_points": n, "seed": SEED},
            "history": hist1,
        },
        ROOT / "checkpoints" / "rf1.pt",
    )
    print("saved rf1.pt")

    # ---------------- reflow coupling ----------------
    print(f"generating {args.reflow_pairs} reflow pairs with 1-RF ...")
    g = torch.Generator(device=device).manual_seed(SAMPLE_SEED)
    x0_pairs = torch.randn(args.reflow_pairs, 2, device=device, generator=g)
    _, x1_pairs = euler_sample(ema1, x0_pairs, steps=100)
    x1_pairs = x1_pairs.to(device)
    np.savez(
        ROOT / "data" / "rf_pairs.npz",
        x0=x0_pairs.detach().cpu().numpy().astype(np.float32),
        x1=x1_pairs.detach().cpu().numpy().astype(np.float32),
    )
    print("saved data/rf_pairs.npz")

    # ---------------- stage 2: 2-RF ----------------
    model2 = VelocityMLP(hidden=args.hidden, depth=args.depth).to(device)
    ema2 = copy.deepcopy(model2).to(device)
    for p in ema2.parameters():
        p.requires_grad_(False)
    hist2 = train_flow(model2, ema2, make_pair_sampler(x0_pairs, x1_pairs), args.steps2, args.batch, args.lr, device, "2rf")

    torch.save(
        {
            "ema": ema2.state_dict(),
            "raw": model2.state_dict(),
            "config": {"hidden": args.hidden, "depth": args.depth, "steps": args.steps2,
                       "batch": args.batch, "n_points": n, "seed": SEED},
            "history": hist2,
        },
        ROOT / "checkpoints" / "rf2.pt",
    )
    print("saved rf2.pt")

    # ---------------- validation ----------------
    x0_vis = x0_pairs[:N_VIS].detach()
    g2 = torch.Generator(device=device).manual_seed(SAMPLE_SEED + 7)
    idx = torch.randint(0, n, (N_VIS,), device=device, generator=g2)
    x1_random = xy[idx]

    print("sampling trajectories ...")
    frames1, _ = euler_sample(ema1, x0_vis, steps=50, record_frames=51)
    frames2, _ = euler_sample(ema2, x0_vis, steps=50, record_frames=51)

    pair_cross_random = crossing_rate(x0_vis.cpu().numpy(), x1_random.cpu().numpy())
    pair_cross_reflow = crossing_rate(x0_vis.cpu().numpy(), x1_pairs[:N_VIS].cpu().numpy())
    curv1 = curvature(frames1)
    curv2 = curvature(frames2)

    step_finals1, step_finals2, cham1, cham2 = [], [], [], []
    for s in STEP_COUNTS:
        _, f1 = euler_sample(ema1, x0_vis, steps=s)
        _, f2 = euler_sample(ema2, x0_vis, steps=s)
        step_finals1.append(f1.numpy())
        step_finals2.append(f2.numpy())
        cham1.append(chamfer(f1.numpy(), target_xy, tree_q))
        cham2.append(chamfer(f2.numpy(), target_xy, tree_q))
        print(f"  steps {s:3d}  1-RF CD {cham1[-1]:.4f}   2-RF CD {cham2[-1]:.4f}")

    # ---- fresh-noise generalization check ----
    # x0_vis comes from the reflow training pairs; test on never-seen noise too.
    print("fresh-noise generalization check ...")
    g3 = torch.Generator(device=device).manual_seed(SAMPLE_SEED + 12345)
    x0_fresh = torch.randn(N_VIS, 2, device=device, generator=g3)
    fresh_steps = [1, 4, 50]
    fresh_sets = {"1rf": [], "2rf": []}
    fresh = {}
    for s in fresh_steps:
        _, f1 = euler_sample(ema1, x0_fresh, steps=s)
        _, f2 = euler_sample(ema2, x0_fresh, steps=s)
        p1, p2 = f1.numpy(), f2.numpy()
        fresh_sets["1rf"].append(p1)
        fresh_sets["2rf"].append(p2)
        fresh[f"chamfer_1rf_{s}"] = chamfer(p1, target_xy, tree_q)
        fresh[f"chamfer_2rf_{s}"] = chamfer(p2, target_xy, tree_q)
        print(f"  fresh {s:3d} steps  1-RF CD {fresh[f'chamfer_1rf_{s}']:.4f}   2-RF CD {fresh[f'chamfer_2rf_{s}']:.4f}")

    stats = {
        "config": vars(args),
        "n_points": n,
        "n_pairs": args.reflow_pairs,
        "history_1rf": hist1,
        "history_2rf": hist2,
        "pair_crossing_random": pair_cross_random,
        "pair_crossing_reflow": pair_cross_reflow,
        "curvature_1rf": curv1,
        "curvature_2rf": curv2,
        "step_counts": STEP_COUNTS,
        "chamfer_1rf": cham1,
        "chamfer_2rf": cham2,
        "fresh_noise": fresh,
    }
    with open(ROOT / "data" / "rf_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("saved data/rf_stats.json")

    np.savez(
        ROOT / "data" / "rf_trajectory.npz",
        x0=x0_vis.cpu().numpy().astype(np.float32),
        x1_random=x1_random.cpu().numpy().astype(np.float32),
        x1_reflow=x1_pairs[:N_VIS].detach().cpu().numpy().astype(np.float32),
        frames_1rf=torch.stack(frames1).numpy().astype(np.float32),
        frames_2rf=torch.stack(frames2).numpy().astype(np.float32),
        step_finals_1rf=np.stack(step_finals1).astype(np.float32),
        step_finals_2rf=np.stack(step_finals2).astype(np.float32),
        step_counts=np.array(STEP_COUNTS, dtype=np.int32),
        target_xy=target_xy.astype(np.float32),
        target_sat=data["sat"].astype(np.float32),
        target_light=data["light"].astype(np.float32),
    )
    print("saved data/rf_trajectory.npz")

    # ---------------- preview ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    globals()["LineCollection"] = LineCollection

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=110)
    rng = np.random.default_rng(0)
    a = x0_vis.cpu().numpy()
    b_random = x1_random.cpu().numpy()
    b_reflow = x1_pairs[:N_VIS].cpu().numpy()
    sel = rng.choice(N_VIS, 700, replace=False)

    ax = axes[0, 0]
    ax.set_facecolor("#0b0d13")
    ax.plot([a[sel, 0], b_random[sel, 0]], [a[sel, 1], b_random[sel, 1]], color="#ff6b6b", lw=0.35, alpha=0.35)
    ax.set_title(f"1-RF random pairing: crossing {pair_cross_random:.3f}", color="white", fontsize=10)
    decorate(ax)

    ax = axes[0, 1]
    ax.set_facecolor("#0b0d13")
    ax.plot([a[sel, 0], b_reflow[sel, 0]], [a[sel, 1], b_reflow[sel, 1]], color="#4dd0e1", lw=0.35, alpha=0.35)
    ax.set_title(f"reflow pairing: crossing {pair_cross_reflow:.3f}", color="white", fontsize=10)
    decorate(ax)

    ax = axes[0, 2]
    ax.set_facecolor("#0b0d13")
    plot_traj(ax, frames1, rng, "#ff8a65")
    ax.set_title(f"1-RF Euler trajectories: extra path {curv1['mean_extra_path']:.3f}", color="white", fontsize=10)
    decorate(ax)

    ax = axes[1, 0]
    ax.set_facecolor("#0b0d13")
    plot_traj(ax, frames2, rng, "#81c784")
    ax.set_title(f"2-RF Euler trajectories: extra path {curv2['mean_extra_path']:.3f}", color="white", fontsize=10)
    decorate(ax)

    ax = axes[1, 1]
    p1, p2 = torch.stack(frames1), torch.stack(frames2)
    r1 = (p1[1:] - p1[:-1]).norm(dim=-1).sum(0) / (p1[-1] - p1[0]).norm(dim=-1).clamp_min(1e-6)
    r2 = (p2[1:] - p2[:-1]).norm(dim=-1).sum(0) / (p2[-1] - p2[0]).norm(dim=-1).clamp_min(1e-6)
    ax.hist(r1.numpy() - 1, bins=60, alpha=0.6, label="1-RF", color="#ff8a65")
    ax.hist(r2.numpy() - 1, bins=60, alpha=0.6, label="2-RF", color="#81c784")
    ax.set_xlabel("path length / chord - 1")
    ax.set_ylabel("points")
    ax.set_title("trajectory straightness")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    ax.plot(STEP_COUNTS, cham1, marker="o", label="1-RF", color="#ff8a65")
    ax.plot(STEP_COUNTS, cham2, marker="o", label="2-RF", color="#81c784")
    ax.set_xscale("log")
    ax.set_xlabel("Euler steps")
    ax.set_ylabel("Chamfer")
    ax.set_title("generation quality vs steps")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = ROOT / "docs" / "preview" / "rf_validate.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("saved", out)
    print("pair crossing random/reflow:", round(pair_cross_random, 4), round(pair_cross_reflow, 4))
    print("curvature 1rf:", curv1)
    print("curvature 2rf:", curv2)


def decorate(ax):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333844")


def plot_traj(ax, frames, rng, color):
    p = torch.stack(frames)
    sel = rng.choice(p.shape[1], 500, replace=False)
    segs = []
    for i in sel:
        pts = p[:, i, :].numpy()
        for t in range(len(pts) - 1):
            segs.append([pts[t], pts[t + 1]])
    ax.add_collection(LineCollection(segs, colors=color, linewidths=0.3, alpha=0.3))


if __name__ == "__main__":
    main()
