"""Step 2: sample one DDIM trajectory, compare step counts, compute mapping stats."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Schedule, VelocityMLP, ddim_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = 8000
SAMPLE_SEED = 20261026
TRAJ_STEPS = 100
TRAJ_FRAMES = 51
STEP_COUNTS = [1, 2, 5, 10, 25, 50, 100, 200]
CROSS_SAMPLE = 300
MIN_ENDPOINT_DIST = 0.01


def crossing_count(p0: np.ndarray, p1: np.ndarray, rng: np.random.Generator):
    """Proper crossings among line segments p0[i] -> p1[i], sampled."""
    n = len(p0)
    sel = rng.choice(n, min(CROSS_SAMPLE, n), replace=False)
    a, b = p0[sel], p1[sel]

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
    count = int(proper[iu].sum())
    return count, len(iu[0])


def displacement_stats(p0, p1, tree_neighbors, rng):
    d = p1 - p0
    length = np.linalg.norm(d, axis=1)
    local = np.linalg.norm(d[:, None, :] - d[tree_neighbors], axis=-1).mean()
    count, pairs = crossing_count(p0, p1, rng)
    return {
        "mean_disp": float(length.mean()),
        "max_disp": float(length.max()),
        "p95_disp": float(np.percentile(length, 95)),
        "smoothness": float(local),
        "crossings": count,
        "cross_pairs": pairs,
        "cross_rate": float(count / pairs) if pairs else 0.0,
    }


def chamfer(p, q, tree_q):
    return float(cKDTree(p).query(q)[0].mean() + tree_q.query(p)[0].mean())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ROOT / "model.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = VelocityMLP(hidden=cfg["hidden"], depth=cfg["depth"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    schedule = Schedule()
    print(f"loaded model: {cfg['n_params']:,} params, trained {cfg['steps']} steps")

    data = np.load(ROOT / "data" / "target_points.npz")
    target_xy = data["xy"]
    tree_q = cKDTree(target_xy)

    gen = torch.Generator(device=device).manual_seed(SAMPLE_SEED)
    x_T = torch.randn(N, 2, device=device, generator=gen)

    print(f"sampling {TRAJ_STEPS} steps ...")
    frames, final = ddim_sample(model, schedule, x_T, steps=TRAJ_STEPS, record_frames=TRAJ_FRAMES)
    frames = torch.stack(frames).numpy()  # [F, N, 2]
    final = final.numpy()
    print("frames", frames.shape)

    rng = np.random.default_rng(SAMPLE_SEED)
    tree_nb = cKDTree(frames[0]).query(frames[0], k=9)[1][:, 1:]  # fixed point identity neighbors

    per_frame = []
    for t in range(len(frames) - 1):
        st = displacement_stats(frames[t], frames[t + 1], tree_nb, rng)
        st["frame"] = t
        st["z"] = t / (len(frames) - 1)
        per_frame.append(st)
        if (t + 1) % 10 == 0:
            print(f"  frame {t+1}/{len(frames)-1}  mean_disp {st['mean_disp']:.4f}  cross {st['cross_rate']:.4f}")

    print("single-step baseline ...")
    _, final_1 = ddim_sample(model, schedule, x_T, steps=1)
    single = displacement_stats(x_T.detach().cpu().numpy(), final_1.numpy(), tree_nb, rng)
    single["chamfer"] = chamfer(final_1.numpy(), target_xy, tree_q)

    print("step-count sweep ...")
    step_finals = []
    step_chamfer = []
    for s in STEP_COUNTS:
        _, f = ddim_sample(model, schedule, x_T, steps=s)
        p = f.numpy()
        step_finals.append(p)
        cd = chamfer(p, target_xy, tree_q)
        step_chamfer.append(cd)
        print(f"  steps {s:4d}  chamfer {cd:.4f}")

    step_finals = np.stack(step_finals)  # [S, N, 2]

    np.savez(
        ROOT / "data" / "trajectory.npz",
        frames=frames.astype(np.float32),
        x_T=x_T.detach().cpu().numpy().astype(np.float32),
        final=final.astype(np.float32),
        final_1=final_1.numpy().astype(np.float32),
        step_finals=step_finals.astype(np.float32),
        step_counts=np.array(STEP_COUNTS, dtype=np.int32),
        target_xy=target_xy.astype(np.float32),
        target_sat=data["sat"].astype(np.float32),
        target_light=data["light"].astype(np.float32),
    )
    print("saved data/trajectory.npz")

    stats = {
        "config": cfg,
        "loss_history": ckpt["history"],
        "sample_seed": SAMPLE_SEED,
        "n_points": N,
        "traj_steps": TRAJ_STEPS,
        "traj_frames": int(frames.shape[0]),
        "step_counts": STEP_COUNTS,
        "step_chamfer": step_chamfer,
        "single_step": single,
        "per_frame": per_frame,
    }
    with open(ROOT / "data" / "sample_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("saved data/sample_stats.json")

    # ---------------- preview ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = plt.figure(figsize=(18, 9), dpi=110)

    # (1) full trajectories
    ax = fig.add_subplot(2, 3, 1)
    ax.set_facecolor("#0b0d13")
    sel = rng.choice(N, 1200, replace=False)
    cmap = plt.get_cmap("turbo")
    segs, cols = [], []
    for i in sel:
        for t in range(len(frames) - 1):
            segs.append([frames[t, i], frames[t + 1, i]])
            cols.append(cmap(t / (len(frames) - 1)))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.25, alpha=0.35))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("multi-step: 1200 point trajectories (color = time)", color="white", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # (2) single-step long mapping lines
    ax = fig.add_subplot(2, 3, 2)
    ax.set_facecolor("#0b0d13")
    sel = rng.choice(N, 900, replace=False)
    ax.plot([x_T.cpu().numpy()[sel, 0], final_1.numpy()[sel, 0]],
            [x_T.cpu().numpy()[sel, 1], final_1.numpy()[sel, 1]],
            color="#ff6b6b", lw=0.25, alpha=0.25)
    ax.scatter(final_1.numpy()[sel, 0], final_1.numpy()[sel, 1], s=0.8, c="#ffd166", alpha=0.7, linewidths=0)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title("1-step: long, collapsing mapping lines", color="white", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # (3) one mid-trajectory displacement field
    ax = fig.add_subplot(2, 3, 3)
    ax.set_facecolor("#0b0d13")
    mid = len(frames) // 2
    sel = rng.choice(N, 500, replace=False)
    p0, p1 = frames[mid][sel], frames[mid + 1][sel]
    ax.plot([p0[:, 0], p1[:, 0]], [p0[:, 1], p1[:, 1]], color="#4dd0e1", lw=0.5, alpha=0.6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_title(f"multi-step frame {mid}: short local moves", color="white", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # (4) displacement curves
    ax = fig.add_subplot(2, 3, 4)
    z = [f["z"] for f in per_frame]
    ax.plot(z, [f["mean_disp"] for f in per_frame], label="mean disp", color="#4dd0e1")
    ax.plot(z, [f["max_disp"] for f in per_frame], label="max disp", color="#ff8a65")
    ax.axhline(single["mean_disp"], color="#ff6b6b", ls="--", label="1-step mean")
    ax.set_xlabel("denoising progress")
    ax.set_ylabel("displacement (normalized units)")
    ax.set_title("per-step displacement: multi-step vs 1-step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (5) smoothness and crossings
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(z, [f["smoothness"] for f in per_frame], label="displacement-field roughness", color="#ba68c8")
    ax.axhline(single["smoothness"], color="#ff6b6b", ls="--", label="1-step")
    ax.set_xlabel("denoising progress")
    ax.set_title("local roughness of the displacement field")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (6) chamfer vs steps
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(STEP_COUNTS, step_chamfer, marker="o", color="#81c784")
    ax.axhline(single["chamfer"], color="#ff6b6b", ls="--", label="1-step")
    ax.set_xscale("log")
    ax.set_xlabel("DDIM steps")
    ax.set_ylabel("Chamfer distance")
    ax.set_title("generation quality vs step count")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    for ax in fig.axes[:3]:
        for sp in ax.spines.values():
            sp.set_color("#333844")

    fig.tight_layout()
    out = ROOT / "assets" / "sample_preview.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("saved", out)

    print("\nsingle-step:", {k: round(v, 4) if isinstance(v, float) else v for k, v in single.items()})


if __name__ == "__main__":
    main()
