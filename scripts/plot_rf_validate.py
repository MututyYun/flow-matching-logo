"""Plot the RF validation figure from saved data (no retraining)."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data" / "rf_trajectory.npz")
s = json.load(open(ROOT / "data" / "rf_stats.json", encoding="utf-8"))

x0 = d["x0"]
x1_random = d["x1_random"]
x1_reflow = d["x1_reflow"]
f1 = d["frames_1rf"]
f2 = d["frames_2rf"]
rng = np.random.default_rng(0)


def decorate(ax):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333844")


def traj(ax, frames, color):
    sel = rng.choice(frames.shape[1], 500, replace=False)
    segs = []
    for i in sel:
        pts = frames[:, i, :]
        for t in range(len(pts) - 1):
            segs.append([pts[t], pts[t + 1]])
    ax.add_collection(LineCollection(segs, colors=color, linewidths=0.3, alpha=0.3))


fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=110)
sel = rng.choice(len(x0), 700, replace=False)

ax = axes[0, 0]
ax.set_facecolor("#0b0d13")
ax.plot([x0[sel, 0], x1_random[sel, 0]], [x0[sel, 1], x1_random[sel, 1]], color="#ff6b6b", lw=0.35, alpha=0.35)
ax.set_title(f"1-RF random pairing: crossing {s['pair_crossing_random']:.3f}", color="white", fontsize=10)
decorate(ax)

ax = axes[0, 1]
ax.set_facecolor("#0b0d13")
ax.plot([x0[sel, 0], x1_reflow[sel, 0]], [x0[sel, 1], x1_reflow[sel, 1]], color="#4dd0e1", lw=0.35, alpha=0.35)
ax.set_title(f"reflow pairing: crossing {s['pair_crossing_reflow']:.3f}", color="white", fontsize=10)
decorate(ax)

ax = axes[0, 2]
ax.set_facecolor("#0b0d13")
traj(ax, f1, "#ff8a65")
ax.set_title(f"1-RF trajectories: extra path {s['curvature_1rf']['mean_extra_path']:.3f}", color="white", fontsize=10)
decorate(ax)

ax = axes[1, 0]
ax.set_facecolor("#0b0d13")
traj(ax, f2, "#81c784")
ax.set_title(f"2-RF trajectories: extra path {s['curvature_2rf']['mean_extra_path']:.4f}", color="white", fontsize=10)
decorate(ax)

ax = axes[1, 1]
r1 = np.linalg.norm(np.diff(f1, axis=0), axis=-1).sum(0) / np.clip(np.linalg.norm(f1[-1] - f1[0], axis=-1), 1e-6, None)
r2 = np.linalg.norm(np.diff(f2, axis=0), axis=-1).sum(0) / np.clip(np.linalg.norm(f2[-1] - f2[0], axis=-1), 1e-6, None)
ax.hist(r1 - 1, bins=60, alpha=0.6, label="1-RF", color="#ff8a65")
ax.hist(r2 - 1, bins=60, alpha=0.6, label="2-RF", color="#81c784")
ax.set_xlabel("path length / chord - 1")
ax.set_ylabel("points")
ax.set_title("trajectory straightness")
ax.legend()
ax.grid(alpha=0.25)

ax = axes[1, 2]
ax.plot(s["step_counts"], s["chamfer_1rf"], marker="o", label="1-RF", color="#ff8a65")
ax.plot(s["step_counts"], s["chamfer_2rf"], marker="o", label="2-RF", color="#81c784")
ax.set_xscale("log")
ax.set_xlabel("Euler steps")
ax.set_ylabel("Chamfer distance")
ax.set_title("generation quality vs steps")
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
out = ROOT / "docs" / "preview" / "rf_validate.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, facecolor="white", bbox_inches="tight")
print("saved", out)
