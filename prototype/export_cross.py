"""导出「随机配对连线 → 交叉处加权平均 → 轨迹弯曲」视频所需数据。

主角：frames_1rf[:, 280]，在 t=0.62 处偏离直线 0.41（弦长 0.80）。
背景：rf_pairs.npz 中随机抽 3000 条 (x0 -> x1) 直线。
另导出离弯曲点最近的 48 条线，供镜头推进后做焦点。
"""
import base64
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "prototype", "cross_data.js")

traj_npz = np.load(os.path.join(ROOT, "data", "rf_trajectory.npz"))
pairs = np.load(os.path.join(ROOT, "data", "rf_pairs.npz"))

H = 1.8
SCALE = 6000.0
TRAJ_ID = 1946
N_LINE = 3000
N_FOCUS = 48

traj_xy = traj_npz["frames_1rf"][:, TRAJ_ID].astype(np.float32)   # (51,2)
n_frames = traj_xy.shape[0]
bend_k = 31                                                       # t = 31/50 = 0.62
bend_xy = traj_xy[bend_k]
bend_p = np.array([bend_xy[0], bend_xy[1], (bend_k / (n_frames - 1)) * H], dtype=np.float32)

rng = np.random.default_rng(7)
sel = rng.choice(len(pairs["x0"]), N_LINE, replace=False)
line_from = pairs["x0"][sel].astype(np.float32)
line_to = pairs["x1"][sel].astype(np.float32)

x0_all = traj_npz["x0"].astype(np.float32)
target = traj_npz["target_xy"].astype(np.float32)

# 线段 A(x0, z=0) -> B(x1, z=H) 到弯曲点 bend_p 的 3D 距离
A = np.stack([line_from[:, 0], line_from[:, 1], np.zeros(N_LINE)], axis=1)
B = np.stack([line_to[:, 0], line_to[:, 1], np.full(N_LINE, H)], axis=1)
AB = B - A
AP = bend_p[None, :] - A
denom = np.maximum((AB * AB).sum(1), 1e-9)
u = np.clip((AP * AB).sum(1) / denom, 0.0, 1.0)
closest = A + AB * u[:, None]
dist = np.linalg.norm(closest - bend_p[None, :], axis=1)
focus = np.argsort(dist)[:N_FOCUS].astype(np.uint16)

allv = np.concatenate([
    line_from.ravel(), line_to.ravel(), traj_xy.ravel(), x0_all.ravel(), target.ravel(),
    bend_p,
])
assert max(abs(float(allv.min())), abs(float(allv.max()))) * SCALE < 32767


def b64_i16(a):
    q = np.clip(np.round(a * SCALE), -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def b64_u16(a):
    return base64.b64encode(a.astype("<u2").tobytes()).decode("ascii")


payload = {
    "scale": SCALE,
    "H": H,
    "nLine": N_LINE,
    "nFocus": N_FOCUS,
    "trajN": n_frames,
    "bendK": bend_k,
    "n": int(x0_all.shape[0]),
    "lineFrom": b64_i16(line_from),
    "lineTo": b64_i16(line_to),
    "traj": b64_i16(traj_xy),
    "x0": b64_i16(x0_all),
    "target": b64_i16(target),
    "focus": b64_u16(focus),
}

with open(OUT, "w", encoding="ascii") as f:
    f.write("window.CROSS = ")
    f.write(json.dumps(payload))
    f.write(";\n")

print(f"bend point: {bend_xy.round(3)} at t={bend_k/(n_frames-1):.2f}, z={bend_p[2]:.2f}")
print(f"focus line distances: {dist[focus].min():.3f} .. {dist[focus].max():.3f}")
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.1f} KB)")
