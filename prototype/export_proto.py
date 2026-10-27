"""把 rf_trajectory.npz 导出成浏览器可直接加载的 data.js（int16 量化 + base64）。

原型阶段用现有的 8000 点数据，不依赖 15000 点重训。
"""
import base64
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "data", "rf_trajectory.npz")
OUT = os.path.join(ROOT, "prototype", "data.js")

d = np.load(NPZ)
x0 = d["x0"].astype(np.float32)
frames1 = d["frames_1rf"].astype(np.float32)
frames2 = d["frames_2rf"].astype(np.float32)
step1 = d["step_finals_1rf"].astype(np.float32)   # (5, N, 2): 1/2/4/10/50 steps
step2 = d["step_finals_2rf"].astype(np.float32)
step_counts = [int(v) for v in d["step_counts"]]
sat = d["target_sat"].astype(np.float32)
light = d["target_light"].astype(np.float32)

n = int(x0.shape[0])
frames = int(frames1.shape[0])

allv = np.concatenate([x0.ravel(), frames1.ravel(), frames2.ravel(), step1.ravel(), step2.ravel()])
vmin, vmax = float(allv.min()), float(allv.max())
scale = 6000.0
assert max(abs(vmin), abs(vmax)) * scale < 32767, "int16 overflow"

rng = np.random.default_rng(42)
n_line = 900
line_idx = np.sort(rng.choice(n, n_line, replace=False)).astype(np.uint16)


def b64_i16(a):
    q = np.clip(np.round(a * scale), -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def b64_f32(a):
    return base64.b64encode(a.astype("<f4").tobytes()).decode("ascii")


def b64_u16(a):
    return base64.b64encode(a.astype("<u2").tobytes()).decode("ascii")


payload = {
    "n": n,
    "frames": frames,
    "nLine": n_line,
    "scale": scale,
    "x0": b64_i16(x0),
    "frames1": b64_i16(frames1),
    "frames2": b64_i16(frames2),
    "step1": b64_i16(step1),
    "step2": b64_i16(step2),
    "stepCounts": step_counts,
    "sat": b64_f32(sat),
    "light": b64_f32(light),
    "lineIdx": b64_u16(line_idx),
}

with open(OUT, "w", encoding="ascii") as f:
    f.write("window.PROTO = ")
    f.write(json.dumps(payload))
    f.write(";\n")

print(f"coord range: [{vmin:.3f}, {vmax:.3f}]  scale={scale}")
print(f"n={n} frames={frames} lines={n_line}")
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1024 / 1024:.2f} MB)")
