"""导出色板染色视频所需数据：底噪点云、三拱门点云、原图明度/饱和度。"""
import base64
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "prototype", "color_data.js")

d = np.load(os.path.join(ROOT, "data", "rf_trajectory.npz"))
x0 = d["x0"].astype(np.float32)
target = d["target_xy"].astype(np.float32)
light = d["target_light"].astype(np.float32)
sat = d["target_sat"].astype(np.float32)
n = int(x0.shape[0])

SCALE = 6000.0
allv = np.concatenate([x0.ravel(), target.ravel()])
assert max(abs(float(allv.min())), abs(float(allv.max()))) * SCALE < 32767


def b64_i16(a):
    q = np.clip(np.round(a * SCALE), -32768, 32767).astype("<i2")
    return base64.b64encode(q.tobytes()).decode("ascii")


def b64_f32(a):
    return base64.b64encode(a.astype("<f4").tobytes()).decode("ascii")


payload = {
    "scale": SCALE,
    "n": n,
    "x0": b64_i16(x0),
    "target": b64_i16(target),
    "light": b64_f32(light),
    "sat": b64_f32(sat),
}

with open(OUT, "w", encoding="ascii") as f:
    f.write("window.COLOR = ")
    f.write(json.dumps(payload))
    f.write(";\n")

print("light range:", float(light.min()), float(light.max()))
print("sat range:", float(sat.min()), float(sat.max()))
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.1f} KB)")
