"""Step 1: rembg -> RGBA -> isolate the "three arches" -> sample 8000 points.

The target is only the central three-arch mark (the four red pillars that form
three arches), without the surrounding ring, lettering or "1958".

Transparent pixels are empty and are never sampled. Coordinates are normalized
to roughly [-1, 1] using the three-arch bounding box, so the mark fills the
frame.
"""
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N_POINTS = 8000
SEED = 20261026
PAD_FRAC = 1.03  # small margin around the mark


def checkerboard(h, w, cell=16):
    yy, xx = np.mgrid[0:h, 0:w]
    board = ((yy // cell + xx // cell) % 2).astype(np.float32)
    return (0.82 + 0.12 * board)[..., None].repeat(3, axis=2)


def get_full_rgba():
    full_path = ROOT / "target" / "target_rgba_full.png"
    if full_path.exists():
        return Image.open(full_path).convert("RGBA")
    from rembg import remove

    src = Image.open(ROOT / "target" / "target.jpg").convert("RGBA")
    out = remove(src)
    out.save(full_path)
    return out


def three_arch_mask(rgb: np.ndarray) -> np.ndarray:
    """Return a uint8 mask of the four pillars forming the three arches."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    red = ((r > 90) & (r > g + 45) & (r > b + 45)).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(red, connectivity=8)
    h = stats[:, cv2.CC_STAT_HEIGHT]
    w = stats[:, cv2.CC_STAT_WIDTH]
    area = stats[:, cv2.CC_STAT_AREA]
    # pillars: tall, narrow, sizeable; excludes the ring (918x918), lettering and 1958
    selected = [
        i for i in range(1, n)
        if h[i] > 300 and w[i] < 200 and area[i] > 5000
    ]
    assert len(selected) == 4, f"expected 4 pillars, found {len(selected)}: {selected}"
    mask = np.isin(lab, selected).astype(np.uint8)
    # include anti-aliased rim
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def main():
    src = Image.open(ROOT / "target" / "target.jpg").convert("RGBA")
    rgb = np.asarray(src)[..., :3]

    full = get_full_rgba()
    if not (ROOT / "target" / "target_rgba_full.png").exists():
        full.save(ROOT / "target" / "target_rgba_full.png")

    mask = three_arch_mask(rgb)
    fg_alpha = np.asarray(full)[..., 3]
    mask = (mask & (fg_alpha > 0)).astype(np.uint8)

    ys, xs = np.where(mask > 0)
    print(f"three-arch pixels: {len(xs)}")
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    print(f"bbox: x {x0}..{x1} (w={x1-x0+1}), y {y0}..{y1} (h={y1-y0+1})")

    # final RGBA: three arches only, everything else transparent
    out = np.zeros_like(np.asarray(src))
    out[..., :3] = rgb
    out[..., 3] = mask * 255
    Image.fromarray(out, "RGBA").save(ROOT / "target" / "target_rgba.png")
    print("saved target_rgba.png (three arches, 4 channels)")

    # ---------- sample points ----------
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(xs), size=N_POINTS, replace=True)
    px = xs[pick].astype(np.float64) + rng.uniform(-0.5, 0.5, N_POINTS)
    py = ys[pick].astype(np.float64) + rng.uniform(-0.5, 0.5, N_POINTS)

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    scale = 0.5 * max(x1 - x0, y1 - y0) * PAD_FRAC
    xy = np.stack([(px - cx) / scale, -(py - cy) / scale], axis=1).astype(np.float32)

    point_rgb = rgb[ys[pick], xs[pick]].astype(np.float32) / 255.0
    mx = point_rgb.max(axis=1)
    mn = point_rgb.min(axis=1)
    light = (mx + mn) / 2.0
    sat = np.where(mx - mn < 1e-6, 0.0, (mx - mn) / np.clip(1.0 - np.abs(2 * light - 1), 1e-6, None))
    hue = np.zeros_like(mx)
    r, g, b = point_rgb[:, 0], point_rgb[:, 1], point_rgb[:, 2]
    d = mx - mn
    m = d > 1e-6
    idx = m & (mx == r)
    hue[idx] = ((g[idx] - b[idx]) / d[idx]) % 6
    idx = m & (mx == g)
    hue[idx] = (b[idx] - r[idx]) / d[idx] + 2
    idx = m & (mx == b)
    hue[idx] = (r[idx] - g[idx]) / d[idx] + 4
    hue = (hue / 6.0) % 1.0

    np.savez(
        ROOT / "data" / "target_points.npz",
        xy=xy,
        rgb=point_rgb,
        sat=sat.astype(np.float32),
        light=light.astype(np.float32),
        hue=hue.astype(np.float32),
        image_size=np.array([rgb.shape[0], rgb.shape[1], cx, cy, scale], dtype=np.float32),
    )
    print("saved data/target_points.npz")
    print("xy range", xy.min(0), "..", xy.max(0))
    print("color mean", point_rgb.mean(0).round(3), "sat mean", round(float(sat.mean()), 3))

    # ---------- preview ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(16, 5.2), dpi=110)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(rgb)
    ax.set_title("target.jpg: take only the three arches")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 2)
    comp = out[..., :3].astype(np.float32) / 255.0
    a = out[..., 3:4].astype(np.float32) / 255.0
    ax.imshow(comp * a + checkerboard(h, w) * (1 - a))
    ax.set_title("target_rgba.png (checker = transparent)")
    ax.axis("off")

    ax = fig.add_subplot(1, 3, 3)
    ax.set_facecolor("#0b0d13")
    ax.scatter(xy[:, 0], xy[:, 1], c=point_rgb, s=2.6, linewidths=0, alpha=0.95)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_title(f"{N_POINTS} sampled points")
    ax.axis("off")

    fig.patch.set_facecolor("white")
    fig.tight_layout()
    out_path = ROOT / "docs" / "preview" / "prepare_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print("saved", out_path)


if __name__ == "__main__":
    main()
