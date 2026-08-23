#!/usr/bin/env python3
"""Stage 2: compose synthetic tiles where instances deliberately touch.

Targets the failure mode flatbug handles worst - separating neighbouring
instances - by generating scenes with a controlled amount of contact, and exact
labels for them.

Placement. The first instance lands anywhere. Each later one either attaches to
an already-placed instance (probability ``--touch-prob``) or is dropped in free
space. Attachment is done by walking the new instance outward from the anchor
along a random bearing and binary-searching the distance until the intersection
is the requested fraction of the smaller instance's area - so "touching" means a
measured ~10% overlap, not a guess.

Labels are *modal*: an instance's polygon is the part of it that remains
visible after anything pasted on top of it, which is what an instance
segmenter should predict. Instances hidden below ``--min-visible`` are dropped
from the labels rather than left as impossible targets.

Backgrounds are cut from real images at spots where the ground truth says no
instance is present, so background statistics (paper texture, scanner
illumination, sticky-trap glue) stay realistic. Alpha edges are feathered,
because a hard paste seam is a shortcut the model would happily learn instead of
the animal's outline.

Usage:
    compose_scenes.py -b <crop-bank> -d <prepared-dataset> -o <out-dir> -n 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np


# ----------------------------------------------------------------- backgrounds
def load_background_pool(data_dir: str, split: str, n: int, tile: int, rng: random.Random) -> list[np.ndarray]:
    """Cut empty patches from real images, guided by the ground-truth polygons."""
    base = os.path.join(data_dir, "insects")
    base = base if os.path.isdir(base) else data_dir
    images = sorted(glob.glob(os.path.join(base, "images", split, "*.jpg")))
    rng.shuffle(images)
    patches: list[np.ndarray] = []
    for path in images:
        if len(patches) >= n:
            break
        image = cv2.imread(path)
        if image is None or min(image.shape[:2]) < tile:
            continue
        h, w = image.shape[:2]
        label = os.path.join(base, "labels", split, os.path.basename(path)[:-4] + ".txt")
        occupied = np.zeros((h, w), np.uint8)
        if os.path.isfile(label):
            for line in open(label):
                vals = line.split()
                if len(vals) < 7:
                    continue
                poly = (np.asarray(vals[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(occupied, [poly], 255)
        for _ in range(12):                      # a few tries per image
            x = rng.randint(0, w - tile); y = rng.randint(0, h - tile)
            if occupied[y:y + tile, x:x + tile].mean() < 1.0:   # essentially empty
                patches.append(image[y:y + tile, x:x + tile].copy())
                break
    return patches


# ------------------------------------------------------------------- instances
def augment_crop(rgba: np.ndarray, target_side: int, rng: random.Random) -> np.ndarray:
    """Flip, rotate by an arbitrary angle, and scale to a target size."""
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    if rng.random() < 0.5:
        rgba = rgba[::-1, :]
    angle = rng.uniform(0, 360)
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    rgba = cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    side = float(np.sqrt(rgba.shape[0] * rgba.shape[1]))
    scale = target_side / max(side, 1)
    if scale < 1.0:                              # only ever scale down - never invent detail
        rgba = cv2.resize(rgba, (max(int(rgba.shape[1] * scale), 8), max(int(rgba.shape[0] * scale), 8)),
                          interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgba)


def paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, feather: int) -> np.ndarray:
    """Alpha-composite at (x, y); returns the binary mask of what was drawn."""
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    out = np.zeros((ch, cw), np.uint8)
    if x1 <= x0 or y1 <= y0:
        return out
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = sub[:, :, 3].astype(np.float32) / 255.0
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    a3 = alpha[:, :, None]
    canvas[y0:y1, x0:x1] = (sub[:, :, :3] * a3 + canvas[y0:y1, x0:x1] * (1 - a3)).astype(np.uint8)
    out[y0:y1, x0:x1] = (sub[:, :, 3] > 127).astype(np.uint8)
    return out


def overlap_fraction(new_mask: np.ndarray, occupied: np.ndarray) -> float:
    area = int(new_mask.sum())
    if area == 0:
        return 1.0
    return float(np.logical_and(new_mask, occupied).sum()) / area


def mask_at(rgba: np.ndarray, x: int, y: int, shape: tuple[int, int]) -> np.ndarray:
    """Binary mask of `rgba` placed at (x, y) on a canvas of `shape`."""
    h, w = rgba.shape[:2]
    ch, cw = shape
    out = np.zeros((ch, cw), np.uint8)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = (rgba[y0 - y:y1 - y, x0 - x:x1 - x, 3] > 127).astype(np.uint8)
    return out


def find_touching_position(
    rgba: np.ndarray, anchor_mask: np.ndarray, target: float, shape: tuple[int, int], rng: random.Random
) -> tuple[int, int] | None:
    """Binary-search outward from the anchor until overlap ~= `target`."""
    ys, xs = np.nonzero(anchor_mask)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = rgba.shape[:2]
    bearing = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(bearing), np.sin(bearing)
    far = float(np.hypot(*anchor_mask.shape) * 0.5 + max(h, w))

    lo, hi = 0.0, far                       # lo overlaps heavily, hi not at all
    best = None
    for _ in range(18):
        mid = (lo + hi) / 2
        x = int(cx + dx * mid - w / 2)
        y = int(cy + dy * mid - h / 2)
        frac = overlap_fraction(mask_at(rgba, x, y, shape), anchor_mask)
        if frac > target:
            lo = mid
        else:
            hi = mid
            best = (x, y)
        if abs(frac - target) < 0.02:
            return (x, y)
    return best


# ---------------------------------------------------------------------- scenes
def compose(bank: list[str], backgrounds: list[np.ndarray], args, rng: random.Random):
    tile = args.tile_size
    canvas = (backgrounds[rng.randrange(len(backgrounds))].copy() if backgrounds
              else np.full((tile, tile, 3), 235, np.uint8))
    canvas = cv2.resize(canvas, (tile, tile))

    n_inst = rng.randint(*args.instances)
    placed_masks: list[np.ndarray] = []
    occupied = np.zeros((tile, tile), np.uint8)
    records = []

    for _ in range(n_inst):
        rgba = cv2.imread(rng.choice(bank), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        target_side = rng.randint(*args.instance_size)
        rgba = augment_crop(rgba, target_side, rng)
        if min(rgba.shape[:2]) < 8 or max(rgba.shape[:2]) >= tile:
            continue

        pos = None
        if placed_masks and rng.random() < args.touch_prob:
            anchor = placed_masks[rng.randrange(len(placed_masks))]
            pos = find_touching_position(rgba, anchor, args.overlap, (tile, tile), rng)
        if pos is None:
            for _ in range(40):                      # free space
                x = rng.randint(0, tile - rgba.shape[1])
                y = rng.randint(0, tile - rgba.shape[0])
                if overlap_fraction(mask_at(rgba, x, y, (tile, tile)), occupied) < 0.01:
                    pos = (x, y); break
        if pos is None:
            continue

        x, y = pos
        cand = mask_at(rgba, x, y, (tile, tile))
        if cand.sum() == 0 or overlap_fraction(cand, occupied) > args.max_overlap:
            continue
        drawn = paste(canvas, rgba, x, y, args.feather)
        # Anything already placed is now partly hidden -> shrink its visible mask.
        for m in placed_masks:
            m &= ~drawn.astype(bool)
        placed_masks.append(drawn.astype(bool))
        records.append(int(drawn.sum()))
        occupied |= drawn

    polys = []
    for m, full_area in zip(placed_masks, records):
        area = int(m.sum())
        if area == 0 or area / max(full_area, 1) < args.min_visible:
            continue                                  # too occluded to be a fair target
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.0, True).squeeze(1)
        if len(c) >= 3:
            polys.append(c)
    return canvas, polys


def main() -> None:  # noqa: D103
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-b", "--bank", required=True, help="crop bank from build_crop_bank.py")
    p.add_argument("-d", "--data-dir", help="prepared dataset, for real background patches")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("-n", "--n-scenes", type=int, default=100)
    p.add_argument("--split", default="train")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--instances", type=int, nargs=2, default=(8, 25), metavar=("MIN", "MAX"))
    p.add_argument("--instance-size", type=int, nargs=2, default=(60, 200), metavar=("MIN", "MAX"),
                   help="target sqrt(area) of pasted instances, in px")
    p.add_argument("--touch-prob", type=float, default=0.5,
                   help="probability an instance is attached to an already-placed one")
    p.add_argument("--overlap", type=float, default=0.10, help="target overlap fraction when touching")
    p.add_argument("--max-overlap", type=float, default=0.35, help="reject placements above this")
    p.add_argument("--min-visible", type=float, default=0.4, help="drop labels hidden below this fraction")
    p.add_argument("--feather", type=int, default=1, help="alpha feather radius in px")
    p.add_argument("--n-backgrounds", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    bank = sorted(glob.glob(os.path.join(args.bank, "crops", "*.png")))
    if not bank:
        raise FileNotFoundError(f"No crops in {args.bank}/crops")
    backgrounds = load_background_pool(args.data_dir, args.split, args.n_backgrounds, args.tile_size, rng) \
        if args.data_dir else []
    print(f"bank {len(bank)} crops | backgrounds {len(backgrounds)}")

    img_dir = os.path.join(args.out, "images", args.split)
    lbl_dir = os.path.join(args.out, "labels", args.split)
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)

    counts = []
    for i in range(args.n_scenes):
        canvas, polys = compose(bank, backgrounds, args, rng)
        name = f"synth_{i:05d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as fh:
            for c in polys:
                norm = (c.astype(np.float64) / [args.tile_size, args.tile_size]).clip(0, 1)
                fh.write("0 " + " ".join(f"{v:.6f}" for v in norm.reshape(-1)) + "\n")
        counts.append(len(polys))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.n_scenes} scenes", flush=True)

    with open(os.path.join(args.out, "compose_config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)
    print(f"\nwrote {args.n_scenes} scenes to {args.out}")
    print(f"  instances/scene: mean {np.mean(counts):.1f}, range {min(counts)}-{max(counts)}")


if __name__ == "__main__":
    main()
