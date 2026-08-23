#!/usr/bin/env python3
"""Stage 1: harvest clean, well-segmented instances into a reusable crop bank.

The point of the bank is that every crop is a *whole, isolated* insect with a
trustworthy mask, because a synthetic scene is only as good as the cut-outs that
go into it. Three filters do the work:

- **size**: large instances are annotated far more carefully than small ones, and
  they can always be scaled down later. Scaling up would invent detail.
- **isolation**: an instance that already touches or overlaps a neighbour drags
  part of that neighbour into its crop, which would poison the very signal we are
  trying to teach. These are rejected.
- **completeness**: instances meeting the image border are truncated, so their
  mask is not the whole animal.

Output is one RGBA PNG per instance (alpha = the polygon mask) plus a manifest.

Usage:
    build_crop_bank.py -d <prepared-dataset> -o <bank-dir> [--split train]
                       [--min-size 128] [--max-crops 5000]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np


def read_polygons(label_path: str, width: int, height: int) -> list[np.ndarray]:
    """YOLO-polygon labels -> pixel-space integer polygons."""
    polys: list[np.ndarray] = []
    if not os.path.isfile(label_path):
        return polys
    with open(label_path) as fh:
        for line in fh:
            vals = line.split()
            if len(vals) < 7:  # class + at least 3 (x, y) pairs
                continue
            coords = np.asarray(vals[1:], dtype=np.float64).reshape(-1, 2)
            coords[:, 0] *= width
            coords[:, 1] *= height
            polys.append(coords.astype(np.int32))
    return polys


def instance_boxes(polys: list[np.ndarray]) -> np.ndarray:
    if not polys:
        return np.zeros((0, 4), dtype=np.int32)
    return np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys], dtype=np.int32)


def boxes_touch(a: np.ndarray, b: np.ndarray, margin: int) -> bool:
    """Do two boxes come within `margin` px of each other?"""
    return not (
        a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1]
    )


def harvest_image(
    image_path: str, label_path: str, min_size: int, isolation_margin: int, border_margin: int
) -> list[tuple[np.ndarray, dict]]:
    """Return (rgba_crop, metadata) for every instance passing the filters."""
    image = cv2.imread(image_path)
    if image is None:
        return []
    height, width = image.shape[:2]
    polys = read_polygons(label_path, width, height)
    if not polys:
        return []
    boxes = instance_boxes(polys)

    out = []
    for i, (poly, box) in enumerate(zip(polys, boxes)):
        x0, y0, x1, y1 = box
        side = float(np.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1)))
        if side < min_size:
            continue
        # Truncated by the image border -> not a whole animal.
        if x0 <= border_margin or y0 <= border_margin or x1 >= width - border_margin or y1 >= height - border_margin:
            continue
        # Any neighbour close enough to intrude into the crop?
        if any(boxes_touch(box, other, isolation_margin) for j, other in enumerate(boxes) if j != i):
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        crop = image[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        if crop.size == 0 or crop_mask.max() == 0:
            continue
        # A polygon much smaller than its box means a sliver or a bad annotation.
        fill_ratio = float((crop_mask > 0).mean())
        if fill_ratio < 0.15:
            continue

        rgba = np.dstack([crop, crop_mask])
        out.append((rgba, {
            "source_image": os.path.basename(image_path),
            "instance_index": i,
            "side_px": round(side, 1),
            "fill_ratio": round(fill_ratio, 3),
            "n_vertices": int(len(poly)),
            "dataset": os.path.basename(image_path).split("_")[0],
        }))
    return out


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--data-dir", required=True, help="fb_prepare_data output (holds data.yaml)")
    parser.add_argument("-o", "--out", required=True, help="destination crop bank directory")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--min-size", type=int, default=128, help="minimum sqrt(box area) in px")
    parser.add_argument("--isolation-margin", type=int, default=12,
                        help="reject an instance if a neighbour is within this many px")
    parser.add_argument("--border-margin", type=int, default=4, help="reject instances touching the image border")
    parser.add_argument("--max-crops", type=int, default=5000)
    parser.add_argument("--max-per-image", type=int, default=8, help="keep the bank diverse across source images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.data_dir
    nested = os.path.join(root, "insects")
    base = nested if os.path.isdir(nested) else root
    image_dir = os.path.join(base, "images", args.split)
    label_dir = os.path.join(base, "labels", args.split)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"No images at {image_dir}")

    images = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    random.Random(args.seed).shuffle(images)
    os.makedirs(os.path.join(args.out, "crops"), exist_ok=True)

    manifest, kept, scanned = [], 0, 0
    for image_path in images:
        if kept >= args.max_crops:
            break
        label_path = os.path.join(label_dir, os.path.basename(image_path)[:-4] + ".txt")
        found = harvest_image(image_path, label_path, args.min_size, args.isolation_margin, args.border_margin)
        scanned += 1
        for rgba, meta in found[: args.max_per_image]:
            if kept >= args.max_crops:
                break
            name = f"{kept:06d}_{meta['dataset']}_{int(meta['side_px'])}px.png"
            cv2.imwrite(os.path.join(args.out, "crops", name), rgba)
            meta["file"] = name
            manifest.append(meta)
            kept += 1
        if scanned % 250 == 0:
            print(f"  scanned {scanned} images, kept {kept} crops", flush=True)

    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    sides = np.array([m["side_px"] for m in manifest]) if manifest else np.zeros(1)
    datasets: dict[str, int] = {}
    for m in manifest:
        datasets[m["dataset"]] = datasets.get(m["dataset"], 0) + 1
    print(f"\nkept {kept} crops from {scanned} images -> {args.out}")
    print(f"  size px: median {np.median(sides):.0f}, range {sides.min():.0f}-{sides.max():.0f}")
    print(f"  sub-datasets represented: {len(datasets)}")
    for k, v in sorted(datasets.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {k:<22}{v:>6}")


if __name__ == "__main__":
    main()
