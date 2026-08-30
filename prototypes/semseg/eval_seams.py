#!/usr/bin/env python3
"""PROTOTYPE - measure whether the outline channel can see INTER-INSTANCE seams.

Overall outline IoU mixes two very different pixel populations. The outer contour of an
isolated animal is easy: insect against background, high contrast. The seam between two
touching animals is hard: insect against insect, often near-zero contrast. A model can
score respectably overall while being blind to every seam, and only the seams decide
whether a watershed can split touching instances - which is the whole reason the outline
channel exists.

This splits the ground-truth outline into `seam` and `outer` pixels and reports recall on
each, without needing an instance-extraction step. Seam pixels are those where the dilated
masks of two *different* instances meet.

Usage:
    eval_seams.py CHECKPOINT [-d DATA] [-n N_CROPS] [--tile 1024]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import OUTLINE_PX, TileSegDataset, WindowSampler  # noqa: E402

SEAM_DILATE = 5  # instances whose dilations meet within this many px count as touching


def seam_and_outer(polygons: list[np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Split the union of instance outlines into inter-instance seams and outer contours.

    Args:
        polygons: Instance polygons in crop-local pixel coordinates.
        shape: (height, width) of the crop.

    Returns:
        Two boolean arrays: seam pixels, and outer-contour pixels.
    """
    h, w = shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SEAM_DILATE + 1, 2 * SEAM_DILATE + 1))
    # count how many distinct instances' dilations cover each pixel
    cover = np.zeros((h, w), np.uint8)
    outline = np.zeros((h, w), np.uint8)
    for c in polygons:
        ci = np.round(c).astype(np.int32)
        one = np.zeros((h, w), np.uint8)
        cv2.fillPoly(one, [ci], 1)
        cover += cv2.dilate(one, k)
        cv2.polylines(outline, [ci], isClosed=True, color=1, thickness=OUTLINE_PX)
    ol = outline > 0
    seam = ol & (cover >= 2)
    return seam, ol & ~seam


def main() -> None:
    """Score seam and outer outline recall for a checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-d", "--data", default="/home/quentin/Desktop/flatbug-dir/flat-bug-data/yolo/insects")
    ap.add_argument("-n", "--n-crops", type=int, default=250)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=0.0, help="cap this process's GPU fraction (0 = no cap)")
    a = ap.parse_args()

    if a.cap > 0:
        torch.cuda.set_per_process_memory_fraction(a.cap)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=3, classes=2)
    model.load_state_dict(ck["model"])
    model.eval().cuda().half()
    print(f"checkpoint epoch {ck['epoch']}  logged {ck['metrics']}\n")

    ds = TileSegDataset(a.data, "val", a.tile, seed=1234, augment_data=False)
    sampler = WindowSampler(len(ds), a.n_crops)
    tp = {"seam": 0, "outer": 0}
    tot = {"seam": 0, "outer": 0}
    fg_tp = fg_n = 0
    n_with_seam = 0
    for g in sampler:
        idx = ds.index[g % len(ds.index)]
        path = ds.images[idx]
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, y, v = ds[g]
        # rebuild polygons in the same crop frame the dataset used
        with torch.no_grad():
            p = torch.sigmoid(model(x[None].cuda().half())).float().cpu()[0].numpy()
        pred_ol = (p[1] > a.threshold) & (v[0].numpy() > 0.5)
        gt_ol = (y[1].numpy() > 0.5) & (v[0].numpy() > 0.5)
        if gt_ol.sum() == 0:
            continue
        # approximate the split using connected components of the GT foreground:
        # seams are outline pixels adjacent to two different foreground components
        fg = (y[0].numpy() > 0.5).astype(np.uint8)
        core = ((y[0].numpy() - y[1].numpy()) > 0.5).astype(np.uint8)
        n_lab, lab = cv2.connectedComponents(core)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SEAM_DILATE + 1, 2 * SEAM_DILATE + 1))
        cover = np.zeros_like(fg, dtype=np.uint8)
        for li in range(1, n_lab):
            cover += cv2.dilate((lab == li).astype(np.uint8), k)
        seam = gt_ol & (cover >= 2)
        outer = gt_ol & ~seam
        if seam.sum():
            n_with_seam += 1
        tp["seam"] += int((pred_ol & seam).sum())
        tot["seam"] += int(seam.sum())
        tp["outer"] += int((pred_ol & outer).sum())
        tot["outer"] += int(outer.sum())
        pf = (p[0] > a.threshold) & (v[0].numpy() > 0.5)
        gf = (y[0].numpy() > 0.5) & (v[0].numpy() > 0.5)
        fg_tp += int((pf & gf).sum())
        fg_n += int(gf.sum())

    print(f"crops scored: {a.n_crops}, of which {n_with_seam} contain at least one inter-instance seam\n")
    print(f"{'outline pixel class':28s} {'pixels':>10s} {'recall':>9s}")
    for k_ in ("outer", "seam"):
        r = tp[k_] / max(tot[k_], 1)
        print(f"{'inter-instance SEAM' if k_ == 'seam' else 'outer contour':28s} {tot[k_]:10d} {r:9.4f}")
    print(f"{'foreground (reference)':28s} {fg_n:10d} {fg_tp / max(fg_n, 1):9.4f}")
    if tot["seam"]:
        print(f"\nseam pixels are {tot['seam'] / max(tot['seam'] + tot['outer'], 1):.1%} of all outline pixels")
        print("Seam recall is the number that decides whether a watershed can split touching animals.")


if __name__ == "__main__":
    main()
