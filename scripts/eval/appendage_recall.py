#!/usr/bin/env python3
"""Score mask quality on thin structures, which mask IoU is almost blind to.

Thin appendages - legs, antennae, wing edges - are only about 7.4% of ground-truth
mask area. A prediction that captures a perfect body and *no* appendages at all
still scores a mask IoU of ~0.93, above the ~0.80 that trained flat-bug models
actually achieve. Mask IoU therefore cannot distinguish "lost every limb" from
"slightly loose boundary", and the spread it reports between models (0.005 across
five checkpoints) says nothing about limb fidelity.

This script splits each ground-truth mask into a core body and its thin parts, and
reports recall separately for each. A pixel is "thin" if a morphological opening
removes it, i.e. it is not covered by any disk of radius 0.4x the instance's own
maximum half-thickness lying inside the mask. The radius is per-instance, so the
decomposition is scale-invariant and a large beetle's legs are judged the same way
as a small fly's.

It also reports the predicted-to-ground-truth area ratio per size bin, which
exposes the opposite failure: annotations drawn conservatively around small
objects, where the model is penalised for being less tight than the annotator.

Input is the per-image CSV that `fb_evaluate` writes, which stores the matched
ground-truth and predicted contours as `contour_1` and `contour_2`.

Usage:
    appendage_recall.py -i EVAL_DIR [EVAL_DIR ...] [--labels A B] [-o out.csv]
"""

from __future__ import annotations

import argparse
import ast
import glob
import os

import cv2
import numpy as np
import pandas as pd

PAD = 6  # Border kept around the union of both contours, so opening is not clipped.
THICKNESS_FRACTION = 0.40  # A part is "thin" below this fraction of the max half-thickness.
SIZE_BINS = [0, 32, 64, 128, 256, 512, np.inf]
SIZE_LABELS = ["<32", "32-64", "64-128", "128-256", "256-512", "512+"]


def decompose(contour: np.ndarray, offset: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise a contour and split it into core-body and thin-appendage pixels.

    Args:
        contour (np.ndarray): Polygon vertices of shape (M, 2) in image coordinates.
        offset (np.ndarray): Origin of the local raster frame, of shape (2,).
        shape (tuple[int, int]): Height and width of the local raster frame.

    Returns:
        (tuple[np.ndarray, np.ndarray]): The filled mask, and a boolean array of thin pixels.
    """
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [(contour - offset).astype(np.int32)], 1)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    radius = max(1, int(round(THICKNESS_FRACTION * distance.max())))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
    return mask, (mask > 0) & (core == 0)


def score_directory(eval_dir: str, min_area: int = 20, max_pixels: int = 4_000_000) -> pd.DataFrame:
    """Score every matched instance in a directory of `fb_evaluate` CSVs.

    Args:
        eval_dir (str): Directory containing one CSV per evaluated image.
        min_area (int): Skip instances whose ground-truth mask is smaller than this, in pixels.
        max_pixels (int): Skip instances whose local raster frame would exceed this, to bound memory.

    Returns:
        (pd.DataFrame): One row per matched instance.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "*.csv"))):
        dataset = os.path.basename(path).split("_")[0]
        try:
            table = pd.read_csv(path, sep=";")
        except Exception:  # noqa: BLE001 - a malformed or empty CSV should not abort the sweep
            continue
        if "IoU" not in table.columns:
            continue
        for _, row in table[(table.idx_1 != -1) & (table.idx_2 != -1)].iterrows():
            try:
                truth = np.array(ast.literal_eval(row.contour_1), float)
                pred = np.array(ast.literal_eval(row.contour_2), float)
            except Exception:  # noqa: BLE001 - skip unparseable contours
                continue
            if len(truth) < 3 or len(pred) < 3:
                continue
            both = np.vstack([truth, pred])
            offset = both.min(0) - PAD
            shape = tuple((both.max(0) - offset + PAD).astype(int)[::-1])
            if min(shape) < 3 or shape[0] * shape[1] > max_pixels:
                continue
            gt_mask, thin = decompose(truth, offset, shape)
            if gt_mask.sum() < min_area or thin.sum() == 0:
                continue
            pred_mask = np.zeros(shape, np.uint8)
            cv2.fillPoly(pred_mask, [(pred - offset).astype(np.int32)], 1)
            core = (gt_mask > 0) & ~thin
            rows.append({
                "dataset": dataset,
                "size": float(np.sqrt(gt_mask.sum())),
                "iou": float((gt_mask & pred_mask).sum() / (gt_mask | pred_mask).sum()),
                "thin_fraction": float(thin.sum() / gt_mask.sum()),
                "recall_core": float((pred_mask[core] > 0).mean()),
                "recall_thin": float((pred_mask[thin] > 0).mean()),
                # IoU obtainable by predicting a perfect body and no appendages at all.
                "iou_without_appendages": float(core.sum() / gt_mask.sum()),
                "pred_over_gt_area": float(pred_mask.sum() / gt_mask.sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    """Score one or more evaluation directories and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Directories of fb_evaluate CSVs")
    parser.add_argument("--labels", nargs="+", default=None, help="Display name per input directory")
    parser.add_argument("-o", "--output", default=None, help="Write the per-instance table here")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.input]
    if len(labels) != len(args.input):
        raise ValueError(f"Got {len(labels)} labels for {len(args.input)} input directories")

    frames = []
    for label, directory in zip(labels, args.input):
        frame = score_directory(directory)
        if frame.empty:
            print(f"{label}: no scorable instances found in {directory}")
            continue
        frame["arm"] = label
        frames.append(frame)
        print(
            f"{label:16s} n={len(frame):6d}  IoU={frame.iou.mean():.4f}  "
            f"core recall={frame.recall_core.mean():.4f}  thin recall={frame.recall_thin.mean():.4f}  "
            f"(IoU with no appendages at all would be {frame.iou_without_appendages.mean():.4f})"
        )
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    combined["bin"] = pd.cut(combined["size"], SIZE_BINS, labels=SIZE_LABELS)

    print("\nThin-appendage recall by instance size")
    print(combined.pivot_table(index="bin", columns="arm", values="recall_thin", observed=True).round(4).to_string())
    print("\nMask IoU by instance size, for reference")
    print(combined.pivot_table(index="bin", columns="arm", values="iou", observed=True).round(4).to_string())
    print("\nPredicted / ground-truth mask area (>1 means the model draws wider than the annotator)")
    print(
        combined.pivot_table(index="bin", columns="arm", values="pred_over_gt_area", observed=True)
        .round(3)
        .to_string()
    )
    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"\nWrote {len(combined)} rows to {args.output}")


if __name__ == "__main__":
    main()
