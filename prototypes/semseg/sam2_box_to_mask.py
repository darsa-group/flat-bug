#!/usr/bin/env python3
"""PROTOTYPE - turn bounding boxes into instance polygons with SAM 2, box-prompted.

ArTaxOr ships bounding boxes only (~15k images, CC BY-NC-SA), and it is the only dataset in
the collection with natural cluttered backgrounds - vegetation, bark, leaf litter. It is
also where the semantic model fails worst, hallucinating insects in twigs and conifer
needles at pixel precision 0.57. Converting its boxes to masks would add exactly the domain
that is missing.

Box prompts are used rather than points because a box is what ArTaxOr provides, and because
box-prompted SAM is far more reliable than point-prompted on cluttered backgrounds.

VALIDATION MATTERS HERE. An earlier test in this project found SAM cutting a moth down to
its body and dropping both wings, and over-segmenting a 49px instance. ArTaxOr's instances
are large (median 528px sqrt-area, the biggest in the collection), which is SAM's best
regime - but the failure mode is appendage amputation, precisely what the flat-bug work has
been trying to preserve. Hence `--validate`, which scores generated masks against real
polygons on the subset that already has them.

Usage:
    sam2_box_to_mask.py -i IMAGE_DIR -o OUT_DIR [--weights sam2.1_b.pt]
    sam2_box_to_mask.py --validate            # score against existing ArTaxOr ground truth
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import read_polygons  # noqa: E402

DEFAULT_WEIGHTS = "/home/quentin/repos/flat-bug-git/sam2.1_b.pt"


MIN_COMPONENT_PX = 20  # ignore specks below this when keeping extra components


def masks_from_boxes(model, img: np.ndarray, boxes: list[list[float]],
                     keep_all_components: bool = True) -> list[list[np.ndarray]]:
    """Prompt SAM 2 with boxes and return the contours of each returned mask.

    SAM returns exactly one mask per box prompt - measured at 1.00 masks per box - so it
    never splits a box into several instances. But 56% of those masks are MULTI-COMPONENT:
    a body plus detached legs, antennae or wing tips. Keeping only the largest contour
    discards 1.7% of mask area on average and up to 11.9%, and by area badly understates the
    harm, since the discarded pieces are exactly the thin structures this project exists to
    preserve.

    Args:
        model: An ultralytics SAM model.
        img: HxWx3 RGB image.
        boxes: List of [x1, y1, x2, y2].
        keep_all_components: Keep every component above ``MIN_COMPONENT_PX``, not just the
            largest.

    Returns:
        One list of contours per box, largest first. An empty list means SAM returned
        nothing for that box.
    """
    if not boxes:
        return []
    res = model(img, bboxes=boxes, verbose=False)
    out = []
    for r in res:
        if r.masks is None:
            continue
        for m in r.masks.data.cpu().numpy():
            mm = (m > 0.5).astype(np.uint8)
            cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = [c[:, 0, :] for c in cs if len(c) >= 3 and cv2.contourArea(c) >= MIN_COMPONENT_PX]
            cs.sort(key=lambda c: -cv2.contourArea(c.astype(np.float32)))
            out.append(cs if keep_all_components else cs[:1])
    return out


def thin_recall(gt: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of the ground truth's thin structures that the prediction covers.

    Args:
        gt: Binary ground-truth mask.
        pred: Binary predicted mask.

    Returns:
        Recall over thin pixels, or NaN if the instance has none.
    """
    dt = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 5)
    r = max(1, int(round(0.40 * dt.max())))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    thin = (gt > 0) & (cv2.morphologyEx(gt.astype(np.uint8), cv2.MORPH_OPEN, k) == 0)
    return float(pred[thin].mean()) if thin.sum() else float("nan")


def validate(model, data: str, pattern: str, n: int) -> None:
    """Score SAM 2 box-prompted masks against existing polygon ground truth.

    Args:
        model: An ultralytics SAM model.
        data: Root of the prepared YOLO layout.
        pattern: Filename prefix selecting the sub-dataset.
        n: Number of images to score.
    """
    files = sorted(glob.glob(os.path.join(data, "images", "val", f"{pattern}*.jpg")))[:n]
    ious, thins, areas = [], [], []
    for f in files:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lf = f.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"
        polys = read_polygons(lf, w, h)
        if not polys:
            continue
        boxes = [[float(p[:, 0].min()), float(p[:, 1].min()),
                  float(p[:, 0].max()), float(p[:, 1].max())] for p in polys]
        got = masks_from_boxes(model, img, boxes)
        for p, cs in zip(polys, got):
            if not cs:
                continue
            g = np.zeros((h, w), np.uint8)
            cv2.fillPoly(g, [np.round(p).astype(np.int32)], 1)
            s = np.zeros((h, w), np.uint8)
            cv2.fillPoly(s, [c.astype(np.int32) for c in cs], 1)
            inter = int((g & s).sum())
            union = int((g | s).sum())
            if union == 0:
                continue
            ious.append(inter / union)
            areas.append(s.sum() / max(g.sum(), 1))
            t = thin_recall(g > 0, s > 0)
            if not np.isnan(t):
                thins.append(t)
    a = np.array(ious)
    print(f"\n{len(a)} instances over {len(files)} {pattern} images\n")
    print(f"   mask IoU vs ground truth : {a.mean():.4f}  (median {np.median(a):.4f}, "
          f"{(a >= 0.7).mean():.1%} above 0.7)")
    print(f"   SAM area / GT area       : {np.mean(areas):.4f}")
    print(f"   thin-structure recall    : {np.mean(thins):.4f}   <- appendage amputation shows up here")


def main() -> None:
    """Validate, or convert a directory of boxes into masks."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--images", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--data", default="/home/quentin/Desktop/flatbug-dir/flat-bug-data/yolo/insects")
    ap.add_argument("--pattern", default="ArTaxOr_")
    ap.add_argument("-n", type=int, default=40)
    a = ap.parse_args()

    from ultralytics import SAM
    model = SAM(a.weights)
    if a.validate:
        validate(model, a.data, a.pattern, a.n)
        return
    raise SystemExit("conversion mode needs the downloaded ArTaxOr boxes; run --validate for now")


if __name__ == "__main__":
    main()
