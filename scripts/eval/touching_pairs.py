#!/usr/bin/env python3
"""Evaluate detection restricted to instances that have a close neighbour.

Overall F1 cannot answer the question the synthetic-scene work exists to
answer. Touching instances are a minority of every dataset, so a model that
stopped merging neighbours entirely would move the headline metric by less
than its run-to-run noise. This script stratifies the same predictions by how
close each ground-truth instance's nearest neighbour is, so the isolated and
the crowded cases are scored separately.

It also reports the failure mode directly. A *merge* is one predicted instance
that covers two or more ground-truth instances - not a missed detection and
not a false positive, but the specific error of failing to split neighbours.
Recall alone scores a merge as one hit and one miss, which understates how
wrong it is.

Usage:
    touching_pairs.py --gt instances_default.json --pred coco_instances.json
                      [--iou 0.5] [--contained 0.5] [--min-size 32]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

# Nearest-neighbour gap, in px, defining each stratum. 0 means the two masks
# touch or overlap; the open-ended last bucket is the isolated case.
BUCKETS = [(0.0, 0.0, "touching"), (0.0, 8.0, "<=8px"), (8.0, 32.0, "8-32px"), (32.0, np.inf, ">32px")]


def _clean(pts: np.ndarray, min_size: float) -> Polygon | None:
    """Build a valid polygon from a point ring, or None if it is unusable."""
    if len(pts) < 3 or max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) < min_size:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0 or poly.geom_type != "Polygon":
        return None
    return poly


def polygons_from_coco(path: str, min_size: float) -> dict[str, list[Polygon]]:
    """Read a COCO instances file into {image stem: [polygon, ...]}.

    Keyed on the file stem rather than the COCO image id, so ground truth and
    predictions produced by different tools still line up.

    Args:
        path: COCO JSON path.
        min_size: Drop instances whose longest side is below this, in px.

    Returns:
        Polygons per image stem.
    """
    with open(path) as fh:
        coco = json.load(fh)
    names = {im["id"]: os.path.splitext(os.path.basename(im["file_name"]))[0] for im in coco.get("images", [])}
    out: dict[str, list[Polygon]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg:
            continue
        ring = seg[0] if isinstance(seg[0], list) else seg
        if len(ring) < 6:
            continue
        poly = _clean(np.asarray(ring, dtype=float).reshape(-1, 2), min_size)
        if poly is not None:
            out[names.get(ann["image_id"], str(ann["image_id"]))].append(poly)
    return out


def polygons_from_yolo(label_dir: str, image_dir: str, min_size: float) -> dict[str, list[Polygon]]:
    """Read YOLO-polygon labels into {image stem: [polygon, ...]}, in pixel space.

    Args:
        label_dir: Directory of `.txt` label files.
        image_dir: Matching images, read for their dimensions only.
        min_size: Drop instances whose longest side is below this, in px.

    Returns:
        Polygons per image stem.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    out: dict[str, list[Polygon]] = defaultdict(list)
    for label in sorted(glob.glob(os.path.join(label_dir, "*.txt"))):
        stem = os.path.splitext(os.path.basename(label))[0]
        image = os.path.join(image_dir, stem + ".jpg")
        try:
            width, height = Image.open(image).size
        except Exception:  # noqa: BLE001 - an unreadable image contributes nothing
            continue
        out[stem] = []
        with open(label) as fh:
            for line in fh:
                vals = line.split()
                if len(vals) < 7:
                    continue
                poly = _clean(np.asarray(vals[1:], float).reshape(-1, 2) * [width, height], min_size)
                if poly is not None:
                    out[stem].append(poly)
    return out


def polygons_from_fb_predict(pred_dir: str, min_size: float) -> dict[str, list[Polygon]]:
    """Read an `fb_predict` output directory into {image stem: [polygon, ...]}.

    `fb_predict` writes one `metadata_*.json` per image, with contours stored
    as [xs, ys] rather than as a point list, so they are transposed here.

    Args:
        pred_dir: Directory written by `fb_predict`.
        min_size: Drop instances whose longest side is below this, in px.

    Returns:
        Polygons per image stem.
    """
    out: dict[str, list[Polygon]] = defaultdict(list)
    for meta in sorted(glob.glob(os.path.join(pred_dir, "**", "metadata_*.json"), recursive=True)):
        with open(meta) as fh:
            data = json.load(fh)
        stem = os.path.splitext(os.path.basename(data["image_path"]))[0]
        out[stem] = []
        for contour in data.get("contours", []):
            pts = np.asarray(contour, dtype=float)
            if pts.ndim != 2 or pts.shape[0] != 2:
                continue
            poly = _clean(pts.T, min_size)
            if poly is not None:
                out[stem].append(poly)
    return out


def neighbour_gaps(polys: list[Polygon]) -> np.ndarray:
    """Distance from each polygon to its nearest other polygon, in px.

    Args:
        polys: Polygons in one image.

    Returns:
        Gap per polygon; inf where an image holds a single instance.
    """
    if len(polys) < 2:
        return np.full(len(polys), np.inf)
    tree = STRtree(polys)
    gaps = np.empty(len(polys))
    for i, p in enumerate(polys):
        # Query a growing window rather than all pairs; most images are sparse.
        best = np.inf
        for radius in (16.0, 64.0, 256.0, 1024.0, np.inf):
            if np.isinf(radius):
                candidates = range(len(polys))
            else:
                candidates = tree.query(p.buffer(radius))
            for j in candidates:
                if int(j) == i:
                    continue
                d = p.distance(polys[int(j)])
                best = min(best, d)
            if np.isfinite(best):
                break
        gaps[i] = best
    return gaps


def match(gt: list[Polygon], pred: list[Polygon], iou_thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedily match predictions to ground truth by IoU, best pair first.

    Args:
        gt: Ground-truth polygons.
        pred: Predicted polygons.
        iou_thr: Minimum IoU for a match.

    Returns:
        (gt_matched, pred_matched) boolean arrays.
    """
    gt_hit = np.zeros(len(gt), bool)
    pred_hit = np.zeros(len(pred), bool)
    if not gt or not pred:
        return gt_hit, pred_hit
    pairs = []
    tree = STRtree(pred)
    for i, g in enumerate(gt):
        for j in tree.query(g):
            j = int(j)
            inter = g.intersection(pred[j]).area
            if inter <= 0:
                continue
            union = g.area + pred[j].area - inter
            if union > 0 and inter / union >= iou_thr:
                pairs.append((inter / union, i, j))
    for _, i, j in sorted(pairs, reverse=True):
        if not gt_hit[i] and not pred_hit[j]:
            gt_hit[i] = pred_hit[j] = True
    return gt_hit, pred_hit


def merged(gt: list[Polygon], pred: list[Polygon], contained: float) -> np.ndarray:
    """Flag ground-truth instances that share one prediction with another instance.

    Args:
        gt: Ground-truth polygons.
        pred: Predicted polygons.
        contained: Fraction of a ground-truth instance that must fall inside a
            prediction for that prediction to be said to cover it.

    Returns:
        Boolean array over ground truth: True where the instance was merged.
    """
    flags = np.zeros(len(gt), bool)
    if not gt or not pred:
        return flags
    tree = STRtree(gt)
    for p in pred:
        covered = [int(i) for i in tree.query(p) if gt[int(i)].intersection(p).area / gt[int(i)].area >= contained]
        if len(covered) >= 2:
            flags[covered] = True
    return flags


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", help="ground-truth COCO json")
    parser.add_argument("--gt-labels", help="directory of YOLO-polygon label files (alternative to --gt)")
    parser.add_argument("--gt-images", help="images matching --gt-labels, read for dimensions only")
    parser.add_argument("--pred", required=True,
                        help="predicted COCO json, or an fb_predict output directory")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match")
    parser.add_argument("--contained", type=float, default=0.5,
                        help="fraction of a GT instance inside a prediction that counts as covered")
    parser.add_argument("--score-missing", action="store_true",
                        help="score ground-truth images with no predictions as all-missed")
    parser.add_argument("--min-size", type=float, default=32.0, help="drop instances below this longest side, in px")
    args = parser.parse_args()

    if args.gt_labels:
        if not args.gt_images:
            parser.error("--gt-labels requires --gt-images")
        gt_by_image = polygons_from_yolo(args.gt_labels, args.gt_images, args.min_size)
    elif args.gt:
        gt_by_image = polygons_from_coco(args.gt, args.min_size)
    else:
        parser.error("provide either --gt or --gt-labels/--gt-images")
    pred_by_image = (polygons_from_fb_predict(args.pred, args.min_size) if os.path.isdir(args.pred)
                     else polygons_from_coco(args.pred, args.min_size))
    missing = sorted(set(gt_by_image) - set(pred_by_image))
    if missing and not args.score_missing:
        # Predictions are often produced for a subset. Scoring the rest as
        # all-missed would silently drown recall - the first run of this script
        # reported 0.9% recall because 2045 of 2047 images had no predictions.
        print(f"note: {len(missing)} ground-truth images have no predictions and are excluded; "
              f"pass --score-missing to score them as all-missed")
        gt_by_image = {k: v for k, v in gt_by_image.items() if k in pred_by_image}
    elif missing:
        print(f"note: {len(missing)} ground-truth images have no predictions; scored as all-missed")

    rows: dict[str, dict[str, int]] = {b[2]: {"n": 0, "tp": 0, "merged": 0} for b in BUCKETS}
    total_pred = total_pred_hit = 0
    for stem, gt in gt_by_image.items():
        pred = pred_by_image.get(stem, [])
        gaps = neighbour_gaps(gt)
        gt_hit, pred_hit = match(gt, pred, args.iou)
        merge_flags = merged(gt, pred, args.contained)
        total_pred += len(pred)
        total_pred_hit += int(pred_hit.sum())
        for lo, hi, name in BUCKETS:
            sel = (gaps <= hi) if lo == hi == 0.0 else ((gaps > lo) & (gaps <= hi))
            rows[name]["n"] += int(sel.sum())
            rows[name]["tp"] += int(gt_hit[sel].sum())
            rows[name]["merged"] += int(merge_flags[sel].sum())

    print(f"\nimages {len(gt_by_image)} | GT instances {sum(r['n'] for r in rows.values())} | predictions {total_pred}")
    print(f"overall precision {total_pred_hit / max(total_pred, 1):.4f}\n")
    print(f"{'neighbour gap':>14s} {'GT':>7s} {'recall':>8s} {'merged':>8s} {'merge rate':>11s}")
    for _, _, name in BUCKETS:
        r = rows[name]
        if r["n"] == 0:
            print(f"{name:>14s} {0:>7d} {'-':>8s} {'-':>8s} {'-':>11s}")
            continue
        print(f"{name:>14s} {r['n']:>7d} {r['tp'] / r['n']:>8.4f} {r['merged']:>8d} {r['merged'] / r['n']:>11.4f}")


if __name__ == "__main__":
    main()
