"""The benchmark's scoring rule, kept apart from the code under test.

A benchmark run predicts with the flat-bug commit being tested, but always scores with THIS file,
so a change to flat-bug's own evaluation code can never move a score. Any change to the rule
below must bump SCORER_VERSION, and scores carrying different versions are not comparable.

The rule (scorer v1), the one used for every end-to-end F1 quoted since August 2026:

* Ground truth and predictions are polygons in full-image pixel coordinates.
* An instance is dropped from BOTH sides when the geometric mean of its bounding-box sides is
  under ``MIN_SIZE_PX`` (32 px). Tiny instances are where annotation is least consistent.
* Predictions are matched one-to-one to ground truth by mask IoU, greedily, best pair first,
  and a pair counts only at IoU >= ``IOU_THRESHOLD`` (0.5).
* Precision = matched predictions / predictions; recall = matched ground truth / ground truth;
  F1 from those. Totals pool instances across images (micro-average), per dataset and overall.
* ``mean_iou`` is the mean IoU of the matched pairs: how good the masks are once found.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree

SCORER_VERSION = "1"
IOU_THRESHOLD = 0.5
MIN_SIZE_PX = 32.0


def clean_polygon(xy: np.ndarray, min_size: float = MIN_SIZE_PX) -> Polygon | None:
    """Return a valid polygon from an (N, 2) array, or None if it is degenerate or too small."""
    if len(xy) < 3:
        return None
    dx, dy = np.ptp(xy[:, 0]), np.ptp(xy[:, 1])
    if float(np.sqrt(max(dx, 1e-9) * max(dy, 1e-9))) < min_size:
        return None
    p = Polygon(xy)
    if not p.is_valid:
        p = p.buffer(0)
    # buffer(0) can split a self-touching outline into a MultiPolygon; v1 drops those, as the
    # August scorer did, rather than guess which part is the animal.
    return p if (not p.is_empty and p.area > 0 and p.geom_type == "Polygon") else None


def match(gt: list[Polygon], pred: list[Polygon], iou_thr: float = IOU_THRESHOLD):
    """Greedy one-to-one matching by IoU, best pair first.

    Returns:
        (gt_matched, pred_matched, pairs): two boolean arrays and the matched (gt index,
        pred index, IoU) triples.
    """
    gt_hit = np.zeros(len(gt), bool)
    pred_hit = np.zeros(len(pred), bool)
    matched: list[tuple[int, int, float]] = []
    if not gt or not pred:
        return gt_hit, pred_hit, matched
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
    for iou, i, j in sorted(pairs, reverse=True):
        if not gt_hit[i] and not pred_hit[j]:
            gt_hit[i] = pred_hit[j] = True
            matched.append((i, j, iou))
    return gt_hit, pred_hit, matched


def gt_polygons(coco_path: str) -> dict[str, list[Polygon]]:
    """Ground-truth polygons per image file name, from a COCO file. Images without annotations map to []."""
    with open(coco_path) as f:
        coco = json.load(f)
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    out: dict[str, list[Polygon]] = {os.path.basename(n): [] for n in by_id.values()}
    for a in coco["annotations"]:
        seg = a.get("segmentation")
        if not seg or not isinstance(seg, list):
            continue
        # One annotation is one instance; if it has several rings, keep the largest.
        polys = [clean_polygon(np.asarray(s, float).reshape(-1, 2)) for s in seg if len(s) >= 6]
        polys = [p for p in polys if p is not None]
        if polys:
            out[os.path.basename(by_id[a["image_id"]])].append(max(polys, key=lambda p: p.area))
    return out


def pred_polygons(pred_dir: str) -> tuple[dict[str, list[Polygon]], dict[str, list[float]]]:
    """Predicted polygons per image file name, from fb_predict's per-image metadata JSON files.

    Returns:
        (polygons, confidences): two dicts keyed by image file name, with parallel lists.
    """
    out: dict[str, list[Polygon]] = {}
    confs: dict[str, list[float]] = {}
    for path in glob.glob(os.path.join(pred_dir, "**", "metadata_*.json"), recursive=True):
        with open(path) as f:
            j = json.load(f)
        sx = j["image_width"] / j["mask_width"]
        sy = j["image_height"] / j["mask_height"]
        polys, cs = [], []
        for c, conf in zip(j["contours"], j["confs"]):
            xy = np.stack([np.asarray(c[0], float) * sx, np.asarray(c[1], float) * sy], axis=1)
            p = clean_polygon(xy)
            if p is not None:
                polys.append(p)
                cs.append(float(conf))
        name = os.path.basename(j["image_path"])
        out[name], confs[name] = polys, cs
    return out, confs


def summarise(c: dict) -> dict:
    """Add precision, recall, F1 and mean IoU to a dict of pooled counts."""
    p = c["tp_pred"] / c["pred"] if c["pred"] else float("nan")
    r = c["tp_gt"] / c["gt"] if c["gt"] else float("nan")
    f1 = 2 * p * r / (p + r) if c["pred"] and c["gt"] and (p + r) > 0 else float("nan")
    miou = c["iou_sum"] / c["tp_gt"] if c["tp_gt"] else float("nan")
    return {"gt": c["gt"], "pred": c["pred"], "tp": c["tp_gt"], "precision": p, "recall": r, "f1": f1,
            "mean_iou": miou}


def score_dataset(gt: dict[str, list[Polygon]], pred: dict[str, list[Polygon]]):
    """Score one dataset.

    Returns:
        (pooled summary, per-image rows, pooled counts, per-image matches) where matches maps an
        image name to its (gt index, pred index, IoU) triples.
    """
    tot = {"gt": 0, "pred": 0, "tp_gt": 0, "tp_pred": 0, "iou_sum": 0.0}
    rows, matches = [], {}
    for name in sorted(gt):
        G, P = gt[name], pred.get(name, [])
        gh, ph, pairs = match(G, P)
        matches[name] = pairs
        c = {"gt": len(G), "pred": len(P), "tp_gt": int(gh.sum()), "tp_pred": int(ph.sum()),
             "iou_sum": float(sum(p[2] for p in pairs))}
        for k in tot:
            tot[k] += c[k]
        rows.append({"image": name, "predicted": name in pred, **summarise(c)})
    return summarise(tot), rows, tot, matches
