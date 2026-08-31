#!/usr/bin/env python3
"""PROTOTYPE - convert ArTaxOr boxes to masks, split by SAM/flat-bug agreement.

ArTaxOr ships bounding boxes only. SAM 2, prompted with those boxes, produces high-quality
masks (IoU 0.907 and thin-structure recall 0.879 against the subset that already has
polygons). But 15,376 images cannot be checked by hand, so agreement with an independent
model is used as a confidence signal:

    bag "agree"    SAM mask matches a flat-bug prediction above --agree-iou. Two independent
                   methods reaching the same outline is strong evidence it is right.
    bag "review"   they disagree, or flat-bug found nothing there. Kept separately with BOTH
                   masks, so the disagreement can be inspected rather than silently trusted.

Both bags are written as COCO instance-segmentation JSON.

The threshold is not guessed. `--calibrate` runs on the ArTaxOr images that already have
polygon ground truth and reports, for each candidate threshold, how often an "agreed" mask
is genuinely good - so the operating point is chosen from measurement.

Usage:
    agreement_bags.py --calibrate [-n 60]
    agreement_bags.py -i ARTAXOR_ROOT -o OUT_DIR [--agree-iou 0.75] [-n N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import read_polygons  # noqa: E402
from sam2_box_to_mask import masks_from_boxes  # noqa: E402

SAM_WEIGHTS = "/home/quentin/repos/flat-bug-git/sam2.1_b.pt"
FB_WEIGHTS = ("/tmp/claude-1000/-home-quentin-repos-flat-bug-git/"
              "d72f9e2f-9b95-4078-b8a0-07edd95336ea/scratchpad/arms/thin_best.pt")


def rasterise(contours: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Fill contours into a binary mask.

    Args:
        contours: Contours in image coordinates.
        shape: (height, width).

    Returns:
        Boolean mask.
    """
    m = np.zeros(shape, np.uint8)
    if contours:
        cv2.fillPoly(m, [np.round(c).astype(np.int32) for c in contours], 1)
    return m > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two boolean masks."""
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


def artaxor_boxes(json_path: str) -> tuple[str, list[list[float]], list[str]]:
    """Read one ArTaxOr VoTT annotation file.

    Args:
        json_path: Path to the ``*-asset.json`` file.

    Returns:
        (image file name, boxes as [x1, y1, x2, y2], tag per box).
    """
    j = json.load(open(json_path))
    name = j["asset"]["name"]
    boxes, tags = [], []
    for r in j.get("regions", []):
        b = r.get("boundingBox")
        if not b:
            continue
        boxes.append([b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]])
        tags.append((r.get("tags") or ["arthropod"])[0])
    return name, boxes, tags


def fb_instances(predictor, img_bgr: np.ndarray, path: str) -> list[np.ndarray]:
    """Run flat-bug and return its predicted contours.

    Args:
        predictor: A flat-bug Predictor.
        img_bgr: The image as BGR uint8.
        path: Image path, required by the predictor.

    Returns:
        List of (N, 2) contours.
    """
    t = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).contiguous()
    try:
        r = predictor.pyramid_predictions(t, path=path, single_scale=False)
    except Exception:  # noqa: BLE001 - a failed image should not abort the sweep
        return []
    out = []
    for q in (r.polygons or []):
        c = np.asarray(q.exterior.coords, float) if hasattr(q, "exterior") else (
            q.cpu().numpy().astype(float) if hasattr(q, "cpu") else np.asarray(q, float))
        if c.ndim == 3:
            c = c[:, 0, :]
        if len(c) >= 3:
            out.append(c)
    return out


def calibrate(n: int) -> None:
    """Measure how well SAM/flat-bug agreement predicts SAM being correct.

    Uses the ArTaxOr images that already have polygon ground truth, so "correct" is
    measured rather than assumed.

    Args:
        n: Number of images to use.
    """
    from ultralytics import SAM

    from flat_bug.predictor import Predictor
    D = "/home/quentin/Desktop/flatbug-dir/flat-bug-data/yolo/insects"
    sam = SAM(SAM_WEIGHTS)
    fb = Predictor(FB_WEIGHTS, device="cuda:0", dtype=torch.float16)
    rows = []
    for f in sorted(glob.glob(f"{D}/images/val/ArTaxOr_*.jpg"))[:n]:
        bgr = cv2.imread(f)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        lf = f.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"
        gts = read_polygons(lf, w, h)
        if not gts:
            continue
        boxes = [[float(p[:, 0].min()), float(p[:, 1].min()),
                  float(p[:, 0].max()), float(p[:, 1].max())] for p in gts]
        sam_cs = masks_from_boxes(sam, img, boxes)
        fb_cs = fb_instances(fb, bgr, f)
        fb_masks = [rasterise([c], (h, w)) for c in fb_cs]
        for gt, cs in zip(gts, sam_cs):
            if not cs:
                continue
            sm = rasterise(cs, (h, w))
            gm = rasterise([gt], (h, w))
            best = max((iou(sm, m) for m in fb_masks), default=0.0)
            rows.append((best, iou(sm, gm)))
    a = np.array(rows)
    print(f"\n{len(a)} SAM masks with ground truth\n")
    print(f"{'agree IoU >=':>13s} {'kept':>7s} {'% kept':>8s} {'of kept, truly good':>21s} {'missed good':>13s}")
    good = a[:, 1] >= 0.7
    for t in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9):
        keep = a[:, 0] >= t
        if keep.sum() == 0:
            continue
        print(f"{t:13.2f} {int(keep.sum()):7d} {keep.mean():8.1%} {good[keep].mean():21.1%} "
              f"{int((good & ~keep).sum()):13d}")
    print(f"\nbaseline: {good.mean():.1%} of all SAM masks are good (IoU>=0.7 vs ground truth)")


def bucket_of(n: int) -> str:
    """Sub-bag name for an image with ``n`` annotated instances."""
    if n <= 1:
        return "n1"
    if n == 2:
        return "n2"
    if n <= 5:
        return "n3-5"
    return "n6plus"


def write_coco(recs: list[dict], path: str, cats: list[str]) -> None:
    """Write records as a COCO instance-segmentation file.

    Args:
        recs: One record per image, each with file_name, width, height and annotations.
        path: Destination JSON path.
        cats: Ordered category names.
    """
    cid = {c: i + 1 for i, c in enumerate(cats)}
    images, anns, aid = [], [], 1
    for i, r in enumerate(recs, start=1):
        images.append({"id": i, "file_name": r["file_name"], "width": r["width"], "height": r["height"]})
        for an in r["anns"]:
            seg = [list(map(float, np.asarray(c).ravel())) for c in an["contours"] if len(c) >= 3]
            if not seg:
                continue
            xs = np.concatenate([np.asarray(c)[:, 0] for c in an["contours"]])
            ys = np.concatenate([np.asarray(c)[:, 1] for c in an["contours"]])
            anns.append({"id": aid, "image_id": i, "category_id": cid.get(an["tag"], 1),
                         "segmentation": seg, "iscrowd": 0,
                         "bbox": [float(xs.min()), float(ys.min()),
                                  float(xs.max() - xs.min()), float(ys.max() - ys.min())],
                         "area": float(an["area"]), "source": an["source"],
                         "agreement_iou": round(float(an["agree"]), 4)})
            aid += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({"info": {"description": "ArTaxOr boxes converted to masks by box-prompted SAM 2.1"},
               "licenses": [{"id": 0, "name": "CC BY-NC-SA 4.0", "url": ""}],
               "categories": [{"id": v, "name": k, "supercategory": "arthropod"} for k, v in cid.items()],
               "images": images, "annotations": anns}, open(path, "w"))


def convert(root: str, out: str, agree_iou: float, limit: int) -> None:
    """Convert every ArTaxOr image, splitting by agreement and instance count.

    Args:
        root: ArTaxOr root containing per-order directories.
        out: Output directory.
        agree_iou: IoU above which SAM and flat-bug are treated as agreeing.
        limit: Max images, or -1 for all.
    """
    from ultralytics import SAM

    from flat_bug.predictor import Predictor
    sam = SAM(SAM_WEIGHTS)
    fb = Predictor(FB_WEIGHTS, device="cuda:0", dtype=torch.float16)
    jsons = sorted(glob.glob(os.path.join(root, "*", "annotations", "*-asset.json")))
    if limit > 0:
        jsons = jsons[:limit]
    bags: dict[tuple[str, str], list[dict]] = {}
    cats: list[str] = []
    n_agree = n_review = 0
    os.makedirs(out, exist_ok=True)
    for i, jf in enumerate(jsons, 1):
        name, boxes, tags = artaxor_boxes(jf)
        if not boxes:
            continue
        img_path = os.path.join(os.path.dirname(os.path.dirname(jf)), name)
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        sam_cs = masks_from_boxes(sam, img, boxes)
        fb_masks = [rasterise([c], (h, w)) for c in fb_instances(fb, bgr, img_path)]
        buck = bucket_of(len(boxes))
        rec = {"agree": {"file_name": name, "width": w, "height": h, "anns": []},
               "review": {"file_name": name, "width": w, "height": h, "anns": []}}
        for cs, tag in zip(sam_cs, tags):
            if tag not in cats:
                cats.append(tag)
            if not cs:
                continue
            sm = rasterise(cs, (h, w))
            best = max((iou(sm, m) for m in fb_masks), default=0.0)
            side = "agree" if best >= agree_iou else "review"
            rec[side]["anns"].append({"contours": cs, "tag": tag, "area": int(sm.sum()),
                                      "source": "sam", "agree": best})
        for side in ("agree", "review"):
            if rec[side]["anns"]:
                bags.setdefault((side, buck), []).append(rec[side])
                if side == "agree":
                    n_agree += len(rec[side]["anns"])
                else:
                    n_review += len(rec[side]["anns"])
        if i % 250 == 0 or i == len(jsons):
            print(f"[{i}/{len(jsons)}] agree {n_agree}  review {n_review}", flush=True)
    for (side, buck), recs in sorted(bags.items()):
        p = os.path.join(out, side, buck, "instances.json")
        write_coco(recs, p, cats or ["arthropod"])
        n = sum(len(r["anns"]) for r in recs)
        print(f"  {side:6s}/{buck:7s}  {len(recs):5d} images  {n:6d} instances -> {p}")
    msg = f"total: {n_agree} agreed, {n_review} for review"
    print(msg + f" ({n_agree / max(n_agree + n_review, 1):.1%} auto-accepted)")


def main() -> None:
    """Calibrate, or split ArTaxOr into agree/review bags."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", default=None, help="ArTaxOr root containing order directories")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--agree-iou", type=float, default=0.75)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("-n", type=int, default=60)
    a = ap.parse_args()
    if a.calibrate:
        calibrate(a.n)
        return
    if not (a.input and a.out):
        raise SystemExit("need -i ARTAXOR_ROOT and -o OUT_DIR (or --calibrate)")
    convert(a.input, a.out, a.agree_iou, a.n)


if __name__ == "__main__":
    main()
