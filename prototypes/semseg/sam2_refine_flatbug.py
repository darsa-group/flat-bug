#!/usr/bin/env python3
"""PROTOTYPE - refine flat-bug instance masks with box-prompted SAM 2, scored by IoS.

flat-bug supplies both the detection and a mask; SAM 2 is prompted with flat-bug's own box
and produces a second mask for the same animal. The two are compared by INTERSECTION OVER
THE SMALLER area rather than IoU:

    IoS = |A and B| / min(|A|, |B|)

IoU punishes a refinement that merely changes extent - if SAM traces legs flat-bug missed,
or trims a halo flat-bug added, IoU falls even though both masks describe the same animal
and one is strictly better. IoS is near 1 whenever one mask essentially contains the other,
so it separates "same object, different extent" from "different object", which is the
distinction that matters when deciding whether a refinement is safe to accept. flat-bug
itself offers IoS as an OVERLAP_METRIC in its NMS for the same reason.

Images here are very large (AgriVolt is about 22,500 x 22,700, over 500 Mpx), so SAM is run
on a crop around each instance rather than the whole frame, magnified so small animals
occupy enough pixels for the mask decoder to resolve them.

Usage:
    sam2_refine_flatbug.py -i PRED_DIR [-n 2] [-o OUT.png]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np

SAM_WEIGHTS = "/home/quentin/repos/flat-bug-git/sam2.1_b.pt"
TARGET = 512      # magnified size of the instance's longest side when prompting SAM
MARGIN = 0.40     # crop margin as a fraction of the box's longest side
MIN_COMPONENT_PX = 20
MIN_REFINE_PX = 50  # skip instances below this sqrt(area); see refine_image


def ios(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller of two boolean masks.

    Args:
        a: First mask.
        b: Second mask.

    Returns:
        |a and b| / min(|a|, |b|), or 0 if either is empty.
    """
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 or nb == 0:
        return 0.0
    return float((a & b).sum() / min(na, nb))


def fb_contour(entry: list) -> np.ndarray:
    """Convert a flat-bug metadata contour into an (N, 2) array.

    flat-bug stores each contour as two parallel coordinate lists.

    Args:
        entry: ``[[x...], [y...]]``.

    Returns:
        (N, 2) float array.
    """
    return np.stack([np.asarray(entry[0], float), np.asarray(entry[1], float)], axis=1)


def refine_image(model, meta_path: str, min_px: float = MIN_REFINE_PX) -> list[dict]:
    """Refine every flat-bug instance in one image with box-prompted SAM 2.

    Instances below ``min_px`` sqrt-area are left alone rather than refined. SAM is weakest
    exactly there - an earlier test in this project found it over-segmenting a 49px instance
    and amputating a moth's wings - and below that size the crop has to be magnified several
    fold before prompting, so SAM decides from interpolated pixels rather than real detail.
    Size is measured as sqrt of the flat-bug mask area, matching flat-bug's own
    MIN_MAX_OBJ_SIZE convention.

    Args:
        model: An ultralytics SAM model.
        meta_path: Path to a flat-bug ``metadata_*.json``.
        min_px: Skip instances whose sqrt(area) is below this.

    Returns:
        One record per REFINED instance. Skipped ones are reported separately by the caller.
    """
    j = json.load(open(meta_path))
    img = cv2.imread(j["image_path"])
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    out = []
    for box, cont, conf in zip(j["boxes"], j["contours"], j["confs"]):
        fb = fb_contour(cont)
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        fb_px = float(np.sqrt(max(abs(cv2.contourArea(fb.astype(np.float32))), 1.0)))
        if fb_px < min_px:  # too small to refine safely; keep flat-bug's mask untouched
            out.append({"skipped": True, "fb_px": fb_px, "conf": float(conf)})
            continue
        m = MARGIN * max(bw, bh)
        cx0, cy0 = int(max(0, x1 - m)), int(max(0, y1 - m))
        cx1, cy1 = int(min(W, x2 + m)), int(min(H, y2 + m))
        crop = img[cy0:cy1, cx0:cx1]
        if crop.size == 0 or min(crop.shape[:2]) < 8:
            continue
        s = float(np.clip(TARGET / max(bw, bh), 1.0, 8.0))
        big = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                         interpolation=cv2.INTER_CUBIC) if s > 1.0 else crop
        pb = [[(x1 - cx0) * s, (y1 - cy0) * s, (x2 - cx0) * s, (y2 - cy0) * s]]
        res = model(big, bboxes=pb, verbose=False)
        sam_cs = []
        for r in res:
            if r.masks is None:
                continue
            for mk in r.masks.data.cpu().numpy():
                mm = (mk > 0.5).astype(np.uint8)
                cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sam_cs += [c[:, 0, :] / s + [cx0, cy0] for c in cs
                           if len(c) >= 3 and cv2.contourArea(c) >= MIN_COMPONENT_PX * s * s]
        if not sam_cs:
            continue
        ch, cw = cy1 - cy0, cx1 - cx0
        fm = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(fm, [np.round(fb - [cx0, cy0]).astype(np.int32)], 1)
        sm = np.zeros((ch, cw), np.uint8)
        cv2.fillPoly(sm, [np.round(c - [cx0, cy0]).astype(np.int32) for c in sam_cs], 1)
        out.append({"skipped": False, "fb_px": fb_px, "ios": ios(fm > 0, sm > 0), "iou": float((fm & sm).sum() / max((fm | sm).sum(), 1)),
                    "fb_area": int(fm.sum()), "sam_area": int(sm.sum()), "conf": float(conf),
                    "crop": (cx0, cy0, cx1, cy1), "fb": fb, "sam": sam_cs, "scale": s})
    return out


def main() -> None:
    """Refine a few images and report the IoS distribution."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--pred-dir", default="/home/quentin/Desktop/AgriVolt/flatbug_yolo26_results")
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--min-size", type=float, default=MIN_REFINE_PX,
                    help="skip instances below this sqrt(area) in px; they keep flat-bug's mask")
    a = ap.parse_args()

    from ultralytics import SAM
    model = SAM(SAM_WEIGHTS)
    metas = sorted(glob.glob(os.path.join(a.pred_dir, "*", "metadata_*.json")))
    seen, chosen = set(), []
    for m in metas:  # one metadata file per image
        d = os.path.dirname(m)
        if d in seen:
            continue
        seen.add(d)
        chosen.append(m)
        if len(chosen) >= a.n:
            break
    allr = []
    for m in chosen:
        r = refine_image(model, m, a.min_size)
        allr += r
        done = [x for x in r if not x["skipped"]]
        i = np.array([x["ios"] for x in done]) if done else np.array([])
        print(f"{os.path.basename(os.path.dirname(m))[:46]:48s} {len(done):4d} refined, "
              f"{len(r) - len(done):3d} skipped (<{a.min_size:.0f}px)  "
              f"median IoS {np.median(i) if len(i) else float('nan'):.4f}", flush=True)
    if not allr:
        print("no instances refined")
        return
    skipped = [x for x in allr if x["skipped"]]
    allr = [x for x in allr if not x["skipped"]]
    if skipped:
        sp = np.array([x["fb_px"] for x in skipped])
        print(f"\nskipped {len(skipped)} instances below {a.min_size:.0f}px "
              f"({len(skipped) / (len(skipped) + len(allr)):.1%}); "
              f"their sqrt-area median {np.median(sp):.0f}px, max {sp.max():.0f}px")
    if not allr:
        print("nothing refined")
        return
    sc = np.array([x["ios"] for x in allr])
    ju = np.array([x["iou"] for x in allr])
    ratio = np.array([x["sam_area"] / max(x["fb_area"], 1) for x in allr])
    print(f"\n{len(allr)} instances\n")
    print(f"   IoS  median {np.median(sc):.4f}   mean {sc.mean():.4f}   "
          f">=0.9 {np.mean(sc >= 0.9):.1%}   >=0.8 {np.mean(sc >= 0.8):.1%}   <0.5 {np.mean(sc < 0.5):.1%}")
    print(f"   IoU  median {np.median(ju):.4f}   mean {ju.mean():.4f}   (shown for contrast)")
    print(f"   SAM area / flat-bug area: median {np.median(ratio):.3f}  "
          f"p10 {np.percentile(ratio, 10):.3f}  p90 {np.percentile(ratio, 90):.3f}")
    print(f"\n   IoS exceeds IoU on {np.mean(sc > ju):.1%} of instances, by {np.mean(sc - ju):.3f} on average -")
    print("   that gap is refinement changing extent rather than changing object.")


if __name__ == "__main__":
    main()
