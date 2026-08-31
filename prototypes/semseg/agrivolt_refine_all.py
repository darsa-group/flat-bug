#!/usr/bin/env python3
"""PROTOTYPE - refine every AgriVolt flat-bug instance with SAM 2 and write one COCO file.

Instances at or above --min-size sqrt-area get SAM 2's mask, prompted with flat-bug's own
box. Smaller ones keep flat-bug's mask untouched: SAM is weakest exactly there - an earlier
test in this project found it over-segmenting a 49px instance and amputating a moth's wings
- and below that size the crop must be magnified severalfold before prompting, so SAM would
be deciding from interpolated pixels rather than real detail. On AgriVolt this affects only
about 0.1% of instances, but the guard costs nothing and matters on denser datasets.

Masks are compared by INTERSECTION OVER THE SMALLER area, not IoU:

    IoS = |A and B| / min(|A|, |B|)

IoU punishes a refinement that only changes extent - trimming a halo or adding legs - even
when both masks describe the same animal. Measured on two AgriVolt images, IoS was 0.989
median against IoU 0.704, and IoS exceeded IoU on 100% of instances. IoS therefore separates
"same object, different extent" from "different object", which is what decides whether a
refinement is safe to accept.

Progress is written per image to a JSONL sidecar, so an interrupted run resumes rather than
restarting: the images are about 503 Mpx each and a full pass takes hours.

Usage:
    agrivolt_refine_all.py -o OUT_DIR [--min-size 50] [--pred-dir DIR] [-n N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

SAM_WEIGHTS = "/home/quentin/repos/flat-bug-git/sam2.1_b.pt"
TARGET = 512
MARGIN = 0.40
MIN_COMPONENT_PX = 20


def ios(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller of two boolean masks.

    Args:
        a: First mask.
        b: Second mask.

    Returns:
        |a and b| / min(|a|, |b|), or 0 if either is empty.
    """
    na, nb = int(a.sum()), int(b.sum())
    return float((a & b).sum() / min(na, nb)) if na and nb else 0.0


def fb_contour(entry: list) -> np.ndarray:
    """Convert a flat-bug metadata contour, stored as two parallel lists, to (N, 2)."""
    return np.stack([np.asarray(entry[0], float), np.asarray(entry[1], float)], axis=1)


def refine_one(model, img: np.ndarray, box: list, fb: np.ndarray) -> tuple[list[np.ndarray], float]:
    """Prompt SAM 2 with one flat-bug box and return its contours plus the IoS.

    Args:
        model: An ultralytics SAM model.
        img: Whole image as RGB uint8.
        box: flat-bug box [x1, y1, x2, y2].
        fb: flat-bug contour, (N, 2).

    Returns:
        (SAM contours in image coordinates, IoS against the flat-bug mask).
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    m = MARGIN * max(bw, bh)
    cx0, cy0 = int(max(0, x1 - m)), int(max(0, y1 - m))
    cx1, cy1 = int(min(W, x2 + m)), int(min(H, y2 + m))
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return [], 0.0
    s = float(np.clip(TARGET / max(bw, bh), 1.0, 8.0))
    big = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)),
                     interpolation=cv2.INTER_CUBIC) if s > 1.0 else crop
    res = model(big, bboxes=[[(x1 - cx0) * s, (y1 - cy0) * s, (x2 - cx0) * s, (y2 - cy0) * s]],
                verbose=False)
    cs = []
    for r in res:
        if r.masks is None:
            continue
        for mk in r.masks.data.cpu().numpy():
            mm = (mk > 0.5).astype(np.uint8)
            found, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs += [c[:, 0, :] / s + [cx0, cy0] for c in found
                   if len(c) >= 3 and cv2.contourArea(c) >= MIN_COMPONENT_PX * s * s]
    if not cs:
        return [], 0.0
    ch, cw = cy1 - cy0, cx1 - cx0
    fm = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(fm, [np.round(fb - [cx0, cy0]).astype(np.int32)], 1)
    sm = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(sm, [np.round(c - [cx0, cy0]).astype(np.int32) for c in cs], 1)
    return cs, ios(fm > 0, sm > 0)


def main() -> None:
    """Refine all images and assemble a single COCO file."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--pred-dir", default="/home/quentin/Desktop/AgriVolt/flatbug_results")
    ap.add_argument("--min-size", type=float, default=50.0,
                    help="sqrt(area) below which an instance keeps flat-bug's mask unrefined")
    ap.add_argument("-n", type=int, default=-1)
    ap.add_argument("--image-root", default="/home/quentin/Desktop/AgriVolt",
                    help="root for resolving relative image_path entries in the metadata")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sidecar = os.path.join(a.out, "per_image.jsonl")
    done = set()
    if os.path.exists(sidecar):
        for line in open(sidecar):
            try:
                done.add(json.loads(line)["file_name"])
            except Exception:  # noqa: BLE001 - a truncated final line is expected after a kill
                pass
        print(f"resuming: {len(done)} images already processed", flush=True)

    from ultralytics import SAM
    model = SAM(SAM_WEIGHTS)
    metas = sorted(glob.glob(os.path.join(a.pred_dir, "*", "metadata_*.json")))
    seen, chosen = set(), []
    for m in metas:
        d = os.path.dirname(m)
        if d not in seen:
            seen.add(d)
            chosen.append(m)
    if a.n > 0:
        chosen = chosen[:a.n]
    t0 = time.time()
    with open(sidecar, "a") as fh:
        for i, mp in enumerate(chosen, 1):
            j = json.load(open(mp))
            # flatbug_results stores a RELATIVE image_path; flatbug_yolo26_results an absolute one
            ipath = j["image_path"]
            if not os.path.isabs(ipath):
                ipath = os.path.join(a.image_root, ipath)
            name = os.path.basename(ipath)
            if name in done:
                continue
            img = cv2.imread(ipath)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            anns = []
            for box, cont, conf in zip(j["boxes"], j["contours"], j["confs"]):
                fb = fb_contour(cont)
                px = float(np.sqrt(max(abs(cv2.contourArea(fb.astype(np.float32))), 1.0)))
                if px < a.min_size:  # keep flat-bug's mask untouched
                    anns.append({"contours": [fb.tolist()], "source": "flatbug",
                                 "ios": None, "fb_px": px, "conf": float(conf)})
                    continue
                cs, v = refine_one(model, img, box, fb)
                if cs:
                    anns.append({"contours": [c.tolist() for c in cs], "source": "sam2",
                                 "ios": round(v, 4), "fb_px": px, "conf": float(conf)})
                else:  # SAM returned nothing: fall back rather than lose the instance
                    anns.append({"contours": [fb.tolist()], "source": "flatbug",
                                 "ios": None, "fb_px": px, "conf": float(conf)})
            fh.write(json.dumps({"file_name": name, "width": j["image_width"],
                                 "height": j["image_height"], "anns": anns}) + "\n")
            fh.flush()
            del img
            if i % 10 == 0 or i == len(chosen):
                el = time.time() - t0
                print(f"[{i}/{len(chosen)}] {el / 60:.0f} min elapsed, "
                      f"~{el / max(i - len(done), 1) * (len(chosen) - i) / 60:.0f} min left", flush=True)

    images, annotations, aid = [], [], 1
    n_sam = n_fb = 0
    for k, line in enumerate(open(sidecar), start=1):
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        images.append({"id": k, "file_name": r["file_name"], "width": r["width"], "height": r["height"]})
        for an in r["anns"]:
            seg = [list(map(float, np.asarray(c).ravel())) for c in an["contours"] if len(c) >= 3]
            if not seg:
                continue
            allc = np.concatenate([np.asarray(c) for c in an["contours"]])
            x0, y0 = allc[:, 0].min(), allc[:, 1].min()
            annotations.append({"id": aid, "image_id": k, "category_id": 1, "segmentation": seg,
                                "iscrowd": 0, "bbox": [float(x0), float(y0),
                                                       float(allc[:, 0].max() - x0),
                                                       float(allc[:, 1].max() - y0)],
                                "area": float(abs(cv2.contourArea(
                                    np.asarray(an["contours"][0], np.float32)))),
                                "source": an["source"], "ios": an["ios"],
                                "fb_sqrt_area_px": round(an["fb_px"], 1),
                                "flatbug_conf": round(an["conf"], 4)})
            aid += 1
            n_sam += an["source"] == "sam2"
            n_fb += an["source"] == "flatbug"
    out = os.path.join(a.out, "agrivolt_refined_coco.json")
    json.dump({"info": {"description": "AgriVolt flat-bug instances refined by box-prompted SAM 2.1; "
                                       f"instances below {a.min_size:.0f}px sqrt-area keep flat-bug's mask"},
               "licenses": [], "categories": [{"id": 1, "name": "arthropod"}],
               "images": images, "annotations": annotations}, open(out, "w"))
    print(f"\n{len(images)} images, {len(annotations)} instances -> {out}")
    print(f"   refined by SAM 2 : {n_sam}  ({n_sam / max(n_sam + n_fb, 1):.1%})")
    print(f"   kept from flat-bug: {n_fb}")


if __name__ == "__main__":
    main()
