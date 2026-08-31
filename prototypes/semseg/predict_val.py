#!/usr/bin/env python3
"""PROTOTYPE - run a semseg checkpoint over every validation image and write error overlays.

Images are processed at NATIVE resolution, tiled with overlap. Tile logits are blended with
a cosine ramp weighted by distance from the tile edge rather than OR-ed together: OR dilates
every seam and bakes a tiling artefact into the foreground, while averaging logits leaves no
visible join.

The overlay marks, at 50% alpha over the original image:
    white  true positive  - predicted foreground that is annotated
    red    false positive - predicted foreground that is not annotated
    blue   false negative - annotated foreground that was not predicted

Usage:
    predict_val.py CHECKPOINT -o OUTDIR [-d DATA] [--tile 1024] [--overlap 256] [-n N]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import read_polygons  # noqa: E402

ALPHA = 0.5


def blend_weights(h: int, w: int, ramp: int) -> np.ndarray:
    """Cosine edge ramp, so overlapping tiles average smoothly instead of showing a seam.

    Args:
        h: Tile height.
        w: Tile width.
        ramp: Ramp width in pixels.

    Returns:
        Float array (h, w) in (0, 1].
    """
    def axis(n: int) -> np.ndarray:
        v = np.ones(n, np.float32)
        r = min(ramp, n // 2)
        if r > 0:
            t = 0.5 * (1 - np.cos(np.linspace(0, np.pi, r, dtype=np.float32)))
            v[:r] = t
            v[-r:] = t[::-1]
        return np.clip(v, 1e-3, 1.0)
    return axis(h)[:, None] * axis(w)[None, :]


@torch.no_grad()
def predict_image(model, img: np.ndarray, tile: int, overlap: int, device: str) -> np.ndarray:
    """Predict foreground probability for one whole image at native resolution.

    Args:
        model: The network.
        img: HxWx3 uint8 RGB image.
        tile: Tile size in pixels.
        overlap: Overlap between adjacent tiles.
        device: Torch device.

    Returns:
        Float array (H, W) of foreground probability.
    """
    h, w = img.shape[:2]
    step = max(32, tile - overlap)
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)
            crop = img[y0c:y1, x0c:x1]
            ch, cw = crop.shape[:2]
            pad = crop
            if ch % 32 or cw % 32:  # the network needs dimensions divisible by 32
                pad = cv2.copyMakeBorder(crop, 0, (-ch) % 32, 0, (-cw) % 32, cv2.BORDER_REFLECT_101)
            x = torch.from_numpy(np.ascontiguousarray(pad)).permute(2, 0, 1)[None].to(device).half() / 255
            p = torch.sigmoid(model(x)[:, 0]).float().cpu().numpy()[0][:ch, :cw]
            bw = blend_weights(ch, cw, overlap // 2)
            acc[y0c:y1, x0c:x1] += p * bw
            wsum[y0c:y1, x0c:x1] += bw
    return acc / np.maximum(wsum, 1e-6)


def main() -> None:
    """Predict over the validation split and write overlays."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("-d", "--data", default="/home/quentin/Desktop/flatbug-dir/flat-bug-data/yolo/insects")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("-n", type=int, default=-1, help="limit number of images")
    ap.add_argument("--max-side", type=int, default=0, help="skip images whose long side exceeds this (0 = no limit)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    n_cls = ck["model"]["segmentation_head.0.weight"].shape[0]
    model = smp.Unet(ck["encoder"], encoder_weights=None, in_channels=3, classes=n_cls)
    model.load_state_dict(ck["model"])
    model.eval().to(dev).half()
    print(f"{os.path.basename(a.checkpoint)}: epoch {ck['epoch']}, {n_cls} channels -> {a.out}", flush=True)

    files = sorted(glob.glob(os.path.join(a.data, "images", "val", "*.jpg")))
    if a.n > 0:
        files = files[:a.n]
    tp_t = fp_t = fn_t = 0
    rows = []
    for i, f in enumerate(files, 1):
        img = cv2.imread(f)
        if img is None:
            continue
        h, w = img.shape[:2]
        if a.max_side and max(h, w) > a.max_side:
            print(f"[{i}/{len(files)}] skip {os.path.basename(f)} ({w}x{h})", flush=True)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prob = predict_image(model, img, a.tile, a.overlap, dev)
        pred = prob > a.threshold

        gt = np.zeros((h, w), np.uint8)
        lf = f.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"
        for c in sorted(read_polygons(lf, w, h), key=lambda q: -cv2.contourArea(q.astype(np.float32))):
            cv2.fillPoly(gt, [np.round(c).astype(np.int32)], 1)
        gtb = gt > 0

        tp, fp, fn = pred & gtb, pred & ~gtb, ~pred & gtb
        tp_t += int(tp.sum())
        fp_t += int(fp.sum())
        fn_t += int(fn.sum())

        col = np.zeros((h, w, 3), np.float32)
        col[tp] = (1.0, 1.0, 1.0)   # white  true positive
        col[fp] = (1.0, 0.0, 0.0)   # red    false positive
        col[fn] = (0.0, 0.0, 1.0)   # blue   false negative
        m = (tp | fp | fn)[..., None].astype(np.float32) * ALPHA
        out = (img.astype(np.float32) / 255) * (1 - m) + col * m
        cv2.imwrite(os.path.join(a.out, os.path.basename(f)),
                    cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append((os.path.basename(f), int(tp.sum()), int(fp.sum()), int(fn.sum())))
        if i % 25 == 0 or i == len(files):
            r = tp_t / max(tp_t + fn_t, 1)
            p_ = tp_t / max(tp_t + fp_t, 1)
            print(f"[{i}/{len(files)}] running recall {r:.4f} precision {p_:.4f}", flush=True)

    with open(os.path.join(a.out, "per_image.csv"), "w") as fh:
        fh.write("image,tp,fp,fn\n")
        for n, tp, fp, fn in rows:
            fh.write(f"{n},{tp},{fp},{fn}\n")
    r = tp_t / max(tp_t + fn_t, 1)
    p_ = tp_t / max(tp_t + fp_t, 1)
    print(f"\n{len(rows)} images. pixel recall {r:.4f}  precision {p_:.4f}  "
          f"F1 {2 * r * p_ / max(r + p_, 1e-9):.4f}  IoU {tp_t / max(tp_t + fp_t + fn_t, 1):.4f}")


if __name__ == "__main__":
    main()
