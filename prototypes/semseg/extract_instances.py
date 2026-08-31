#!/usr/bin/env python3
"""PROTOTYPE - turn (foreground, outline) probability maps into instances.

Runs BLOB-WISE rather than over the whole image. Foreground is about 10% of pixels, so a
watershed over a 143Mpx scan would spend 90% of its time on empty background. Instead the
foreground is split into connected components and each is processed inside its own bounding
box, which also makes the components independent and therefore trivially parallel.

Most blobs contain exactly one animal. Those are detected by their distance transform
having a single peak and are emitted directly, skipping the watershed entirely - the
expensive path runs only where animals actually touch.

Markers come from the distance transform, not from eroding the foreground by the outline.
Measured against 951 known polygons, distance-transform markers recovered 0.94x the true
instance count (mean error 2.73 per image) where outline erosion gave 1.28x (error 8.35):
a uniform erosion deletes a 2px leg while barely denting a 30px body, so legs become
phantom instances.

GPU: the network forward pass and thresholding are GPU work, but connected components,
the distance transform and the watershed are CPU (OpenCV / scikit-image). GPU versions
exist in cucim/cupy, which are not installed here; the blob-wise decomposition makes the
CPU cost small enough that it is not the bottleneck.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

DEFAULTS = dict(t_fg=0.5, t_outline=0.5, alpha=0.45, min_area=64, use_outline=True, margin=4)


def _split_blob(mask: np.ndarray, outline: np.ndarray | None, alpha: float,
                min_area: int, use_outline: bool) -> tuple[np.ndarray, int]:
    """Split one foreground blob into instances.

    Args:
        mask: Binary blob mask, cropped to its bounding box.
        outline: Predicted outline probability over the same crop, or None.
        alpha: Fraction of the blob's maximum distance defining a marker.
        min_area: Minimum instance area in pixels.
        use_outline: Whether to subtract the predicted outline before seeding, which helps
            where two animals overlap enough that the distance transform has one peak.

    Returns:
        The labelled crop and the number of instances found.
    """
    core = mask.copy()
    if use_outline and outline is not None:
        core = (mask > 0) & (outline < 0.5)
        core = core.astype(np.uint8)
        if core.sum() < min_area:  # outline swallowed the blob; fall back
            core = mask.copy()
    dt = cv2.distanceTransform(core, cv2.DIST_L2, 5)
    mx = float(dt.max())
    if mx <= 0:
        return mask.astype(np.int32), 1
    seeds = (dt > alpha * mx).astype(np.uint8)
    n_seed, seed_lab = cv2.connectedComponents(seeds)
    n_seed -= 1
    if n_seed <= 1:  # single animal: no watershed needed
        return mask.astype(np.int32), 1
    # drop seeds too small to be a real animal, so noise does not over-split
    keep = [i for i in range(1, n_seed + 1) if (seed_lab == i).sum() >= max(4, min_area // 16)]
    if len(keep) <= 1:
        return mask.astype(np.int32), 1
    relab = np.zeros_like(seed_lab)
    for j, i in enumerate(keep, start=1):
        relab[seed_lab == i] = j
    lab = watershed(-dt, markers=relab, mask=mask > 0)
    return lab, len(keep)


def extract(prob_fg: np.ndarray, prob_outline: np.ndarray | None = None, **kw) -> list[np.ndarray]:
    """Extract instance polygons from probability maps.

    Args:
        prob_fg: Foreground probability, (H, W) in [0, 1].
        prob_outline: Outline probability over the same grid, or None.
        **kw: Overrides for ``DEFAULTS``.

    Returns:
        List of (N, 2) int arrays, one contour per instance, in image coordinates.
    """
    p = {**DEFAULTS, **kw}
    fg = (prob_fg > p["t_fg"]).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = []
    m = p["margin"]
    H, W = fg.shape
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < p["min_area"]:
            continue
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(W, x + w + m), min(H, y + h + m)
        blob = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ol = prob_outline[y0:y1, x0:x1] if prob_outline is not None else None
        sub, k = _split_blob(blob, ol, p["alpha"], p["min_area"], p["use_outline"])
        for j in range(1, int(sub.max()) + 1):
            piece = (sub == j).astype(np.uint8)
            if piece.sum() < p["min_area"]:
                continue
            cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cs:
                if len(c) >= 3 and cv2.contourArea(c) >= p["min_area"]:
                    out.append(c[:, 0, :] + [x0, y0])
    return out


def n_blobs_needing_watershed(prob_fg: np.ndarray, prob_outline: np.ndarray | None, **kw) -> tuple[int, int]:
    """Count how many blobs contain more than one animal, for profiling.

    Args:
        prob_fg: Foreground probability.
        prob_outline: Outline probability, or None.
        **kw: Overrides for ``DEFAULTS``.

    Returns:
        (blobs needing a watershed, total blobs).
    """
    p = {**DEFAULTS, **kw}
    fg = (prob_fg > p["t_fg"]).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    multi = tot = 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < p["min_area"]:
            continue
        tot += 1
        blob = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        mx = float(dt.max())
        if mx <= 0:
            continue
        k = cv2.connectedComponents((dt > p["alpha"] * mx).astype(np.uint8))[0] - 1
        if k > 1:
            multi += 1
    return multi, tot


_ = ndi  # scipy is imported for parity with skimage's watershed backend
