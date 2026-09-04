"""Datasets whose masks are not trustworthy: keep their boxes, ignore their masks.

Some sources give reliable bounding boxes and unreliable polygons. bugbox-bulk is the
motivating case - its outlines are a mix of coarse mask-traced blobs and 5-point hulls, some
of which enclose two animals as one instance - and the original ArTaxOr release is boxes only.
Training a segmentation head on those polygons teaches it a wrong boundary; discarding the
images altogether throws away good detection signal.

So: name those datasets in the config, and every polygon in them is downgraded to its own
bounding box. The instance still contributes fully to box, class and DFL loss. It contributes
nothing to the mask loss, and it is excluded from mask metrics.

Why the rectangle is kept at all rather than dropped: the augmentation pipeline transforms
segments alongside boxes (crops, flips, rotations), and `polygons2masks_overlap` needs a
polygon per instance to build a well-formed target tensor. The rectangle keeps those code
paths intact. It is never used as a training target - `has_mask` is what the loss reads.

A missing mask is NOT a neutral signal. With overlap_mask=True the loss builds
`gt_mask = masks_i == (mask_idx + 1)`, which for an absent instance is all-False, and BCE then
actively trains the model to predict empty masks where a real animal is. That is worse than
not training on it at all, which is why the flag has to reach the loss rather than simply
leaving the polygon out.
"""

from __future__ import annotations

import os
import re

import numpy as np

from flat_bug import logger


def compile_bbox_only(names: list[str] | tuple[str, ...] | None) -> re.Pattern | None:
    """Build a basename-prefix matcher for the named datasets.

    flat-bug identifies a dataset by the prefix of the image basename (`ArTaxOr_1234.jpg`), the
    same convention `fb_exclude_datasets` uses.
    """
    if not names:
        return None
    return re.compile(r"^(" + "|".join(re.escape(str(n)) for n in names) + r")")


def is_bbox_only(im_file: str, pattern: re.Pattern | None) -> bool:
    """Does this image belong to a dataset declared bbox-only?"""
    return bool(pattern) and bool(pattern.match(os.path.basename(im_file)))


def downgrade_labels(labels: list[dict], im_files: list[str], pattern: re.Pattern | None) -> int:
    """Replace polygons with their bounding rectangle for bbox-only datasets, in place.

    Returns the number of images downgraded. Every label gains a `has_mask` flag, so downstream
    code can rely on it being present rather than testing for its absence.
    """
    n_img = n_inst = 0
    for label, f in zip(labels, im_files):
        bbox_only = is_bbox_only(f, pattern)
        label["has_mask"] = not bbox_only
        if not bbox_only:
            continue
        segs = label.get("segments")
        if segs is None or len(segs) == 0:
            n_img += 1
            continue
        rects = []
        for s in segs:
            s = np.asarray(s, dtype=np.float32).reshape(-1, 2)
            if s.size == 0:
                rects.append(s)
                continue
            x0, y0 = s[:, 0].min(), s[:, 1].min()
            x1, y1 = s[:, 0].max(), s[:, 1].max()
            # Closed rectangle, matching the point order the augmentation pipeline expects.
            rects.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))
            n_inst += 1
        label["segments"] = rects
        n_img += 1
    if n_img:
        logger.info(f"bbox-only datasets: downgraded {n_inst} polygons to boxes across {n_img} images")
    return n_img
