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

Why the rectangle has to exist before ultralytics counts
--------------------------------------------------------
A bbox-only dataset may ship label files that are pure detection labels - five fields per line,
no polygon at all. Every one of artaxor-bbox's 7118 annotations is like that. Ultralytics'
`YOLODataset.get_labels` then finds `len(boxes) != len(segments)` over the whole corpus and
applies its documented fallback:

    "To resolve this only boxes will be used and all segments will be removed."

That removal is GLOBAL - every image in the run loses its masks, not only the bbox-only ones -
so the segmentation head trains on nothing. It is a warning rather than an error, so the run
continues to completion emitting zero losses and zero metrics, which is how it cost a 2.5 hour
training job before anyone noticed.

Hence `fill_missing_segments`, called from `cache_labels` BEFORE that count. `downgrade_labels`
then handles the opposite case - a bbox-only dataset that does carry polygons - and sets
`has_mask` for both.
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


def fill_missing_segments(labels: list[dict], pattern: re.Pattern | None) -> int:
    """Give every bbox-only instance a rectangle polygon, so box and segment counts agree.

    Must run inside `cache_labels`, i.e. BEFORE `YOLODataset.get_labels` compares the totals -
    see the module docstring for what happens otherwise. At this stage boxes are normalised
    xywh (what `verify_image_label` produces) and segments are normalised xy point arrays.

    Returns the number of instances given a synthetic polygon.
    """
    if pattern is None:
        return 0
    n_inst = n_img = 0
    for label in labels:
        if not is_bbox_only(label.get("im_file", ""), pattern):
            continue
        boxes = np.asarray(label.get("bboxes"), dtype=np.float32).reshape(-1, 4)
        segs = label.get("segments") or []
        if len(segs) == len(boxes):
            continue
        if len(segs):
            # Partly segmented: refuse rather than guess which polygon belongs to which box.
            raise ValueError(
                f"{label.get('im_file')}: {len(segs)} segments for {len(boxes)} boxes. A "
                "bbox-only dataset must be either fully segmented or not segmented at all."
            )
        rects = []
        for cx, cy, w, h in boxes:
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            # Closed rectangle, matching the point order the augmentation pipeline expects.
            rects.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32))
            n_inst += 1
        label["segments"] = rects
        n_img += 1
    if n_img:
        logger.info(
            f"bbox-only datasets: synthesised {n_inst} rectangle polygons across {n_img} "
            "box-only images, so segment and box counts reconcile"
        )
    return n_inst
