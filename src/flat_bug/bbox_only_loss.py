"""Exclude bbox-only images from the segmentation loss, without touching anything else.

`calculate_segmentation_loss` receives no `batch`, so the per-image `has_mask` flag is stashed
by wrapping the criterion's `__call__` and read by the patched loop. Both patches are applied
to the CLASS, because YOLO26 wraps two `v8SegmentationLoss` instances inside `E2ELoss`
(one2many and one2one) and an instance-level patch would miss both.

Two things this must get right:

  * An image without masks is SKIPPED, not fed an all-zero target. `gt_mask = masks_i ==
    (mask_idx + 1)` is all-False for such an instance, and BCE on that trains the model to
    predict empty masks where a real animal is - actively harmful, not merely uninformative.
  * The normaliser changes from `fg_mask.sum()` (all positives) to the positives of masked
    images only. Leaving it as `fg_mask.sum()` would silently shrink the mask loss in
    proportion to how much bbox-only data is in the batch, which would make the loss weight
    depend on dataset mixture rather than on anything meaningful.

With every image masked, this is arithmetically identical to the stock loss - `test_bbox_only`
pins that.
"""

from __future__ import annotations

import contextlib
import threading

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import crop_mask, v8SegmentationLoss
from ultralytics.utils.ops import xyxy2xywh

_state = threading.local()

# Weight on the containment penalty for bbox-only instances. 0 reproduces the original
# behaviour exactly, where such instances contribute nothing to the mask loss.
CONTAINMENT_WEIGHT = 0.0


def set_containment_weight(w: float) -> None:
    """Set the containment weight for bbox-only instances. See `_containment_loss`."""
    global CONTAINMENT_WEIGHT
    CONTAINMENT_WEIGHT = float(w)


def _containment_loss(pred_masks_i, proto_i, xyxy_i, area_i):
    """Penalise predicted mask probability that falls OUTSIDE the ground-truth box.

    A bbox-only instance has no mask to imitate, but its box still says where the animal is
    NOT. Without that constraint the mask head gets no signal at all on these images and, on
    the domain they belong to, learns to over-cover: measured on artaxor-seg, the arm trained
    with bbox-only data produced masks 1.28x the true area, spilling 23.8% of their area
    outside the animal against 11.3% for the control.

    Note this is information the standard loss never uses for ANY instance. Upstream
    `single_mask_loss` applies `crop_mask(loss, xyxy)`, which zeroes the BCE outside the target
    box, so mask predicted outside the truth is unpenalised even when a real mask is available.

    The target outside the box is 0, so this is BCE against zeros restricted to the exterior,
    normalised per instance the same way `single_mask_loss` normalises - mean over the plane
    divided by box area - so the two terms are on a comparable scale.
    """
    pred = torch.einsum("in,nhw->ihw", pred_masks_i, proto_i)
    inside = crop_mask(torch.ones_like(pred), xyxy_i)
    outside = 1.0 - inside
    bce = F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred), reduction="none")
    return ((bce * outside).mean(dim=(1, 2)) / area_i).sum()


def _current_has_mask():
    return getattr(_state, "has_mask", None)


def _patched_calculate_segmentation_loss(
    self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
):
    """As upstream, but images flagged has_mask=False are excluded from the mask loss.

    With CONTAINMENT_WEIGHT > 0 they instead contribute a penalty on mask predicted outside
    their ground-truth box - the one thing their annotation does tell us. The two terms are
    normalised over their own populations so the weight controls the balance directly.
    """
    _, _, mask_h, mask_w = proto.shape
    loss = 0
    n_valid = 0
    contain = 0
    n_contain = 0

    has_mask = _current_has_mask()
    target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
    marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
    mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

    for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
        fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
        usable = has_mask is None or bool(has_mask[i])
        if usable and fg_mask_i.any():
            mask_idx = target_gt_idx_i[fg_mask_i]
            if self.overlap:
                gt_mask = (masks_i == (mask_idx + 1).view(-1, 1, 1)).float()
            else:
                gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
            loss += self.single_mask_loss(
                gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
            )
            n_valid += int(fg_mask_i.sum())
        elif CONTAINMENT_WEIGHT and fg_mask_i.any():
            # bbox-only image: no mask to imitate, but the box says where the animal is not.
            contain += _containment_loss(
                pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
            )
            n_contain += int(fg_mask_i.sum())
        else:
            # WARNING: keeps DDP from reporting unused gradients; also the branch a bbox-only
            # image takes when containment is off, contributing exactly zero.
            loss += (proto * 0).sum() + (pred_masks * 0).sum()

    out = loss / max(n_valid, 1)
    if CONTAINMENT_WEIGHT and n_contain:
        out = out + CONTAINMENT_WEIGHT * contain / n_contain
    return out


def _wrap_call(orig):
    def call(self, preds, batch):
        prev = getattr(_state, "has_mask", None)
        _state.has_mask = batch.get("has_mask") if isinstance(batch, dict) else None
        try:
            return orig(self, preds, batch)
        finally:
            _state.has_mask = prev
    return call


_ENABLED = False


def enable_bbox_only_segmentation_loss() -> None:
    """Apply the patch for the life of the process. Idempotent.

    Called only when at least one bbox-only dataset is configured, so a run without the feature
    is bit-for-bit what it was before this module existed.
    """
    global _ENABLED
    if _ENABLED:
        return
    v8SegmentationLoss.calculate_segmentation_loss = _patched_calculate_segmentation_loss
    v8SegmentationLoss.__call__ = _wrap_call(v8SegmentationLoss.__call__)
    _ENABLED = True


@contextlib.contextmanager
def bbox_only_segmentation_loss():
    """Enable per-image mask-loss masking for the duration of the block (used by tests)."""
    o_calc = v8SegmentationLoss.calculate_segmentation_loss
    o_call = v8SegmentationLoss.__call__
    v8SegmentationLoss.calculate_segmentation_loss = _patched_calculate_segmentation_loss
    v8SegmentationLoss.__call__ = _wrap_call(o_call)
    try:
        yield
    finally:
        v8SegmentationLoss.calculate_segmentation_loss = o_calc
        v8SegmentationLoss.__call__ = o_call
