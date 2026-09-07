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

# Weight on the box-projection loss for bbox-only instances. 0 reproduces the original
# behaviour exactly, where such instances contribute nothing to the mask loss.
PROJECTION_WEIGHT = 0.0


def set_projection_weight(w: float) -> None:
    """Set the projection weight for bbox-only instances. See `_projection_loss`."""
    global PROJECTION_WEIGHT
    PROJECTION_WEIGHT = float(w)


def _current_has_mask():
    return getattr(_state, "has_mask", None)


def _projection_loss(pred_masks_i, proto_i, xyxy_i):
    """Make the predicted mask's own extent match the ground-truth box.

    A bbox-only instance has no mask to imitate, but its box states exactly where the mask
    should begin and end. Deriving a box from the predicted mask and comparing it with the
    annotation gives a signal in BOTH directions, which is what a one-sided "do not spill
    outside the box" penalty cannot do: that penalty is minimised by predicting nothing, and
    with no positive mask supervision on these images an empty mask is exactly what it would
    train for.

    The extent is taken as a differentiable projection rather than a hard argmax. Reducing the
    mask probability by max along each axis gives its silhouette on x and on y; the target
    silhouettes are 1 across the box's span and 0 elsewhere. Matching them with a dice loss
    forces the mask to reach both edges of the box and to stop there. This is the projection
    term of BoxInst (Tian et al., CVPR 2021), which is the established way to supervise a mask
    head from boxes alone.

    Returns:
        Summed loss over the instances, one dice term per axis.
    """
    pred = torch.einsum("in,nhw->ihw", pred_masks_i, proto_i).sigmoid()
    target = crop_mask(torch.ones_like(pred), xyxy_i)

    def dice(a, b, eps=1e-5):
        num = 2.0 * (a * b).sum(dim=1)
        den = (a * a).sum(dim=1) + (b * b).sum(dim=1) + eps
        return 1.0 - num / den

    # silhouette on each axis: max over the other axis
    loss_x = dice(pred.max(dim=1).values, target.max(dim=1).values)
    loss_y = dice(pred.max(dim=2).values, target.max(dim=2).values)
    return (loss_x + loss_y).sum()


def _patched_calculate_segmentation_loss(
    self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
):
    """As upstream, but images flagged has_mask=False are excluded from the mask loss.

    With PROJECTION_WEIGHT > 0 they instead contribute a loss that makes the mask's own extent
    match their ground-truth box - the one mask-shaped fact their annotation contains. The two
    terms are normalised over their own populations, so the weight is a direct ratio between
    them rather than something that drifts with the dataset mixture.
    """
    _, _, mask_h, mask_w = proto.shape
    loss = 0
    n_valid = 0
    proj = 0
    n_proj = 0

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
        elif PROJECTION_WEIGHT and fg_mask_i.any():
            # bbox-only image: no mask to imitate, but its box fixes where the mask must start
            # and stop.
            proj += _projection_loss(pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i])
            n_proj += int(fg_mask_i.sum())
        else:
            # WARNING: keeps DDP from reporting unused gradients; also the branch a bbox-only
            # image takes when the projection loss is off, contributing exactly zero.
            loss += (proto * 0).sum() + (pred_masks * 0).sum()

    out = loss / max(n_valid, 1)
    if PROJECTION_WEIGHT and n_proj:
        out = out + PROJECTION_WEIGHT * proj / n_proj
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
