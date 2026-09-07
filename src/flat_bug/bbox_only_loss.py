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

# Weight on the pairwise colour-affinity term. Supplies the SHAPE that a box cannot; without
# it the projection term alone is satisfied by a mask that fills the box.
PAIRWISE_WEIGHT = 0.0


def set_projection_weight(w: float) -> None:
    """Set the projection weight for bbox-only instances. See `_projection_loss`."""
    global PROJECTION_WEIGHT
    PROJECTION_WEIGHT = float(w)


def set_pairwise_weight(w: float) -> None:
    """Set the colour-affinity weight for bbox-only instances. See `_pairwise_loss`."""
    global PAIRWISE_WEIGHT
    PAIRWISE_WEIGHT = float(w)


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


def _current_img():
    return getattr(_state, "img", None)


def _neighbours(x, k=3, dilation=2):
    """Values of each pixel's k*k neighbourhood, centre removed. (n,H,W) -> (n,K,H,W)."""
    n, h, w = x.shape
    unf = F.unfold(x.unsqueeze(1), k, dilation=dilation, padding=dilation * (k // 2))
    unf = unf.view(n, k * k, h, w)
    centre = k * k // 2
    idx = [i for i in range(k * k) if i != centre]
    return unf[:, idx]


def colour_affinity(img, k=3, dilation=2, theta=2.0):
    """How alike is each pixel to each of its neighbours? (3,H,W) -> (K,H,W) in (0, 1].

    Similarity is exp(-||c_i - c_j|| / theta) on the raw channels. It depends only on the
    image, so it is computed once per image and reused by every instance in it.
    """
    c, h, w = img.shape
    unf = F.unfold(img.unsqueeze(0), k, dilation=dilation, padding=dilation * (k // 2))
    unf = unf.view(1, c, k * k, h, w)
    centre = k * k // 2
    dist = (unf - unf[:, :, centre : centre + 1]).norm(dim=1)     # (1,k*k,H,W)
    idx = [i for i in range(k * k) if i != centre]
    return torch.exp(-dist[0, idx] / theta)


def _pairwise_loss(pred_masks_i, proto_i, xyxy_i, sim, threshold=0.3, chunk=32):
    """Neighbouring pixels that LOOK alike should be labelled alike.

    A box fixes an instance's extent but says nothing about its shape, so the projection loss
    alone is satisfied by a mask that simply fills the box - which on this corpus would
    over-cover by 1.79x, since a real mask occupies a median 0.558 of its own box. The shape
    has to come from the image, and the oldest usable prior is that adjacent pixels of similar
    colour belong to the same thing.

    For each neighbour pair (i, j) that looks alike, the loss is -log P(y_i = y_j), where
    P(same) = P_i.P_j + (1-P_i).(1-P_j), evaluated in log space for stability. Pairs that look
    DIFFERENT are dropped rather than pushed apart, so the term never invents a boundary; it
    only forbids one where the image says there is none. That abstention is what keeps it
    honest on low-contrast data - measured animal-vs-background LAB distance is above 25 on 24
    of 26 datasets, but only 17.0 on PeMaToEuroPep and 24.8 on AMI-traps.

    Restricted to the interior of the ground-truth box: outside it the answer is already known
    to be background, which the projection term handles.
    """
    total = pred_masks_i.new_zeros(())
    denom = pred_masks_i.new_zeros(())
    for a in range(0, pred_masks_i.shape[0], chunk):
        pm, xy = pred_masks_i[a : a + chunk], xyxy_i[a : a + chunk]
        pred = torch.einsum("in,nhw->ihw", pm, proto_i)
        lp, ln = F.logsigmoid(pred), F.logsigmoid(-pred)
        log_same = torch.logaddexp(
            lp.unsqueeze(1) + _neighbours(lp), ln.unsqueeze(1) + _neighbours(ln)
        )
        w = (sim.unsqueeze(0) >= threshold).to(pred.dtype) * crop_mask(
            torch.ones_like(pred), xy
        ).unsqueeze(1)
        total = total - (log_same * w).sum()
        denom = denom + w.sum()
    return total / denom.clamp_min(1.0)


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
    pair = 0
    n_pair = 0
    img = _current_img() if PAIRWISE_WEIGHT else None

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
        elif (PROJECTION_WEIGHT or PAIRWISE_WEIGHT) and fg_mask_i.any():
            # bbox-only image: no mask to imitate. Its box fixes where the mask starts and
            # stops; the image itself supplies the shape in between.
            if PROJECTION_WEIGHT:
                proj += _projection_loss(pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i])
                n_proj += int(fg_mask_i.sum())
            if PAIRWISE_WEIGHT and img is not None and i < img.shape[0]:
                # one similarity map per image, shared by all its instances
                im = img[i].float()
                if im.max() > 1.5:
                    im = im / 255.0
                im = F.interpolate(
                    im.unsqueeze(0), size=(mask_h, mask_w), mode="bilinear", align_corners=False
                )[0]
                pair += _pairwise_loss(
                    pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], colour_affinity(im)
                )
                n_pair += 1
        else:
            # WARNING: keeps DDP from reporting unused gradients; also the branch a bbox-only
            # image takes when the projection loss is off, contributing exactly zero.
            loss += (proto * 0).sum() + (pred_masks * 0).sum()

    out = loss / max(n_valid, 1)
    if PROJECTION_WEIGHT and n_proj:
        out = out + PROJECTION_WEIGHT * proj / n_proj
    if PAIRWISE_WEIGHT and n_pair:
        out = out + PAIRWISE_WEIGHT * pair / n_pair
    return out


def _wrap_call(orig):
    def call(self, preds, batch):
        prev_m = getattr(_state, "has_mask", None)
        prev_i = getattr(_state, "img", None)
        is_d = isinstance(batch, dict)
        _state.has_mask = batch.get("has_mask") if is_d else None
        # The pairwise term needs the pixels, which the loss signature does not carry.
        _state.img = batch.get("img") if is_d else None
        try:
            return orig(self, preds, batch)
        finally:
            _state.has_mask = prev_m
            _state.img = prev_i
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
