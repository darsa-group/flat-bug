"""Memory-bounded validation loss for dense flat-bug images.

Why this exists. YOLOv26's one2many head assigns many positive anchors per ground-truth
instance. On a flat-bug crop holding a hundred animals that is tens of thousands of
positives, and ultralytics' segmentation loss materialises three tensors of shape
(n_positives, mask_h, mask_w) at once - the boolean ground-truth expansion, the prototype
product, and the per-pixel BCE. At 50k positives and a 160x160 prototype that is over 15 GB,
which is why the trainer used to suppress the validation loss entirely and log four zeros.

The fix is arithmetic, not statistical: `single_mask_loss` reduces with `.sum()` over
instances, so summing over chunks of instances gives the same number to floating-point
associativity, while peak memory is bounded by the chunk instead of by the crowd. That
matters here because the whole point of the synthetic-scenes experiment is crowded images,
and a validation signal that vanishes exactly on the hard cases is no signal at all.

The patch is applied only around the validation pass, so the training path stays bit-for-bit
what it was for every earlier run.
"""

from __future__ import annotations

import contextlib

import torch
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import xyxy2xywh

# 1024 positives x 160 x 160 x 4 bytes is about 105 MB per tensor, so roughly 0.3 GB live
# across the three, which sits comfortably beside the model on any of the L40S cards.
DEFAULT_CHUNK = 1024


def _chunked_calculate_segmentation_loss(
    self: v8SegmentationLoss,
    fg_mask: torch.Tensor,
    masks: torch.Tensor,
    target_gt_idx: torch.Tensor,
    target_bboxes: torch.Tensor,
    batch_idx: torch.Tensor,
    proto: torch.Tensor,
    pred_masks: torch.Tensor,
    imgsz: torch.Tensor,
    _chunk: int = DEFAULT_CHUNK,
) -> torch.Tensor:
    """Identical to `v8SegmentationLoss.calculate_segmentation_loss`, in chunks of positives."""
    _, _, mask_h, mask_w = proto.shape
    loss = 0

    target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
    marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
    mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

    for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
        fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
        if fg_mask_i.any():
            mask_idx = target_gt_idx_i[fg_mask_i]
            pred_i, xyxy_i, area_i = pred_masks_i[fg_mask_i], mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
            if not self.overlap:
                # Gathered once per image; indexing it per chunk is cheap.
                image_masks = masks[batch_idx.view(-1) == i]
            for start in range(0, mask_idx.shape[0], _chunk):
                stop = start + _chunk
                idx = mask_idx[start:stop]
                if self.overlap:
                    gt_mask = (masks_i == (idx + 1).view(-1, 1, 1)).float()
                else:
                    gt_mask = image_masks[idx]
                loss += self.single_mask_loss(gt_mask, pred_i[start:stop], proto_i, xyxy_i[start:stop], area_i[start:stop])
        else:
            # WARNING: prevents Multi-GPU DDP 'unused gradient' errors, do not remove.
            loss += (proto * 0).sum() + (pred_masks * 0).sum()

    return loss / fg_mask.sum()


@contextlib.contextmanager
def bounded_segmentation_loss(chunk: int = DEFAULT_CHUNK):
    """Swap in the chunked segmentation loss for the duration of the block."""
    original = v8SegmentationLoss.calculate_segmentation_loss

    def _patched(self, *args, **kwargs):
        return _chunked_calculate_segmentation_loss(self, *args, _chunk=chunk, **kwargs)

    v8SegmentationLoss.calculate_segmentation_loss = _patched
    try:
        yield
    finally:
        v8SegmentationLoss.calculate_segmentation_loss = original
