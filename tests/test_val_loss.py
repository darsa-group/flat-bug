"""The chunked validation segmentation loss must equal the loss it replaces.

The whole justification for chunking is that `single_mask_loss` reduces with `.sum()` over
instances, so the split is invisible to the result. If that ever stops being true - because
ultralytics changes the reduction, say - the validation losses in results.csv silently become
a different quantity from the training losses beside them, and the A/B built on them is void.
"""

import pytest
import torch

from flat_bug.val_loss import bounded_segmentation_loss

ultralytics_loss = pytest.importorskip("ultralytics.utils.loss")


class _Criterion:
    """The two attributes `calculate_segmentation_loss` actually touches."""

    def __init__(self, overlap: bool):
        self.overlap = overlap
        self.single_mask_loss = ultralytics_loss.v8SegmentationLoss.single_mask_loss


def _inputs(overlap: bool, n_anchors: int = 300, n_gt: int = 7, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    bs, nm, mh, mw = 2, 32, 32, 32
    fg_mask = torch.rand(bs, n_anchors, generator=g) > 0.35
    fg_mask[0, 0] = True  # guarantee at least one positive image
    target_gt_idx = torch.randint(0, n_gt, (bs, n_anchors), generator=g)
    # Boxes covering a random sub-rectangle of a 128 px image, in xyxy.
    xy1 = torch.rand(bs, n_anchors, 2, generator=g) * 60
    target_bboxes = torch.cat([xy1, xy1 + 20 + torch.rand(bs, n_anchors, 2, generator=g) * 40], dim=2)
    proto = torch.randn(bs, nm, mh, mw, generator=g)
    pred_masks = torch.randn(bs, n_anchors, nm, generator=g)
    imgsz = torch.tensor([128.0, 128.0])
    if overlap:
        masks = torch.randint(0, n_gt + 1, (bs, mh, mw), generator=g).float()
        batch_idx = torch.zeros(bs * n_gt, 1)
    else:
        masks = torch.randint(0, 2, (bs * n_gt, mh, mw), generator=g).float()
        batch_idx = torch.arange(bs).repeat_interleave(n_gt).view(-1, 1).float()
    return fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("chunk", [1, 7, 64, 100000])
def test_chunked_matches_unchunked(overlap: bool, chunk: int):
    """Any chunk size, including one bigger than the batch, gives the reference value."""
    criterion = _Criterion(overlap)
    args = _inputs(overlap)

    reference = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(criterion, *args)
    with bounded_segmentation_loss(chunk=chunk):
        chunked = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(criterion, *args)

    assert torch.allclose(reference, chunked, rtol=1e-5, atol=1e-6), f"{reference} != {chunked}"


def test_patch_is_reverted_even_on_error():
    """A failure inside the block must not leave the class method swapped out."""
    original = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss
    with pytest.raises(RuntimeError):
        with bounded_segmentation_loss():
            raise RuntimeError("boom")
    assert ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss is original


def test_image_with_no_positive_anchors_still_contributes():
    """The DDP-guard branch must survive chunking, or multi-GPU runs break."""
    criterion = _Criterion(overlap=True)
    fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz = _inputs(True)
    fg_mask[1] = False  # second image has nothing assigned

    reference = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(
        criterion, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    )
    with bounded_segmentation_loss(chunk=8):
        chunked = ultralytics_loss.v8SegmentationLoss.calculate_segmentation_loss(
            criterion, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
        )
    assert torch.allclose(reference, chunked, rtol=1e-5, atol=1e-6)
