"""Custom loss functions for flatbug.

Currently this provides :class:`ThinWeightedSegmentationLoss`, which up-weights the
thin parts of an instance (legs, antennae, wings) in the mask loss.

Motivation: thin appendages account for only ~7.4% of ground-truth mask area, so a
prediction that captures a perfect body and *no* appendages at all still scores a
mask IoU of ~0.93. Uniform per-pixel BCE therefore has almost no gradient signal for
them, and measured appendage-pixel recall (0.57) is far below core-body recall (0.93).
"""

from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask


def thin_structure_map(mask: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """Identify the thin parts of a binary mask by morphological opening.

    A pixel is "thin" if it is removed by an opening with a ``kernel``-sized structuring
    element, i.e. it is not covered by any disk of radius ``kernel // 2`` lying inside the
    mask. For an insect this selects the appendages and leaves the body.

    Erosion is computed as ``-maxpool(-x)`` so that the whole operation is a pair of
    max-pools and runs on GPU without leaving the graph-free target tensors.

    Args:
        mask (torch.Tensor): Binary (0/1) float masks of shape (N, H, W).
        kernel (int): Size of the square structuring element. Must be odd.

    Returns:
        (torch.Tensor): Float tensor of shape (N, H, W), 1 on thin pixels and 0 elsewhere.
    """
    if kernel % 2 != 1:
        raise ValueError(f"kernel must be odd, got {kernel}")
    x = mask.unsqueeze(1)
    eroded = -F.max_pool2d(-x, kernel, stride=1, padding=kernel // 2)
    opened = F.max_pool2d(eroded, kernel, stride=1, padding=kernel // 2)
    return (x - opened).clamp_(0, 1).squeeze(1)


class ThinWeightedSegmentationLoss(v8SegmentationLoss):
    """Segmentation loss that up-weights thin structures in the mask term.

    Identical to :class:`~ultralytics.utils.loss.v8SegmentationLoss` except that the
    per-pixel mask BCE is multiplied by ``1 + thin_weight`` on pixels belonging to a thin
    part of the ground-truth instance. The result is renormalised by the mean weight
    inside the box crop, so the mask-loss magnitude stays comparable to an unweighted run
    and ``box``/``cls``/``dfl`` gain balancing does not need to be retuned.

    Note:
        This only has an effect on appendages that are actually present in the training
        target. At ``mask_ratio=4`` only ~68% of appendage pixels survive rasterisation
        (~48% for instances below 32px), so this should be paired with ``mask_ratio=2``.
    """

    def __init__(self, model, thin_weight: float = 4.0, thin_kernel: int = 5, **kwargs):
        """Initialise the loss.

        Args:
            model (torch.nn.Module): The de-paralleled model, as for the base class.
            thin_weight (float): Extra weight given to thin pixels. 0 reproduces the base class.
            thin_kernel (int): Structuring element size used to define "thin". Must be odd.
            **kwargs: Forwarded to :class:`~ultralytics.utils.loss.v8SegmentationLoss`.
        """
        super().__init__(model, **kwargs)
        self.thin_weight = float(thin_weight)
        self.thin_kernel = int(thin_kernel)

    def single_mask_loss(
        self, gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the thin-structure-weighted mask loss for the instances of one image.

        Shadows the base class' static method of the same name; the call site in
        ``calculate_segmentation_loss`` invokes it as ``self.single_mask_loss(...)``.

        The weight is passed to ``binary_cross_entropy_with_logits`` rather than applied
        afterwards, which avoids materialising a second (N, H, W) tensor. This matters at
        ``mask_ratio=2``, where these tensors are four times larger than the stock setting.

        Args:
            gt_mask (torch.Tensor): Ground truth masks of shape (N, H, W).
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth boxes in xyxy format, scaled to mask size, of shape (N, 4).
            area (torch.Tensor): Normalised area of each ground truth box, of shape (N,).

        Returns:
            (torch.Tensor): The summed mask loss over the instances.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        if self.thin_weight <= 0:
            loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
            return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

        weight = 1.0 + self.thin_weight * thin_structure_map(gt_mask, self.thin_kernel)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, weight=weight, reduction="none")

        # Box indicator, built once and reused, mirroring ultralytics' crop_mask.
        h, w = gt_mask.shape[-2:]
        x1, y1, x2, y2 = torch.chunk(xyxy[:, :, None], 4, 1)
        cols = torch.arange(w, device=gt_mask.device, dtype=x1.dtype)[None, None, :]
        rows = torch.arange(h, device=gt_mask.device, dtype=x1.dtype)[None, :, None]
        inside = ((cols >= x1) * (cols < x2) * (rows >= y1) * (rows < y2)).to(loss.dtype)

        n_inside = inside.sum(dim=(1, 2)).clamp_min(1.0)
        # Mean weight within the crop, so the magnitude matches an unweighted run and the
        # box/cls/dfl gain balance carries over from previous experiments unchanged.
        mean_weight = ((weight * inside).sum(dim=(1, 2)) / n_inside).clamp_min(1e-6)
        return ((loss * inside).sum(dim=(1, 2)) / (h * w) / mean_weight / area).sum()




def build_thin_criterion(model, thin_weight: float = 4.0, thin_kernel: int = 5):
    """Build a thin-structure-weighted criterion matching the model's head type.

    Mirrors :meth:`ultralytics.nn.tasks.SegmentationModel.init_criterion`: end-to-end heads
    (yolo26 and later) wrap the loss in :class:`~ultralytics.utils.loss.E2ELoss`, which
    instantiates it twice with different ``tal_topk`` for the one2many and one2one branches.

    Args:
        model (torch.nn.Module): The de-paralleled segmentation model.
        thin_weight (float): Extra weight given to thin pixels.
        thin_kernel (int): Structuring element size used to define "thin". Must be odd.

    Returns:
        (E2ELoss | ThinWeightedSegmentationLoss): The criterion to assign to the model.
    """
    loss_fn = partial(ThinWeightedSegmentationLoss, thin_weight=thin_weight, thin_kernel=thin_kernel)
    if getattr(model, "end2end", False):
        return E2ELoss(model, loss_fn)
    return loss_fn(model)


class ThinCriterionFactory:
    """Picklable replacement for a model's ``init_criterion`` method.

    Assigning a closure to ``model.init_criterion`` breaks checkpoint saving, because
    ultralytics deep-copies and pickles the whole model. This class is defined at module
    level so it pickles by reference; the self-reference to ``model`` is resolved by
    pickle's memo, since the model is the object being serialised anyway.

    Construction of the criterion stays lazy (it happens on the first ``loss()`` call),
    which matters because :class:`~ultralytics.utils.loss.v8DetectionLoss` reads
    ``model.args``, and the trainer only attaches those after the model is built.
    """

    def __init__(self, model, thin_weight: float = 4.0, thin_kernel: int = 5):
        """Store the model and the thin-structure hyperparameters.

        Args:
            model (torch.nn.Module): The segmentation model this criterion belongs to.
            thin_weight (float): Extra weight given to thin pixels.
            thin_kernel (int): Structuring element size used to define "thin". Must be odd.
        """
        self.model = model
        self.thin_weight = float(thin_weight)
        self.thin_kernel = int(thin_kernel)

    def __call__(self):
        """Build the criterion.

        Returns:
            (E2ELoss | ThinWeightedSegmentationLoss): The criterion for the stored model.
        """
        return build_thin_criterion(self.model, self.thin_weight, self.thin_kernel)
