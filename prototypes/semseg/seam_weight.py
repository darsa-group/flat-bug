#!/usr/bin/env python3
"""PROTOTYPE - per-pixel loss weighting for inter-instance seams. Opt-in, off by default.

Inter-instance seams are 0.0111% of pixels and 1.2% of outline pixels, and under the
current per-channel ``pos_weight=[2, 8]`` they carry roughly 0.076% of the total BCE
signal. A model can reach a near-optimal loss while getting every seam wrong, which is what
the measured gap between seam recall (0.537) and outer-contour recall (0.745) suggests is
happening. Only seams decide whether a watershed can split touching animals.

This is Ronneberger's U-Net weight map in spirit - ``w = 1 + w0 * exp(-d^2 / 2*sigma^2)``
around a boundary between two instances.

NOTE ON A REJECTED SHORTCUT. Deriving seams from the target alone - "an outline pixel with
foreground on both sides lies between two instances" - would have avoided touching the
dataset, but measured against the exact definition it scored precision 0.058 at recall
0.630 over 120 crops: a local foreground fraction cannot tell "between two animals" from
"inside a dense cluster of one". Weighting that 30x would have put almost all the extra
signal on non-seams. Seams are therefore computed from the polygons, where instance
identity is unambiguous.

A flat multiplier on the seam ring itself would need ~500x to reach even a 5% share of the
BCE signal, which invites the opposite failure - painting seams everywhere, since a false
seam would cost almost nothing against a missed one. Spreading the weight over a Gaussian
neighbourhood raises the signal share with a far gentler ``w0``, as the original does.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

DEFAULT_W0 = 30.0
DEFAULT_SIGMA = 6.0
SEAM_DILATE = 5  # two instances whose dilations meet within this many px count as touching


def seam_from_polygons(polygons: list[np.ndarray], shape: tuple[int, int], outline_px: int) -> np.ndarray:
    """Mark outline pixels lying between two different instances.

    Args:
        polygons: Instance polygons in crop-local pixel coordinates.
        shape: (height, width) of the crop.
        outline_px: Outline thickness, matching the target rasterisation.

    Returns:
        Float array (H, W), 1.0 on seam pixels.
    """
    h, w = shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SEAM_DILATE + 1, 2 * SEAM_DILATE + 1))
    cover = np.zeros((h, w), np.uint16)
    outline = np.zeros((h, w), np.uint8)
    for c in polygons:
        ci = np.round(c).astype(np.int32)
        one = np.zeros((h, w), np.uint8)
        cv2.fillPoly(one, [ci], 1)
        cover += cv2.dilate(one, k)
        cv2.polylines(outline, [ci], isClosed=True, color=1, thickness=outline_px)
    return ((outline > 0) & (cover >= 2)).astype(np.float32)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur.

    Args:
        x: Tensor of shape (B, 1, H, W).
        sigma: Standard deviation in pixels.

    Returns:
        The blurred tensor, same shape.
    """
    r = max(1, int(round(3 * sigma)))
    g = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(x, g.view(1, 1, -1, 1), padding=(r, 0))


def weight_from_seam(seam: torch.Tensor, w0: float = DEFAULT_W0, sigma: float = DEFAULT_SIGMA) -> torch.Tensor:
    """Spread a seam mask into a smooth per-pixel loss weight.

    Args:
        seam: Seam mask of shape (B, 1, H, W).
        w0: Peak extra weight at a seam. 0 returns an all-ones map.
        sigma: Spatial spread in pixels.

    Returns:
        Weights of shape (B, 1, H, W), 1.0 away from seams and up to ``1 + w0`` at one.
    """
    if w0 <= 0:
        return torch.ones_like(seam)
    spread = _gaussian_blur(seam, sigma)
    peak = spread.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return 1.0 + w0 * (spread / peak)
