#!/usr/bin/env python3
"""Tile dataset for two-channel semantic segmentation of arthropods.

Targets are (foreground, outline). The outline channel is what later separates touching
instances: eroding foreground by the outline leaves one connected core per animal, which
seeds a watershed.

Edge handling differs deliberately from flat-bug's detection pipeline. `FixInstances`
uses `area_thr=0.975`, deleting any instance not almost entirely inside the crop, because
a truncated box is a wrong regression target. Here those pixels are still arthropod, so
partial instances are KEPT in the foreground channel; only the outline is suppressed along
the crop border, since the image edge is not an animal's outline.
"""

from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

OUTLINE_PX = 3  # total outline thickness at native resolution


def _read_polygons(label_path: str, w: int, h: int) -> list[np.ndarray]:
    """Read a YOLO-segmentation label file and return polygons in pixel coordinates.

    Args:
        label_path: Path to the ``.txt`` label file.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        A list of (N, 2) float arrays, one per instance.
    """
    out = []
    if not os.path.isfile(label_path):
        return out
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            c = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2) * np.array([w, h], np.float32)
        except ValueError:
            continue
        if len(c) >= 3:
            out.append(c)
    return out


def rasterise(polygons: list[np.ndarray], shape: tuple[int, int], border_is_edge: bool = True) -> np.ndarray:
    """Rasterise instance polygons into a (2, H, W) float target.

    Instances are painted in order of decreasing area so that thin structures belonging to
    small animals are not overwritten by the bodies of large ones - the same appendage
    pixels the mask loss was reweighted to protect.

    Args:
        polygons: Instance polygons in crop-local pixel coordinates.
        shape: (height, width) of the crop.
        border_is_edge: If True, suppress outline along the crop border, where an
            instance's apparent edge is an artefact of cropping rather than its outline.

    Returns:
        Float array of shape (2, H, W): channel 0 foreground, channel 1 outline.
    """
    h, w = shape
    fg = np.zeros((h, w), np.uint8)
    ol = np.zeros((h, w), np.uint8)
    order = sorted(polygons, key=lambda p: -cv2.contourArea(p.astype(np.float32)))
    for c in order:
        ci = np.round(c).astype(np.int32)
        cv2.fillPoly(fg, [ci], 1)
        cv2.polylines(ol, [ci], isClosed=True, color=1, thickness=OUTLINE_PX)
    if border_is_edge:
        b = OUTLINE_PX
        ol[:b, :] = 0
        ol[-b:, :] = 0
        ol[:, :b] = 0
        ol[:, -b:] = 0
    # The outline belongs to the animal, so keep it inside the foreground.
    fg = np.maximum(fg, ol * (fg > 0))
    return np.stack([fg, ol]).astype(np.float32)


class TileSegDataset(Dataset):
    """Random 1024px crops from the flat-bug YOLO layout, with (foreground, outline) targets."""

    def __init__(self, root: str, split: str = "train", tile: int = 1024, length: int = 4000, seed: int = 0):
        """Build the dataset index.

        Args:
            root: Directory containing ``images/<split>`` and ``labels/<split>``.
            split: ``train`` or ``val``.
            tile: Crop size in pixels.
            length: Number of crops per epoch (crops are sampled on the fly).
            seed: Base seed for crop sampling.
        """
        self.images = sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))
        if not self.images:
            raise FileNotFoundError(f"no images under {os.path.join(root, 'images', split)}")
        self.root, self.split, self.tile, self.length, self.seed = root, split, tile, length, seed
        self.train = split == "train"

    def __len__(self) -> int:
        """Number of crops drawn per epoch."""
        return self.length

    def _label_path(self, img: str) -> str:
        return img.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"

    def __getitem__(self, i: int):
        """Return one (image, target) pair as float tensors."""
        rng = random.Random((self.seed << 20) ^ i ^ (os.getpid() << 8))
        path = self.images[rng.randrange(len(self.images))]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.tile, self.tile, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        polys = _read_polygons(self._label_path(path), W, H)
        t = self.tile
        # Many sub-datasets ship images smaller than a tile (wehrli2025 is 572px). Zero
        # padding would spend most of the tile on black and teach the model that the pad
        # colour is background, so upscale instead - which also acts as scale augmentation.
        if min(H, W) < t:
            f = t / min(H, W)
            img = cv2.resize(img, (max(t, int(round(W * f))), max(t, int(round(H * f)))), interpolation=cv2.INTER_LINEAR)
            polys = [p * f for p in polys]
            H, W = img.shape[:2]
        # Bias crops toward annotated regions, otherwise most tiles of a huge scan are empty.
        if polys and rng.random() < 0.92:
            c = polys[rng.randrange(len(polys))].mean(0)
            x0 = int(c[0] - t / 2 + rng.randint(-t // 4, t // 4))
            y0 = int(c[1] - t / 2 + rng.randint(-t // 4, t // 4))
        else:
            x0 = rng.randint(0, max(0, W - t))
            y0 = rng.randint(0, max(0, H - t))
        x0 = max(0, min(x0, max(0, W - t)))
        y0 = max(0, min(y0, max(0, H - t)))
        crop = img[y0:y0 + t, x0:x0 + t]
        ch, cw = crop.shape[:2]
        if ch < t or cw < t:  # pad small images rather than skip them
            pad = np.zeros((t, t, 3), np.uint8)
            pad[:ch, :cw] = crop
            crop = pad
        local = []
        for p in polys:
            q = p - np.array([x0, y0], np.float32)
            if q[:, 0].max() < 0 or q[:, 1].max() < 0 or q[:, 0].min() > t or q[:, 1].min() > t:
                continue
            local.append(q)  # kept even if only partly inside: those pixels are still arthropod
        target = rasterise(local, (t, t), border_is_edge=(cw >= t and ch >= t))
        if self.train:
            if rng.random() < 0.5:
                crop = crop[:, ::-1]
                target = target[:, :, ::-1]
            if rng.random() < 0.5:
                crop = crop[::-1]
                target = target[:, ::-1]
            k = rng.randrange(4)
            if k:
                crop = np.rot90(crop, k)
                target = np.rot90(target, k, axes=(1, 2))
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float().div_(255)
        y = torch.from_numpy(np.ascontiguousarray(target))
        return x, y
