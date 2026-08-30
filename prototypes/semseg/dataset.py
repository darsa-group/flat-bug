#!/usr/bin/env python3
"""PROTOTYPE - not part of the mainstream flat-bug pipeline.

Tile dataset for two-channel semantic segmentation of arthropods, as an alternative to
the detect-then-segment architecture. Targets are (foreground, outline); eroding the
foreground by the outline leaves one connected core per animal to seed a watershed, so
touching instances separate without a detection head, NMS or a scale pyramid.

Nothing here is imported by `flat_bug`. It reads the same prepared YOLO layout that
`fb_prepare_data` writes, and nothing else is shared.

SCALE POLICY: crops are taken at NATIVE resolution. No rescaling of any kind, so an
instance occupies exactly as many pixels as it does in the source image and the model
sees the true size distribution of each imaging system. Images smaller than one tile are
reflect-padded rather than upscaled, which keeps native scale at the cost of some
synthetic texture at the border.

EDGE POLICY: deliberately the opposite of flat-bug's detection pipeline, where
`FixInstances(area_thr=0.975)` deletes any instance not almost wholly inside the crop
because a truncated box is a wrong regression target. Here those pixels are still
arthropod, so partial instances are KEPT in the foreground channel; only the outline is
suppressed along the crop border, since the image edge is not an animal's outline.

SAMPLING: a coverage list is built in which every image appears at least once, repeated in
proportion to how many tiles it contains (capped, see MAX_REPEATS). Flat-bug weights by
``area * n_instances`` and draws ``2 * n_images`` samples per epoch, which is similar in
spirit; tile count is used here because it is exactly "how many crops are needed to cover
this image once".

An epoch is a *window* over that list, not a full pass: `WindowSampler` hands out a
rotating slice so coverage accumulates over several epochs. This decouples "how often do I
get a metric and a checkpoint" from "how long until every image has been seen", which
otherwise forces 76-minute epochs and a 10-point learning curve.

AUGMENTATIONS (all label-preserving; see `augment()` for the rationale of each):
    - dihedral: horizontal/vertical flip and 90-degree rotation  (p=1, lossless)
    - HSV jitter: h 0.015, s 0.7, v 0.4                          (flat-bug's own values)
    - channel inversion                                          (p=0.25, as flat-bug)
Deliberately ABSENT: any scaling, arbitrary-angle rotation, shear or perspective. Those
resample the image, and at native resolution a 2px leg does not survive interpolation -
the thin structures are the whole point of the outline channel.
"""

from __future__ import annotations

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

OUTLINE_PX = 3  # outline thickness in pixels, at native resolution
HSV_GAINS = (0.015, 0.7, 0.4)  # matches flat-bug's hsv_h / hsv_s / hsv_v
P_INVERT = 0.25  # matches flat-bug's RandomColorInv
P_NEAR_INSTANCE = 0.92  # fraction of crops centred on an annotation
MAX_REPEATS = 48  # cap on per-epoch repeats, so one 143Mpx scan cannot dominate an epoch


def read_polygons(label_path: str, w: int, h: int) -> list[np.ndarray]:
    """Read a YOLO-segmentation label file, returning polygons in pixel coordinates.

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


def rasterise(polygons: list[np.ndarray], shape: tuple[int, int], suppress_border: bool = True) -> np.ndarray:
    """Rasterise instance polygons into a (2, H, W) float target.

    Instances are painted in order of decreasing area so a small animal's legs are not
    overwritten by a large animal's body. 18.2% of annotated instances overlap another,
    so this fires often even though shared pixels are only 0.63% of annotated area.

    Args:
        polygons: Instance polygons in crop-local pixel coordinates.
        shape: (height, width) of the crop.
        suppress_border: If True, erase outline along the crop border, where an
            instance's apparent edge is an artefact of cropping.

    Returns:
        Float array (2, H, W): channel 0 foreground, channel 1 outline.
    """
    h, w = shape
    fg = np.zeros((h, w), np.uint8)
    ol = np.zeros((h, w), np.uint8)
    for c in sorted(polygons, key=lambda p: -cv2.contourArea(p.astype(np.float32))):
        ci = np.round(c).astype(np.int32)
        cv2.fillPoly(fg, [ci], 1)
        cv2.polylines(ol, [ci], isClosed=True, color=1, thickness=OUTLINE_PX)
    if suppress_border:
        b = OUTLINE_PX
        ol[:b, :] = 0
        ol[-b:, :] = 0
        ol[:, :b] = 0
        ol[:, -b:] = 0
    fg = np.maximum(fg, ol * (fg > 0))  # the outline belongs to the animal
    return np.stack([fg, ol]).astype(np.float32)


def augment(img: np.ndarray, target: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Apply the label-preserving augmentations listed in the module docstring.

    Only dihedral transforms touch geometry, so the target needs no interpolation and thin
    structures survive exactly. Photometric jitter matches flat-bug's own hyperparameters
    so the two pipelines see comparable colour variation.

    Args:
        img: HxWx3 uint8 RGB crop.
        target: (2, H, W) float target.
        rng: Seeded RNG.

    Returns:
        The augmented (img, target).
    """
    if rng.random() < 0.5:
        img, target = img[:, ::-1], target[:, :, ::-1]
    if rng.random() < 0.5:
        img, target = img[::-1], target[:, ::-1]
    k = rng.randrange(4)
    if k:
        img, target = np.rot90(img, k), np.rot90(target, k, axes=(1, 2))
    img = np.ascontiguousarray(img)
    hg, sg, vg = HSV_GAINS
    if hg or sg or vg:
        r = np.array([rng.uniform(-1, 1) * hg + 1, rng.uniform(-1, 1) * sg + 1, rng.uniform(-1, 1) * vg + 1])
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
        dt = img.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut = (cv2.LUT(hue, ((x * r[0]) % 180).astype(dt)),
               cv2.LUT(sat, np.clip(x * r[1], 0, 255).astype(dt)),
               cv2.LUT(val, np.clip(x * r[2], 0, 255).astype(dt)))
        img = cv2.cvtColor(cv2.merge(lut), cv2.COLOR_HSV2RGB)
    if rng.random() < P_INVERT:
        img = 255 - img
    return img, np.ascontiguousarray(target)


class TileSegDataset(Dataset):
    """Native-resolution crops from the flat-bug YOLO layout, with (foreground, outline) targets."""

    def __init__(self, root: str, split: str = "train", tile: int = 1024,
                 length: int | None = None, seed: int = 0, augment_data: bool | None = None):
        """Build the dataset index.

        Args:
            root: Directory containing ``images/<split>`` and ``labels/<split>``.
            split: ``train`` or ``val``.
            tile: Crop size in pixels, taken at native resolution.
            length: Approximate epoch length. None uses one crop per tile of every image.
            seed: Base seed for crop sampling.
            augment_data: Force augmentation on or off. Defaults to on for ``train``.
        """
        self.images = sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))
        if not self.images:
            raise FileNotFoundError(f"no images under {os.path.join(root, 'images', split)}")
        self.root, self.split, self.tile, self.seed = root, split, tile, seed
        self.augment_data = (split == "train") if augment_data is None else augment_data
        self.index = self._build_index(length)

    def _build_index(self, length: int | None) -> list[int]:
        """Repeat each image in proportion to its tile count, so every image is seen.

        Args:
            length: Desired epoch length. If None, the natural sum of repeats is used;
                otherwise repeats are scaled to approximately this many crops, but never
                below one per image so coverage is guaranteed.

        Returns:
            List of image indices forming one epoch.
        """
        from PIL import Image  # header-only read, no decode

        Image.MAX_IMAGE_PIXELS = None  # several scans exceed PIL's decompression-bomb limit

        reps = []
        for p in self.images:
            try:
                w, h = Image.open(p).size
            except Exception:  # noqa: BLE001 - an unreadable image still gets one slot
                w = h = self.tile
            reps.append(min(MAX_REPEATS, max(1, (w // self.tile) * (h // self.tile))))
        if length is not None:
            scale = max(0.0, (length - len(reps)) / max(sum(reps) - len(reps), 1))
            reps = [max(1, int(round(1 + (r - 1) * scale))) for r in reps]
        index = [i for i, r in enumerate(reps) for _ in range(r)]
        # Shuffle, or the rotating window walks the SORTED file list and each epoch trains on
        # one alphabetically-adjacent - hence domain-homogeneous - slice. That produces
        # sequential domain shift and catastrophic forgetting, and pins validation to whichever
        # datasets sort first.
        random.Random(20260830).shuffle(index)
        return index

    def __len__(self) -> int:
        """Length of the full coverage list (one pass over every image's tiles)."""
        return len(self.index)

    def _label_path(self, img: str) -> str:
        return img.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"

    def __getitem__(self, i: int):
        """Return one (image, target) pair as float tensors."""
        # `i` is a global step that keeps increasing across epochs, so both the image
        # (cycling through the coverage list) and the crop position vary every pass.
        rng = random.Random((self.seed << 20) ^ (i * 2654435761) & 0xFFFFFFFF)
        path = self.images[self.index[i % len(self.index)]]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.tile, self.tile, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        polys = read_polygons(self._label_path(path), W, H)
        t = self.tile
        if polys and rng.random() < P_NEAR_INSTANCE:  # else most tiles of a large scan are empty
            c = polys[rng.randrange(len(polys))].mean(0)
            x0 = int(c[0] - t / 2 + rng.randint(-t // 4, t // 4))
            y0 = int(c[1] - t / 2 + rng.randint(-t // 4, t // 4))
        else:
            x0, y0 = rng.randint(0, max(0, W - t)), rng.randint(0, max(0, H - t))
        x0 = max(0, min(x0, max(0, W - t)))
        y0 = max(0, min(y0, max(0, H - t)))
        crop = img[y0:y0 + t, x0:x0 + t]
        ch, cw = crop.shape[:2]
        whole = cw >= t and ch >= t
        if not whole:  # reflect-pad, preserving native scale (never upscale)
            crop = cv2.copyMakeBorder(crop, 0, t - ch, 0, t - cw, cv2.BORDER_REFLECT_101)
        local = [p - np.array([x0, y0], np.float32) for p in polys]
        local = [p for p in local
                 if p[:, 0].max() >= 0 and p[:, 1].max() >= 0 and p[:, 0].min() <= t and p[:, 1].min() <= t]
        target = rasterise(local, (t, t), suppress_border=whole)
        if not whole:  # padded region carries no annotation, so it must not be scored as background
            valid = np.zeros((t, t), np.float32)
            valid[:ch, :cw] = 1.0
        else:
            valid = np.ones((t, t), np.float32)
        if self.augment_data:
            stacked = np.concatenate([target, valid[None]], 0)
            crop, stacked = augment(crop, stacked, rng)
            target, valid = stacked[:2], stacked[2]
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float().div_(255)
        y = torch.from_numpy(np.ascontiguousarray(target))
        v = torch.from_numpy(np.ascontiguousarray(valid))[None]
        return x, y, v


class WindowSampler(torch.utils.data.Sampler):
    """Yield a rotating window of the coverage list, so epochs are short but coverage accumulates.

    Epoch ``e`` yields global indices ``[e*epoch_len, (e+1)*epoch_len)``. The dataset maps a
    global index onto the coverage list modulo its length, so every image is still visited
    once per full cycle - it just takes ``len(coverage) / epoch_len`` epochs rather than one.
    """

    def __init__(self, n_total: int, epoch_len: int):
        """Initialise the sampler.

        Args:
            n_total: Length of the dataset's coverage list.
            epoch_len: Number of crops per epoch.
        """
        self.n_total = n_total
        self.epoch_len = epoch_len
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the window to the given epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Crops per epoch."""
        return self.epoch_len

    def __iter__(self):
        """Yield this epoch's global indices."""
        start = self.epoch * self.epoch_len
        return iter(range(start, start + self.epoch_len))
