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

SYNTHETIC SCENES: optionally, a fraction of crops are composed by flat-bug's
``SceneComposer`` instead of read from disk. The motivation is specific - inter-instance
seams are 0.0111% of real pixels and appear in only 82 of 250 validation crops, and seams
are the only thing that decides whether a watershed can split touching animals. Composed
scenes raise seam density about ninefold and their ground truth is exact at precisely the
place real annotation is least reliable. Off by default (``synth_prob=0``).

AUGMENTATIONS (all label-preserving; see `augment()` for the rationale of each):
    - dihedral: horizontal/vertical flip and 90-degree rotation  (p=1, lossless)
    - HSV jitter: h 0.015, s 0.7, v 0.4                          (flat-bug's own values)
    - channel inversion                                          (p=0.25, as flat-bug)
OPTIONAL, all off by default (see `augment()`):
    - rotation at an arbitrary angle. Safe here because the target is re-rasterised from
      ROTATED POLYGONS rather than resampled, so it stays exact and only the image is
      interpolated. flat-bug uses degrees=180 for the same reason.
    - Gaussian blur, sigma in [0, 2]. Calibrated: high-frequency detail spans 10x across
      the sub-datasets (0.71 for Massid45 and ubc-scanned-sticky-cards up to 6.91 for
      broto2025), and sigma 2 takes the sharpest down past the blurriest, so the range
      covers the real spread without leaving it.
    - Gaussian noise, scaled to each crop's own contrast, since contrast spans 12x
      (std 7.5 to 93.5) and an absolute noise level would be trivial on one dataset and
      overwhelming on another.

Still deliberately absent: scaling (breaks the native-resolution contract with inference)
and elastic deformation (most likely to produce anatomically implausible animals, and the
hardest to calibrate).
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
DIST_FLOOR = 0.05  # smallest normalising divisor, so a 1px-thick leg cannot blow up the map
HSV_GAINS = (0.015, 0.7, 0.4)  # matches flat-bug's hsv_h / hsv_s / hsv_v
P_INVERT = 0.25  # matches flat-bug's RandomColorInv
P_BLUR = 0.0  # probability of Gaussian blur; set by the trainer
BLUR_SIGMA = (0.4, 2.0)  # spans the measured 10x range of high-frequency detail
P_NOISE = 0.0  # probability of Gaussian noise; set by the trainer
NOISE_FRAC = (0.01, 0.06)  # std as a fraction of the crop's own contrast
P_ROTATE = 0.0  # probability of arbitrary-angle rotation; set by the trainer
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


def rotate_crop_and_polygons(img: np.ndarray, polygons: list[np.ndarray], angle: float
                             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rotate an image about its centre and transform the polygons to match.

    The target is later re-rasterised from the rotated polygons, so it is exact - only the
    image is interpolated. This is what makes arbitrary-angle rotation safe for thin
    structures that would not survive resampling of a rasterised mask.

    Args:
        img: HxWx3 uint8 crop.
        polygons: Instance polygons in crop-local pixel coordinates.
        angle: Rotation in degrees.

    Returns:
        The rotated image and the transformed polygons.
    """
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    rot = []
    for c in polygons:
        q = np.concatenate([c, np.ones((len(c), 1), np.float32)], 1) @ mat.T
        rot.append(q.astype(np.float32))
    return out, rot


def distance_map(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Per-instance normalised distance transform: 0 at the boundary, 1 at the centre.

    A better watershed marker generator than eroding the foreground by a uniform outline.
    Measured against 951 known polygons, thresholding this map recovered 0.94x the true
    instance count (mean error 2.73 per image) where erosion gave 1.28x (error 8.35): a
    uniform 3px erosion deletes a 2px leg while barely denting a 30px body, so legs become
    phantom instances. The distance transform is scale-relative and does not.

    Args:
        polygons: Instance polygons in crop-local pixel coordinates.
        shape: (height, width) of the crop.

    Returns:
        Float array (H, W) in [0, 1].
    """
    h, w = shape
    out = np.zeros((h, w), np.float32)
    for c in sorted(polygons, key=lambda q: -cv2.contourArea(q.astype(np.float32))):
        one = np.zeros((h, w), np.uint8)
        cv2.fillPoly(one, [np.round(c).astype(np.int32)], 1)
        if one.sum() == 0:
            continue
        d = cv2.distanceTransform(one, cv2.DIST_L2, 5)
        mx = float(d.max())
        if mx <= 0:
            continue
        out = np.maximum(out, d / max(mx, DIST_FLOOR))
    return np.clip(out, 0, 1)


def instance_weight_map(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel weight of 1/sqrt(area), so small animals are not drowned by large ones.

    The loss pools Dice over the whole batch and averages BCE over pixels, so a 12px insect
    contributes about 0.003% of the signal and missing it entirely is nearly free. This is
    the same correction ultralytics applies in `single_mask_loss` via its `/ area` term.
    1/sqrt(area) rather than 1/area: the model already over-predicts small blobs (69% of its
    sub-16px components are spurious), so equalising fully would likely buy recall with
    hallucinations.

    Args:
        polygons: Instance polygons in crop-local pixel coordinates.
        shape: (height, width) of the crop.

    Returns:
        Float array (H, W), mean 1 over foreground, 1.0 on background.
    """
    h, w = shape
    out = np.ones((h, w), np.float32)
    acc = np.zeros((h, w), np.float32)
    for c in sorted(polygons, key=lambda q: -cv2.contourArea(q.astype(np.float32))):
        one = np.zeros((h, w), np.uint8)
        cv2.fillPoly(one, [np.round(c).astype(np.int32)], 1)
        a = float(one.sum())
        if a <= 0:
            continue
        acc = np.where(one > 0, 1.0 / np.sqrt(max(a, 1.0)), acc)
    fg = acc > 0
    if fg.any():
        acc[fg] /= acc[fg].mean()  # mean 1 over foreground, so overall loss scale is preserved
        out[fg] = acc[fg]
    return out


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
    if P_BLUR and rng.random() < P_BLUR:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*BLUR_SIGMA))
    if P_NOISE and rng.random() < P_NOISE:
        # relative to this crop's contrast: an absolute level would be trivial on a
        # high-contrast dataset and overwhelming on a low-contrast one.
        sd = max(1.0, float(img.std())) * rng.uniform(*NOISE_FRAC)
        noise = np.random.default_rng(rng.randrange(2 ** 32)).normal(0, sd, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, np.ascontiguousarray(target)


class TileSegDataset(Dataset):
    """Native-resolution crops from the flat-bug YOLO layout, with (foreground, outline) targets."""

    def __init__(self, root: str, split: str = "train", tile: int = 1024,
                 length: int | None = None, seed: int = 0, augment_data: bool | None = None,
                 seam_channel: bool = False, dist_channel: bool = False,
                 inst_weight: bool = False, synth_bank: str | None = None,
                 synth_cache: str | None = None, synth_prob: float = 0.0,
                 synth_touch_prob: float = 0.92, synth_coverage: float = 0.30,
                 synth_max_instances: int = 90):
        """Build the dataset index.

        Args:
            root: Directory containing ``images/<split>`` and ``labels/<split>``.
            split: ``train`` or ``val``.
            tile: Crop size in pixels, taken at native resolution.
            length: Approximate epoch length. None uses one crop per tile of every image.
            seed: Base seed for crop sampling.
            augment_data: Force augmentation on or off. Defaults to on for ``train``.
            seam_channel: If True the target gains a channel marking inter-instance seams,
                for loss weighting. Off by default, so existing runs are unchanged.
            dist_channel: If True a per-instance normalised distance map is added as a third
                PREDICTED channel, for watershed markers.
            inst_weight: If True a 1/sqrt(area) per-pixel weight channel is added.
            synth_bank: Crop-bank directory for synthetic scenes, or None to disable.
            synth_cache: Background cache directory for synthetic scenes.
            synth_prob: Probability that a crop is a composed scene rather than a real one.
            synth_touch_prob: Probability a pasted instance is placed touching another.
            synth_coverage: Target fraction of the tile covered by instances.
            synth_max_instances: Cap on instances per composed scene.
        """
        self.images = sorted(glob.glob(os.path.join(root, "images", split, "*.jpg")))
        if not self.images:
            raise FileNotFoundError(f"no images under {os.path.join(root, 'images', split)}")
        self.root, self.split, self.tile, self.seed = root, split, tile, seed
        self.augment_data = (split == "train") if augment_data is None else augment_data
        self.seam_channel = seam_channel
        self.dist_channel = dist_channel
        self.inst_weight = inst_weight
        # channel layout, so the trainer cannot mis-slice it
        self.n_pred = 3 if dist_channel else 2
        i = self.n_pred
        self.idx_seam = (i := i + 1) - 1 if seam_channel else None
        self.idx_instw = (i := i + 1) - 1 if inst_weight else None
        self.n_channels = i
        self.synth_prob = float(synth_prob) if (synth_bank and synth_cache) else 0.0
        self._synth_args = dict(bank_dir=synth_bank, cache_dir=synth_cache, tile=tile,
                                coverage=synth_coverage, touch_prob=synth_touch_prob,
                                overlap=0.12, max_overlap=0.40, min_visible=0.45,
                                max_instances=synth_max_instances, amodal_labels=False)
        self._composer = None  # built lazily, so dataloader workers each get their own
        self.index = self._build_index(length)


    def _extra_channels(self, local, t):
        """Build the optional distance / seam / instance-weight channels."""
        chans = []
        if self.dist_channel:
            chans.append(distance_map(local, (t, t))[None])
        if self.seam_channel:
            from seam_weight import seam_from_polygons
            chans.append(seam_from_polygons(local, (t, t), OUTLINE_PX)[None])
        if self.inst_weight:
            chans.append(instance_weight_map(local, (t, t))[None])
        return chans

    def _scene(self, rng: random.Random):
        """Compose one synthetic scene.

        Returns:
            (RGB uint8 image, list of polygons in pixel coordinates).
        """
        if self._composer is None:
            import sys as _sys
            if "/home/quentin/repos/flat-bug-git/src" not in _sys.path:
                _sys.path.insert(0, "/home/quentin/repos/flat-bug-git/src")
            from flat_bug.synthetic import SceneComposer
            self._composer = SceneComposer(**self._synth_args)
        img, polys, _ = self._composer.compose(rng)
        t = self.tile
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [q.reshape(-1, 2) * t for q in polys]

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
        if self.synth_prob > 0 and rng.random() < self.synth_prob:
            crop, local = self._scene(rng)
            t = self.tile
            target = rasterise(local, (t, t), suppress_border=True)
            ex = self._extra_channels(local, t)
            if ex:
                target = np.concatenate([target[:2], *ex], 0) if self.dist_channel else \
                    np.concatenate([target, *ex], 0)
            valid = np.ones((t, t), np.float32)
            if self.augment_data:
                nch = target.shape[0]
                stacked = np.concatenate([target, valid[None]], 0)
                crop, stacked = augment(crop, stacked, rng)
                target, valid = stacked[:nch], stacked[nch]
            x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float().div_(255)
            return (x, torch.from_numpy(np.ascontiguousarray(target)),
                    torch.from_numpy(np.ascontiguousarray(valid))[None])
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
        if self.augment_data and P_ROTATE and rng.random() < P_ROTATE:
            crop, local = rotate_crop_and_polygons(crop, local, rng.uniform(0, 360))
        target = rasterise(local, (t, t), suppress_border=whole)
        ex = self._extra_channels(local, t)
        if ex:
            target = np.concatenate([target[:2], *ex], 0) if self.dist_channel else \
                np.concatenate([target, *ex], 0)
        if not whole:  # padded region carries no annotation, so it must not be scored as background
            valid = np.zeros((t, t), np.float32)
            valid[:ch, :cw] = 1.0
        else:
            valid = np.ones((t, t), np.float32)
        if self.augment_data:
            nch = target.shape[0]  # 2, or 3 when the seam channel is enabled
            stacked = np.concatenate([target, valid[None]], 0)
            crop, stacked = augment(crop, stacked, rng)
            target, valid = stacked[:nch], stacked[nch]
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
