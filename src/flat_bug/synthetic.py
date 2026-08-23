"""Synthetic scenes of touching instances, composed at training time.

The model's weakest point is separating neighbouring animals that touch. Real
data supplies few such pairs and never labels them as a special case, so this
module manufactures them: well-segmented cut-outs are scaled down, rotated and
placed so that a controlled fraction of them come into contact.

Two rules keep the scenes honest.

**Stay in domain.** Instances and background are drawn from the same CVAT
sub-dataset (`fb_prepare_data` prefixes every filename with it). Mixing them
would paste an EntoScan specimen from a white well-plate onto a green pan-trap
lid; the illumination, colour temperature and focus all disagree, and a model
can learn that seam instead of the animal's outline.

**Prefer large crops, but fall back.** Large instances carry trustworthy masks,
yet an absolute size floor excludes whole domains: sticky-pi, wehrli2025 and
Massid45 have 0.0-0.4% of their instances above 256px, and those are exactly the
dense traps where neighbours touch. So size is a *preference*, applied at
sampling time and relative to each domain:

    q_i = min(1, size_i / s_star)        P(crop i | domain d) ~ q_i ** alpha

`q` saturates at `s_star` so every sufficiently large crop is equally preferred
- an unsaturated ``size ** alpha`` would put nearly all the mass on the single
largest crop in the domain. Every ``q_i > 0``, so a domain with nothing large
does not drop out; it concentrates on its own upper tail.

See `scripts/synthetic/build_crop_bank.py` for how the crop bank is harvested.
"""

from __future__ import annotations

import glob
import json
import os
import random

import cv2
import numpy as np

from flat_bug import logger


def dataset_of(path: str) -> str:
    """Return the CVAT sub-dataset a prepared file belongs to.

    `fb_prepare_data` prefixes every merged filename with its sub-dataset name.

    Args:
        path: Path to a prepared image or label file.

    Returns:
        The sub-dataset name.
    """
    return os.path.basename(path).split("_")[0]


def augment_crop(rgba: np.ndarray, target_side: int, rng: random.Random) -> np.ndarray:
    """Flip, rotate by an arbitrary angle and scale a cut-out to a target size.

    Args:
        rgba: The cut-out, with the instance mask in the alpha channel.
        target_side: Desired sqrt(area) of the result, in pixels.
        rng: Random source.

    Returns:
        The transformed cut-out. Never upscaled - scaling up would invent detail.
    """
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    if rng.random() < 0.5:
        rgba = rgba[::-1, :]
    angle = rng.uniform(0, 360)
    h, w = rgba.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    rgba = cv2.warpAffine(rgba, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    side = float(np.sqrt(rgba.shape[0] * rgba.shape[1]))
    scale = target_side / max(side, 1)
    if scale < 1.0:
        rgba = cv2.resize(
            rgba,
            (max(int(rgba.shape[1] * scale), 8), max(int(rgba.shape[0] * scale), 8)),
            interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(rgba)


def mask_at(rgba: np.ndarray, x: int, y: int, shape: tuple[int, int]) -> np.ndarray:
    """Binary mask of `rgba` placed at (x, y) on a canvas of `shape`.

    Args:
        rgba: The cut-out.
        x: Left coordinate.
        y: Top coordinate.
        shape: (height, width) of the canvas.

    Returns:
        A uint8 mask, 1 where the instance covers the canvas.
    """
    h, w = rgba.shape[:2]
    ch, cw = shape
    out = np.zeros((ch, cw), np.uint8)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = (rgba[y0 - y : y1 - y, x0 - x : x1 - x, 3] > 127).astype(np.uint8)
    return out


def overlap_fraction(new_mask: np.ndarray, occupied: np.ndarray) -> float:
    """Fraction of `new_mask` that falls on already-occupied pixels.

    Args:
        new_mask: Candidate placement mask.
        occupied: Union of what is already placed.

    Returns:
        Overlap as a fraction of the candidate's area; 1.0 for an empty candidate.
    """
    area = int(new_mask.sum())
    if area == 0:
        return 1.0
    return float(np.logical_and(new_mask, occupied).sum()) / area


def paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, feather: int) -> np.ndarray:
    """Alpha-composite `rgba` onto `canvas` at (x, y), in place.

    Args:
        canvas: BGR canvas, modified in place.
        rgba: The cut-out.
        x: Left coordinate.
        y: Top coordinate.
        feather: Radius in px over which to soften the alpha edge.

    Returns:
        Binary mask of what was actually drawn.
    """
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    out = np.zeros((ch, cw), np.uint8)
    if x1 <= x0 or y1 <= y0:
        return out
    sub = rgba[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = sub[:, :, 3].astype(np.float32) / 255.0
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    a3 = alpha[:, :, None]
    canvas[y0:y1, x0:x1] = (sub[:, :, :3] * a3 + canvas[y0:y1, x0:x1] * (1 - a3)).astype(np.uint8)
    out[y0:y1, x0:x1] = (sub[:, :, 3] > 127).astype(np.uint8)
    return out


def find_touching_position(
    rgba: np.ndarray, anchor_mask: np.ndarray, target: float, shape: tuple[int, int], rng: random.Random
) -> tuple[int, int] | None:
    """Binary-search outward from an anchor until the overlap is about `target`.

    Args:
        rgba: The cut-out to place.
        anchor_mask: Mask of the instance to touch.
        target: Desired overlap as a fraction of the new instance's area.
        shape: (height, width) of the canvas.
        rng: Random source, used to pick the bearing.

    Returns:
        Top-left placement, or None if no position was found.
    """
    ys, xs = np.nonzero(anchor_mask)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = rgba.shape[:2]
    bearing = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(bearing), np.sin(bearing)
    far = float(np.hypot(*anchor_mask.shape) * 0.5 + max(h, w))

    lo, hi = 0.0, far
    best = None
    for _ in range(18):
        mid = (lo + hi) / 2
        x = int(cx + dx * mid - w / 2)
        y = int(cy + dy * mid - h / 2)
        frac = overlap_fraction(mask_at(rgba, x, y, shape), anchor_mask)
        if frac > target:
            lo = mid
        else:
            hi = mid
            best = (x, y)
        if abs(frac - target) < 0.02:
            return (x, y)
    return best


def crop_weights(sizes: np.ndarray, s_star: float, alpha: float) -> np.ndarray:
    """Sampling probabilities over a domain's crops: P ~ min(1, size/s_star) ** alpha.

    Args:
        sizes: Longest side, in px, of each crop in the domain.
        s_star: Size at which a mask is treated as fully trustworthy.
        alpha: Strength of the size preference. 0 is uniform; large is near-argmax.

    Returns:
        A probability vector over the crops.
    """
    q = np.minimum(1.0, sizes / s_star)
    w = q**alpha
    total = w.sum()
    return w / total if total > 0 else np.full(len(sizes), 1.0 / max(len(sizes), 1))


def sample_target_size(real_sizes: np.ndarray, native: float, rng: random.Random) -> float:
    """Draw a paste size from a domain's own size distribution, truncated at `native`.

    Truncation is what guarantees a crop is never upscaled.

    Args:
        real_sizes: Longest side of every real instance in the domain.
        native: The crop's own longest side, in px.
        rng: Random source.

    Returns:
        The size, in px, to scale the crop to.
    """
    usable = real_sizes[real_sizes <= native]
    if usable.size == 0:
        return float(min(native, real_sizes.min() if real_sizes.size else native))
    return float(usable[rng.randrange(usable.size)])


def _instance_sizes(data_dir: str, split: str) -> dict[str, list[float]]:
    """Longest side (px) of every real instance, per sub-dataset."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    base = os.path.join(data_dir, "insects")
    base = base if os.path.isdir(base) else data_dir
    sizes: dict[str, list[float]] = {}
    for label in sorted(glob.glob(os.path.join(base, "labels", split, "*.txt"))):
        image = os.path.join(base, "images", split, os.path.basename(label)[:-4] + ".jpg")
        try:
            width, height = Image.open(image).size
        except Exception:  # noqa: BLE001 - a corrupt or missing image just contributes nothing
            continue
        ds = dataset_of(label)
        with open(label) as fh:
            for line in fh:
                vals = line.split()
                if len(vals) < 7:
                    continue
                coords = np.asarray(vals[1:], float).reshape(-1, 2) * [width, height]
                sizes.setdefault(ds, []).append(float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))))
    return sizes


def build_cache(
    data_dir: str,
    cache_dir: str,
    split: str = "train",
    tile: int = 1024,
    per_dataset: int = 6,
    clearance: int = 32,
    screen_weights: str | None = None,
    screen_device: str = "cuda:0",
) -> None:
    """Cut empty background patches and record per-domain instance sizes.

    A patch is accepted only if it contains *zero* annotated pixels once the
    ground truth is dilated by `clearance`. That trusts the source labels, which
    is not always enough: running flatbug over 114 candidate patches found
    animals in 6 of them, concentrated in datasets that annotate a single focal
    specimen (ArTaxOr) or under-annotate dense traps (AMI-traps, PeMaToEuroPep).
    Pass `screen_weights` to drop those too - an unlabelled animal in a
    background is a false negative baked into the target.

    Patches are written as one memmappable `.npy`, so dataloader workers share
    the pages instead of each holding a copy of several hundred megabytes.

    Args:
        data_dir: Prepared dataset (the `fb_prepare_data` output).
        cache_dir: Destination directory.
        split: Which split to cut backgrounds from.
        tile: Patch size, in px.
        per_dataset: How many patches to keep per sub-dataset.
        clearance: Dilation applied to the ground truth before testing emptiness.
        screen_weights: Optional flatbug checkpoint used to reject contaminated patches.
        screen_device: Device for the screening pass.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(data_dir, "insects")
    base = base if os.path.isdir(base) else data_dir
    images = sorted(glob.glob(os.path.join(base, "images", split, "*.jpg")))
    random.Random(0).shuffle(images)
    want = per_dataset * (2 if screen_weights else 1)
    kernel = np.ones((clearance * 2 + 1, clearance * 2 + 1), np.uint8)
    rng = random.Random(0)

    patches: list[np.ndarray] = []
    index: dict[str, list[int]] = {}
    for path in images:
        ds = dataset_of(path)
        if len(index.get(ds, [])) >= want:
            continue
        image = cv2.imread(path)
        if image is None or min(image.shape[:2]) < tile:
            continue
        h, w = image.shape[:2]
        label = os.path.join(base, "labels", split, os.path.basename(path)[:-4] + ".txt")
        occupied = np.zeros((h, w), np.uint8)
        if os.path.isfile(label):
            with open(label) as fh:
                for line in fh:
                    vals = line.split()
                    if len(vals) < 7:
                        continue
                    poly = (np.asarray(vals[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                    cv2.fillPoly(occupied, [poly], 255)
        if occupied.any():
            occupied = cv2.dilate(occupied, kernel)
        for _ in range(15):
            x, y = rng.randint(0, w - tile), rng.randint(0, h - tile)
            if not occupied[y : y + tile, x : x + tile].any():
                index.setdefault(ds, []).append(len(patches))
                patches.append(image[y : y + tile, x : x + tile].copy())
                break

    if screen_weights:
        patches, index = _screen(patches, index, screen_weights, screen_device, per_dataset)

    array = np.stack(patches) if patches else np.zeros((0, tile, tile, 3), np.uint8)
    np.save(os.path.join(cache_dir, "backgrounds.npy"), array)
    with open(os.path.join(cache_dir, "backgrounds.json"), "w") as fh:
        json.dump(index, fh)
    with open(os.path.join(cache_dir, "real_sizes.json"), "w") as fh:
        json.dump(_instance_sizes(data_dir, split), fh)
    logger.info(f"synthetic cache: {len(patches)} background patches over {len(index)} sub-datasets -> {cache_dir}")


def _screen(
    patches: list[np.ndarray], index: dict[str, list[int]], weights: str, device: str, keep: int
) -> tuple[list[np.ndarray], dict[str, list[int]]]:
    """Drop candidate patches in which flatbug finds an animal."""
    import tempfile

    from flat_bug.predictor import Predictor

    predictor = Predictor(weights, device=device)
    out_patches: list[np.ndarray] = []
    out_index: dict[str, list[int]] = {}
    dropped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for ds, ids in index.items():
            for i in ids:
                if len(out_index.get(ds, [])) >= keep:
                    break
                path = os.path.join(tmp, f"{ds}_{i}.jpg")
                cv2.imwrite(path, patches[i], [cv2.IMWRITE_JPEG_QUALITY, 95])
                if len(predictor.pyramid_predictions(path, single_scale=True).boxes) == 0:
                    out_index.setdefault(ds, []).append(len(out_patches))
                    out_patches.append(patches[i])
                else:
                    dropped += 1
    logger.info(f"synthetic cache: screening dropped {dropped} contaminated patches")
    return out_patches, out_index


class SceneComposer:
    """Composes one synthetic scene per call, from a crop bank and a background cache.

    Heavy state (the background array) is memmapped, so forked or spawned
    dataloader workers share the same physical pages.
    """

    def __init__(
        self,
        bank_dir: str,
        cache_dir: str,
        tile: int = 1024,
        alpha: float = 4.0,
        s_star: float = 256.0,
        tau: float = 0.5,
        coverage: float = 0.15,
        touch_prob: float = 0.6,
        overlap: float = 0.10,
        max_overlap: float = 0.35,
        min_visible: float = 0.4,
        feather: int = 1,
        max_instances: int = 150,
    ) -> None:
        """Load the bank manifest and memmap the background cache.

        Args:
            bank_dir: Crop bank from `build_crop_bank.py`.
            cache_dir: Background cache from `build_cache`.
            tile: Scene size, in px.
            alpha: Size preference exponent.
            s_star: Size at which a mask is treated as fully trustworthy.
            tau: Domain prior temper: P(d) ~ (real instances) ** tau.
            coverage: Target fraction of the tile covered by instances.
            touch_prob: Probability that a placement is attempted against a neighbour.
            overlap: Target overlap fraction when touching.
            max_overlap: Reject placements above this overlap.
            min_visible: Drop labels hidden below this fraction of their full area.
            feather: Alpha feather radius, in px.
            max_instances: Hard cap, matching the trainer's `fb_max_instances`.
        """
        self.tile = tile
        self.alpha, self.s_star, self.tau = alpha, s_star, tau
        self.coverage, self.touch_prob = coverage, touch_prob
        self.overlap, self.max_overlap = overlap, max_overlap
        self.min_visible, self.feather = min_visible, feather
        self.max_instances = max_instances

        with open(os.path.join(bank_dir, "manifest.json")) as fh:
            manifest = json.load(fh)
        bank: dict[str, list[tuple[str, float]]] = {}
        for m in manifest:
            size = float(m.get("max_px") or m["side_px"])
            bank.setdefault(m["dataset"], []).append((os.path.join(bank_dir, "crops", m["file"]), size))
        self.paths = {d: [e[0] for e in v] for d, v in bank.items()}
        self.probs = {
            d: crop_weights(np.array([e[1] for e in v]), s_star, alpha) for d, v in bank.items()
        }
        self.native = {d: np.array([e[1] for e in v]) for d, v in bank.items()}

        self.backgrounds = np.load(os.path.join(cache_dir, "backgrounds.npy"), mmap_mode="r")
        with open(os.path.join(cache_dir, "backgrounds.json")) as fh:
            self.bg_index = json.load(fh)
        with open(os.path.join(cache_dir, "real_sizes.json")) as fh:
            self.real_sizes = {k: np.asarray(v) for k, v in json.load(fh).items()}

        self.domains = sorted(set(self.paths) & set(self.bg_index))
        if not self.domains:
            raise RuntimeError("No sub-dataset has both crops and background patches")
        # Quality deliberately does not enter this term: down-weighting the poorly
        # resolved domains would drop exactly the ones this dataset exists to cover.
        counts = np.array([max(len(self.real_sizes.get(d, ())), 1) for d in self.domains], dtype=float)
        w = counts**tau
        self.domain_probs = w / w.sum()

    def compose(self, rng: random.Random) -> tuple[np.ndarray, list[np.ndarray], str]:
        """Build one scene.

        Args:
            rng: Random source.

        Returns:
            (BGR image, list of normalised polygons, sub-dataset name).
        """
        tile = self.tile
        ds = self.domains[int(np.searchsorted(np.cumsum(self.domain_probs), rng.random()))]
        ids = self.bg_index[ds]
        canvas = np.array(self.backgrounds[ids[rng.randrange(len(ids))]])
        if canvas.shape[0] != tile or canvas.shape[1] != tile:
            canvas = cv2.resize(canvas, (tile, tile))

        sizes_d = self.real_sizes.get(ds, np.array([tile / 8.0]))
        probs, paths, native = self.probs[ds], self.paths[ds], self.native[ds]

        # Crowd by area, not by count: a fixed count covers a twentieth of a
        # sticky-card tile and half of an ArTaxOr one. Solving
        # n * E[t^2] = coverage * tile^2 keeps the visual density comparable, and
        # at small native scales that means many more instances per tile - which
        # is where touching pairs come from.
        typical = float(np.median(sizes_d)) if sizes_d.size else tile / 8.0
        n_inst = int(round(self.coverage * tile * tile / max(typical * typical, 1.0)))
        n_inst = int(max(4, min(self.max_instances, n_inst)))

        cumulative = np.cumsum(probs)
        placed: list[np.ndarray] = []
        occupied = np.zeros((tile, tile), np.uint8)
        full_areas: list[int] = []

        for _ in range(n_inst):
            idx = min(int(np.searchsorted(cumulative, rng.random())), len(paths) - 1)
            rgba = cv2.imread(paths[idx], cv2.IMREAD_UNCHANGED)
            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                continue
            target = int(round(sample_target_size(sizes_d, float(native[idx]), rng)))
            if target < 8:
                continue
            rgba = augment_crop(rgba, target, rng)
            if min(rgba.shape[:2]) < 8 or max(rgba.shape[:2]) >= tile:
                continue

            pos = None
            if placed and rng.random() < self.touch_prob:
                anchor = placed[rng.randrange(len(placed))]
                pos = find_touching_position(rgba, anchor, self.overlap, (tile, tile), rng)
            if pos is None:
                for _ in range(40):
                    x = rng.randint(0, tile - rgba.shape[1])
                    y = rng.randint(0, tile - rgba.shape[0])
                    if overlap_fraction(mask_at(rgba, x, y, (tile, tile)), occupied) < 0.01:
                        pos = (x, y)
                        break
            if pos is None:
                continue

            x, y = pos
            candidate = mask_at(rgba, x, y, (tile, tile))
            if candidate.sum() == 0 or overlap_fraction(candidate, occupied) > self.max_overlap:
                continue
            drawn = paste(canvas, rgba, x, y, self.feather)
            for m in placed:
                m &= ~drawn.astype(bool)
            placed.append(drawn.astype(bool))
            full_areas.append(int(candidate.sum()))
            occupied |= drawn

        polys: list[np.ndarray] = []
        for mask, full in zip(placed, full_areas):
            if full == 0 or mask.sum() / full < self.min_visible:
                continue
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < 16:
                continue
            polys.append(contour.reshape(-1, 2).astype(np.float32) / tile)
        return canvas, polys, ds
