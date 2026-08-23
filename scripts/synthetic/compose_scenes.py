#!/usr/bin/env python3
"""Stage 2: compose synthetic tiles where instances deliberately touch.

Targets the failure mode flatbug handles worst - separating neighbouring
instances - by generating scenes with a controlled amount of contact, and exact
labels for them.

Placement. The first instance lands anywhere. Each later one either attaches to
an already-placed instance (probability ``--touch-prob``) or is dropped in free
space. Attachment is done by walking the new instance outward from the anchor
along a random bearing and binary-searching the distance until the intersection
is the requested fraction of the smaller instance's area - so "touching" means a
measured ~10% overlap, not a guess.

Labels are *modal*: an instance's polygon is the part of it that remains
visible after anything pasted on top of it, which is what an instance
segmenter should predict. Instances hidden below ``--min-visible`` are dropped
from the labels rather than left as impossible targets.

Backgrounds are cut from real images at spots where the ground truth says no
instance is present, so background statistics (paper texture, scanner
illumination, sticky-trap glue) stay realistic. Alpha edges are feathered,
because a hard paste seam is a shortcut the model would happily learn instead of
the animal's outline.

Usage:
    compose_scenes.py -b <crop-bank> -d <prepared-dataset> -o <out-dir> -n 200
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np


# ----------------------------------------------------------------- backgrounds
def dataset_of(path: str) -> str:
    """`fb_prepare_data` prefixes every file with its CVAT sub-dataset name."""
    return os.path.basename(path).split("_")[0]


def load_background_pool(
    data_dir: str, split: str, per_dataset: int, tile: int, rng: random.Random, clearance: int = 32
) -> dict[str, list[np.ndarray]]:
    """Cut empty patches from real images, keyed by sub-dataset.

    A patch is accepted only if it contains *zero* annotated pixels once the
    ground-truth masks are dilated by `clearance`. The earlier version allowed a
    small mean occupancy, which let genuinely unlabelled animals through - and an
    unlabelled insect in a background is a false negative baked into the labels.
    This cannot catch animals the *source annotation* missed entirely - measured
    at 6/114 patches (12 insects) over the flat-bug training split. Pass
    `--screen-weights` to run flatbug over the candidates and drop those too.
    """
    base = os.path.join(data_dir, "insects")
    base = base if os.path.isdir(base) else data_dir
    images = sorted(glob.glob(os.path.join(base, "images", split, "*.jpg")))
    rng.shuffle(images)
    pool: dict[str, list[np.ndarray]] = {}
    kernel = np.ones((clearance * 2 + 1, clearance * 2 + 1), np.uint8)

    for path in images:
        ds = dataset_of(path)
        if len(pool.get(ds, [])) >= per_dataset:
            continue
        if all(len(v) >= per_dataset for v in pool.values()) and len(pool) > 12:
            pass  # keep going; other datasets may still be unfilled
        image = cv2.imread(path)
        if image is None or min(image.shape[:2]) < tile:
            continue
        h, w = image.shape[:2]
        label = os.path.join(base, "labels", split, os.path.basename(path)[:-4] + ".txt")
        occupied = np.zeros((h, w), np.uint8)
        if os.path.isfile(label):
            for line in open(label):
                vals = line.split()
                if len(vals) < 7:
                    continue
                poly = (np.asarray(vals[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(occupied, [poly], 255)
        if occupied.any():
            occupied = cv2.dilate(occupied, kernel)
        for _ in range(15):
            x = rng.randint(0, w - tile); y = rng.randint(0, h - tile)
            if not occupied[y:y + tile, x:x + tile].any():       # strictly empty
                pool.setdefault(ds, []).append(image[y:y + tile, x:x + tile].copy())
                break
    return pool


# ------------------------------------------------------------------- instances
def augment_crop(rgba: np.ndarray, target_side: int, rng: random.Random) -> np.ndarray:
    """Flip, rotate by an arbitrary angle, and scale to a target size."""
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
    if scale < 1.0:                              # only ever scale down - never invent detail
        rgba = cv2.resize(rgba, (max(int(rgba.shape[1] * scale), 8), max(int(rgba.shape[0] * scale), 8)),
                          interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgba)


def paste(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int, feather: int) -> np.ndarray:
    """Alpha-composite at (x, y); returns the binary mask of what was drawn."""
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    out = np.zeros((ch, cw), np.uint8)
    if x1 <= x0 or y1 <= y0:
        return out
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = sub[:, :, 3].astype(np.float32) / 255.0
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    a3 = alpha[:, :, None]
    canvas[y0:y1, x0:x1] = (sub[:, :, :3] * a3 + canvas[y0:y1, x0:x1] * (1 - a3)).astype(np.uint8)
    out[y0:y1, x0:x1] = (sub[:, :, 3] > 127).astype(np.uint8)
    return out


def overlap_fraction(new_mask: np.ndarray, occupied: np.ndarray) -> float:
    area = int(new_mask.sum())
    if area == 0:
        return 1.0
    return float(np.logical_and(new_mask, occupied).sum()) / area


def mask_at(rgba: np.ndarray, x: int, y: int, shape: tuple[int, int]) -> np.ndarray:
    """Binary mask of `rgba` placed at (x, y) on a canvas of `shape`."""
    h, w = rgba.shape[:2]
    ch, cw = shape
    out = np.zeros((ch, cw), np.uint8)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cw), min(y + h, ch)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = (rgba[y0 - y:y1 - y, x0 - x:x1 - x, 3] > 127).astype(np.uint8)
    return out


def find_touching_position(
    rgba: np.ndarray, anchor_mask: np.ndarray, target: float, shape: tuple[int, int], rng: random.Random
) -> tuple[int, int] | None:
    """Binary-search outward from the anchor until overlap ~= `target`."""
    ys, xs = np.nonzero(anchor_mask)
    if len(xs) == 0:
        return None
    cx, cy = float(xs.mean()), float(ys.mean())
    h, w = rgba.shape[:2]
    bearing = rng.uniform(0, 2 * np.pi)
    dx, dy = np.cos(bearing), np.sin(bearing)
    far = float(np.hypot(*anchor_mask.shape) * 0.5 + max(h, w))

    lo, hi = 0.0, far                       # lo overlaps heavily, hi not at all
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


# ---------------------------------------------------------------------- scenes
def screen_backgrounds(
    pool: dict[str, list[np.ndarray]], weights: str, device: str, keep: int
) -> dict[str, list[np.ndarray]]:
    """Drop candidate patches in which flatbug finds an animal.

    The zero-annotated-pixel rule in `load_background_pool` only trusts the
    source labels. Datasets that annotate one focal specimen per image (ArTaxOr)
    or that under-annotate dense traps (AMI-traps, PeMaToEuroPep) leak real
    animals into "empty" patches, and an unlabelled animal in a background is a
    false negative baked into the training target - precisely the wrong lesson
    for a model we are trying to make *more* willing to split neighbours.
    """
    import tempfile

    from flat_bug.predictor import Predictor  # heavy + needs a GPU; import only when asked

    predictor = Predictor(weights, device=device)
    cleaned: dict[str, list[np.ndarray]] = {}
    dropped = 0
    with tempfile.TemporaryDirectory() as tmp:
        for ds, patches in pool.items():
            for i, patch in enumerate(patches):
                if len(cleaned.get(ds, [])) >= keep:
                    break
                path = os.path.join(tmp, f"{ds}_{i}.jpg")
                cv2.imwrite(path, patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if len(predictor.pyramid_predictions(path, single_scale=True).boxes) == 0:
                    cleaned.setdefault(ds, []).append(patch)
                else:
                    dropped += 1
    print(f"screened backgrounds: dropped {dropped} contaminated patches, "
          f"{sum(len(v) for v in cleaned.values())} kept across {len(cleaned)} sub-datasets")
    return cleaned


def load_bank(bank_dir: str) -> dict[str, list[tuple[str, float]]]:
    """Read the manifest into {sub-dataset: [(crop_path, longest_side_px), ...]}."""
    with open(os.path.join(bank_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    out: dict[str, list[tuple[str, float]]] = {}
    for m in manifest:
        size = float(m.get("max_px") or m["side_px"])
        out.setdefault(m["dataset"], []).append((os.path.join(bank_dir, "crops", m["file"]), size))
    return out


def domain_size_distributions(data_dir: str, split: str, cache: str) -> dict[str, np.ndarray]:
    """Longest-side (px) of every real instance, per sub-dataset.

    This is the distribution a synthetic scene should imitate: pasting an
    ArTaxOr-sized beetle onto a sticky card would be domain-matched in
    provenance and wrong in scale.
    """
    if os.path.isfile(cache):
        with open(cache) as fh:
            return {k: np.asarray(v) for k, v in json.load(fh).items()}
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    base = os.path.join(data_dir, "insects")
    base = base if os.path.isdir(base) else data_dir
    sizes: dict[str, list[float]] = {}
    for label in sorted(glob.glob(os.path.join(base, "labels", split, "*.txt"))):
        image = os.path.join(base, "images", split, os.path.basename(label)[:-4] + ".jpg")
        try:
            width, height = Image.open(image).size
        except Exception:
            continue
        ds = dataset_of(label)
        for line in open(label):
            vals = line.split()
            if len(vals) < 7:
                continue
            coords = np.asarray(vals[1:], float).reshape(-1, 2) * [width, height]
            sizes.setdefault(ds, []).append(
                float(max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1])))
            )
    with open(cache, "w") as fh:
        json.dump(sizes, fh)
    return {k: np.asarray(v) for k, v in sizes.items()}


def crop_weights(entries: list[tuple[str, float]], s_star: float, alpha: float) -> np.ndarray:
    """P(crop) proportional to q^alpha, with q = min(1, size / s_star).

    q saturates at s_star so every sufficiently large crop is equally preferred -
    an unsaturated size**alpha would put nearly all the mass on the single
    biggest crop in the domain. Because q > 0 for every crop, the distribution
    stays well defined when a domain has nothing large: it simply concentrates
    on that domain's own upper tail rather than dropping out. That is the whole
    "prefer large, fall back to small" rule.
    """
    q = np.minimum(1.0, np.array([e[1] for e in entries]) / s_star)
    w = q ** alpha
    total = w.sum()
    return w / total if total > 0 else np.full(len(entries), 1.0 / len(entries))


def sample_target_size(real_sizes: np.ndarray, native: float, rng: random.Random) -> float:
    """Draw a paste size from the domain's own size distribution, truncated at `native`.

    Truncation is what guarantees we never upscale a crop and invent detail.
    """
    usable = real_sizes[real_sizes <= native]
    if usable.size == 0:
        return float(min(native, real_sizes.min() if real_sizes.size else native))
    return float(usable[rng.randrange(usable.size)])


def compose(bank_by_ds: dict[str, list[tuple[str, float]]], backgrounds: dict[str, list[np.ndarray]],
            real_sizes: dict[str, np.ndarray], args, rng: random.Random):
    """Build one scene. Instances and background come from the same sub-dataset.

    Mixing them would paste an EntoScan specimen from a white well-plate onto a
    green pan-trap lid: the illumination, colour temperature and focus all
    disagree, making the composite trivially separable from a real image, and a
    model can learn that seam instead of the animal's outline.
    """
    tile = args.tile_size
    eligible = sorted(set(bank_by_ds) & set(backgrounds)) if not args.mix_backgrounds else sorted(bank_by_ds)
    if not eligible:
        raise RuntimeError("No sub-dataset has both crops and backgrounds; use --mix-backgrounds")
    # P(domain) proportional to its share of real instances, tempered by `tau`.
    # tau=1 reproduces the real (Zipfian) prior, tau=0 is uniform; 0.5 is the
    # usual compromise. Quality does NOT enter here on purpose - down-weighting
    # the poorly resolved domains would drop exactly the ones we set out to cover.
    weights = np.array([max(len(real_sizes.get(d, ())), 1) ** args.tau for d in eligible])
    ds = eligible[int(rng.choices(range(len(eligible)), weights=weights, k=1)[0])]
    bank = bank_by_ds[ds]
    probs = crop_weights(bank, args.s_star, args.alpha)
    sizes_d = real_sizes.get(ds, np.array([args.tile_size / 8.0]))
    if args.mix_backgrounds:
        pool = [p for v in backgrounds.values() for p in v]
    else:
        pool = backgrounds[ds]
    canvas = pool[rng.randrange(len(pool))].copy() if pool else np.full((tile, tile, 3), 235, np.uint8)
    canvas = cv2.resize(canvas, (tile, tile))

    # Crowd by area, not by count. A fixed count means a sticky-card scene of
    # 60px insects covers a twentieth of the tile while an ArTaxOr scene of
    # 220px ones covers half. Solving `n * E[t^2] = coverage * tile^2` keeps the
    # visual density comparable, and at small native scales that means many more
    # instances per tile - which is exactly where touching pairs come from.
    typical = float(np.median(sizes_d)) if sizes_d.size else tile / 8.0
    n_inst = int(round(args.coverage * tile * tile / max(typical * typical, 1.0)))
    n_inst = max(args.instances[0], min(args.instances[1], n_inst))
    n_inst = rng.randint(max(args.instances[0], int(n_inst * 0.75)), max(args.instances[0] + 1, n_inst))
    placed_masks: list[np.ndarray] = []
    occupied = np.zeros((tile, tile), np.uint8)
    records = []

    for _ in range(n_inst):
        idx = int(np.searchsorted(np.cumsum(probs), rng.random()))
        path, native = bank[min(idx, len(bank) - 1)]
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.shape[2] != 4:
            continue
        target_side = int(round(sample_target_size(sizes_d, native, rng)))
        if target_side < 8:
            continue
        rgba = augment_crop(rgba, target_side, rng)
        if min(rgba.shape[:2]) < 8 or max(rgba.shape[:2]) >= tile:
            continue

        pos = None
        if placed_masks and rng.random() < args.touch_prob:
            anchor = placed_masks[rng.randrange(len(placed_masks))]
            pos = find_touching_position(rgba, anchor, args.overlap, (tile, tile), rng)
        if pos is None:
            for _ in range(40):                      # free space
                x = rng.randint(0, tile - rgba.shape[1])
                y = rng.randint(0, tile - rgba.shape[0])
                if overlap_fraction(mask_at(rgba, x, y, (tile, tile)), occupied) < 0.01:
                    pos = (x, y); break
        if pos is None:
            continue

        x, y = pos
        cand = mask_at(rgba, x, y, (tile, tile))
        if cand.sum() == 0 or overlap_fraction(cand, occupied) > args.max_overlap:
            continue
        drawn = paste(canvas, rgba, x, y, args.feather)
        # Anything already placed is now partly hidden -> shrink its visible mask.
        for m in placed_masks:
            m &= ~drawn.astype(bool)
        placed_masks.append(drawn.astype(bool))
        records.append(int(drawn.sum()))
        occupied |= drawn

    polys = []
    for m, full_area in zip(placed_masks, records):
        area = int(m.sum())
        if area == 0 or area / max(full_area, 1) < args.min_visible:
            continue                                  # too occluded to be a fair target
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.0, True).squeeze(1)
        if len(c) >= 3:
            polys.append(c)
    return canvas, polys, ds


def main() -> None:  # noqa: D103
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-b", "--bank", required=True, help="crop bank from build_crop_bank.py")
    p.add_argument("-d", "--data-dir", help="prepared dataset, for real background patches")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("-n", "--n-scenes", type=int, default=100)
    p.add_argument("--split", default="train")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--instances", type=int, nargs=2, default=(8, 200), metavar=("MIN", "MAX"),
                   help="clamp on the count implied by --coverage")
    p.add_argument("--instance-size", type=int, nargs=2, default=(60, 200), metavar=("MIN", "MAX"),
                   help="target sqrt(area) of pasted instances, in px")
    p.add_argument("--touch-prob", type=float, default=0.5,
                   help="probability an instance is attached to an already-placed one")
    p.add_argument("--overlap", type=float, default=0.10, help="target overlap fraction when touching")
    p.add_argument("--max-overlap", type=float, default=0.35, help="reject placements above this")
    p.add_argument("--min-visible", type=float, default=0.4, help="drop labels hidden below this fraction")
    p.add_argument("--feather", type=int, default=1, help="alpha feather radius in px")
    p.add_argument("--backgrounds-per-dataset", type=int, default=6)
    p.add_argument("--alpha", type=float, default=4.0,
                   help="size preference: P(crop) ~ min(1, size/s_star)**alpha. 0=uniform, large=strict")
    p.add_argument("--s-star", type=float, default=256.0,
                   help="size (px) at which a mask is treated as fully trustworthy")
    p.add_argument("--tau", type=float, default=0.5,
                   help="domain prior temper: P(d) ~ (real instances)**tau. 1=proportional, 0=uniform")
    p.add_argument("--coverage", type=float, default=0.15,
                   help="target fraction of the tile covered by instances; drives how crowded scenes are")
    p.add_argument("--screen-weights", default=None,
                   help="flatbug .pt used to reject background patches containing unlabelled animals")
    p.add_argument("--screen-device", default="cuda:0")
    p.add_argument("--mix-backgrounds", action="store_true",
                   help="Allow instances on backgrounds from other sub-datasets (domain mismatch)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    crops = sorted(glob.glob(os.path.join(args.bank, "crops", "*.png")))
    if not crops:
        raise FileNotFoundError(f"No crops in {args.bank}/crops")
    bank_by_ds: dict[str, list[str]] = {}
    for c in crops:
        bank_by_ds.setdefault(os.path.basename(c).split("_")[1], []).append(c)
    bank_by_ds = load_bank(args.bank)
    real_sizes = domain_size_distributions(args.data_dir, args.split,
                                           os.path.join(args.bank, "real_sizes.json")) if args.data_dir else {}
    # Oversample when screening: flatbug rejects some candidates, and we still
    # want `backgrounds_per_dataset` survivors per sub-dataset.
    want = args.backgrounds_per_dataset * (2 if args.screen_weights else 1)
    backgrounds = load_background_pool(args.data_dir, args.split, want,
                                       args.tile_size, rng) if args.data_dir else {}
    if backgrounds and args.screen_weights:
        backgrounds = screen_backgrounds(backgrounds, args.screen_weights, args.screen_device,
                                         args.backgrounds_per_dataset)
    shared = sorted(set(bank_by_ds) & set(backgrounds))
    print(f"bank {len(crops)} crops across {len(bank_by_ds)} sub-datasets | "
          f"backgrounds for {len(backgrounds)} sub-datasets | usable (both): {len(shared)}")
    print("  usable: " + ", ".join(f"{d}({len(bank_by_ds[d])}c/{len(backgrounds[d])}b)" for d in shared))

    img_dir = os.path.join(args.out, "images", args.split)
    lbl_dir = os.path.join(args.out, "labels", args.split)
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)

    counts, used = [], {}
    for i in range(args.n_scenes):
        canvas, polys, ds = compose(bank_by_ds, backgrounds, real_sizes, args, rng)
        name = f"{ds}_synth_{i:05d}"
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as fh:
            for c in polys:
                norm = (c.astype(np.float64) / [args.tile_size, args.tile_size]).clip(0, 1)
                fh.write("0 " + " ".join(f"{v:.6f}" for v in norm.reshape(-1)) + "\n")
        counts.append(len(polys)); used[ds] = used.get(ds, 0) + 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.n_scenes} scenes", flush=True)

    with open(os.path.join(args.out, "compose_config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=1)
    print(f"\nwrote {args.n_scenes} scenes to {args.out}")
    print(f"  instances/scene: mean {np.mean(counts):.1f}, range {min(counts)}-{max(counts)}")
    print("  scenes per sub-dataset: " + ", ".join(f"{k}={v}" for k, v in sorted(used.items())))


if __name__ == "__main__":
    main()
