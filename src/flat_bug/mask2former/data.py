"""YOLO-polygon → Mask2Former dataset adapter (prototype).

Reads the label layout produced by ``fb_prepare_data`` (ultralytics YOLO polygon
segmentation format) and yields per-image samples in the shape expected by
``Mask2FormerForUniversalSegmentation``:

- ``pixel_values``: ``(3, H, W)`` float32, ImageNet-normalized
- ``mask_labels``:  ``(N, H, W)`` float32, one binary mask per instance
  (float, not bool: the loss point-samples them with ``grid_sample``, which has no bool kernel)
- ``class_labels``: ``(N,)`` int64, all zeros (single "insect" class)

Two datasets are provided:

- `FlatBugM2FAugmentedDataset` (and its validation twin) reuse flatbug's own YOLO
  training pipeline verbatim - random crop with rescaling, affine transform,
  HSV/colour inversion, flips, and the inpainting that erases instances the crop
  cut through - swapping only the final packing step, because Mask2Former wants one
  binary mask per instance where YOLO wants a single overlapped index mask. Image
  oversampling by area and instance count comes along with it. This is the default.
- `FlatBugM2FDataset` is the deterministic no-augmentation fallback
  (center-crop-then-resize), useful for debugging and for reproducible evaluation.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Sequence
from copy import deepcopy

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from ultralytics.data.augment import Format
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace

from flat_bug.datasets import FlatBugYOLODataset, FlatBugYOLOValidationDataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _label_path_for_image(image_path: str) -> str:
    """Ultralytics convention: swap ``/images/`` for ``/labels/``, replace ext with ``.txt``."""
    parts = image_path.rsplit(os.sep + "images" + os.sep, 1)
    if len(parts) == 2:
        base = parts[0] + os.sep + "labels" + os.sep + parts[1]
    else:
        base = image_path
    return os.path.splitext(base)[0] + ".txt"


def _read_polygons(label_path: str, w: int, h: int) -> list[np.ndarray]:
    """Read YOLO polygon labels (normalized) and return pixel-space int polygons."""
    if not os.path.isfile(label_path):
        return []
    polys: list[np.ndarray] = []
    with open(label_path) as f:
        for line in f:
            vals = line.strip().split()
            # class + at least 3 (x, y) pairs
            if len(vals) < 7:
                continue
            coords = np.asarray(vals[1:], dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            polys.append(coords.astype(np.int32))
    return polys


def resolve_image_dir(data_dir: str, split: str) -> str:
    """Locate the image directory of a split in a dataset written by ``fb_prepare_data``.

    That layout nests the data one level below the manifest::

        <data_dir>/data.yaml            # path: insects, train: images/train, val: images/val
        <data_dir>/insects/images/train
        <data_dir>/insects/labels/train

    Following ultralytics, a relative ``path`` is resolved against the directory
    holding ``data.yaml`` - not the working directory. Passing the nested
    directory (``<data_dir>/insects``) directly works too.
    """
    candidates = []
    data_yaml = os.path.join(data_dir, "data.yaml")
    if os.path.isfile(data_yaml):
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f) or {}
        rel = cfg.get(split) or os.path.join("images", split)
        root = cfg.get("path") or os.curdir
        if not os.path.isabs(root):
            root = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), root)
        candidates.append(rel if os.path.isabs(rel) else os.path.join(root, rel))
    # Also accept being pointed straight at the directory holding images/ and labels/.
    candidates.append(os.path.join(data_dir, "images", split))

    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.normpath(candidate)
    raise FileNotFoundError(
        f"No '{split}' image directory for {data_dir}; tried " + ", ".join(repr(c) for c in candidates)
    )


class FlatBugM2FDataset(Dataset):
    """Minimal YOLO-polygon → Mask2Former dataset."""

    def __init__(self, data_dir: str, split: str = "train", image_size: int = 512):
        """Index the images of one split.

        Args:
            data_dir: A dataset directory produced by ``fb_prepare_data``.
            split: Either ``"train"`` or ``"val"``.
            image_size: Side length of the square samples yielded by the dataset.

        """
        assert split in ("train", "val"), split
        self.image_size = image_size

        image_dir = resolve_image_dir(data_dir, split)
        self.image_files = sorted(
            f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(image_dir, f"*.{ext}"))
        )
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_files[idx]
        bgr = cv2.imread(image_path)
        if bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Center-square-crop then resize; deterministic, avoids augmentation stack.
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        image = image[y0 : y0 + s, x0 : x0 + s]
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        polys = _read_polygons(_label_path_for_image(image_path), w, h)
        masks: list[np.ndarray] = []
        for poly in polys:
            m = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(m, [poly], color=1)
            m = m[y0 : y0 + s, x0 : x0 + s]
            m = cv2.resize(m, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            if m.any():
                masks.append(m.astype(np.float32))

        pixel_values = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD

        if masks:
            mask_labels = torch.from_numpy(np.stack(masks))
            class_labels = torch.zeros(len(masks), dtype=torch.long)
        else:
            mask_labels = torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
        }


def collate(batch: list[dict]) -> dict:
    """Stack pixel tensors; keep per-image mask/class lists (variable N)."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask_labels": [b["mask_labels"] for b in batch],
        "class_labels": [b["class_labels"] for b in batch],
    }


#: How the augmented pipeline packs a sample: one binary mask per instance at full
#: resolution, where YOLO wants a single overlapped index mask at ``mask_ratio``.
M2F_FORMAT = {
    "bbox_format": "xywh",
    "normalize": True,
    "return_mask": True,
    "return_keypoint": False,
    "batch_idx": False,
    "mask_ratio": 1,
    "mask_overlap": False,
}


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """uint8 ``(3, H, W)`` RGB -> ImageNet-normalized float32."""
    return (image.float().div(255.0) - IMAGENET_MEAN) / IMAGENET_STD


def yolo_labels_to_m2f(labels: dict) -> dict:
    """Repack one augmented ultralytics sample into Mask2Former's input contract."""
    return {
        "pixel_values": normalize_image(labels["img"]),
        # Float, not bool: the loss point-samples the targets with `grid_sample`.
        "mask_labels": labels["masks"].float(),
        "class_labels": labels["cls"].reshape(-1).long(),
    }


def default_hyperparameters(image_size: int, **overrides) -> IterableSimpleNamespace:
    """Ultralytics hyperparameters driving the augmentations, with flatbug's task defaults."""
    hyp = deepcopy(DEFAULT_CFG)
    hyp.task = "segment"
    hyp.imgsz = image_size
    for key, value in overrides.items():
        if getattr(hyp, key, None) is None and not hasattr(hyp, key):
            raise KeyError(f"Unknown hyperparameter {key!r}")
        setattr(hyp, key, value)
    return hyp


class _M2FFormat:
    """Swap the pipeline's final packing step for Mask2Former's, leaving every augmentation intact."""

    def build_transforms(self, hyp: IterableSimpleNamespace):  # noqa: D102
        pipeline = super().build_transforms(hyp)
        pipeline.transforms[-1] = Format(**M2F_FORMAT)
        return pipeline

    def __getitem__(self, index: int) -> dict:
        return yolo_labels_to_m2f(super().__getitem__(index))


class FlatBugM2FAugmentedDataset(_M2FFormat, FlatBugYOLODataset):
    """Flatbug's YOLO *training* augmentations, packed for Mask2Former.

    Inherits the full stack: `RandomCrop` (which rescales), `FlatBugRandomPerspective`
    (rotation and scaling, inpainting the borders it introduces), `CenterCrop`,
    `RandomHSV`, `RandomColorInv`, both `RandomFlip`s, and `FixInstances` - which
    inpaints away instances the crop cut through so they are not left in the image
    as unlabelled objects. Images are oversampled by area x instance count.
    """


class FlatBugM2FAugmentedValidationDataset(_M2FFormat, FlatBugYOLOValidationDataset):
    """Flatbug's YOLO *validation* pipeline (crop, center-crop, `FixInstances`), packed for Mask2Former."""


def exclude_pattern(exclude_datasets: Sequence[str] | None) -> str:
    """Regex excluding whole sub-datasets by filename prefix, as `flat_bug.trainers` does.

    `fb_prepare_data` prefixes every file with its CVAT sub-dataset name, so holding a
    sub-dataset out of training is a match on that prefix.
    """
    return f"^(?!({'|'.join(exclude_datasets)}))" if exclude_datasets else ""


def build_augmented_dataset(
    data_dir: str,
    split: str = "train",
    image_size: int = 512,
    max_instances: int | float | None = None,
    batch_size: int = 1,
    max_images: int | None = None,
    exclude_datasets: Sequence[str] | None = None,
    **hyperparameter_overrides,
) -> FlatBugM2FAugmentedDataset | FlatBugM2FAugmentedValidationDataset:
    """Build the augmented dataset for one split of an ``fb_prepare_data`` output tree.

    Args:
        data_dir: The prepared dataset (the directory holding ``data.yaml``), or the
            directory holding ``images/`` and ``labels/`` directly.
        split: ``"train"`` or ``"val"``. The training split gets the full augmentation
            stack; the validation split gets the deterministic crop pipeline.
        image_size: Side length of the square samples.
        max_instances: Cap on instances per sample, applied by `FixInstances`.
        batch_size: Only used by ultralytics for rectangular batching bookkeeping.
        max_images: Keep only the first N images (alphabetically). ``None`` keeps all.
        exclude_datasets: Sub-dataset name prefixes to hold out, as in `flat_bug.trainers`.
        hyperparameter_overrides: Ultralytics hyperparameters to override, e.g.
            ``fliplr=0.5``, ``hsv_v=0.4``.

    Returns:
        The dataset, yielding ``pixel_values`` / ``mask_labels`` / ``class_labels``.

    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    image_dir = resolve_image_dir(data_dir, split)

    data_yaml = os.path.join(data_dir, "data.yaml")
    if os.path.isfile(data_yaml):
        with open(data_yaml) as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data.setdefault("names", ["insect"])
    data.setdefault("nc", len(data["names"]))
    data.setdefault("channels", 3)

    is_train = split == "train"
    dataset_class = FlatBugM2FAugmentedDataset if is_train else FlatBugM2FAugmentedValidationDataset
    return dataset_class(
        data=data,
        img_path=image_dir,
        imgsz=image_size,
        cache=False,
        augment=is_train,
        hyp=default_hyperparameters(image_size, **hyperparameter_overrides),
        rect=not is_train,
        batch_size=batch_size,
        pad=0.0 if is_train else 0.5,
        single_cls=False,
        max_instances=max_instances if is_train else np.inf,
        task="segment",
        subset_args={"n": max_images, "pattern": exclude_pattern(exclude_datasets)},
    )


def build_background_dataset(
    image_dir: str,
    image_size: int = 512,
    batch_size: int = 1,
    **hyperparameter_overrides,
) -> FlatBugM2FAugmentedDataset:
    """Build an instance-free dataset from a directory of background images.

    The images go through the *same* augmentation pipeline as the training data, so
    the model cannot separate negatives from positives on crop statistics or colour
    alone. No label files are needed - ultralytics treats a missing label file as a
    background image - and it is an error if any of them turns out to be labelled,
    which would silently poison the negatives.

    Args:
        image_dir: A directory of images that contain no insects.
        image_size: Side length of the square samples; match the training size.
        batch_size: Only used by ultralytics for batching bookkeeping.
        hyperparameter_overrides: Ultralytics hyperparameters to override.

    Returns:
        The dataset, yielding samples with zero instances.

    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Background image directory not found: {image_dir}")
    dataset = FlatBugM2FAugmentedDataset(
        data={"names": ["insect"], "nc": 1, "channels": 3},
        img_path=image_dir,
        imgsz=image_size,
        cache=False,
        augment=True,
        hyp=default_hyperparameters(image_size, **hyperparameter_overrides),
        rect=False,
        batch_size=batch_size,
        pad=0.0,
        single_cls=False,
        max_instances=None,
        task="segment",
    )
    labelled = sum(int(len(label["cls"]) > 0) for label in dataset.labels)
    if labelled:
        raise ValueError(
            f"{labelled} of {len(dataset.labels)} images in {image_dir} carry labels; "
            "background images must be instance-free."
        )
    return dataset


class BackgroundMixDataset(Dataset):
    """Interleave a training dataset with insect-free background images at a fixed rate.

    In-domain random crops already supply plenty of empty regions, but they are empty
    *flatbug* regions. Out-of-domain false positives - printed text, debris, packaging
    - need negatives drawn from those domains, which is what this mixes in.

    The plan is fixed at construction, so every epoch sees each foreground sample
    exactly once and the realized background fraction is exact rather than expected.
    """

    def __init__(
        self,
        foreground: Dataset,
        background: Dataset,
        background_fraction: float = 0.2,
        seed: int = 0,
    ):
        """Mix `background` into `foreground`.

        Args:
            foreground: The training dataset.
            background: An instance-free dataset, e.g. from `build_background_dataset`.
            background_fraction: Proportion of the epoch drawn from `background`, in [0, 1).
            seed: Seed for the (fixed) interleaving plan.

        """
        if not 0 <= background_fraction < 1:
            raise ValueError(f"background_fraction must be in [0, 1), got {background_fraction}")
        if background_fraction and len(background) == 0:  # type: ignore[arg-type]
            raise ValueError("background dataset is empty but background_fraction > 0")
        self.foreground = foreground
        self.background = background
        self.background_fraction = background_fraction

        n_foreground = len(foreground)  # type: ignore[arg-type]
        total = round(n_foreground / (1 - background_fraction)) if background_fraction else n_foreground
        n_background = total - n_foreground

        rng = np.random.default_rng(seed)
        is_background = np.zeros(total, dtype=bool)
        if n_background:
            is_background[rng.choice(total, size=n_background, replace=False)] = True
        self._is_background = is_background
        # Positions within each source; backgrounds cycle if there are fewer of them.
        self._foreground_index = np.cumsum(~is_background) - 1
        self._background_index = (np.cumsum(is_background) - 1) % max(len(background), 1)  # type: ignore[arg-type]

    def __len__(self) -> int:
        return len(self._is_background)

    def __getitem__(self, index: int) -> dict:
        if self._is_background[index]:
            return self.background[int(self._background_index[index])]
        return self.foreground[int(self._foreground_index[index])]
