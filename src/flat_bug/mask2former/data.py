"""YOLO-polygon → Mask2Former dataset adapter (prototype).

Reads the label layout produced by ``fb_prepare_data`` (ultralytics YOLO polygon
segmentation format) and yields per-image samples in the shape expected by
``Mask2FormerForUniversalSegmentation``:

- ``pixel_values``: ``(3, H, W)`` float32, ImageNet-normalized
- ``mask_labels``:  ``(N, H, W)`` bool, one binary mask per instance
- ``class_labels``: ``(N,)`` int64, all zeros (single "insect" class)

Deliberately minimal: center-crop-then-resize, no augmentation, no oversampling.
Enough to prove the training plumbing end-to-end.
"""

from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

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


class FlatBugM2FDataset(Dataset):
    """Minimal YOLO-polygon → Mask2Former dataset."""

    def __init__(self, data_dir: str, split: str = "train", image_size: int = 512):
        assert split in ("train", "val"), split
        self.image_size = image_size

        image_dir = self._resolve_image_dir(data_dir, split)
        self.image_files = sorted(
            f for ext in ("jpg", "jpeg", "png") for f in glob.glob(os.path.join(image_dir, f"*.{ext}"))
        )
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

    @staticmethod
    def _resolve_image_dir(data_dir: str, split: str) -> str:
        data_yaml = os.path.join(data_dir, "data.yaml")
        if os.path.isfile(data_yaml):
            with open(data_yaml) as f:
                cfg = yaml.safe_load(f) or {}
            rel = cfg.get(split, f"images/{split}")
            root = cfg.get("path", os.path.dirname(os.path.abspath(data_yaml)))
            if not os.path.isabs(rel):
                rel = os.path.join(root, rel)
            if os.path.isdir(rel):
                return rel
        # Fallback: canonical layout used by fb_prepare_data
        return os.path.join(data_dir, "images", split)

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
                masks.append(m.astype(bool))

        pixel_values = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD

        if masks:
            mask_labels = torch.from_numpy(np.stack(masks))
            class_labels = torch.zeros(len(masks), dtype=torch.long)
        else:
            mask_labels = torch.zeros((0, self.image_size, self.image_size), dtype=torch.bool)
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
