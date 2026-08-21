"""Tests for the Mask2Former prototype (dataset adapter + tiled inference).

The model itself is stubbed out, so these run without ``transformers`` (or a
download) and cover the parts we actually wrote: the YOLO-polygon adapter, the
per-query decoding, the tiling arithmetic and the conversion to
`flat_bug.predictor.TensorPredictions`.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch
import yaml
from ultralytics.utils.instance import Instances

from flat_bug.augmentations import remove_instances
from flat_bug.coco_utils import fb_to_coco
from flat_bug.mask2former.data import (
    BackgroundMixDataset,
    FlatBugM2FDataset,
    build_augmented_dataset,
    build_background_dataset,
    collate,
    exclude_pattern,
)
from flat_bug.mask2former.predict import M2FPredictor

N_QUERIES = 4
TILE_SIZE = 64
MASK_STRIDE = 4
EDGE_CASE_MARGIN = 4


class _StubOutput:  # noqa: D101
    def __init__(self, class_queries_logits, masks_queries_logits):  # noqa: D107
        self.class_queries_logits = class_queries_logits
        self.masks_queries_logits = masks_queries_logits


class _StubModel(torch.nn.Module):
    """Emits one confident query with a square mask, and otherwise "no object".

    Mimics the output signature of ``Mask2FormerForUniversalSegmentation``:
    ``(batch, n_queries, n_classes + 1)`` class logits and
    ``(batch, n_queries, tile / 4, tile / 4)`` mask logits.
    """

    def __init__(self, square: tuple[int, int, int, int], tile_size: int = TILE_SIZE):  # noqa: D107
        super().__init__()
        self.square = square  # (x1, y1, x2, y2), in tile pixel coordinates
        self.tile_size = tile_size
        # A parameter so that `.to(device=..., dtype=...)` has something to act on.
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(1)))

    def forward(self, pixel_values: torch.Tensor) -> _StubOutput:  # noqa: D102
        batch = pixel_values.shape[0]
        device = pixel_values.device
        # Class logits: query 0 is a confident "insect", the rest are "no object".
        class_logits = torch.zeros((batch, N_QUERIES, 2), device=device)
        class_logits[:, 0, 0] = 10.0
        class_logits[:, 1:, 1] = 10.0
        # Mask logits: a negative field with a positive square for query 0.
        low = self.tile_size // MASK_STRIDE
        mask_logits = torch.full((batch, N_QUERIES, low, low), -10.0, device=device)
        x1, y1, x2, y2 = (c // MASK_STRIDE for c in self.square)
        mask_logits[:, 0, y1:y2, x1:x2] = 10.0
        return _StubOutput(class_logits, mask_logits)


def _predictor(square=(16, 16, 48, 48), **kwargs) -> M2FPredictor:  # noqa: D103
    defaults = dict(
        tile_size=TILE_SIZE,
        minimum_tile_overlap=TILE_SIZE // 2,
        score_threshold=0.5,
        min_max_obj_size=(4, 10**8),
        edge_case_margin=EDGE_CASE_MARGIN,
    )
    defaults.update(kwargs)
    return M2FPredictor(_StubModel(square), **defaults)


def _write_yolo_sample(directory: str, split: str, size: int = 128, n: int = 1) -> None:
    """Write `n` images + YOLO-polygon labels in the layout `fb_prepare_data` produces."""
    image_dir = os.path.join(directory, "images", split)
    label_dir = os.path.join(directory, "labels", split)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    for i in range(n):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        x0, y0 = size // 4 + i, size // 4
        x1, y1 = size * 3 // 4 + i, size * 3 // 4
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), -1)
        stem = "sample" if n == 1 else f"sample_{i:03d}"
        cv2.imwrite(os.path.join(image_dir, f"{stem}.jpg"), image)
        # A square polygon covering the same region, normalized to [0, 1].
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        coords = " ".join(f"{x / size} {y / size}" for x, y in corners)
        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as f:
            f.write(f"0 {coords}\n")


def _write_prepared_dataset(root: str, size: int = 128, n: int = 1) -> None:
    """Reproduce the tree `fb_prepare_data` writes from the CVAT-cloned COCO sub-datasets.

    The images sit one level below the manifest, under the directory named by the
    manifest's (relative) ``path`` key.
    """
    manifest = {"path": "insects", "train": "images/train", "val": "images/val", "nc": 1, "names": ["insect"]}
    with open(os.path.join(root, "data.yaml"), "w") as f:
        yaml.safe_dump(manifest, f)
    for split in ("train", "val"):
        _write_yolo_sample(os.path.join(root, manifest["path"]), split, size=size, n=n)


def test_dataset_reads_fb_prepare_data_layout():
    """`fb_train_m2f -d <prepared>` must find the CVAT-derived data, not just a flat directory."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp)
        for split in ("train", "val"):
            # Pointed at the manifest ...
            dataset = FlatBugM2FDataset(tmp, split=split)
            assert len(dataset) == 1
            # ... and the labels alongside them must resolve too, or training sees no instances.
            assert dataset[0]["mask_labels"].shape[0] == 1
            # ... or straight at the nested directory.
            assert len(FlatBugM2FDataset(os.path.join(tmp, "insects"), split=split)) == 1


def test_dataset_reports_the_paths_it_tried():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="tried"):
            FlatBugM2FDataset(tmp, split="train")


def test_dataset_yields_mask2former_batches():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_yolo_sample(tmp, "train")
        dataset = FlatBugM2FDataset(tmp, split="train", image_size=64)
        assert len(dataset) == 1

        sample = dataset[0]
        assert sample["pixel_values"].shape == (3, 64, 64)
        assert sample["mask_labels"].shape == (1, 64, 64)
        # Float, not bool: the Mask2Former loss point-samples the targets with `grid_sample`.
        assert sample["mask_labels"].dtype == torch.float32
        assert sample["class_labels"].tolist() == [0]
        # The image is square, so the center crop is a no-op and the polygon simply scales.
        assert sample["mask_labels"][0].sum() == pytest.approx(32 * 32, rel=0.1)

        batch = collate([sample, sample])
        assert batch["pixel_values"].shape == (2, 3, 64, 64)
        assert len(batch["mask_labels"]) == 2


def test_dataset_handles_unlabelled_images():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_yolo_sample(tmp, "train")
        os.remove(os.path.join(tmp, "labels", "train", "sample.txt"))
        sample = FlatBugM2FDataset(tmp, split="train", image_size=64)[0]
        assert sample["mask_labels"].shape == (0, 64, 64)
        assert sample["class_labels"].shape == (0,)


def test_decode_tile_scores_and_shapes():  # noqa: D103
    predictor = _predictor()
    outputs = predictor._model(torch.zeros((1, 3, TILE_SIZE, TILE_SIZE)))
    masks, scores, classes = predictor._decode_tile(
        outputs.class_queries_logits[0], outputs.masks_queries_logits[0]
    )
    # Only the single confident query survives.
    assert masks.shape == (1, TILE_SIZE, TILE_SIZE)
    assert classes.tolist() == [0]
    assert 0 < scores[0] <= 1
    # The mask covers the square the stub drew.
    assert masks[0].sum() == pytest.approx((48 - 16) ** 2, rel=0.05)


def test_predict_maps_detections_back_to_image_coordinates():
    """One scale, one tile: the mask must land where the model drew it."""
    square = (16, 16, 48, 48)
    predictor = _predictor(square)
    image = torch.zeros((3, 48, 48), dtype=torch.uint8)
    predictions = predictor(image, path="synthetic.jpg", single_scale=True)

    assert len(predictions) == 1
    x1, y1, x2, y2 = predictions.boxes[0].tolist()
    assert (x1, y1) == pytest.approx(square[:2], abs=2)
    assert (x2, y2) == pytest.approx(square[2:], abs=2)
    assert predictions.contours[0].shape[1] == 2
    assert 0 < predictions.confs[0] <= 1


def test_predict_stitches_detections_from_every_tile():
    """The stub draws its square in every tile, so the stitching arithmetic is fully observable."""
    square = (16, 16, 48, 48)
    predictor = _predictor(square)
    image = torch.zeros((3, 96, 96), dtype=torch.uint8)
    predictions = predictor(image, path="synthetic.jpg", single_scale=True)

    # 96 px tiled by 64-wide tiles overlapping by at least 32 -> offsets [0, 32], so 2 x 2 tiles.
    tile_offsets = [0, 32]
    expected = sorted((square[0] + ox, square[1] + oy) for oy in tile_offsets for ox in tile_offsets)
    assert len(predictions) == len(expected)
    found = sorted((box[0].item(), box[1].item()) for box in predictions.boxes)
    for (fx, fy), (ex, ey) in zip(found, expected):
        assert (fx, fy) == pytest.approx((ex, ey), abs=2)


def test_scale_ladder_matches_the_yolo_pyramid():
    """Coarsest scale fits the whole image in one tile; finest is native resolution."""
    predictor = _predictor()
    ladder = predictor.scales(4000)
    assert ladder[0] == pytest.approx(TILE_SIZE / 4000)
    assert ladder[-1] == 1.0
    assert all(a < b for a, b in zip(ladder, ladder[1:])), "scales must increase"
    # An image smaller than a tile is upsampled to fill one, and needs no ladder.
    assert predictor.scales(TILE_SIZE // 2) == [2.0]
    assert predictor.scales(4000, single_scale=True) == [1.0]


def test_pyramid_merges_overlapping_detections_across_scales():
    """Every scale sees the same object, so stitching plus NMS must not multiply it."""
    predictor = _predictor()
    image = torch.zeros((3, 96, 96), dtype=torch.uint8)
    multi = predictor(image, path="synthetic.jpg")
    single = predictor(image, path="synthetic.jpg", single_scale=True)
    assert len(multi) < len(single), "the pyramid should de-duplicate, not accumulate"
    for box in multi.boxes:
        x1, y1, x2, y2 = box.tolist()
        assert 0 <= x1 < x2 <= 96
        assert 0 <= y1 < y2 <= 96


def test_predictions_serialize_to_coco():  # noqa: D103
    predictor = _predictor()
    predictions = predictor(torch.zeros((3, 48, 48), dtype=torch.uint8), path="synthetic.jpg")
    data = predictions.json_data
    assert data["image_width"] == 48 and data["image_height"] == 48

    coco = fb_to_coco(data, {})
    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == len(predictions)
    for annotation in coco["annotations"]:
        assert len(annotation["segmentation"][0]) >= 6  # at least three (x, y) pairs
        assert 0 < annotation["conf"] <= 1


def test_empty_predictions_are_representable():  # noqa: D103
    # No query passes the score threshold, so the result must still be a usable object.
    predictor = _predictor(score_threshold=0.999999)
    predictions = predictor(torch.zeros((3, 48, 48), dtype=torch.uint8), path="synthetic.jpg")
    assert len(predictions) == 0
    assert predictions.json_data["contours"] == []


# --- Augmented pipeline -------------------------------------------------------
# The point of these is that the augmentations are flatbug's own, not a
# reimplementation: same crops, rescaling, colour jitter, flips and inpainting the
# YOLO models train on. Only the final packing step differs.

TRAIN_PIPELINE = [
    "RandomCrop", "FlatBugRandomPerspective", "CenterCrop", "RandomHSV",
    "RandomColorInv", "RandomFlip", "RandomFlip", "FixInstances", "Format",
]
VAL_PIPELINE = ["RandomCrop", "CenterCrop", "FixInstances", "Format"]


def _augmented(root: str, split: str, image_size: int = 64):
    return build_augmented_dataset(root, split=split, image_size=image_size, batch_size=1)


def test_augmented_training_split_reuses_the_flatbug_pipeline():
    """Every flatbug augmentation must be present; only the packing step is swapped."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256)
        dataset = _augmented(tmp, "train")
        assert [type(t).__name__ for t in dataset.transforms.transforms] == TRAIN_PIPELINE
        packer = dataset.transforms.transforms[-1]
        # Mask2Former needs one binary mask per instance, not YOLO's overlapped index mask.
        assert packer.mask_overlap is False
        assert packer.mask_ratio == 1


def test_augmented_validation_split_uses_the_deterministic_pipeline():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256)
        dataset = _augmented(tmp, "val")
        assert [type(t).__name__ for t in dataset.transforms.transforms] == VAL_PIPELINE


def test_augmented_dataset_yields_mask2former_batches():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256, n=4)
        dataset = _augmented(tmp, "train")
        sample = dataset[0]
        assert sample["pixel_values"].shape == (3, 64, 64)
        assert sample["pixel_values"].dtype == torch.float32
        assert sample["mask_labels"].dtype == torch.float32
        assert sample["mask_labels"].shape[1:] == (64, 64)
        assert sample["class_labels"].dtype == torch.int64
        assert sample["class_labels"].shape[0] == sample["mask_labels"].shape[0]

        # Ragged instance counts must survive collation, including empty crops.
        batch = collate([dataset[i] for i in range(4)])
        assert batch["pixel_values"].shape == (4, 3, 64, 64)
        assert len(batch["mask_labels"]) == len(batch["class_labels"]) == 4


def test_augmentation_actually_varies_between_reads():
    """A deterministic pipeline here would mean the augmentations silently did nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256)
        dataset = _augmented(tmp, "train")
        draws = [dataset[0]["pixel_values"] for _ in range(8)]
        assert any(not torch.equal(draws[0], other) for other in draws[1:])


def test_oversampling_is_inherited_from_the_yolo_dataset():
    """Images are oversampled by area x instance count, so len() exceeds the file count."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256)
        dataset = _augmented(tmp, "train")
        assert len(dataset) >= len(dataset.im_files)


def test_inpainting_erases_instances_the_crop_cut_through():
    """`FixInstances` must not just drop a clipped instance - it must paint it out.

    Otherwise the object stays visible in the image as an unlabelled target, which
    teaches the model to suppress exactly what it should detect.
    """
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    # A bright blob straddling the left edge, so only half of it is inside the image.
    cv2.rectangle(image, (0, 40), (20, 80), (255, 255, 255), -1)
    before = image.copy()

    segment = np.array([[[-20, 40], [20, 40], [20, 80], [-20, 80]]], dtype=np.float32)
    bbox = np.array([[0.0, 60.0, 40.0, 40.0]], dtype=np.float32)  # xywh, center-based
    labels = {
        "img": image,
        "instances": Instances(bbox, segment, None, bbox_format="xywh", normalized=False),
        "cls": np.zeros((1, 1), dtype=np.float32),
    }

    out = remove_instances(labels, area_thr=0.975, max_targets=None, min_size=0)

    assert len(out["instances"]) == 0, "a half-cropped instance should not be kept as a label"
    assert not np.array_equal(before, out["img"]), "the clipped instance was dropped but left in the image"
    # The blob region specifically must have been painted over.
    assert out["img"][40:80, 0:20].mean() < before[40:80, 0:20].mean()


# --- Out-of-domain background negatives ---------------------------------------


def _write_background_images(directory: str, n: int = 5, size: int = 256) -> str:
    """Insect-free images, deliberately textured so they are not trivially blank."""
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        cv2.putText(image, f"540-{i:02d}", (20, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
        cv2.imwrite(os.path.join(directory, f"background_{i:03d}.jpg"), image)
    return directory


def test_background_dataset_yields_no_instances():
    """Background images need no label files - a missing one already means "background"."""
    with tempfile.TemporaryDirectory() as tmp:
        background = build_background_dataset(_write_background_images(tmp), image_size=64)
        assert len(background) > 0
        for i in range(min(4, len(background))):
            sample = background[i]
            assert sample["mask_labels"].shape[0] == 0
            assert sample["class_labels"].shape[0] == 0
            assert sample["pixel_values"].shape == (3, 64, 64)


def test_background_dataset_rejects_labelled_images():
    """Pointing at a labelled directory would silently poison the negatives."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_yolo_sample(tmp, "train", size=256, n=2)
        with pytest.raises(ValueError, match="instance-free"):
            build_background_dataset(os.path.join(tmp, "images", "train"), image_size=64)


def test_background_mix_hits_the_requested_fraction():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256, n=4)
        background_dir = _write_background_images(os.path.join(tmp, "bg"), n=5)
        foreground = _augmented(tmp, "train")
        background = build_background_dataset(background_dir, image_size=64)

        mixed = BackgroundMixDataset(foreground, background, background_fraction=0.2)
        realized = mixed._is_background.mean()
        assert realized == pytest.approx(0.2, abs=0.05)
        # Every foreground sample is still seen exactly once per epoch.
        assert (~mixed._is_background).sum() == len(foreground)


def test_background_mix_is_deterministic_and_covers_both_sources():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256, n=4)
        background_dir = _write_background_images(os.path.join(tmp, "bg"), n=5)
        foreground = _augmented(tmp, "train")
        background = build_background_dataset(background_dir, image_size=64)

        a = BackgroundMixDataset(foreground, background, background_fraction=0.25, seed=7)
        b = BackgroundMixDataset(foreground, background, background_fraction=0.25, seed=7)
        assert np.array_equal(a._is_background, b._is_background)
        assert a._is_background.any() and not a._is_background.all()

        # Samples from both sources are shaped identically, so they batch together.
        batch = collate([a[i] for i in range(len(a))])
        assert batch["pixel_values"].shape[1:] == (3, 64, 64)
        assert any(m.shape[0] == 0 for m in batch["mask_labels"])


def test_background_fraction_must_be_a_proportion():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256, n=4)
        foreground = _augmented(tmp, "train")
        with pytest.raises(ValueError, match="background_fraction"):
            BackgroundMixDataset(foreground, foreground, background_fraction=1.0)


def test_sub_datasets_can_be_held_out_by_prefix():
    """`fb_prepare_data` prefixes files with their CVAT sub-dataset, as flatbug's trainer relies on."""
    import re

    pattern = re.compile(exclude_pattern(["Diopsis", "CollembolAI"]))
    assert pattern.search("ALUS_2020_batch1_img_000.jpg")
    assert not pattern.search("Diopsis_trap_04_img_000.jpg")
    assert not pattern.search("CollembolAI_run2_img_000.jpg")
    # No exclusions means an empty pattern, which matches everything.
    assert re.compile(exclude_pattern(None)).search("anything.jpg")


def test_max_images_truncates_a_split():  # noqa: D103
    with tempfile.TemporaryDirectory() as tmp:
        _write_prepared_dataset(tmp, size=256, n=4)
        assert len(_augmented(tmp, "train").im_files) == 4
        limited = build_augmented_dataset(tmp, split="train", image_size=64, batch_size=1, max_images=2)
        assert len(limited.im_files) == 2
