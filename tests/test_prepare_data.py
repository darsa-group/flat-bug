"""Tests for assembling one YOLO dataset out of the many CVAT sub-datasets of a project.

`fb_clone_data` leaves one directory per completed CVAT task; `fb_prepare_data`
folds all of them into a single train/val split plus a merged COCO ground truth.
The risk is silent loss: a sub-dataset that lands entirely in one split must
still contribute its annotations to that split's ground truth.
"""

import hashlib
import json
import os
import sys
import tempfile

import cv2
import numpy as np
import pytest

from flat_bug.cli import fb_prepare_data

VALIDATION_PROPORTION = 0.5


def _split_of(image_bytes: bytes) -> str:
    """Replicate the md5-based pseudorandom split so the tests are deterministic."""
    p = int(hashlib.md5(image_bytes).hexdigest()[0:4], 16) / int("ffff", 16)
    return "val" if p < VALIDATION_PROPORTION else "train"


def _render(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 60, (128, 128, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (60, 70), (255, 255, 255), -1)
    cv2.rectangle(image, (80, 75), (110, 110), (255, 255, 255), -1)
    return cv2.imencode(".jpg", image)[1].tobytes()


def _make_task(root: str, name: str, blobs: list[bytes]) -> int:
    """Write one CVAT-task-shaped COCO sub-dataset. Returns its annotation count."""
    task_dir = os.path.join(root, name)
    os.makedirs(task_dir, exist_ok=True)
    boxes = [(20, 20, 60, 70), (80, 75, 110, 110)]
    images, annotations = [], []
    for i, blob in enumerate(blobs):
        file_name = f"{name}_{i:03d}.jpg"
        with open(os.path.join(task_dir, file_name), "wb") as f:
            f.write(blob)
        images.append({"id": i + 1, "file_name": file_name, "width": 128, "height": 128})
        for x1, y1, x2, y2 in boxes:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": i + 1,
                "category_id": 1,
                "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
                "area": (x2 - x1) * (y2 - y1),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
            })
    with open(os.path.join(task_dir, "instances_default.json"), "w") as f:
        json.dump({
            "licenses": [], "info": {}, "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "insect", "supercategory": ""}],
        }, f)
    return len(annotations)


def _run_prepare(source: str, target: str) -> None:
    """Drive the real CLI, passing ``-p`` explicitly - argparse used to hand it over as a string."""
    argv = sys.argv
    sys.argv = ["fb_prepare_data", "-i", source, "-o", target, "-p", str(VALIDATION_PROPORTION)]
    try:
        fb_prepare_data.main()
    finally:
        sys.argv = argv


def _merged(target: str, split: str) -> dict:
    path = os.path.join(target, fb_prepare_data.DATASET_NAME, "labels", split, "instances_default.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def project():
    """Two sub-datasets: one whose images all land in train, one that spans both splits."""
    pool = [_render(seed) for seed in range(40)]
    train_only = [b for b in pool if _split_of(b) == "train"]
    mixed = [b for b in pool if _split_of(b) == "val"][:4] + train_only[6:10]
    assert len(train_only[:6]) == 6 and len(mixed) == 8, "not enough fixtures generated"
    with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
        n = _make_task(source, "task_all_train", train_only[:6])
        n += _make_task(source, "task_mixed", mixed)
        _run_prepare(source, target)
        yield target, n


def test_every_sub_dataset_reaches_the_training_ground_truth(project):
    """A sub-dataset with no validation images must still contribute its train annotations."""
    target, _ = project
    sources = {i["file_name"].split("_task_")[0] for i in _merged(target, "train")["images"]}
    assert sources == {"task_all_train", "task_mixed"}


def test_no_images_or_annotations_are_lost_in_the_merge(project):  # noqa: D103
    target, total_annotations = project
    train, val = _merged(target, "train"), _merged(target, "val")
    assert len(train["images"]) + len(val["images"]) == 14
    assert len(train["annotations"]) + len(val["annotations"]) == total_annotations


def test_ids_are_remapped_without_collisions_across_sub_datasets(project):  # noqa: D103
    target, _ = project
    for split in ("train", "val"):
        coco = _merged(target, split)
        image_ids = [i["id"] for i in coco["images"]]
        annotation_ids = [a["id"] for a in coco["annotations"]]
        assert len(set(image_ids)) == len(image_ids)
        assert len(set(annotation_ids)) == len(annotation_ids)
        assert all(a["image_id"] in set(image_ids) for a in coco["annotations"])


def test_images_and_labels_line_up_on_disk(project):  # noqa: D103
    target, _ = project
    root = os.path.join(target, fb_prepare_data.DATASET_NAME)
    for split in ("train", "val"):
        images = {os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "images", split))}
        labels = {
            os.path.splitext(f)[0]
            for f in os.listdir(os.path.join(root, "labels", split))
            if f.endswith(".txt")
        }
        assert images == labels, f"image/label mismatch in {split}"


def test_an_empty_split_is_still_readable_json():
    """Merging nothing must produce an empty COCO document, not the literal `null`."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "instances_default.json")
        fb_prepare_data.merge_cocos([], out)
        with open(out) as f:
            coco = json.load(f)
        assert coco is not None
        assert coco["images"] == [] and coco["annotations"] == []
        assert coco["categories"][0]["name"] == "insect"
