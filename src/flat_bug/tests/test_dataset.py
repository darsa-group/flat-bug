import os
import tempfile
import glob
import unittest

from copy import deepcopy

import numpy as np

from ultralytics.data import build_dataloader
from ultralytics.data.utils import verify_image_label
from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.plotting import plot_images

from flat_bug.datasets import MyYOLODataset, MyYOLOValidationDataset

from .remote_lfs_fallback import check_file_with_remote_fallback

TEST_DIR = os.path.dirname(__file__)
ASSET_DIR = os.path.join(TEST_DIR, "assets")
ASSET_NAME = "ALUS_Non-miteArachnids_Unknown_2020_11_03_4545"

IMAGE_ASSET = os.path.join(ASSET_DIR, ASSET_NAME + ".jpg")
LABEL_ASSET = os.path.join(ASSET_DIR, ASSET_NAME + ".txt")
check_file_with_remote_fallback(IMAGE_ASSET)
check_file_with_remote_fallback(LABEL_ASSET)

ASSET_DATA = {
    "names": ["insect"],
    "nc": 1,
    "path": ASSET_DIR,
    "train": ASSET_DIR,
    "val": ASSET_DIR,
}

BATCH_SIZE = 1
N_WORKERS = 0
RANK = -1

TEST_CFG = deepcopy(DEFAULT_CFG)
setattr(TEST_CFG, "task", "segment")

def mock_verify_image_label(image_path : str, label_path : str) -> dict:
    args = (image_path, label_path, "unit_test", False, 1, 0, 0)
    im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = verify_image_label(args)
    label = {
        "im_file": im_file,
        "shape": shape,
        "cls": lb[:, 0:1],  
        "bboxes": lb[:, 1:],
        "segments": segments,
        "keypoints": keypoint,
        "normalized": True,
        "bbox_format": "xywh",
    }
    label["instances"] = Instances(np.array(label["bboxes"]), np.array(resample_segments(label["segments"])), label["keypoints"], bbox_format=label["bbox_format"], normalized=label["normalized"])
    return label

def create_train_dataset(args : IterableSimpleNamespace) -> MyYOLODataset:
    return MyYOLODataset(
        data=ASSET_DATA,
        img_path=ASSET_DIR,
        imgsz=1024,
        cache=False,
        augment=True,
        hyp=args,
        rect=args.rect,
        batch_size=BATCH_SIZE,
        pad=0.0,
        single_cls=args.single_cls or False,
        max_instances=None,
        task="segment",
        subset_args={"n" : 1, "pattern" : ASSET_NAME}
    )

def create_validation_dataset(args : IterableSimpleNamespace) -> MyYOLOValidationDataset:
    return MyYOLOValidationDataset(
        data=ASSET_DATA,
        img_path=ASSET_DIR,
        imgsz=1024,
        cache=False,
        augment=False,
        hyp=args,
        rect=True,
        batch_size=BATCH_SIZE,
        pad=0.5,
        single_cls=args.single_cls or False,
        max_instances=np.Inf,
        task="segment",
        subset_args={"n" : 1, "pattern" : ASSET_NAME}
    )

def test_plot_batch(batch, ni):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=f.name,
            on_plot=os.remove,
        )

class TestDataset(unittest.TestCase):
    def test_train_dataset(self):
        args = IterableSimpleNamespace(**TEST_CFG) if not isinstance(TEST_CFG, IterableSimpleNamespace) else TEST_CFG
        dataset = create_train_dataset(args)
        dataloader = build_dataloader(dataset, BATCH_SIZE, N_WORKERS, True, RANK)
        dataloader_iter = dataloader.iterator

        for batch, _ in zip(dataloader_iter, range(1)):
            self.assertEqual(batch["img"].shape[0], BATCH_SIZE)

        try:
            test_plot_batch(batch, 0)
        except Exception as e:
            self.fail(f"Failed to plot training batch: {e}")

    def test_validation_dataset(self):
        args = IterableSimpleNamespace(**TEST_CFG) if not isinstance(TEST_CFG, IterableSimpleNamespace) else TEST_CFG
        dataset = create_validation_dataset(args)
        dataloader = build_dataloader(dataset, BATCH_SIZE, N_WORKERS, True, RANK)
        dataloader_iter = dataloader.iterator

        for batch, _ in zip(dataloader_iter, range(1)):
            self.assertEqual(batch["img"].shape[0], BATCH_SIZE)

        try:
            test_plot_batch(batch, 0)
        except Exception as e:
            self.fail(f"Failed to plot validation batch: {e}")

    def test_verify_image_label(self):
        label = mock_verify_image_label(IMAGE_ASSET, LABEL_ASSET)
        self.assertIsInstance(label, dict)
        self.assertIn("im_file", label)
        self.assertIn("shape", label)
        self.assertIn("cls", label)
        self.assertIn("bboxes", label)
        self.assertIn("segments", label)
        self.assertIn("keypoints", label)
        self.assertIn("normalized", label)
        self.assertIn("bbox_format", label)
        self.assertIn("instances", label)

    @classmethod
    def tearDownClass(cls):
        # Clean caches i.e. files ending with .cache or .cache.lock in the directory of this script
        cache_files = glob.glob(os.path.join(TEST_DIR, "*.cache*"))
        for cache_file in cache_files:
            os.remove(cache_file)

if __name__ == "__main__":
    unittest.main()
    
