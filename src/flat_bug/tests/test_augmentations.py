import unittest

import os, math

from copy import deepcopy

from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data.base import Path
from ultralytics.data.utils import verify_image_label
from ultralytics.data.augment import Compose
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments

from flat_bug.datasets import train_augmentation_pipeline, validation_augmentation_pipeline

from flat_bug.tests.remote_lfs_fallback import check_file_with_remote_fallback

TEST_HYP = {
    "hsv_h": 0.5,
    "hsv_s": 0.5,
    "hsv_v": 0.5,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mask_ratio": 2,
    "overlap_mask": 0.5,
    "max_instances": 1000,
    "min_size": 4,
    "imgsz": 1024,
    "use_segments": True,
    "use_keypoints": False
}

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_NAME = "ALUS_Non-miteArachnids_Unknown_2020_11_03_4545"

TEST_IMG = os.path.join(ASSET_DIR, ASSET_NAME + ".jpg")
TEST_LABEL = os.path.join(ASSET_DIR, ASSET_NAME + ".txt")
check_file_with_remote_fallback(TEST_IMG)
check_file_with_remote_fallback(TEST_LABEL)

def generate_train_augmentation_pipeline(hyp):
    hyp = IterableSimpleNamespace(**hyp)
    return train_augmentation_pipeline(
        hyperparameters=hyp,
        image_size=hyp.imgsz,
        max_instances=hyp.max_instances,
        min_size=hyp.min_size,
        use_segments=hyp.use_segments,
        use_keypoints=hyp.use_keypoints
    )

def generate_validation_augmentation_pipeline(hyp):
    hyp = IterableSimpleNamespace(**hyp)
    return validation_augmentation_pipeline(
        image_size=hyp.imgsz,
        min_size=hyp.min_size,
        use_segments=hyp.use_segments,
        use_keypoints=hyp.use_keypoints
    )

def mock_verify_image_label(image_path, label_path):
    try:
        args = (image_path, label_path, "unit_test", False, 1, 0, 0)
        im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg = verify_image_label(args)
    except ValueError as e:
        args = (image_path, label_path, "unit_test", False, 1, 0, 0, True)
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

def mock_yolo_base_dataset_load_image(image_path, imgsz, rect_mode=False):
    """
    Loads an image from the given path and resizes it if necessary.

    Args:
        image_path (str or Path): Path to the image file.
        imgsz (int): Desired image size for resizing.
        rect_mode (bool): Whether to maintain the aspect ratio when resizing.

    Returns:
        tuple: (im, (h0, w0), resized_shape)
            - im: The loaded and possibly resized image.
            - (h0, w0): Original height and width of the image.
            - resized_shape: Shape of the resized image.
    """
    f = Path(image_path)
    
    if not f.exists():
        raise FileNotFoundError(f"Image Not Found {f}")

    im = cv2.imread(str(f))  # Read image using OpenCV
    if im is None:
        raise FileNotFoundError(f"Image Not Found {f}")
    
    h0, w0 = im.shape[:2]  # Original height and width
    
    if rect_mode:  # Resize while maintaining aspect ratio
        r = imgsz / max(h0, w0)  # Ratio
        if r != 1:  # If sizes are not equal
            w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    elif not (h0 == w0 == imgsz):  # Resize by stretching image to square imgsz
        im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    
    return im, (h0, w0), im.shape[:2]

def make_empty(obj : Any) -> Any:
    if isinstance(obj, np.ndarray):
        obj = np.empty((0, *obj.shape[1:]), dtype=obj.dtype)
    elif isinstance(obj, torch.Tensor):
        obj = torch.empty((0, *obj.shape[1:]), dtype=obj.dtype)
    elif isinstance(obj, list):
        obj = []
    return obj

class TestMockYOLOHelpers(unittest.TestCase):
    def test_mock_yolo_base_dataset_load_image(self):
        loaded_img, _, _ = mock_yolo_base_dataset_load_image(TEST_IMG, TEST_HYP["imgsz"])
        self.assertIsInstance(loaded_img, np.ndarray, msg=f"Expected {np.ndarray} object, got {type(loaded_img)}")
        self.assertEqual(loaded_img.shape, (TEST_HYP["imgsz"], TEST_HYP["imgsz"], 3), msg=f"Expected image shape ({TEST_HYP['imgsz']}, {TEST_HYP['imgsz']}, 3), got {loaded_img.shape}")

    def test_mock_verify_image_label(self):
        result = mock_verify_image_label(TEST_IMG, TEST_LABEL)
        self.assertIsInstance(result, dict, msg=f"Expected {dict} object, got {type(result)}")
        correct = {
            "im_file": str,
            "shape": tuple,
            "cls": np.ndarray,
            "bboxes": np.ndarray,
            "segments": list,
            "keypoints": None,
            "normalized": bool,
            "bbox_format": str,
            "instances": Instances
        }
        for k, v in correct.items():
            self.assertTrue(k in result, msg=f"Missing key '{k}' in result")
            if v is None:
                continue
            self.assertIsInstance(result[k], v, msg=f"Invalid type for key '{k}'. Expected {v}, got {type(result[k])}")

class TestAugmentations(unittest.TestCase):
    def test_generate_train_augmentation_pipeline(self):
        pipeline = generate_train_augmentation_pipeline(TEST_HYP)
        self.assertTrue(isinstance(pipeline, Compose), msg=f"Expected {Compose} object, got {type(pipeline)}")

    def test_generate_validation_augmentation_pipeline(self):
        pipeline = generate_validation_augmentation_pipeline(TEST_HYP)
        self.assertTrue(isinstance(pipeline, Compose), msg=f"Expected {Compose} object, got {type(pipeline)}")

    def test_train_augmentation_pipeline(self):
        pipeline = generate_train_augmentation_pipeline(TEST_HYP)
        loaded_img, _, _ = mock_yolo_base_dataset_load_image(TEST_IMG, int(TEST_HYP["imgsz"] * 2))
        pipeline_input = mock_verify_image_label(TEST_IMG, TEST_LABEL)
        pipeline_input["img"] = loaded_img
        # Normal pipeline execution
        try:
            out = pipeline(deepcopy(pipeline_input))
            # DEBUG SAVE THE OUTPUT IMAGE
            # out_img = out["img"].permute(1, 2, 0).flip(2).numpy()
            # out_img = np.ascontiguousarray(out_img).astype(np.uint8)
            # polys = out["instances"].segments
            # polys = [poly.astype(int) for poly in polys]
            # cv2.drawContours(out_img, polys, -1, (0, 255, 0), 2)
            # cv2.imwrite(ASSET_DIR + "/test_train_augmentation_pipeline.jpg", out_img)
        except Exception as e:
            raise type(e)("Failed to execute training augmentation pipeline on image with labels due to:\n\t" + str(e))
        self.assertIsInstance(out, dict, msg="Invalid output of training augmentation pipeline on image with labels")
        # Simulate empty labels
        empty_pipeline_input = deepcopy(pipeline_input)
        for k, v in empty_pipeline_input.items():
            if k == "img":
                continue
            if isinstance(v, Instances):
                empty_pipeline_input[k] = Instances(make_empty(v.bboxes), make_empty(v.segments), make_empty(v.keypoints), bbox_format=empty_pipeline_input["bbox_format"], normalized=v.normalized)
            else:
                empty_pipeline_input[k] = make_empty(v)
        try:
            out = pipeline(empty_pipeline_input)
        except Exception as e:
            raise type(e)("Failed to execute training augmentation pipeline on image without labels due to:\n\t" + str(e))
        self.assertIsInstance(out, dict, msg="Invalid output of training augmentation pipeline on image without labels")

    def test_validation_augmentation_pipeline(self):
        pipeline = generate_validation_augmentation_pipeline(TEST_HYP)
        loaded_img, _, _ = mock_yolo_base_dataset_load_image(TEST_IMG, int(TEST_HYP["imgsz"] * 2))
        pipeline_input = mock_verify_image_label(TEST_IMG, TEST_LABEL)
        pipeline_input["img"] = loaded_img
        # Normal pipeline execution
        try:
            out = pipeline(deepcopy(pipeline_input))
        except Exception as e:
            raise type(e)("Failed to execute validation augmentation pipeline on image with labels due to:\n\t" + str(e))
        self.assertIsInstance(out, dict, msg="Invalid output of validation augmentation pipeline on image with labels")
        # Simulate empty labels
        empty_pipeline_input = deepcopy(pipeline_input)
        for k, v in empty_pipeline_input.items():
            if k == "img":
                continue
            if isinstance(v, Instances):
                empty_pipeline_input[k] = Instances(make_empty(v.bboxes), make_empty(v.segments), make_empty(v.keypoints), bbox_format=empty_pipeline_input["bbox_format"], normalized=v.normalized)
            else:
                empty_pipeline_input[k] = make_empty(v)
        try:
            out = pipeline(empty_pipeline_input)
        except Exception as e:
            raise type(e)("Failed to execute validation augmentation pipeline on image without labels due to:\n\t" + str(e))
        self.assertIsInstance(out, dict, msg="Invalid output of validation augmentation pipeline on image without labels")


if __name__ == "__main__":
    unittest.main()