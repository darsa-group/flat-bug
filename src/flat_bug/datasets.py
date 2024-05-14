import copy
import os
import cv2
import numpy as np
from pathlib import Path
import glob
import re

from typing import List, Dict, Optional

from ultralytics.data import YOLODataset
from ultralytics.data.augment import RandomFlip, RandomHSV, Compose, Format
from flat_bug.augmentations import MyCrop, RandomCrop, MyRandomPerspective, RandomColorInv

HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def get_datasets(files : List[str]) -> Dict[str, List[str]]:
    file_dataset = [re.match(r"[^_]+", os.path.basename(f)).group(0) for f in files]
    datasets = list(set(file_dataset))
    datasets = {d : [] for d in datasets}
    for file, fd in zip(files, file_dataset):
        datasets[fd].append(file)
    return datasets

def subset(self, n : Optional[int]=None, pattern : Optional[str]=None):
    """
    Subsets the dataset to the first 'n' elements that match the pattern.

    Args:
        n (int, optional): The number of elements to keep. Defaults to None; keep all.
        pattern (str, optional): A regex pattern to match the filenames. Defaults to None; match all.
    """
    if pattern is None and (n is None or n == -1):
        return self
    # Compile the regex pattern
    pattern = re.compile(pattern) if pattern else None
    # Create a match function that returns True if the filename matches the pattern or the pattern is None
    match_fn = (lambda x: pattern.search(os.path.basename(x))) if pattern else (lambda x: True)
    # Get the indices of the elements that match the pattern
    indices = [i for i, f in enumerate(self.im_files) if match_fn(f)]
    # If n is not None, keep only the first n elements
    if n is not None and n != -1:
        indices = indices[:n]
    # Subset the images
    self.im_files = [f for i, f in enumerate(self.im_files) if i in indices]

def hook_get_labels_with_subset(obj, args):
    if not isinstance(args, dict):
        raise ValueError("args must be a dictionary")
    if not isinstance(obj, MyYOLODataset):
        raise ValueError("obj must be an instance of MyYOLODataset")
    def subset_then_get():
        subset(obj, **args)
        obj.get_labels = getattr(super(type(obj), obj), "get_labels")
        return obj.get_labels()
    obj.get_labels = subset_then_get

class MyYOLODataset(YOLODataset):
    def __init__(self, max_instances, classes=None, subset_args=None, *args, **kwargs):
        self._max_instances = max_instances
        self._include_classes = classes # Only used so the class list is visible in the subset method
        if subset_args is not None:
            hook_get_labels_with_subset(self, subset_args)
        super().__init__(classes=classes, *args, **kwargs)

    def _debug_write_loaded_images(self, out, index):
        m = np.ascontiguousarray(out["masks"].detach().numpy().transpose(1, 2, 0)) * 255
        m = cv2.cvtColor(cv2.resize(m, (self.imgsz, self.imgsz)), cv2.COLOR_GRAY2BGR)
        n = np.ascontiguousarray(out["img"].detach().numpy().transpose(1, 2, 0)) * 255
        bbs = out["bboxes"].detach().numpy() * self.imgsz
        bbs = bbs.astype(int)
        for k in range(bbs.shape[0]):
            x, y, w, h = bbs[k, :]
            n = cv2.rectangle(n, (x - w // 2, y - w // 2), (x + w // 2, y + h // 2), 255, 3)
        cv2.imwrite("/tmp/test/%i-img.jpg" % index, n + m / 3)

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, resized hw)

        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]

        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)

            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw

            # # fixme
            # if os.path.basename(self.im_files[i]).startswith("2023-"):
            #     im = cv2.resize(im, (round(w0 / 4.0), round(h0 / 4.0)), interpolation=cv2.INTER_AREA)
            #     # print("not cached", f, (h0, w0), im.shape[:2])
            #     h0, w0 = im.shape[:2]

            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        # print("cached", f, self.im_hw0[i], self.im_hw[i])
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def build_transforms(self, hyp=None):

        return Compose([
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomColorInv(),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr),
            MyRandomPerspective(imgsz=self.imgsz, degrees=180, translate=0),
            RandomCrop(self.imgsz, max_targets=self._max_instances),
            # MyAlbumentations(self.imgsz),
            # LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False),
            Format(bbox_format="xywh",
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask)
        ])

    def __getitem__(self, index):
        out = self.transforms(self.get_image_and_label(index))
        return out


class MyYOLOValidationDataset(MyYOLODataset):
    _resample_n = 5

    def build_transforms(self, hyp=None):
        return Compose([
            MyRandomPerspective(imgsz=self.imgsz, degrees=0, scale=(.1, 1), translate=0),
            RandomCrop(self.imgsz, max_targets=self._max_instances),
            Format(bbox_format="xywh",
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask)
        ])

    def __len__(self):
        return super().__len__() * self._resample_n

    #

    def __getitem__(self, index):
        i = index % super().__len__()
        return super().__getitem__(i)
