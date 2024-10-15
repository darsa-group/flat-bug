import os, stat, re, tempfile

from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from typing import Union, List, Tuple, Dict, Optional, Self

from ultralytics.utils import IterableSimpleNamespace
from ultralytics.data import YOLODataset
from ultralytics.data.dataset import LOGGER
from ultralytics.data.augment import RandomFlip, RandomHSV, Compose, Format
from flat_bug.augmentations import CenterCrop, RandomCrop, MyRandomPerspective, RandomColorInv, FixInstances

HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

def get_area(image_path):
    image = Image.open(image_path)
    return image.size[0] * image.size[1]

def calculate_image_weights(image_paths : List[str]) -> List[float]:
    """
    Calculate normalized weights for each image based on the file sizes,
    normalized by the minimum file size, so that the values are between 1 and infinity.
    
    Args:
        image_paths (list of str): List of image file paths.
    
    Returns:
        list of float: normalized weights for each image.
    """
    file_sizes = [get_area(path) + 1 for path in image_paths]
    min_size = min(file_sizes)
    return [(size / min_size) for size in file_sizes]

def reweight(
        weights : List[float], 
        target_sum : Union[float, int]
    ) -> List[float]:
    """
    Reweights the provided list of weights so that their sum equals the target sum.
    
    Args:
        weights (list of float): List of weights to reweight.
        target_sum (float): Desired sum of the weights.
    
    Returns:
        list of float: Reweighted weights.
    """
    sum_weights = sum(weights)
    return [max(round(w * target_sum / sum_weights), 1) for w in weights]

def generate_indices(
        weights : List[float], 
        target_size : Optional[int]=None
    ) -> List[int]:
    """
    Deterministically generates a list of indices based on the provided weights to oversample the items.
    
    Args:
        weights (list of float): List of weights for each item.
        target_size (int): Desired size of the output list. If None, the size of the output is approximately the sum of the weights.

    Returns:
        list of int: List of indices to oversample the items.
    """
    n = len(weights)
    weights = [max(round(w), 1) for w in weights]
    indices = []

    if target_size is not None:
        for _ in range(10):
            if abs(sum(weights) - target_size)/target_size < 0.01:
                break
            weights = reweight(weights, target_size)

    for i, w in enumerate(weights):
        indices.extend([i] * int(max(round(w), 1)))

    return indices

def get_datasets(files : List[str]) -> Dict[str, List[str]]:
    file_dataset = [re.match(r"[^_]+", os.path.basename(f)).group(0) for f in files]
    datasets = list(set(file_dataset))
    datasets = {d : [] for d in datasets}
    for file, fd in zip(files, file_dataset):
        datasets[fd].append(file)
    return datasets


def subset(
        self : "MyYOLODataset", 
        n : Optional[int]=None, 
        pattern : Optional[str]=None
    ):
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
    # Create a match function that returns Truthy if the filename matches the pattern or the pattern is None
    match_fn = (lambda x: pattern.search(os.path.basename(x))) if pattern else (lambda x: True)
    # Get the indices of the elements that match the pattern
    indices = [i for i, f in enumerate(self.im_files) if match_fn(f)]
    # If n is not None, keep only the first n elements
    if n is not None and n != -1:
        indices = indices[:n]
    # Subset the images
    self.im_files = [f for i, f in enumerate(self.im_files) if i in indices]

def hook_get_labels_with_subset(
        obj : "MyYOLODataset", 
        args : Dict
    ):
    if not isinstance(args, dict):
        raise ValueError("args must be a dictionary")
    if not isinstance(obj, MyYOLODataset):
        raise ValueError("obj must be an instance of MyYOLODataset")
    def subset_then_get():
        subset(obj, **args)
        obj.get_labels = getattr(super(type(obj), obj), "get_labels")
        return obj.get_labels()
    obj.get_labels = subset_then_get

class PrintNumInstances:
    def __init__(self, title : str):
        self.fmt = f'({"{num:>5}"}) ({"{imsize:^10}"}) | {title}'

    def __call__(self, labels : Dict):
        n = len(labels["instances"]) if "instances" in labels else labels["masks"].max().item()
        print(self.fmt.format(num=n, imsize="x".join([str(d) for d in labels["img"].shape])))
        return labels

def train_augmentation_pipeline(
        hyperparameters : IterableSimpleNamespace, 
        image_size : int, 
        max_instances : Optional[int], 
        min_size : int, 
        use_segments : bool, 
        use_keypoints : bool
    ) -> Compose:
    return Compose([
        RandomCrop(imsize=int(image_size * 1.5)), # Crop to slightly larger than needed for training
        MyRandomPerspective(imgsz=int(image_size * 1.5), degrees=180, translate=0, scale=0), # Affine transformation at same size as above
        CenterCrop(image_size), # Crop to needed size
        RandomHSV(hgain=hyperparameters.hsv_h, sgain=hyperparameters.hsv_s, vgain=hyperparameters.hsv_v),
        RandomColorInv(p=0.25),
        RandomFlip(direction="vertical", p=hyperparameters.flipud),
        RandomFlip(direction="horizontal", p=hyperparameters.fliplr),
        FixInstances(area_thr=0.975, max_targets=max_instances, min_size=min_size), # Remove instances outside crop
        Format( # YOLO-native preprocessing
            bbox_format="xywh",
            normalize=True,
            return_mask=use_segments,
            return_keypoint=use_keypoints,
            batch_idx=True,
            mask_ratio=hyperparameters.mask_ratio,
            mask_overlap=hyperparameters.overlap_mask
        ),
    ])

def validation_augmentation_pipeline(
        image_size : int, 
        min_size : int, 
        use_segments : bool, 
        use_keypoints : bool
    ) -> Compose:
    return Compose([
        RandomCrop(imsize=int(image_size * 1.5)),
        CenterCrop(image_size),
        FixInstances(area_thr=0.975, max_targets=None, min_size=min_size),
        Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=use_segments,
            return_keypoint=use_keypoints,
            batch_idx=True,
            mask_ratio=1,
            mask_overlap=True
        )
    ])


class MyYOLODataset(YOLODataset):
    _min_size : int=32 # What is the minimum size of an instance to be considered (width or height in pixels after augmentations)
    _oversample_factor : int=2 # How much do we allow the dataset to grow when oversampling - this is done to ensure larger images are not underrepresented

    def __init__(
            self : Self, 
            max_instances : Optional[int], 
            classes : None=None, 
            subset_args : Optional[Dict]=None, 
            *args, 
            **kwargs
        ):
        self._max_instances = max_instances
        self._include_classes = classes # Only used so the class list is visible in the subset method
        if subset_args is not None:
            hook_get_labels_with_subset(self, subset_args)
        super().__init__(classes=classes, *args, **kwargs)
        self.sample_weights = [image_weight * len(label_i["cls"]) for label_i, image_weight in zip(self.labels, calculate_image_weights(self.im_files))]
        self.__indices = generate_indices(self.sample_weights, target_size=len(self.im_files) * self._oversample_factor)

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

    def load_image(
            self : Self, 
            i : Union[int, slice]
        ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
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

            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        # print("cached", f, self.im_hw0[i], self.im_hw[i])
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def build_transforms(
            self : Self, 
            hyp : IterableSimpleNamespace
        ) -> Compose:
        return train_augmentation_pipeline(
            hyperparameters=hyp, 
            image_size=self.imgsz, 
            max_instances=self._max_instances, 
            min_size=self._min_size, 
            use_segments=self.use_segments, 
            use_keypoints=self.use_keypoints
        )

    def cache_labels(
            self : Self, 
            path : Path=Path("./labels.cache")
        ):
        """
        OBS: DO NOT USE THIS FUNCTION MANUALLY.
        """
        LOGGER.warning("!! OBS !! ==>>== Flat-bug doesn't use the .cache-file! ==<<== !! OBS !!")

        # To bypass the creation of .cache files we use a temporary dummy file, which is set to read-only, causing a check in ultralytics to bail on creating the file
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        # The path passed to the superclass `cache_labels` method must be a pathlib.Path object
        unwriteable_tmp_path = Path(tmp_file.name)
        
        # Change the file to read-only
        os.chmod(str(unwriteable_tmp_path), stat.S_IREAD)

        # Before calling the superclass `cache_labels` method, we need to create a dummy `<unwriteable_tmp_path>.cache.npy` file
        temporary_dummy_numpy_cache_file = unwriteable_tmp_path.with_suffix(".cache.npy")
        np.save(temporary_dummy_numpy_cache_file, np.array([0]))
        
        # Call the superclass `cache_labels` method with the temporary read-only pathlib.Path object 
        return_val = super().cache_labels(path=unwriteable_tmp_path)

        # Remove the temporary file if it still exists
        if os.path.exists(unwriteable_tmp_path):
            os.remove(unwriteable_tmp_path)
        # Remove the temporary numpy cache file if it still exists
        if os.path.exists(temporary_dummy_numpy_cache_file):
            os.remove(temporary_dummy_numpy_cache_file)
        
        return return_val

    
    def __len__(self):
        return len(self.__indices)

    def __getitem__(self, index):
        return self.transforms(self.get_image_and_label(self.__indices[index]))


class MyYOLOValidationDataset(MyYOLODataset):
    _resample_n : int= 5

    def build_transforms(
            self : Self, 
            hyp : IterableSimpleNamespace
        ) -> Compose:
        return validation_augmentation_pipeline(
            image_size=self.imgsz, 
            min_size=self._min_size, 
            use_segments=self.use_segments, 
            use_keypoints=self.use_keypoints
        )

    def __len__(self):
        return super().__len__() * self._resample_n

    def __getitem__(self, index):
        i = index % super().__len__()
        return super().__getitem__(i)
