"""Modified YOLO dataset used for training flatbug."""

import os
import re
import stat
import tempfile
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PIL import Image
from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, RandomFlip, RandomHSV
from ultralytics.data.dataset import LOGGER
from ultralytics.utils import IterableSimpleNamespace

from flat_bug.augmentations import (
    CenterCrop,
    FixInstances,
    FlatBugRandomPerspective,
    MaybeZoomCrop,
    RandomColorInv,
    RandomCrop,
    ZoomCrop,
)
from flat_bug.bbox_only import compile_bbox_only, downgrade_labels, fill_missing_segments

HELP_URL = "See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes


def get_area(image_path):  # noqa: D103
    with Image.open(image_path) as image:
        return image.size[0] * image.size[1]


def pyramid_tile_count(w: int, h: int, tile: int = 1024, overlap: int = 384,
                       edge: int = 16, increment: float = 2 / 3) -> int:
    """Tiles the inference pyramid would run over an image of this size.

    Area understates the work for a large image and overstates it for a small one: the
    pyramid pads by EDGE_CASE_MARGIN, overlaps tiles by MINIMUM_TILE_OVERLAP, and repeats the
    whole grid at every scale from TILE/max_dim up to native. Counting tiles captures all
    three, so an image is weighted by what it actually costs to infer rather than by how many
    pixels it happens to contain.

    Mirrors `Predictor.pyramid_predictions`; keep the two in step if the ladder changes.
    """
    from flat_bug.geometric import calculate_tile_offsets
    pad = 4 * edge
    w, h = w + pad, h + pad
    s = tile / max(w, h)
    scales = [s] if s >= 1 else []
    if s < 1:
        while s <= 0.9:
            scales.append(s)
            s /= increment
        if s != 1:
            scales.append(1.0)
    n = 0
    for sc in scales:
        sw = max(round(w * sc / 4) * 4, tile)
        sh = max(round(h * sc / 4) * 4, tile)
        n += len(calculate_tile_offsets((sw, sh), tile, overlap))
    return max(n, 1)


def calculate_pyramid_weights(image_paths: list[str]) -> list[float]:
    """Per-image weight equal to its pyramid tile count."""
    out = []
    for p in image_paths:
        with Image.open(p) as im:
            w, h = im.size
        out.append(float(pyramid_tile_count(w, h)))
    return out


def calculate_image_weights(image_paths: list[str]) -> list[float]:
    """Calculate normalized weights for each image based on the file sizes.

    Normalized by the minimum file size, so that the values are between 1 and infinity.

    Args:
        image_paths: `list` of image file paths.

    Returns:
        normalized weights for each image.

    """
    file_sizes = [get_area(path) + 1 for path in image_paths]
    min_size = min(file_sizes)
    return [(size / min_size) for size in file_sizes]


def reweight(weights: list[float], target_sum: float | int) -> list[float]:
    """Reweights the provided list of weights so that their sum equals the target sum.

    Args:
        weights: `list` of weights to reweight.
        target_sum: Desired sum of the weights.

    Returns:
        Reweighted weights.

    """
    sum_weights = sum(weights)
    return [max(round(w * target_sum / sum_weights), 1) for w in weights]


def generate_indices(weights: list[float], target_size: int | None = None) -> list[int]:
    """Deterministically generates a list of indices based on the provided weights to oversample the items.

    Args:
        weights: `list` of weights for each item.
        target_size: Desired length of the output `list`.
            If `None`, the size of the output is approximately the sum of the weights.

    Returns:
        `list` of indices to oversample the items.

    """
    # n = len(weights)
    weights = [max(round(w), 1) for w in weights]
    indices = []

    if target_size is not None:
        for _ in range(10):
            if abs(sum(weights) - target_size) / target_size < 0.01:
                break
            weights = reweight(weights, target_size)

    for i, w in enumerate(weights):
        indices.extend([i] * int(max(round(w), 1)))

    return indices


def get_datasets(files: list[str]) -> dict[str, list[str]]:  # noqa: D103
    file_dataset = [mtch.group(0) for f in files if (mtch := re.match(r"[^_]+", os.path.basename(f)))]
    datasets = list(set(file_dataset))
    datasets = {d: [] for d in datasets}
    for file, fd in zip(files, file_dataset):
        datasets[fd].append(file)
    return datasets


def subset(self: "FlatBugYOLODataset", n: int | None = None, pattern: str | None = None):
    """Subsets the dataset to the first 'n' elements that match the pattern.

    Args:
        self: A `FlatBugYOLODataset` instance.
        n: The number of elements to keep. Defaults to None; keep all.
        pattern: A regex pattern to match the filenames. Defaults to None; match all.

    """
    if pattern is None and (n is None or n == -1):
        return self
    if pattern is not None:
        cp = re.compile(pattern)

        def _match_pattern(x):
            return bool(cp.search(os.path.basename(x)))

        match_fn = _match_pattern
    else:

        def _match_all(_):
            return True

        match_fn = _match_all
    # Get the indices of the elements that match the pattern
    indices = [i for i, f in enumerate(self.im_files) if match_fn(f)]
    # If n is not None, keep only the first n elements
    if n is not None and n != -1:
        indices = indices[:n]
    # Subset the images
    self.im_files = [f for i, f in enumerate(self.im_files) if i in indices]


def hook_get_labels_with_subset(  # noqa: D103
    obj: "FlatBugYOLODataset", args: dict
):
    if not isinstance(args, dict):
        raise ValueError("args must be a dictionary")
    if not isinstance(obj, FlatBugYOLODataset):
        raise ValueError("obj must be an instance of FlatBugYOLODataset")

    def subset_then_get():
        subset(obj, **args)
        obj.get_labels = getattr(super(type(obj), obj), "get_labels")
        return obj.get_labels()

    obj.get_labels = subset_then_get


class PrintNumInstances:  # noqa: D101
    def __init__(self, title: str):  # noqa: D107
        self.fmt = f"({'{num:>5}'}) ({'{imsize:^10}'}) | {title}"

    def __call__(self, labels: dict):  # noqa: D102
        n = len(labels["instances"]) if "instances" in labels else labels["masks"].max().item()
        print(self.fmt.format(num=n, imsize="x".join([str(d) for d in labels["img"].shape])))
        return labels


def train_augmentation_pipeline(  # noqa: D103
    hyperparameters: IterableSimpleNamespace,
    image_size: int,
    max_instances: int | float | None,
    min_size: int,
    use_segments: bool,
    use_keypoints: bool,
    zoom_prob: float = 0.0,
    zoom_min_px: int = 100,
    zoom_occupancy: tuple[float, float] = (0.22, 0.45),
    zoom_min_scale: float = 1.0,
    zoom_jitter: float = 0.25,
) -> Compose:
    crop_size = int(image_size * 1.5)
    # A fraction of crops are magnified onto a single large instance, so thin appendages are
    # trained at a resolution where they exist. See ZoomCrop for why instances below
    # `min_size` are dropped and inpainted BEFORE magnification rather than after.
    crop = RandomCrop(imsize=crop_size)
    if zoom_prob > 0:
        crop = MaybeZoomCrop(
            crop,
            ZoomCrop(
                imsize=crop_size,
                min_px=zoom_min_px,
                drop_below=min_size,
                occupancy=tuple(zoom_occupancy),
                min_scale=zoom_min_scale,
                jitter=zoom_jitter,
            ),
            p=zoom_prob,
        )
    return Compose([
        # Crop to slightly larger than needed for training
        crop,
        # Affine transformation at same size as above
        FlatBugRandomPerspective(imgsz=int(image_size * 1.5), degrees=180, translate=0, scale=0),
        # Crop to needed size
        CenterCrop(image_size),
        RandomHSV(hgain=hyperparameters.hsv_h, sgain=hyperparameters.hsv_s, vgain=hyperparameters.hsv_v),
        RandomColorInv(p=0.25),
        RandomFlip(direction="vertical", p=hyperparameters.flipud),
        RandomFlip(direction="horizontal", p=hyperparameters.fliplr),
        # Remove instances outside crop
        FixInstances(area_thr=0.975, max_targets=max_instances, min_size=min_size),
        # YOLO-native preprocessing
        Format(
            bbox_format="xywh",
            normalize=True,
            return_mask=use_segments,
            return_keypoint=use_keypoints,
            batch_idx=True,
            mask_ratio=hyperparameters.mask_ratio,
            mask_overlap=hyperparameters.overlap_mask,
        ),
    ])


def validation_augmentation_pipeline(  # noqa: D103
    image_size: int, min_size: int, use_segments: bool, use_keypoints: bool
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
            mask_overlap=True,
        ),
    ])


class FlatBugYOLODataset(YOLODataset):  # noqa: D101
    # What is the minimum size of an instance to be considered (width or height in pixels after augmentations)
    _min_size: int = 32

    # How much do we allow the dataset to grow when oversampling, used to ensure larger images are not underrepresented
    _oversample_factor: int = 2

    # Fraction of training crops magnified onto one instance, and the smallest instance
    # (longest box side, original pixels) eligible to be magnified onto.
    _zoom_prob: float = 0.0
    _zoom_min_px: int = 100
    _zoom_occupancy: tuple[float, float] = (0.22, 0.45)
    _zoom_min_scale: float = 1.0
    _zoom_jitter: float = 0.25

    # Which image-weighting scheme decides how often an image is drawn, and the epoch length
    _sample_weight: str = "current"
    _samples_per_epoch: int | None = None

    def __init__(  # noqa: D107
        self, max_instances: int | float | None, classes: None = None, subset_args: dict | None = None,
        bbox_only_datasets: list[str] | None = None,
        zoom_prob: float = 0.0, zoom_min_px: int = 100,
        zoom_occupancy: tuple[float, float] | list[float] | None = None,
        zoom_min_scale: float = 1.0, zoom_jitter: float = 0.25,
        sample_weight: str = "current", samples_per_epoch: int | None = None,
        *args, **kwargs
    ):
        self._max_instances = max_instances
        self._zoom_prob = float(zoom_prob)
        self._zoom_min_px = int(zoom_min_px)
        if zoom_occupancy is not None:
            self._zoom_occupancy = (float(zoom_occupancy[0]), float(zoom_occupancy[1]))
        self._zoom_min_scale = float(zoom_min_scale)
        self._zoom_jitter = float(zoom_jitter)
        self._sample_weight = str(sample_weight)
        self._samples_per_epoch = int(samples_per_epoch) if samples_per_epoch else None
        self._include_classes = classes  # Only used so the class list is visible in the subset method
        self._bbox_only = compile_bbox_only(bbox_only_datasets)
        if subset_args is not None:
            hook_get_labels_with_subset(self, subset_args)
        if "data" in kwargs:
            if "channels" not in kwargs["data"]:
                kwargs["data"]["channels"] = 3
        super().__init__(classes=classes, *args, **kwargs)
        # After labels exist, before sample weights: polygons in bbox-only datasets become
        # their own bounding rectangle, and every label gains a `has_mask` flag.
        downgrade_labels(self.labels, self.im_files, self._bbox_only)
        # How often each image appears in an epoch. The scheme is selectable so that
        # sampling can be compared as an experimental arm; see scripts/training/splits.
        #   current  area x instances - the historical default
        #   area     area alone, dropping the instance-count factor
        #   pyramid  tiles the inference pyramid would run over the image
        #   uniform  every image once
        # Comparing arms requires an IDENTICAL epoch length: ultralytics drives warmup and
        # lrf off the epoch index, so arms with different epoch lengths would silently train
        # under different learning-rate schedules as well as different budgets.
        areas = calculate_image_weights(self.im_files)
        if self._sample_weight == "uniform":
            self.sample_weights = [1.0 for _ in self.labels]
        elif self._sample_weight == "area":
            self.sample_weights = list(areas)
        elif self._sample_weight == "pyramid":
            # what the image costs at INFERENCE: tiles summed over the pyramid, including
            # overlap and every scale, rather than a bare pixel count
            self.sample_weights = calculate_pyramid_weights(self.im_files)
        elif self._sample_weight == "current":
            self.sample_weights = [
                a * len(label_i["cls"]) for label_i, a in zip(self.labels, areas)
            ]
        else:
            raise ValueError(f"Unknown fb_sample_weight: {self._sample_weight!r}")
        target = self._samples_per_epoch or (len(self.im_files) * self._oversample_factor)
        self.__indices = generate_indices(self.sample_weights, target_size=int(target))
        LOGGER.info(
            f"sampler '{self._sample_weight}': {len(self.im_files)} images -> "
            f"{len(self.__indices)} samples/epoch (target {int(target)})"
        )

    def _debug_write_loaded_images(self, out, index):
        m = np.ascontiguousarray(out["masks"].detach().numpy().transpose(1, 2, 0)) * 255
        m = cv2.cvtColor(cv2.resize(m, (self.imgsz, self.imgsz)), cv2.COLOR_GRAY2BGR)
        n = np.ascontiguousarray(out["img"].detach().numpy().transpose(1, 2, 0)) * 255
        bbs = out["bboxes"].detach().numpy() * self.imgsz
        bbs = bbs.astype(int)
        for k in range(bbs.shape[0]):
            x, y, w, h = bbs[k, :]
            n = cv2.rectangle(n, (x - w // 2, y - w // 2), (x + w // 2, y + h // 2), 255, 3)
        cv2.imwrite(f"/tmp/test/{index}-img.jpg", n + m / 3)

    def load_image(self, i: int | slice) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image.

        Args:
            i: Image index.

        Returns:
            im, hw_original, hw_resized

        """
        # Loads 1 image from dataset index 'i', returns (im, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        f = cast(Path, f)
        fn = cast(Path, fn)

        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = cast(np.ndarray, np.load(fn))

            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw

            return im, (h0, w0), im.shape[:2]  # type: ignore
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # type: ignore

    def build_transforms(  # noqa: D102
        self, hyp: IterableSimpleNamespace
    ) -> Compose:
        return train_augmentation_pipeline(
            hyperparameters=hyp,
            image_size=self.imgsz,
            max_instances=self._max_instances,
            min_size=self._min_size,
            use_segments=self.use_segments,
            use_keypoints=self.use_keypoints,
            zoom_prob=self._zoom_prob,
            zoom_min_px=self._zoom_min_px,
            zoom_occupancy=self._zoom_occupancy,
            zoom_min_scale=self._zoom_min_scale,
            zoom_jitter=self._zoom_jitter,
        )

    def cache_labels(self, path: Path = Path("./labels.cache")):
        """OBS: DO NOT USE THIS FUNCTION MANUALLY."""
        LOGGER.warning("!! OBS !! ==>>== Flat-bug doesn't use the .cache-file! ==<<== !! OBS !!")

        # To bypass the creation of .cache files we use a temporary dummy file, which is set to read-only,
        # causing a check in ultralytics to bail on creating the file
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        # The path passed to the superclass `cache_labels` method must be a pathlib.Path object
        unwriteable_tmp_path = Path(tmp_file.name)

        # Change the file to read-only
        os.chmod(str(unwriteable_tmp_path), stat.S_IREAD)

        # Before calling the superclass `cache_labels` method,
        # we need to create a dummy `<unwriteable_tmp_path>.cache.npy` file
        temporary_dummy_numpy_cache_file = unwriteable_tmp_path.with_suffix(".cache.npy")
        np.save(temporary_dummy_numpy_cache_file, np.array([0]))

        # Call the superclass `cache_labels` method with the temporary read-only pathlib.Path object
        return_val = super().cache_labels(path=unwriteable_tmp_path)

        # Box-only labels get their rectangle here, while we still can: the caller,
        # `YOLODataset.get_labels`, compares total boxes against total segments immediately
        # after this returns and silently drops the segments of EVERY image if they disagree.
        fill_missing_segments(return_val.get("labels", []), self._bbox_only)

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


class FlatBugYOLOValidationDataset(FlatBugYOLODataset):  # noqa: D101
    _resample_n: int = 5

    def build_transforms(  # noqa: D102
        self, hyp: IterableSimpleNamespace
    ) -> Compose:
        return validation_augmentation_pipeline(
            image_size=self.imgsz,
            min_size=self._min_size,
            use_segments=self.use_segments,
            use_keypoints=self.use_keypoints,
        )

    def __len__(self):
        return super().__len__() * self._resample_n

    def __getitem__(self, index):
        i = index % super().__len__()
        return super().__getitem__(i)
