"""Implementation of the public flatbug `Predictor` and `TensorPredictions`."""

import atexit
import base64
import json
import os
import pathlib
import queue
import threading
import uuid
from concurrent.futures import Future, as_completed, wait
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.types
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch._prims_common import DeviceLikeType
from torchvision.io import ImageReadMode, decode_image
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from flat_bug import download_from_repository, logger
from flat_bug.config import CFG_PARAMS, DEFAULT_CFG, read_cfg
from flat_bug.geometric import (
    calculate_tile_offsets,
    chw2hwc_uint8,
    contours_to_masks,
    create_contour_mask,
    find_contours,
    poly_area,
    scale_contour,
    simplify_contour,
)
from flat_bug.nms import nms_boxes, nms_masks, nms_polygons
from flat_bug.yolo_helpers import (
    ResultsWithTiles,
    merge_tile_results,
    offset_box,
    postprocess,
    resize_masks,
    stack_masks,
)


class AsyncExecutor:  # noqa: D101
    def __init__(self, max_workers=None, backlog=10000):  # noqa: D107
        self.limit = max_workers or max(1, min(16, (os.cpu_count() or 2) // 2))
        self._queue = queue.Queue(maxsize=backlog)
        self._threads, self._active, self._lock = [], set(), threading.Lock()
        atexit.register(self.flush)

    def _init_pool(self):
        with self._lock:
            if not self._threads:
                for i in range(self.limit):
                    thread = threading.Thread(
                        target=self._work, 
                        daemon=True, 
                        name=f"Async-{i}"
                    )
                    thread.start()
                    self._threads.append(thread)

    def _work(self):
        while True:
            fn, future = self._queue.get()
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        future.set_result(fn())
                    except Exception as e:
                        future.set_exception(e)
                        logger.error(f"Async task failed: {e}", exc_info=True)
            finally:
                with self._lock:
                    self._active.discard(future)
                self._queue.task_done()

    def submit(self, fn, *args, **kwargs):
        """Submit a call to be executed asynchronously."""
        if not self._threads:
            self._init_pool()
        future = Future()
        with self._lock:
            self._active.add(future)
        # Blocks here if backlog is full
        self._queue.put((lambda: fn(*args, **kwargs), future))
        return future

    def flush(self, progress=False):
        """Wait for all pending futures to finish."""
        with self._lock:
            pending = list(self._active)
        if not pending:
            return

        if progress and tqdm:
            for _ in tqdm(as_completed(pending), total=len(pending), desc="Finishing pending executions."):
                pass
        else:
            wait(pending)


_executor = AsyncExecutor()


class Prepared_Results:
    """Class for containing the results from a single `Predictor._detect_instances` call.

    This should probably not be its own class, but just a TensorPredictions object with a single element instead,
    but this would require altering the `TensorPredictions._combine_predictions` function
    to handle a single element differently or pass a flag or something.
    """

    def __init__(self, predictions: ResultsWithTiles, scale: tuple[float, float], device, dtype):  # noqa: D107
        self.wh_scale = torch.tensor(scale, device=device, dtype=dtype).unsqueeze(0)
        self._predictions = predictions
        assert self._predictions.boxes is not None and isinstance(self._predictions.boxes.data, torch.Tensor)
        self._predictions.boxes.data[:, :4] /= self.wh_scale.repeat(1, 2)
        self._predictions.polygons = self._predictions.polygons._apply(
            lambda poly: (poly + torch.roll(poly, 1, dims=0)) / (2 * self.wh_scale)
        )
        self.scale = sum(scale) / 2
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self._predictions)

    def __getitem__(self, i):
        elems = self._predictions[i]
        assert isinstance(elems, ResultsWithTiles)
        return Prepared_Results(elems, (self.scale, self.scale), self.device, self.dtype)

    # Properties for accessing the data
    @property
    def contours(self):  # noqa: D102
        assert self._predictions.masks is not None
        return [
            torch.as_tensor(c) if c is not None else torch.tensor([], dtype=torch.long, device=self.device)
            for c in self._predictions.masks.xy
        ]

    @property
    def masks(self) -> torch.Tensor | np.ndarray:  # noqa: D102
        assert self._predictions.masks is not None
        return self._predictions.masks.data

    @property
    def boxes(self) -> torch.Tensor | np.ndarray:  # noqa: D102
        assert self._predictions.boxes is not None
        return torch.as_tensor(self._predictions.boxes.xyxy)

    @property
    def confs(self) -> torch.Tensor | np.ndarray:  # noqa: D102
        assert self._predictions.boxes is not None
        return torch.as_tensor(self._predictions.boxes.conf)

    @property
    def classes(self) -> torch.Tensor:
        """Not implemented properly."""
        ### OBS: This is not really implemented, but exists just so that the the rest of the code already handles
        # the multiclass case, but this function will need to be changed for it to work properly ###
        # Currently this function is pretty redundant, since the localizer only has a single class.
        # If there were more classes, the function should do some kind of argmax on self._predictions.boxes.cls
        # (I assume these are class probabilities).
        assert self._predictions.boxes is not None
        return torch.ones_like(torch.as_tensor(self._predictions.boxes.cls))


# Class for containing the results from multiple _detect_instances calls
class TensorPredictions:
    """Result class for combining the results from multiple YOLOv8 detections at different scales into a single object.

    `TensorPredictions` handles a rather complex merging procedure,
    resizing to remove image padding and scaling effects on the masks and boxes,
    and non-maximum suppression using mask-IoU or mask-IoS.

    `TensorPredictions` also allows for easy conversion from mask to contours and back, plotting of the results,
    and (de-)serialization to save and load the results to/from disk.
    """

    DUPLICATE_THRESHOLD = 1
    PREFER_POLYGONS = True  # If True, will use shapely Polygons instead of masks for NMS and drawing
    # These are simply initialized here to decrease clutter in the __init__ function and arguments
    mask_width = None
    mask_height = None
    device = None
    dtype = None
    CONSTANTS = (
        "image",
        "image_path",
        "device",
        "dtype",
        "time",
        "mask_height",
        "mask_width",
        "BOX_IS_EQUAL_MARGIN",
        "PREFER_POLYGONS",
    )

    def __init__(
        self,
        predictions: list[Prepared_Results] | None = None,
        image: torch.Tensor | None = None,
        image_path: str | None = None,
        time: bool = False,
        **kwargs,
    ):
        """Create a `TensorPredictions` instance from scratch.

        You probably don't want to use this method manually. If you want to load saved results use:

        ```
        prediction = TensorPredictions.load(...)
        ```

        Args:
            predictions: Predictions from multiple `Predictor._detect_instances` calls.
            image: The image where the predictions originate.
            image_path: Path to the source file for `image`, can be used as a substitute.
            time: Whether operations (such as initialization, NMS, etc.) should be timed.
            kwargs: Additional configuration arguments.

        """
        self.time = time
        start = end = None

        if self.time and predictions is not None and len(predictions) > 0:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # Allow passing of keyword arguments to set attributes
        for k, v in kwargs.items():
            if k in self.CONSTANTS:
                setattr(self, k, v)
            else:
                logger.warning(f"WARNING: Unknown keyword argument {k}={v} for TensorPredictions is ignored!")

        # Device and dtype are None by default, but they may be set by the user or
        # passed by **kwargs, so we check if they are None and if so set them to the default values
        # Then we check that they are the same for all predictions and the image (if they are not None)
        if predictions is not None and len(predictions) > 0:
            # Check that all predictions have the same device and dtype
            elem = predictions[0]
            if self.device is None:
                self.device = elem.device
            if self.dtype is None:
                self.dtype = elem.dtype
            for pi, p in enumerate(predictions):
                assert p.device == self.device, RuntimeError(
                    f"predictions[{pi}].device {p.device} != device {self.device}"
                )
                assert p.dtype == self.dtype, RuntimeError(f"predictions[{pi}].dtype {p.dtype} != dtype {self.dtype}")
            if image is not None:
                assert image.device == self.device, RuntimeError(f"image.device {image.device} != device {self.device}")
        else:
            self.device, self.dtype = torch.device("cpu"), torch.float32

        self.image_path = image_path
        if image is None:
            if self.image_path is None:
                raise ValueError('Either `image` or `image_path` must be specified.')
            self.image = decode_image(
                input=self.image_path, 
                mode=ImageReadMode.RGB, 
                apply_exif_orientation=True
            ).to(self.device)
        else:
            self.image = image.to(self.device)

        # Combine the predictions
        if predictions is not None and len(predictions) > 0:
            self._combine_predictions(predictions)
        else:
            # If there are no predictions, set other attributes to empty tensors or lists.
            # Ensures correct type and device for the attributes when there are no predictions
            self.masks, self.polygons, self.boxes, self.confs, self.classes, self.scales = (
                torch.empty((0, 0), device=self.device, dtype=self.dtype),
                [],
                torch.empty((0, 4), device=self.device, dtype=self.dtype),
                torch.empty((0,), device=self.device, dtype=self.dtype),
                torch.empty((0,), device=self.device, dtype=self.dtype),
                [],
            )

        if self.time and predictions is not None and len(predictions) > 0:
            assert end is not None and start is not None
            end.record()
            torch.cuda.synchronize()
            logger.info(f"Initializing TensorPredictions took {start.elapsed_time(end) / 1000:.3f} s")

    def _combine_predictions(self, predictions: list[Prepared_Results]):
        """Combine a list of Prepared_Results from multiple `Predictor._detect_instances` calls.

        This function is used in-place during initialization of a `TensorPrediction` instance.

        Args:
            predictions: A list of Prepared_Results objects.
            offset: A vector of length 2 containing the x and y offset of the image.

        """
        start = end = end_duplication_removal = end_mask_combination = None
        if self.time:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            end_duplication_removal = torch.cuda.Event(enable_timing=True)
            end_mask_combination = torch.cuda.Event(enable_timing=True)
            start.record()

        self.boxes = torch.cat([torch.as_tensor(p.boxes) for p in predictions])  # Nx4
        self.confs = torch.cat([torch.as_tensor(p.confs) for p in predictions])  # N
        self.scales = [p.scale for p in predictions for _ in range(len(p))]  # N

        ## Duplicate removal ##
        valid_indices = nms_boxes(
            self.boxes,
            self.confs,
            overlap_threshold=self.DUPLICATE_THRESHOLD
        )
        # Subset the boxes and confidences to the valid indices
        self.boxes = self.boxes[valid_indices]
        self.confs = self.confs[valid_indices]
        # Divide the valid indices into each prediction object
        n_detections = [len(p) for p in predictions]
        # max_indices = cumsum(n_detections)
        max_indices = cast(list[int], np.cumsum(n_detections).tolist())
        valid_chunked = [
            valid_indices[
                (valid_indices < max_indices[i]) & (valid_indices >= (max_indices[i - 1] if i > 0 else 0))
            ] - (max_indices[i] - n_detections[i])
            for i in range(len(predictions))
        ]

        if self.time:
            assert end_duplication_removal is not None
            end_duplication_removal.record()

        # For the remaining attributes we remove the duplicates before combining them
        # NxMHxMW - MH and MW are proportional to the original image size
        self.masks = stack_masks([p.masks[nd] for p, nd in zip(predictions, valid_chunked)])
        mhw = self.masks.shape[1:]
        assert len(mhw) == 2
        self.mask_height, self.mask_width = map(round, mhw)

        if self.time:
            assert end_mask_combination is not None
            end_mask_combination.record()

        # Set the target shape of the masks to the shape of the image passed to the TensorPredictions object
        self.masks.orig_shape = self.image.shape[1:]

        poly_lists = [p._predictions.polygons.to_list() for p in predictions]
        self.polygons: list[torch.Tensor] = [
            p[int(nd_i.item()) if isinstance(nd, torch.Tensor) else int(nd)]
            for p, nd in zip(poly_lists, valid_chunked)
            for nd_i in nd
        ]
        self.classes = torch.cat([p.classes[nd] for p, nd in zip(predictions, valid_chunked)])  # N
        self.scales = [predictions[i].scale for i, p in enumerate(valid_chunked) for _ in range(len(p))]  # N

        # Sort the polygons, masks, boxes, classes, scales and confidences by confidence
        sorted_indices = self.confs.argsort(descending=True)
        self.masks = self.masks[sorted_indices]
        self.polygons = [self.polygons[i] for i in sorted_indices]
        self.boxes = self.boxes[sorted_indices]
        self.classes = self.classes[sorted_indices]
        self.scales = [self.scales[i] for i in sorted_indices]
        self.confs = self.confs[sorted_indices]

        # # Check that everything is the correct size
        assert len(self) == len(self.boxes), RuntimeError(f"{len(self)=} != {len(self.boxes)=}")
        assert len(self) == len(self.confs), RuntimeError(f"{len(self)=} != {len(self.confs)=}")
        assert len(self) == len(self.classes), RuntimeError(f"{len(self)=} != {len(self.classes)=}")
        assert len(self) == len(self.scales), RuntimeError(f"{len(self)=} != {len(self.scales)=}")
        if self.time:
            assert (
                start is not None
                and end is not None
                and end_duplication_removal is not None
                and end_mask_combination is not None
            )
            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end) / 1000
            duplication_removal = start.elapsed_time(end_duplication_removal) / 1000
            mask_combination = end_duplication_removal.elapsed_time(end_mask_combination) / 1000
            logger.info(
                f"Combining {len(predictions)} predictions into a single TensorPredictions object took {total:.3f} s |"
                f" Duplication removal: {duplication_removal:.3f} s | Mask combination: {mask_combination:.3f} s"
            )

    def offset_scale_pad(self, offset: torch.Tensor, scale: float, pad: int = 0):
        """Scale and offset the detections to real image coordinates in-place.

        Since the image may be padded, the masks and boxes should be offset by the padding-width and scaled
        by the `scale_before` factor to match the original image size. Also pads the boxes by pad pixels to be safe.

        Args:
            offset: A vector of length 2 containing the x and y offset of the image.
                Useful for removing image-padding effects.
            scale: The scale factor of the image.
            pad: The number of pixels to pad the boxes by. Defaults to 0. (Not to be confused with image-padding,
                this is about expanding the boxes a bit to ensure they cover the entire mask)

        Returns:
            The `TensorPredictions` object with the masks, polygons and boxes offset, scaled and padded.

        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if len(self) > 0:
            offset = offset.to(self.device)
            if any(offset > 0):
                raise NotImplementedError("Positive offsets are not implemented yet")
            # Boxes is easy
            self.boxes = offset_box(self.boxes, offset)  # Add the offsets to the box-coordinates
            self.boxes[:, :4] = (self.boxes[:, :4] * scale).round()  # Multiply the box-coordinates by the scale factor
            # Pad the boxes a bit to be safe
            self.boxes[:, :2] -= pad
            self.boxes[:, 2:] += pad
            self.boxes = self.boxes.long()
            # Clamp the boxes to the image size
            self.boxes[:, 0:4:2] = self.boxes[:, 0:4:2].clamp(0, self.image.shape[2] - 1)
            self.boxes[:, 1:4:2] = self.boxes[:, 1:4:2].clamp(0, self.image.shape[1] - 1)

            self.polygons = [(poly + offset.unsqueeze(0)) * scale for poly in self.polygons]

            # However masks are more complicated since they don't have the same size as the image
            image_shape = torch.tensor(  # Get the shape of the original image
                [self.image.shape[1], self.image.shape[2]], 
                device=self.device,
                dtype=self.dtype
            ) 
            # Calculate the normalized offset 
            # i.e. the offset as a fraction of the scaled and padded image size, 
            # here the scaled and padded image size is calculated from the original image shape
            # (but it would probably be easier just to pass it...)
            offset_norm = -offset / (image_shape / scale - 2 * offset)
            orig_mask_shape = torch.tensor(
                [self.masks.shape[1], self.masks.shape[2]],
                device=self.device, dtype=self.dtype
            ) - 1
            # Convert the normalized offset to the coordinates of the masks
            offset_mask_coords = offset_norm * orig_mask_shape
            # Round the coordinates to the nearest integer and convert to long (needed for indexing)
            offset_mask_coords = torch.round(offset_mask_coords).long()
            self.masks.data = torch.as_tensor(self.masks.data)[
                :,
                offset_mask_coords[0] : (-(offset_mask_coords[0] + 1) if offset_mask_coords[0] != 0 else None),
                offset_mask_coords[1] : (-(offset_mask_coords[1] + 1) if offset_mask_coords[1] != 0 else None),
            ]  # Slice out the padded parts of the masks

        if self.time:
            end.record()
            torch.cuda.synchronize()
            logger.info(f"Offsetting, scaling and padding took {start.elapsed_time(end) / 1000:.3f} s")

        return self

    def fix_boxes(self):
        """Set the boxes to match the masks in-place.

        It is not strictly needed, but can be used as a sanity check to see if the boxes match the masks.
        The discrepancy between the boxes and the masks comes about by all the scaling and smoothing of the masks.

        TODO: Should probably be removed.
        """
        if self.PREFER_POLYGONS:
            raise NotImplementedError("`fix_boxes` is not implemented for polygons")
        mask_data = torch.as_tensor(self.masks.data)
        nonzero_indices = mask_data.nonzero()
        mask_size = torch.tensor([mask_data.shape[1], mask_data.shape[2]], device=self.device, dtype=self.dtype)
        image_size = torch.tensor([self.image.shape[1], self.image.shape[2]], device=self.device, dtype=self.dtype)
        mask_to_image_scale = image_size / mask_size
        for i in range(len(self)):
            this_mask_nz = nonzero_indices[nonzero_indices[:, 0] == i][:, 1:]
            if len(this_mask_nz) == 0:
                self.boxes[i] = torch.tensor([0, 0, 0, 0], device=self.device, dtype=self.dtype)
            else:
                self.boxes[i] = torch.tensor(
                    [
                        this_mask_nz[:, 1].min(),
                        this_mask_nz[:, 0].min(),
                        this_mask_nz[:, 1].max(),
                        this_mask_nz[:, 0].max(),
                    ],
                    device=self.device,
                    dtype=self.dtype,
                ) * mask_to_image_scale.repeat(2)
        self.boxes[:, :2] = self.boxes[:, :2].floor()
        self.boxes[:, 2:] = self.boxes[:, 2:].ceil()
        self.boxes[:, 0:4:2] = self.boxes[:, 0:4:2].clamp(0, self.image.shape[2])
        self.boxes[:, 1:4:2] = self.boxes[:, 1:4:2].clamp(0, self.image.shape[1])
        return self

    def non_max_suppression(self, overlap_threshold: float, metric: str, **kwargs):
        """Perform non-max suppression (NMS) in-place.

        Either uses polygons (most likely) or masks.
        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            len_before = len(self)

        metric = metric.strip().lower()

        # Skip if there are no instances to remove
        if len(self) > 1:
            # Perform non-maximum suppression on the polygons or masks
            if self.PREFER_POLYGONS:
                nms_ind = nms_polygons(
                    polygons=self.polygons,
                    scores=self.confs,
                    overlap_threshold=overlap_threshold,
                    overlap_fn=metric,
                    return_indices=True,
                    boxes=self.boxes,
                    **kwargs,
                )
            else:
                image_to_mask_scale = torch.tensor(
                    [self.image.shape[1] / self.masks.data.shape[1], self.image.shape[2] / self.masks.data.shape[2]],
                    device=self.device,
                    dtype=self.dtype,
                )
                nms_ind: torch.Tensor = nms_masks(
                    masks=torch.as_tensor(self.masks.data),
                    scores=self.confs,
                    overlap_threshold=overlap_threshold,
                    overlap_fn=metric,
                    return_indices=True,
                    boxes=self.boxes / image_to_mask_scale.repeat(2).unsqueeze(0),
                    **kwargs,
                )
            # Remove the instances that were not selected
            self = self[nms_ind.sort().values]
        else:
            nms_ind = torch.empty((0,))

        if self.time:
            end.record()
            torch.cuda.synchronize()
            logger.info(
                f"Non-maximum suppression took {start.elapsed_time(end) / 1000:.3f}s "
                f"for removing {len_before - len(nms_ind)} elements of {len_before} elements"
            )
        return self

    @property
    def contours(self) -> list[torch.Tensor]:
        """Wraps the `openCV.findContours` function.

        `openCV.contourArea` is used to select the largest contour for each mask.
        """
        if self.PREFER_POLYGONS:
            return self.polygons
        else:
            return [
                self.contour_to_image_coordinates(
                    find_contours(create_contour_mask(mask), largest_only=True, simplify=False)
                )
                for mask in self.masks.data
            ]

    @contours.setter
    def contours(self, value: list[torch.Tensor | np.ndarray]):
        assert self.mask_height is not None and self.mask_width is not None
        if self.PREFER_POLYGONS:
            if not isinstance(value, list):
                raise RuntimeError(f"Unknown type `{type(value)}` for `contours` - should be a list of polygons")
            image_h, image_w = self.image.shape[1:]
            contour_scaling = [(image_h - 1) / (self.mask_height - 1), (image_w - 1) / (self.mask_width - 1)]
            for i in range(len(value)):
                if not isinstance(value[i], np.ndarray):
                    value[i] = np.array(value[i])
                if value[i].shape[1] != 2:
                    if value[i].shape[0] == 2:
                        value[i] = np.transpose(value[i], (1, 0))
                    else:
                        raise RuntimeError(f"Unknown shape `{value[i].shape}` for `contours[{i}]` - should be (N, 2)")
                value[i] = torch.from_numpy(
                    scale_contour(
                        contour=np.asarray(value[i]),
                        scale=contour_scaling,
                        expand_by_one=True
                    )
                ).long().to(self.device)
            self.polygons = cast(list[torch.Tensor], value)
            self.masks = torch.stack([
                torch.empty((0, 0), device=self.device, dtype=self.dtype)
                for _ in range(len(value))
            ])  # Initialize empty masks
        else:
            self.masks = contours_to_masks(
                list(map(torch.as_tensor, value)),
                self.mask_height, self.mask_width
            ).to(self.device)

    @property
    def areas(self):
        """Detection areas (either from polygons or masks)."""
        if self.PREFER_POLYGONS:
            return [poly_area(poly) for poly in self.polygons]
        else:
            return self.masks.sum(1).sum(1).tolist()

    def contour_to_image_coordinates(self, contour: torch.Tensor, scale: float = 1) -> torch.Tensor:
        """Convert a contour from mask coordinates to image coordinates.

        Args:
            contour: The contour to convert.
            scale: The scale factor to apply to the contour. Defaults to 1.

        Returns:
            The contour in image coordinates.

        """
        assert self.mask_height is not None and self.mask_width is not None
        image_h, image_w = self.image.shape[1:]
        mask_to_image_scale = [(image_h - 1) / (self.mask_height - 1), (image_w - 1) / (self.mask_width - 1)]
        mask_to_image_scale = torch.tensor(mask_to_image_scale, device=self.device, dtype=torch.float32) * scale
        scaled_contour = scale_contour(contour.cpu().numpy(), mask_to_image_scale.cpu().numpy(), True)
        scaled_contour = simplify_contour(scaled_contour, (mask_to_image_scale / 2).mean().item())
        scaled_contour = torch.tensor(scaled_contour, device=self.device, dtype=torch.long).squeeze(1)

        return scaled_contour

    def flip(self, direction: str = "vertical"):
        """Flips the masks, polygons and boxes along the specified axis in-place.

        Args:
            direction: The axis to flip the masks, polygons and boxes along.
                Defaults to "vertical". Should be one of "vertical", "y", "horizontal" or "x".

        Returns:
            The `TensorPredictions` instance with the masks, polygons and boxes flipped.

        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        mask_data = torch.as_tensor(self.masks.data)
        if direction == "vertical" or direction == "y":
            if mask_data.dim() == 3:
                self.masks.data = torch.flip(mask_data, [1])
            self.boxes[:, 1] = self.image.shape[1] - self.boxes[:, 1]
            self.boxes[:, 3] = self.image.shape[1] - self.boxes[:, 3]
            for i in range(len(self)):
                self.polygons[i][:, 1] = self.image.shape[1] - self.polygons[i][:, 1]
        elif direction == "horizontal" or direction == "x":
            if mask_data.dim() == 3:
                self.masks.data = torch.flip(mask_data, [2])
            self.boxes[:, 0] = self.image.shape[2] - self.boxes[:, 0]
            self.boxes[:, 2] = self.image.shape[2] - self.boxes[:, 2]
            for i in range(len(self)):
                self.polygons[i][:, 0] = self.image.shape[2] - self.polygons[i][:, 0]
        else:
            raise RuntimeError(f"Unknown direction `{direction}` - should be 'vertical'/'y' or 'horizontal'/'x'")

        if self.time:
            end.record()
            torch.cuda.synchronize()
            logger.info(f"Flipping masks, polygons and boxes {direction} took {start.elapsed_time(end) / 1000:.3f} s")

        return self

    def __len__(self) -> int:
        return len(self.polygons)

    def new(self):  # noqa: D102
        return TensorPredictions([], **{k: self.__dict__[k] for k in self.CONSTANTS if k in self.__dict__})

    def __getitem__(self, i):
        """Flexible indexing for TensorPredictions.

        Can be used to get a single element, a slice, or an iterable of indices (e.g. a list, tuple, tensor).
        """
        new_tp = self.new()
        for k, v in self.__dict__.items():
            if k not in self.CONSTANTS:
                # Check if 'i' is a slice
                if isinstance(i, slice):
                    new_value = v[i]
                # Check if 'i' is an iterable
                elif hasattr(i, "__iter__"):
                    if isinstance(i, torch.Tensor):
                        # Just to be super safe we cast to float, then round, then cast to long, then to list
                        i = i.float().round().long().tolist()
                    if not all(isinstance(j, int) for j in i) or all(isinstance(j, float) and (j % 1) == 0 for j in i): # type: ignore
                        raise RuntimeError(f"Unknown type or non-integer float for {i}: {type(i).__name__}")
                    i = [int(j) for j in i]  # type: ignore
                    # If v is a tensor, we can just index it with the list
                    if isinstance(v, torch.Tensor):
                        new_value = v[i]
                    # Otherwise if v is a list, and we need to take the elements from the list
                    elif isinstance(v, list):
                        new_value = [v[j] for j in i]
                    # Otherwise we hope that v supports flexible indexing as well
                    else:
                        try:
                            new_value = v[i]
                        except Exception as e:
                            raise RuntimeError(
                                f"Unknown type for {k}: {type(v)} does not support flexible indexing"
                            ) from e
                else:
                    # Otherwise, assume it's an index
                    if isinstance(i, torch.Tensor) and len(i) == 1:
                        i = i.item()
                    if isinstance(i, float):
                        assert (i % 1) == 0, RuntimeError(f"Index {i} is not an integer")
                        i = int(i)
                    assert isinstance(i, int), RuntimeError(f"Unknown type for {i}: {type(i)}")
                    # If it's a tensor, we need to unsqueeze it to make it a 1-element tensor
                    if isinstance(v, torch.Tensor):
                        new_value = v[i].unsqueeze(0)
                    # Otherwise we assume v is a list, we can just take the element and put it in a list
                    elif isinstance(v, list):
                        new_value = [v[i]]
                    else:
                        raise RuntimeError(f"Unknown type for {k}: {type(v)}")
                setattr(new_tp, k, new_value)
        return new_tp

    def plot(
        self,
        linewidth: int = 2,
        masks: bool = True,
        boxes: bool = True,
        confidence: bool = True,
        outpath: str | None = None,
        scale: float = 1,
        contour_color: tuple[int, int, int] = (255, 0, 0),
        box_color: tuple[int, int, int] = (0, 0, 0),
        alpha: float = 0.3,
        wait: bool = False,
    ):
        """Visualizes `flatbug` predictions from a `TensorPredictions` object.

        Args:
            linewidth: Linewidth of the segmentation countours and bounding boxes.
                Default to 2.
            masks: Flag indicating whether segmentation contours should be included.
                Default to True.
            boxes: Flag indicating whether bounding boxes should be included, if False confidences are also omitted.
                Defaults to True.
            confidence: Flag indicating whether detection confidences should be included,
                if boxes is False, this argument is ignored. Defaults to True.
            outpath: Where should the visualization be saved? If outpath is None, then the rasterized visualization is
                returned as a `cv2.UMat`/`np.ndarray` (shape: HWC, colors: BGR). Defaults to None.
            scale: Render the visualization at a scale relative to the image size
                (from which the predictions originate).
                **OBS**: Large images and/or scales can be very slow to render. Defaults to 1.
            contour_color: RGB color ([0, 255]) to use for contour border and fill.
                Defaults to `(255, 0, 0)` (red).
            box_color: RGB color ([0, 255]) to use for bounding box and confidence text color.
                Defaults to `(0, 0, 0)` (black).
            alpha: Transparency of the contour fill ([0, 1]).
                Defaults to 0.3.
            wait: If `False` (default) returns a future immediately, otherwise block and return the actual result.

        Returns:
            If outpath is supplied, it is returned. Otherwise the rasterized visualization is returned
            as as a `cv2.UMat`/`np.ndarray` (shape: HWC, colors: BGR).
            **OBS**: If `wait=True` then a future is returned instead.

        """
        params = locals()
        params.pop("self", None)
        params.pop("wait", None)
        data = {
            "image": self.image_path or self.image.detach().cpu().clone(),
            "bboxes": self.boxes.detach().cpu().clone(),
            "contours": [poly.detach().cpu().clone() for poly in self.polygons],
            "confs": self.confs.detach().cpu().clone(),
        }
        if outpath and outpath.lower().endswith(".svg"):
            retval = _executor.submit(TensorPredictions._plot_svg, **data, **params)
        else:
            retval = _executor.submit(TensorPredictions._plot_image, **data, **params)
        if wait:
            _executor.flush()
        return outpath or retval

    @staticmethod
    def _box_to_svg_element(
        box: torch.Tensor,
        scale: float = 1.0,
        color: tuple[int, int, int] = (0, 0, 0),
        linewidth: float | int = 2,
        label: str | None = None,
        label_fontsize: float | int = 12,
        background_image: Any | None = None,  # expected to be a NumPy array in BGR
    ) -> str:
        # Convert box color (RGB tuple) to hex.
        hex_color = f"#{''.join(hs if len(hs) == 2 else hs + '0' for v in color if len(hs := hex(v)[2:]))}"

        if scale != 1:
            box = (box.float() * scale).round().long()
        xmin, ymin, xmax, ymax = box.tolist()
        width = xmax - xmin
        height = ymax - ymin

        # Build the rectangle SVG element.
        rect_svg = f'<rect x="{xmin}" y="{ymin}" width="{width}" height="{height}" style="stroke:{hex_color};stroke-width:{linewidth};fill:none"/>'  # noqa: E501

        if label is not None:
            avg_char_width = label_fontsize * 0.6  # 0.6: Arbitrarly chosen value for ~avg. character aspect ratio
            text_width = int(len(label) * avg_char_width)
            text_height = label_fontsize
            offset = (linewidth * 3) // 2  # offset in pixels above the box
            text_y = ymin - offset
            if text_y < text_height:
                text_y = ymax + text_height
            label_color = color

            # If a background image is provided, sample the region where the label will appear.
            if background_image is not None:
                ih, iw = background_image.shape[:2]
                text_x = xmin
                # Ensure we don't sample outside the image.
                region_x_end = min(text_x + round(1 * text_width), iw)
                region_y_end = min(text_y + round(1 * text_height), ih)
                if region_x_end > text_x and region_y_end > text_y:
                    region = background_image[text_y:region_y_end, text_x:region_x_end]
                    avg_brightness = np.mean(region)
                    # If the sampled area is dark, switch the label to white.
                    label_color = (0, 0, 0) if avg_brightness > 150 else (255, 255, 255)

            # Convert the label color to hex.
            label_hex = f"#{''.join(hs if len(hs) == 2 else hs + '0' for v in label_color if len(hs := hex(v)[2:]))}"
            # Create a <text> element. Note that we use a fixed font size (12px) and family.
            text_svg = (
                f'<text x="{xmin}" y="{text_y}" fill="{label_hex}" '
                f'style="font-size:{label_fontsize}px;font-family:Arial;font-weight:bold;">{label}</text>'
            )
            out = f"<g>{rect_svg}{text_svg}</g>"
        else:
            out = rect_svg

        return out

    @staticmethod
    def _contour_to_svg_element(
        contour: torch.Tensor | Any,
        scale: float = 1.0,
        color: tuple[int, int, int] = (255, 0, 0),
        linewidth: int | float = 2,
        alpha: int | float = 0.33,
    ):
        d_list = []
        hex_color = f"#{''.join(hs if len(hs) == 2 else hs + '0' for v in color if len(hs := hex(v)[2:]))}"

        stroke_colour = hex_color
        fill_colour = hex_color
        if alpha > 1:
            alpha = alpha / 255
        for i in range(len(contour)):
            name = i
            x, y = (contour[i] * scale).round().long().tolist()
            d_list.append(f"{x},{y}")
        d_str = " ".join(d_list)
        return (
            f'<path name="{name}" '
            f'style="stroke:{stroke_colour};stroke-width:{linewidth};stroke-opacity:1;'
            f'fill:{fill_colour};fill-opacity:{alpha}" d="M{d_str} Z"/>'
        )

    @staticmethod
    def _plot_svg(
        image: torch.Tensor | str,
        bboxes: torch.Tensor,
        contours: torch.Tensor,
        confs: torch.Tensor,
        linewidth: int = 2,
        masks: bool = True,
        boxes: bool = True,
        confidence: bool = True,
        outpath: str | None = None,
        scale: float = 1,
        contour_color: tuple[int, int, int] = (255, 0, 0),
        box_color: tuple[int, int, int] = (0, 0, 0),
        alpha: float = 0.3,
    ):
        embed_jpeg = True

        if isinstance(image, str):
            tensor_image = decode_image(input=image, mode=ImageReadMode.RGB, apply_exif_orientation=True)
        else:
            tensor_image = image
        np_image = torchvision.transforms.ConvertImageDtype(torch.uint8)(tensor_image).permute(1, 2, 0).cpu().numpy()
        if scale != 1:
            np_image = cv2.resize(np_image, (0, 0), fx=scale, fy=scale)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        try:
            height, width = np_image.shape[0:2]
            ((_, text_height), _) = cv2.getTextSize("0", cv2.FONT_HERSHEY_SIMPLEX, 1 * scale, 2)

            encoded_string = base64.b64encode(cv2.imencode(".jpg", np_image)[1])
            desc = ""
            content = []
            content.append(
                f'<svg width="{width}" height="{height}" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg">'
            )

            # Embed the background image.
            if embed_jpeg:
                content.append(
                    f'<image {desc} width="{int(width)}" height="{int(height)}" x="0" y="0" '
                    f'xlink:href="data:image/jpeg;base64,{encoded_string.decode("utf-8")}"/>'
                )
            if masks:
                for cont in contours:
                    content.append(TensorPredictions._contour_to_svg_element(
                        cont,
                        scale=scale,
                        color=contour_color,
                        linewidth=linewidth,
                        alpha=alpha
                    ))
            if boxes:
                for box, conf in zip(bboxes, confs):
                    lbl = f"{conf.item():.1%}" if confidence else None
                    # Pass the background image so the function can sample the area behind the label.
                    content.append(TensorPredictions._box_to_svg_element(
                        box,
                        scale=scale,
                        color=box_color,
                        linewidth=linewidth,
                        label=lbl,
                        background_image=image,
                        label_fontsize=text_height,
                    ))
            content.append("</svg>")

            if outpath:
                with open(outpath, "w+") as f:
                    f.writelines(content)
                return None

        except Exception as e:
            raise e

        return content

    @staticmethod
    def _plot_image(
        image: torch.Tensor | str,
        bboxes: torch.Tensor,
        contours: torch.Tensor,
        confs: torch.Tensor,
        linewidth: int = 2,
        masks: bool = True,
        boxes: bool = True,
        confidence: bool = True,
        outpath: str | None = None,
        scale: float = 1,
        contour_color: tuple[int, int, int] = (255, 0, 0),
        box_color: tuple[int, int, int] = (0, 0, 0),
        alpha: float = 0.3,
    ) -> cv2.UMat | None:
        if isinstance(image, str):
            tensor_image = decode_image(input=image, mode=ImageReadMode.RGB, apply_exif_orientation=True)
        else:
            tensor_image = image
        np_image = cast(
            np.ndarray,
            torchvision.transforms.ConvertImageDtype(torch.uint8)(tensor_image).permute(1, 2, 0).cpu().numpy(),
        )
        if scale != 1:
            np_image = cv2.resize(np_image, (0, 0), fx=scale, fy=scale)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # Convert colors from RGB to BGR
        contour_color = (contour_color[-1], contour_color[-2], contour_color[-3])
        box_color = (box_color[-1], box_color[-2], box_color[-3])

        if len(contours) > 0:
            # Draw masks
            if masks:
                smpl_contours = [
                    simplify_contour((c * scale).round().to(torch.int32).cpu().numpy(), scale / 2)
                    for c in contours
                ]
                ih, iw = np_image.shape[:2]
                _alpha = int(255 * alpha)

                poly_alpha = np.zeros((ih, iw, 1), dtype=np.int32)
                for i, c in enumerate(smpl_contours):
                    this_poly_alpha = np.zeros((ih, iw, 1), dtype=np.uint8)
                    cv2.drawContours(this_poly_alpha, [c], -1, 1, -1)
                    poly_alpha += this_poly_alpha * _alpha
                poly_alpha = poly_alpha.clip(0, 255) / 255

                # Create a red fill for the polygons
                poly_fill = np.zeros_like(np_image)
                for i, channel_color in enumerate(contour_color):
                    poly_fill[:, :, i] = channel_color
                # Add the polygons to the image by blending the fill and the image using the alpha mask
                np_image = (np_image.astype(np.float32) * (1 - poly_alpha) + poly_fill * poly_alpha)
                np_image = np_image.round().astype(np.uint8)
                # Draw the contours
                for i, c in enumerate(smpl_contours):
                    cv2.drawContours(np_image, [c], -1, contour_color, linewidth)

            # Draw boxes and confidences
            if boxes:
                for box, conf in zip(bboxes, confs):
                    box = box * scale
                    box[:2] = box[:2].floor()
                    box[2:] = box[2:].ceil()
                    box = box.long()
                    start_point = (int(box[0]), int(box[1]))
                    end_point = (int(box[2]), int(box[3]))
                    cv2.rectangle(np_image, start_point, end_point, box_color, linewidth)  # black box
                    if confidence:
                        # Get the width and height of the text
                        (text_width, text_height), _ = cv2.getTextSize(
                            f"{conf * 100:.3g}%",
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1 * scale,
                            thickness=max(1, round(2 * scale)),
                        )
                        # Calculate the text position
                        xp, yp = start_point[0], start_point[1] - linewidth * 2
                        if yp < text_height:
                            yp = end_point[1] + text_height + linewidth * 2
                        # Get the average color intensity behind the text
                        avg_color = np.mean(np_image[yp : yp + text_height, xp : xp + text_width])
                        # Draw the text
                        cv2.putText(
                            img=np_image,
                            text=f"{conf * 100:.3g}%",
                            org=(xp, yp),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1 * scale,
                            color=(0, 0, 0) if avg_color > 150 else (255, 255, 255),
                            thickness=max(1, round(2 * scale)),
                        )

        # Save or show the image
        if outpath:
            cv2.imwrite(outpath, np_image)
            return None
        else:
            return cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)  # type: ignore

    @property
    def crops(self) -> list[torch.Tensor]:
        """Detection crops."""
        return [self.image[:, y1:y2, x1:x2] for x1, y1, x2, y2 in self.boxes.long().tolist()]

    @property
    def crop_masks(self) -> list[torch.Tensor]:
        """Masks for detection crops."""
        if self.PREFER_POLYGONS:
            return [
                contours_to_masks([contour.round().long() - box[:2]], box[3] - box[1], box[2] - box[0])
                for contour, box in zip(self.contours, self.boxes.long())
            ]
        else:
            return [
                resize_masks(mask, self.image.shape[1:])[box[1] : box[3], box[0] : box[2]]  # type: ignore - TODO: fixme
                for mask, box in zip(self.masks, self.boxes.long())  # type: ignore - TODO: fixme
            ]

    @staticmethod
    def _save_1_crop(
        crop: torch.Tensor,
        mask: torch.Tensor | None,
        path: str,
    ) -> str:
        Image.fromarray(
            obj=chw2hwc_uint8(crop, mask).detach().cpu().numpy(), mode="RGB" if mask is None else "RGBA"
        ).save(path, compress_level=1)
        return path

    def save_crops(
        self,
        outdir: str,
        basename: str | None = None,
        mask: bool = False,
        identifier: str | None = None,
        wait: bool = False,
    ) -> list[str]:
        """Save prediction crops."""
        if outdir is None or not os.path.exists(outdir) or not os.path.isdir(outdir):
            raise RuntimeError(f"Invalid outdir {outdir}, does not exist or is not a directory")
        if self.image_path is not None:
            if basename is None:
                assert self.image_path is not None, RuntimeError("Cannot save crops without image_path")
                basename, _ = os.path.splitext(os.path.basename(self.image_path))
            _, image_ext = os.path.splitext(os.path.basename(self.image_path))
        else:
            basename, image_ext = str(uuid.uuid4()), ".jpg"
        if mask:
            image_ext = ".png"
        if identifier is None:
            identifier_field = ""
        else:
            identifier_field = f"UUID_{identifier}"

        crops = self.crops
        if mask:
            crop_masks = self.crop_masks
        else:
            crop_masks = [None] * len(crops)
        crop_paths = [
            os.path.join(outdir, f"crop_{basename}_CROPNUMBER_{i}_{identifier_field}{image_ext}")
            for i in range(len(crops))
        ]

        for crop, _mask, path in zip(crops, crop_masks, crop_paths):
            if isinstance(_mask, torch.Tensor):
                _mask = _mask.detach().cpu().clone()
            _executor.submit(self._save_1_crop, crop.detach().cpu().clone(), _mask, path)
        if wait:
            _executor.flush()

        return crop_paths

    @property
    def json_data(self):
        """JSON-compatible dictionary with instance state data."""
        boxes = self.boxes.cpu().tolist()
        contours = [c.T.cpu().tolist() for c in self.contours]
        confs = self.confs.float().cpu().tolist()
        classes = self.classes.cpu().long().tolist()
        scales = self.scales
        areas = self.areas
        mdata = self.masks.data
        return {
            "boxes": boxes,
            "contours": contours,
            "confs": confs,
            "classes": classes,
            "scales": scales,
            "areas": areas,
            "image_path": self.image_path,
            "image_width": self.image.shape[2],
            "image_height": self.image.shape[1],
            "mask_width": self.image.shape[2] if self.PREFER_POLYGONS else mdata.shape[2],
            "mask_height": self.image.shape[1] if self.PREFER_POLYGONS else mdata.shape[1],
            "identifier": None,
        }

    def serialize(
        self, outpath: str, save_json: bool = True, save_pt: bool = False, identifier: str | None = None
    ) -> None:
        """Serialize the `TensorPredictions` object to a .pt file and/or a .json file.

        The .pt file contains an exact copy of the `TensorPredictions` object, while the .json file
        contains the data in a more human-readable format, which can be
        deserialized into a `TensorPredictions` object using the 'load' function.

        Args:
            outpath: The path to save the serialized data to. Defaults to None.
            save_json: Whether to save the .json file. Defaults to True. Recommended.
            save_pt: Whether to save the .pt file. Defaults to False. Rather disk space wasteful.
            identifier: An identifier for the serialized data. Defaults to None.

        """
        assert len(outpath) > 0, RuntimeError("Cannot serialize with empty outpath")
        assert os.path.exists(os.path.dirname(outpath)), RuntimeError(
            f"Invalide outpath {outpath}, directory does not exist"
        )

        # Check that the outpath doesn't have a file-extension
        outpath, ext = os.path.splitext(outpath)
        if ext != "" and len(ext) < 5:
            logger.warning(
                f"serializer outpath ({outpath}) should not have a file-extension for 'TensorPredictions.serialize'!"
            )
        else:
            outpath = f"{outpath}{ext}"

        pt_path = f"{outpath}.pt"
        json_path = f"{outpath}.json"

        if save_pt:
            if os.path.exists(pt_path):
                logger.warning(f"Pickle ({pt_path}) already exists, overwriting!")
            torch.save(self, pt_path)

        if save_json:
            if os.path.exists(json_path):
                logger.warning(f"JSON ({json_path}) already exists, overwriting!")
            json_data = self.json_data
            json_data["identifier"] = (identifier if identifier else self.image_path,)
            with open(json_path, "w") as f:
                json.dump(json_data, f)

    @classmethod
    def load(cls, data: str | dict, device: DeviceLikeType | None = None, dtype: torch.types._dtype | None = None):
        """Deserializes a TensorPredictions object from a .pt or .json file, or a dictionary.

        OBS: Mutates and returns the current object.

        Args:
            data: The path to the file to load or a dictionary with the deserialized json data.
            device: The device to load the data to. Defaults to None. If None, the device is set to "cpu".
            dtype: The data type to load the data as. Defaults to None. If None, the data type is set to torch.float32.

        Returns:
            This object with the deserialized data.

        """
        if isinstance(data, str):
            path = data
            assert os.path.isfile(path), RuntimeError(f"Invalid path: {path}")
            # Check whether the path is a .pt file or a .json file
            _, ext = os.path.splitext(path)
            if ext == ".pt":
                # When loading from .pt we get an exact copy of the saved TensorPredictions object
                self = torch.load(path)
                return self
            elif ext == ".json":
                with open(path) as f:
                    data = json.load(f)
            else:
                raise RuntimeError(f"Unknown file-extension: {ext} for path: {path}")
        assert not isinstance(data, str)

        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        empty_image = torch.zeros((3, data["image_height"], data["image_width"]), device=device, dtype=dtype) + 255  # type: ignore
        inst = cls(image=empty_image, device=device, dtype=dtype)
        setattr(
            inst, "PREFER_POLYGONS", True
        )  # Since we only store contours in the .json file, we prefer polygons on loading

        # Load constants
        for k, v in data.items():
            if k in inst.CONSTANTS:
                setattr(inst, k, v)

        # Load the data
        for k, v in data.items():
            # Skip constants in second round
            if k in inst.CONSTANTS:
                continue
            # Skip dynamically computed class property attributes
            if k in ["areas"]:
                continue
            # Skip the identifier
            if k in ["identifier", "image_height", "image_width"]:
                continue
            # Catch attributes that don't need special treatment
            elif k in ["scales", "contours"]:
                pass
            # Bounding boxes are easy (as usual)
            elif k == "boxes":
                v = torch.tensor(v, device=inst.device, dtype=inst.dtype)
            # While masks are a bit more complicated
            # Confidences and classes are 1-d tensors (arrays)
            elif k in ["confs", "classes"]:
                v = torch.tensor(v, device=inst.device, dtype=inst.dtype)
            else:
                raise RuntimeError(f"Unknown key in json file: {k}")
            setattr(inst, k, v)

        return inst

    def save(
        self,
        output_directory: str,
        overview: bool | str = True,
        crops: bool | str = True,
        metadata: bool | str = True,
        fast: bool = False,
        mask_crops: bool = False,
        identifier: str | None = None,
        basename: str | None = None,
        wait: bool = False,
    ) -> str | None:
        """Save the serialized prediction results, crops, and overview to the given output directory.

        TODO: Add the identifier to the names of the files,
        so that we can save multiple predictions for the same image or images with the same name.

        Args:
            output_directory: The directory to save the prediction results to.
            overview: Whether to save the overview image. Defaults to True.
                If a string is given, it is interpreted as a path to a directory to save the overview image to.
            crops: Whether to save the crops. Defaults to True.
                If a string is given, it is interpreted as a path to a directory to save the crops to.
            metadata: Whether to save the metadata. Defaults to True.
                If a string is given, it is interpreted as a path to a directory to save the metadata to.
            fast: Whether to use the fast version of the overview image. Defaults to False.
                Saves the overview image at half the resolution.
            mask_crops: Whether to mask the crops. Defaults to False.
            identifier: An identifier for the serialized data. Defaults to None.
            basename: The base name of the image. Defaults to None.
                If None, the base name is extracted from the image path, which must be set in this case.
            wait: If true `save` blocks execution until results are finished saving,
                otherwise results will be saved asynchronously.

        Returns:
            The path to the directory containing the serialized data.
            The crops and overview image(s) are also saved here by default.
            If the standard location is not used at all, the directory is not created and None is returned instead.

        """
        if basename is None:
            if self.image_path is None:
                raise ValueError("Unable to save prediction with unknown source file, when `basename` is not supplied.")
            # Get the base name of the image
            basename = os.path.splitext(os.path.basename(self.image_path))[0]

        prediction_directory = os.path.join(output_directory, basename)
        # Create the prediction directory if it does not exist and it is needed
        # (i.e. if we are saving crops, overview, or metadata to a standard location)
        prediction_directory_is_used = (overview is True) or (crops is True) or (metadata is True)
        if prediction_directory_is_used:
            os.makedirs(prediction_directory, exist_ok=True)

        # Save overview
        if overview:
            # Check if the overview path is overwritten and make sure the directory exists and is a directory
            overview_directory = overview if isinstance(overview, str) else prediction_directory
            os.makedirs(overview_directory, exist_ok=True)
            assert os.path.isdir(overview_directory), RuntimeError(f"Invalid path for overview: {overview_directory}")
            overview_path = os.path.join(overview_directory, f"overview_{basename}_UUID_{identifier}.jpg")
            scale, linewidth = 1, 2
            if fast:
                scale = min(1 / 2, 3072 / max(self.image.shape[1:]))
                linewidth = 1
            self.plot(outpath=overview_path, linewidth=linewidth, scale=scale)

        # Save crops
        if crops:
            # Check if the crops path is overwritten and make sure the directory exists and is a directory
            crop_directory = crops if isinstance(crops, str) else os.path.join(prediction_directory, "crops")
            os.makedirs(crop_directory, exist_ok=True)
            assert os.path.isdir(crop_directory), RuntimeError(f"Invalid path for crops: {crop_directory}")
            self.save_crops(outdir=crop_directory, basename=basename, mask=mask_crops, identifier=identifier)

        # Save json
        if metadata:
            # Check if the metadata path is overwritten and make sure the directory exists and is a directory
            metadata_directory = metadata if isinstance(metadata, str) else prediction_directory
            os.makedirs(metadata_directory, exist_ok=True)
            assert os.path.isdir(metadata_directory), RuntimeError(f"Invalid path for metadata: {metadata_directory}")
            metadata_path = os.path.join(metadata_directory, f"metadata_{basename}_UUID_{identifier}")
            self.serialize(outpath=metadata_path, identifier=identifier)

        if wait:
            _executor.flush()

        return prediction_directory if prediction_directory_is_used else None


def _process_batch(
    image: torch.Tensor,
    offsets: list[tuple[tuple[int, int], tuple[int, int]]],
    tile_size: int,
    batch_start_idx: int,
    batch_size: int,
    device: DeviceLikeType | None = None,
    model: torch.nn.Module = None,  # type: ignore # TODO: fixthis!
    time: bool = False,
    callback: str = "__call__",
    **kwargs: Any,  # Swallow any extra arguments
) -> tuple[torch.Tensor, Any, tuple[int, int, int] | None]:
    start_batch_event = end_fetch_event = end_forward_event = end_batch_event = start_batch_event = (
        current_device_stream
    ) = None
    if time:
        start_batch_event = torch.cuda.Event(enable_timing=True)
        end_fetch_event = torch.cuda.Event(enable_timing=True)
        end_forward_event = torch.cuda.Event(enable_timing=True)
        end_batch_event = torch.cuda.Event(enable_timing=True)
        current_device_stream = torch.cuda.current_stream(device=device)
        start_batch_event.record(current_device_stream)

    # Get the offsets for the current batch and extract and stack the corresponding tiles
    batch = torch.stack([
        image[:, o[0] : (o[0] + tile_size), o[1] : (o[1] + tile_size)]
        for (m, n), o in offsets[batch_start_idx : min((batch_start_idx + batch_size), len(offsets))]
    ], dim=0)
    
    if time:
        assert current_device_stream is not None and end_fetch_event is not None
        end_fetch_event.record(current_device_stream)

    # Forward pass the model on the batch tiles
    with torch.inference_mode():
        batch_outputs = getattr(model, callback)(batch)

    if time:
        assert current_device_stream is not None and end_forward_event is not None
        end_forward_event.record(current_device_stream)
        assert current_device_stream is not None and end_batch_event is not None
        end_batch_event.record(current_device_stream)
        torch.cuda.synchronize(device=device)
        assert current_device_stream is not None and start_batch_event is not None and end_fetch_event is not None
        batch_time = start_batch_event.elapsed_time(end_batch_event) / 1000  # Convert to seconds
        fetch_time = start_batch_event.elapsed_time(end_fetch_event) / 1000  # Convert to seconds
        forward_time = end_fetch_event.elapsed_time(end_forward_event) / 1000  # Convert to seconds

    # Return the postprocessed batch outputs and optionally the timing
    if time:
        return batch, batch_outputs, (batch_time, fetch_time, forward_time)  # type: ignore
    else:
        return batch, batch_outputs, None


class Predictor:
    """A flatbug predictor.

    The flatbug is built to be used primarily via calling the instance itself
    (which is an alias for `Predictor.pyramid_predictions`):

    ```
    model = Predictor(...)
    prediction = model(image)
    ```

    The flatbug predictor returns an object of type `TensorPredictions` which
    is designed for use in a vertically integrated computer vision pipeline
    (i.e. intermediary results are not written to disk, but kept in GPU or CPU
    RAM to avoid I/O), but also has export and visualization functionality.
    """

    HYPERPARAMETERS: list[str] = CFG_PARAMS
    """
    The available hyperparameters for the predictor. \\
    These can be set using the `set_hyperparameters` class method.
    """

    # Hyperparameters, set to None so they are visible in the class
    MIN_MAX_OBJ_SIZE: tuple[int, int] = None  # type: ignore
    """
    Defines the minimum and maximum object size as seen in a single tile. \\
    Size is defined as the square root of the pixel area of the bounding box.
    """
    MAX_MASK_SIZE: int = None  # type: ignore
    """
    Defines the maximum size of the segmentation masks. \\
    Only applies if PREFER_POLYGONS is False.
    """
    SCORE_THRESHOLD: float = None  # type: ignore
    """
    The score threshold for the predictions. \\
    TODO: This should be called CONFIDENCE_THRESHOLD.
    """
    OVERLAP_THRESHOLD: float = None  # type: ignore
    """
    The overlap (e.g. IOU) threshold used to determine if two instances are duplicates. \\
    """
    MINIMUM_TILE_OVERLAP: int = None  # type: ignore
    """
    The minimum - but not necessarily the maximum - overlap between tiles \\
    in a single layer of the pyramid. Increasing this value will increase \\
    the computation time, but may improve the detection of large instances. 
    """
    EDGE_CASE_MARGIN: int = None  # type: ignore
    """
    The margin to add to the edge of the image to catch instances that are \\
    split between tiles. The margin is added to the edge of the image, such \\
    that instances on the true edge of the images are not removed.
    """
    PREFER_POLYGONS: bool = None  # type: ignore
    """
    Whether to prefer representing the instance segmentation using polygons \\
    instead of masks. This is a much more compact representation, but cannot \\
    represent complex shapes (like holes in the mask), only concave polygons.
    """
    EXPERIMENTAL_NMS_OPTIMIZATION: bool = None  # type: ignore
    """
    Enables an experimental optimization for the NMS step. \\
    This optimization improves the performance of the NMS step when there are \\
    many instances in a large image and CUDA is available.
    """
    OVERLAP_METRIC: str = None  # type: ignore
    """
    Metric to use for NMS. One of "IOU" or "IOS", more might be added in the future.
    """
    TIME: bool = None  # type: ignore
    """
    Whether to time the different parts of the prediction process. \\
    Enabling this will print a verbose output of the timing of the different \\
    parts of the prediction process.
    """
    TILE_SIZE: int = None  # type: ignore
    """
    The size of the tiles to split the image into. \\
    This is defined by the model and should probably not be changed.
    """
    BATCH_SIZE: int = None  # type: ignore
    """
    The batch size to use for the prediction. \\
    This determines how many tiles are processed in parallel. \\
    Increasing this value may improve performance, but will also increase memory usage.
    """

    # Enable debug mode, only for development
    DEBUG = False

    def __init__(
        self,
        model: str | pathlib.Path = "flat_bug_M_v2.pt",
        cfg: dict | str | Path | None = None,
        device: str | torch.device | int | list[str | torch.device | int] = torch.device("cpu"),
        dtype: torch.types._dtype | str = torch.float32,
    ):
        """Instantiate a flatbug predictor.

        Args:
            model: Path to a local weight file, or the name of a weight file in the flatbug model zoo.
            cfg: A dictionary or a path to a YAML containing the flatbug config for this model instance.
            device: Which device to run the model on.
            dtype: Which dtype to run the model on.

        """
        cfg = read_cfg(cfg, strict=True) if isinstance(cfg, (str, Path)) else (cfg or {})
        self.set_hyperparameters(**{**DEFAULT_CFG, **cfg})

        self._multi_gpu = isinstance(device, (list, tuple))
        self._devices = [torch.device(device)] if not self._multi_gpu else [torch.device(d) for d in device]  # type: ignore
        if len(self._devices) > 1:
            # TODO: Implement  single-producer multi-consumer model for _detect_instances in the multi-gpu case
            raise NotImplementedError("Multi-GPU is not implemented yet")
        self._device = self._devices[0]
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            raise ValueError(f"Dtype '{dtype}' is not supported.")
        self._dtype = dtype

        self._model: torch.nn.Module
        if isinstance(model, str):
            if not os.path.exists(model):
                success = download_from_repository(
                    "models/" + "/".join(os.path.normpath(model).split(os.path.sep)), model, False
                )
                if not success:
                    raise FileNotFoundError(f"No such model or file: '{model}'")

            yolo = YOLO(model, "segment", verbose=True)
            assert isinstance(yolo.model, torch.nn.Module)
            self._model = yolo.model
            self._model.to(self._device, dtype=self._dtype)
            self._model.eval()
        elif isinstance(model, torch.nn.Module):
            self._model = model
        else:
            raise RuntimeError(f"Unknown model type: {type(model)}")
        self._model = self._model.to(self._device, dtype=self._dtype)
        self._model.eval()

        self._yolo_predictor = None

    def set_hyperparameters(self, **kwargs):
        """Set the hyperparameters in-place for the predictor.

        Args:
            **kwargs: The hyperparameters to set.

        Returns:
            This instance with the new hyperparameters.

        """
        for k, v in kwargs.items():
            if k in self.HYPERPARAMETERS:
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown hyperparameter: {k}")
        return self

    def _detect_instances(self, image: torch.Tensor, scale: float = 1.0, max_scale: bool = False) -> Prepared_Results:
        TILE_SIZE = self.TILE_SIZE
        this_MIN_MAX_OBJ_SIZE = list(self.MIN_MAX_OBJ_SIZE)
        this_EDGE_CASE_MARGIN = self.EDGE_CASE_MARGIN
        # If we are at the top level, we don't want to remove large instances
        #   - since there are no layers above to detect them as small instances
        if max_scale:
            this_MIN_MAX_OBJ_SIZE[1] = int(1e9)
            this_EDGE_CASE_MARGIN = 0

        if self.TIME:
            start_detect = torch.cuda.Event(enable_timing=True)
            end_detect = torch.cuda.Event(enable_timing=True)
            main_stream = torch.cuda.current_stream(device=self._device)
            start_detect.record(main_stream)

        orig_h, orig_w = image.shape[1:]
        w, h = orig_w, orig_h
        padded = False
        h_pad, w_pad = 0, 0
        pad_lrtb = 0, 0, 0, 0
        real_scale = 1, 1

        # Check dimensions and channels
        assert image.device == self._device, RuntimeError(f"image.device {image.device} != self._device {self._device}")
        assert image.dtype == self._dtype, RuntimeError(f"image.dtype {image.dtype} != self._dtype {self._dtype}")

        # Resize if scale is not 1
        if scale != 1:
            h, w = round(orig_h * scale / 4) * 4, round(orig_w * scale / 4) * 4
            real_scale = w / orig_w, h / orig_h
            resize = transforms.Resize((h, w), antialias=True)
            image = resize(image)
            h, w = image.shape[1:]

        # If any of the sides are smaller than the TILE_SIZE, pad to TILE_SIZE
        if w < TILE_SIZE or h < TILE_SIZE:
            padded = True
            w_pad = max(0, TILE_SIZE - w) // 2
            h_pad = max(0, TILE_SIZE - h) // 2
            pad_lrtb = w_pad, w_pad + (w % 2 == 1), h_pad, h_pad + (h % 2 == 1)
            image = torch.nn.functional.pad(image, pad_lrtb, mode="constant", value=0)  # Pad with black
            h, w = image.shape[1:]

        offsets = calculate_tile_offsets(
            image_size=(w, h), tile_size=TILE_SIZE, minimum_overlap=self.MINIMUM_TILE_OVERLAP
        )

        hyperparams = {
            "image": image,
            "batch_size": self.BATCH_SIZE,
            "tile_size": TILE_SIZE,
            "edge_case_margin": this_EDGE_CASE_MARGIN,
            "score_threshold": self.SCORE_THRESHOLD,
            "overlap_threshold": self.OVERLAP_THRESHOLD,
            "overlap_metric": self.OVERLAP_METRIC,
            "min_max_object_size": this_MIN_MAX_OBJ_SIZE,
            "time": self.TIME,
        }

        if self.TIME:
            # Initialize timing calculations
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            batch_times, fetch_times, forward_times, postprocess_times = [], [], [], []
            start_event.record(main_stream)

        postprocessed_results: list[Results] = [None for _ in range(len(offsets))]  # type: ignore
        batches = 0
        with torch.no_grad():
            for batch_start_idx in range(0, len(offsets), self.BATCH_SIZE):
                batch_kwargs = {
                    "offsets" : offsets,
                    "batch_start_idx" : batch_start_idx,
                    **hyperparams
                }
                batches += 1

                batch, raw_results, timing = _process_batch(model=self._model, device=self._device, **batch_kwargs)
                if self.TIME:
                    postprocess_start = torch.cuda.Event(enable_timing=True)
                    postprocess_end = torch.cuda.Event(enable_timing=True)
                    postprocess_start.record(main_stream)
                this_postprocessed_results = postprocess(
                    raw_results,
                    imgs=batch,
                    max_det=1000,
                    min_confidence=self.SCORE_THRESHOLD,
                    overlap_threshold=self.OVERLAP_THRESHOLD,
                    overlap_metric=self.OVERLAP_METRIC,
                    nms=3,
                    valid_size_range=self.MIN_MAX_OBJ_SIZE,
                    edge_margin=self.EDGE_CASE_MARGIN,
                )
                for batch_index in range(len(this_postprocessed_results)):
                    tr = Results(**this_postprocessed_results[batch_index])  # type: ignore
                    tr.orig_img = None  # type: ignore # Comment this line if we want debug output.
                    postprocessed_results[batch_start_idx + batch_index] = tr
                if self.TIME:
                    assert timing is not None
                    batch_times.append(timing[0])
                    fetch_times.append(timing[1])
                    forward_times.append(timing[2])
                    postprocess_end.record(main_stream)
                    torch.cuda.synchronize(device=self._device)
                    postprocess_times.append(postprocess_start.elapsed_time(postprocess_end) / 1000)

        if self.TIME:
            # Finish timing calculations
            end_event.record(main_stream)
            torch.cuda.synchronize(device=self._device)
            total_elapsed = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            fetch_time, forward_time, postprocess_time = sum(fetch_times), sum(forward_times), sum(postprocess_times)
            total_batch_time = sum(batch_times) + postprocess_time
            overhead_prop = (total_elapsed - total_batch_time) / total_elapsed
            fetch_prop, forward_prop, postprocess_prop = (
                fetch_time / total_batch_time,
                forward_time / total_batch_time,
                postprocess_time / total_batch_time,
            )

        # ruff: disable[E501]
        # DEBUG #####
        # if self.DEBUG:
        #     logger.info(f'Number of tiles processed before merging and plotting: {len(postprocessed_results)}')
        # for i in range(len(postprocessed_results)):
        #     postprocessed_results[i].orig_img = (postprocessed_results[i].orig_img.detach().contiguous() * 255).to(torch.uint8).cpu().numpy() # Needed for compatibility with the Results.plot function
        #     postprocessed_results[i].names = ["?" for _ in range(10)]
        # fig, axs = plt.subplots(y_n_tiles, x_n_tiles, figsize=(x_n_tiles * 5, y_n_tiles * 5))
        # axs = axs.flatten() if len(offsets) > 1 else [axs]
        # postprocessed_results : List[Results] = postprocessed_results
        # [axs[i].imshow(p.plot(pil=False, masks=True, probs=False, labels=False, kpt_line=False)) for i, p in enumerate(postprocessed_results)]
        # plt.savefig(os.path.join(f"debug_{scale:.3f}_fraw.png"), dpi=300)
        # for i in range(len(postprocessed_results)):
        #     postprocessed_results[i].orig_img = torch.tensor(postprocessed_results[i].orig_img).squeeze(0).to(dtype=self._dtype, device=self._device) / 255.0 # Backtransform
        ###############
        # ruff: enable[E501]

        ## Combine the results from the tiles
        MASK_SIZE = 256  # Defined by the YOLOv8 model segmentation architecture
        MASK_TO_IMG_RATIO = MASK_SIZE / torch.tensor(
            [TILE_SIZE, TILE_SIZE], dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        box_offsetters = torch.tensor(
            [[o[1][0] - pad_lrtb[2], o[1][1] - pad_lrtb[0]] for o in offsets], dtype=torch.float32, device=self._device
        )
        mask_offsetters = torch.round(box_offsetters * MASK_TO_IMG_RATIO).long()
        new_mask_size = (
            (mask_offsetters.max(dim=0).values + MASK_SIZE)
            - torch.tensor(pad_lrtb[1::2][::-1], dtype=torch.long, device=self._device) * MASK_TO_IMG_RATIO[0]
        ).tolist()
        orig_img = (
            image[
                :,
                pad_lrtb[2] : (-pad_lrtb[3] if pad_lrtb[3] != 0 else None),
                pad_lrtb[0] : (-pad_lrtb[1] if pad_lrtb[1] != 0 else None),
            ]
            if padded
            else image
        )

        merged_results = merge_tile_results(
            results=postprocessed_results,
            orig_img=orig_img.permute(1, 2, 0),
            box_offsetters=box_offsetters.to(self._dtype),
            mask_offsetters=mask_offsetters,
            new_shape=new_mask_size,
            clamp_boxes=(h - sum(pad_lrtb[2:]), w - sum(pad_lrtb[:2])),
            max_mask_size=self.MAX_MASK_SIZE,
            exclude_masks=self.PREFER_POLYGONS,
        )

        # ruff: disable[E501]
        #### DEBUG #####
        # if self.DEBUG:
        # logger.info(f'Number of tiles processed after merging and filtering: {len(ps)}')
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ps.orig_img = (ps.orig_img.detach().contiguous() * 255).to(torch.uint8).cpu().numpy() # Needed for compatibility with the Results.plot function
        # # ps.boxes.data[:, :4] /= scale
        # logger.info(ps.orig_img.shape)
        # ax.imshow(ps.plot(pil=False, masks=True, probs=False, labels=False, kpt_line=False))
        # plt.savefig(f"debug_{scale:.3f}_merged.png", dpi=300)
        # ps.orig_img = torch.tensor(ps.orig_img).squeeze(0).to(dtype=self._dtype, device=self._device) / 255.0 # Backtransform
        # # ps.boxes.data[:, :4] *= scale
        #################
        # ruff: enable[E501]

        if self.TIME:
            end_detect.record(main_stream)
            torch.cuda.synchronize(device=self._device)
            total_detect_time = start_detect.elapsed_time(end_detect) / 1000  # Convert to seconds
            pred_prop = total_elapsed / total_detect_time
            logger.info(
                f"Prediction time: {total_elapsed:.3f}s/{pred_prop:>4.1%}"
                f" (overhead: {overhead_prop:>4.1%}) |"
                f" Fetch {fetch_prop:>4.1%} |"
                f" Forward {forward_prop:>4.1%} |"
                f" Postprocess {postprocess_prop:>4.1%} |"
                f" Tiles {len(offsets)}"
            )
            if hasattr(self, "total_detection_time"):
                self.total_detection_time += total_detect_time
            if hasattr(self, "total_forward_time"):
                self.total_forward_time += forward_time
        return Prepared_Results(
            predictions=merged_results,
            scale=real_scale,
            device=self._device,
            dtype=self._dtype
        )

    def pyramid_predictions(
        self,
        image: torch.Tensor | str,
        path: str | None = None,
        scale_increment: float = 2 / 3,
        scale_before: float | int = 1,
        single_scale: bool = False,
    ) -> TensorPredictions:
        """Perform inference on an image at multiple scales and return the predictions.

        Args:
            image: The image to run inference on. If a string is given, the image is read from the path.
                If it is a `torch.Tensor`, the path must be provided.
                We assume that floating point images are in the range [0, 1]
                and integer images are in the range [0, integer_type_max].
                *(see https://github.com/pytorch/vision/blob/6d7851bd5e2bedc294e40e90532f0e375fcfee04/torchvision/transforms/_functional_tensor.py#L66)*
            path: The path to the image. Defaults to None. Must be provided if `image` is a `torch.Tensor`.
            scale_increment: The scale increment to use when resizing the image. Defaults to 2/3.
            scale_before: The scale to apply before running inference. Defaults to 1.
            single_scale: Whether to run inference on a single scale. Defaults to False.

        Returns:
            The predictions for the image.

        """
        if self.TIME:
            start_pyramid, end_pyramid = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_pyramid.record()

        real_path = image if isinstance(image, str) else path
        if isinstance(image, str):
            tensor_image : torch.Tensor = decode_image(
                input=image,
                mode=ImageReadMode.RGB,
                apply_exif_orientation=True
            )
        elif isinstance(image, torch.Tensor):
            tensor_image = image.cpu()
            logger.debug(
                "Input image source file not specified for prediction, "
                "saving the prediction will require specifying the source file basename."
            )
        else:
            raise TypeError(f"Unknown type for image: {type(image)}, expected str or torch.Tensor")

        c, h, w = tensor_image.shape
        transform_list = []

        if scale_before != 1:
            w, h = int(w * scale_before), int(h * scale_before)
            resize = transforms.Resize((h, w), antialias=True)
            transform_list.append(resize)

        if tensor_image.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            transform_list.append(transforms.ConvertImageDtype(self._dtype))

        # A border is always added now, to avoid edge-cases on the actual edge of the image.
        # I.e. only detections on internal edges of tiles should be removed, not detections on the edge of the image.
        edge_case_margin_padding_multiplier = 2
        padding_offset = (
            torch.tensor((self.EDGE_CASE_MARGIN, self.EDGE_CASE_MARGIN), dtype=self._dtype)
            * edge_case_margin_padding_multiplier
        )

        if padding_offset.sum() > 0:
            padding_for_edge_cases = transforms.Pad(
                padding=self.EDGE_CASE_MARGIN * edge_case_margin_padding_multiplier,
                fill=0,
                padding_mode='constant'
            )
            transform_list.append(padding_for_edge_cases)
        else:
            padding_offset[:] = 0
        
        transformed_image = (
            transforms.Compose(transform_list)(tensor_image) if transform_list else tensor_image
        ).to(device=self._device,  dtype=self._dtype)

        assert len(transformed_image.shape) == 3, RuntimeError(
            f"transformed_image.shape {transformed_image.shape} != 3"
        )
        assert transformed_image.shape[0] == 3, RuntimeError(
            f"transformed_image.shape[0] {transformed_image.shape[0]} != 3. "
            "The image is probably supplied in WxHxC instead of CxWxH, try image.permute(2, 1, 0) before passing it."
        )

        max_dim = max(transformed_image.shape[1:])
        # min_dim = min(transformed_image.shape[1:])

        scales = []

        if single_scale:
            scales = [1]
        else:
            s = self.TILE_SIZE / max_dim

            if s >= 1:
                scales.append(s)
            else:
                while s <= 0.9:  # Cut off at 90%, to avoid having s~1 and s=1.
                    scales.append(s)
                    s /= scale_increment
                if s != 1:
                    scales.append(1.0)

        logger.debug(f"Running inference on scales: {scales}")

        if self.TIME:
            self.total_detection_time, self.total_forward_time = 0, 0
        all_preds = [
            self._detect_instances(transformed_image, scale=s, max_scale=s == min(scales))
            for s in reversed(scales)
        ]

        if self.TIME:
            if self.total_detection_time > 0:
                perc_forward = f"{self.total_forward_time / self.total_detection_time * 100:.3g}"
            else:
                perc_forward = "N/A"
            logger.info(f"Total detection time: {self.total_detection_time:.3f}s ({perc_forward}% forward)")

        all_preds = (
            TensorPredictions(
                predictions=all_preds,
                image=tensor_image.to(self._device),
                image_path=real_path,
                dtype=self._dtype,
                device=self._device,
                time=self.TIME,
                PREFER_POLYGONS=self.PREFER_POLYGONS,
            )
            .offset_scale_pad(
                offset=-padding_offset,
                scale=1 / scale_before,
                pad=5,  # pad the boxes a bit to ensure they encapsulate the masks
            )
            .non_max_suppression(
                overlap_threshold=self.OVERLAP_THRESHOLD,
                metric=self.OVERLAP_METRIC,
                group_first=self.EXPERIMENTAL_NMS_OPTIMIZATION,
            )
        )

        if self.TIME:
            # Finish timing calculations
            end_pyramid.record()
            torch.cuda.synchronize()
            total_pyramid_time = start_pyramid.elapsed_time(end_pyramid) / 1000
            logger.info(
                f"Total pyramid time: {total_pyramid_time:.3f}s"
                f" ({self.total_detection_time / total_pyramid_time * 100:.3g}% detection |"
                f" {self.total_forward_time / total_pyramid_time * 100:.3g}% forward)"
            )

        return all_preds

    def __call__(
        self,
        image: torch.Tensor | str,
        path: str | None = None,
        scale_increment: float = 2 / 3,
        scale_before: float | int = 1,
        single_scale: bool = False,
    ) -> TensorPredictions:
        """Perform inference on an image at multiple scales and return the predictions.

        Args:
            image: The image to run inference on. If a string is given, the image is read from the path.
                If it is a `torch.Tensor`, the path must be provided. We assume that floating point images
                are in the range [0, 1] and integer images are in the range [0, integer_type_max].
                *(see https://github.com/pytorch/vision/blob/6d7851bd5e2bedc294e40e90532f0e375fcfee04/torchvision/transforms/_functional_tensor.py#L66)*
            path: The path to the image. Defaults to None. Must be provided if `image` is a `torch.Tensor`.
            scale_increment: The scale increment to use when resizing the image. Defaults to 2/3.
            scale_before: The scale to apply before running inference. Defaults to 1.
            single_scale: Whether to run inference on a single scale. Defaults to False.

        Returns:
            The predictions for the image.

        """
        params = locals()
        params.pop("self", None)
        return self.pyramid_predictions(**params)
