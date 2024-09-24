import base64
import os.path
import pathlib
import shutil
import tempfile
import json
import math

from typing import Union, List, Tuple, Optional, Any, Self
import torch.types
from torch._prims_common import DeviceLikeType

import numpy as np
import cv2
from PIL import Image

import torch
import torchvision
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms

from ultralytics import YOLO
from ultralytics.engine.results import Results

# from flat_bug.yolo_helpers import *
from flat_bug import logger
from flat_bug.yolo_helpers import ResultsWithTiles, stack_masks, offset_box, resize_mask, postprocess, merge_tile_results
from flat_bug.geometric import find_contours, contours_to_masks, simplify_contour, create_contour_mask, scale_contour, chw2hwc_uint8
from flat_bug.nms import nms_masks, nms_polygons, detect_duplicate_boxes
from flat_bug.config import read_cfg, DEFAULT_CFG, CFG_PARAMS
from flat_bug.augmentations import InpaintPad


# Class for containing the results from a single _detect_instances call - This should probably not be its own class, but just a TensorPredictions object with a single element instead, but this would require altering the TensorPredictions._combine_predictions function to handle a single element differently or pass a flag or something
class Prepared_Results:
    def __init__(self, predictions: "ResultsWithTiles", scale: Tuple[float, float], device, dtype):
        self.wh_scale = torch.tensor(scale, device=device, dtype=dtype).unsqueeze(0)
        self._predictions = predictions
        self._predictions.boxes.data[:, :4] /= self.wh_scale.repeat(1, 2)
        self._predictions.polygons = [(poly + torch.roll(poly, 1, dims=0)) / (2 * self.wh_scale) for poly in self._predictions.polygons]
        # self._predictions.polygons = [torch.tensor(shapely.affinity.scale(Polygon(poly.cpu().numpy()), self.wh_scale[0][0].item(), self.wh_scale[0][1].item(), origin="centroid").exterior.coords, device=device, dtype=dtype) for poly in self._predictions.polygons]
        self.scale = sum(scale) / 2
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self._predictions)

    def __getitem__(self, i) -> "Prepared_Results":
        return Prepared_Results(self._predictions[i], self.scale, self.device, self.dtype)

    # Properties for accessing the data
    @property
    def contours(self) -> List["torch.Tensor"]:
        return [c if c is not None else torch.tensor([], dtype=torch.long, device=self.device) for c in self._predictions.masks.xy]

    @property
    def masks(self) -> Union["torch.Tensor", "np.ndarray"]:
        return self._predictions.masks.data

    @property
    def boxes(self) -> Union["torch.Tensor", "np.ndarray"]:
        return self._predictions.boxes.xyxy

    @property
    def confs(self) -> Union["torch.Tensor", "np.ndarray"]:
        return self._predictions.boxes.conf

    @property
    def classes(self) -> "torch.Tensor":
        ### OBS: This is not really implemented, but exists just so that the the rest of the code already handles the multiclass case, but this function will need to be changed for it to work properly ### 
        # Currently this function is pretty redundant, since the localizer only has a single class. 
        # If there were more classes, the function should do some kind of argmax on self._predictions.boxes.cls (I assume these are class probabilities).
        return torch.ones_like(self._predictions.boxes.cls)

# Class for containing the results from multiple _detect_instances calls
class TensorPredictions:
    """
    Result handling class for combining the results from multiple YOLOv8 detections at different scales into a single object.

    `TensorPredictions` handles a rather complex merging procedure, resizing to remove image padding and scaling effects on the masks and boxes, and non-maximum suppression using mask-IoU or mask-IoS.

    `TensorPredictions` also allows for easy conversion from mask to contours and back, plotting of the results, and (de-)serialization to save and load the results to/from disk.
    """
    BOX_IS_EQUAL_MARGIN = 0  # How many pixels the boxes can differ by and still be considered equal? Used for removing duplicates before merging overlapping masks.
    PREFER_POLYGONS = False  # If True, will use shapely Polygons instead of masks for NMS and drawing
    # These are simply initialized here to decrease clutter in the __init__ function and arguments
    mask_width = None
    mask_height = None
    device = None
    dtype = None
    CONSTANTS = ["image", "image_path", "device", "dtype", "time", "mask_height", "mask_width", "CONSTANTS",
                 "BOX_IS_EQUAL_MARGIN",
                 "PREFER_POLYGONS"]  # Attributes that should not be changed after initialization - should 'contours' be here?

    def __init__(
            self, 
            predictions : Optional[list[Prepared_Results]]=None,
            image : Optional["torch.Tensor"]=None,
            image_path : Optional[str] = None, 
            time : bool=False, 
            **kwargs
        ):
        # Set option flags
        self.time = time

        # Timing could probably be hidden in a decorator...
        if self.time and len(predictions) > 0:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # Allow passing of keyword arguments to set attributes
        for k, v in kwargs.items():
            if k in self.CONSTANTS:
                setattr(self, k, v)
            else:
                logger.warning(f"WARNING: Unknown keyword argument {k}={v} for TensorPredictions is ignored!")

        # Device and dtype are None by default, but they may be set by the user or passwed by **kwargs, so we check if they are None and if so set them to the default values
        # Then we check that they are the same for all predictions and the image (if they are not None)
        no_predictions = predictions is None or len(predictions) == 0
        if not no_predictions:
            # Check that all predictions have the same device and dtype
            if self.device is None:
                self.device = predictions[0].device
            if self.dtype is None:
                self.dtype = predictions[0].dtype
            for pi, p in enumerate(predictions):
                assert p.device == self.device, RuntimeError(f"predictions[{pi}].device {p.device} != device {self.device}")
                assert p.dtype == self.dtype, RuntimeError(f"predictions[{pi}].dtype {p.dtype} != dtype {self.dtype}")
            if not image is None:
                assert image.device == self.device, RuntimeError(f"image.device {image.device} != device {self.device}")

        # Set attributes
        self.image = image
        self.image_path = image_path

        # Combine the predictions
        if not no_predictions:
            self._combine_predictions(predictions)
        else:
            # If there are no predictions, set other attributes to empty tensors or lists - ensures correct type and device for the attributes when there are no predictions
            self.masks, self.polygons, self.boxes, self.confs, self.classes, self.scales = torch.empty((0, 0), device=self.device, dtype=self.dtype), [], torch.empty((0, 4), device=self.device, dtype=self.dtype), torch.empty(0, device=self.device, dtype=self.dtype), torch.empty(0, device=self.device, dtype=self.dtype), []

        if self.time and len(predictions) > 0:
            end.record()
            torch.cuda.synchronize()
            logger.info(f'Initializing TensorPredictions took {start.elapsed_time(end) / 1000:.3f} s')

    def _combine_predictions(
            self, 
            predictions: list[Prepared_Results]
        ):
        """
        Combines a list of Prepared_Results from multiple _detect_instances calls into a single TensorPredictions object.

        Args:
            predictions (list[Prepared_Results]): A list of Prepared_Results objects.
            offset (torch.Tensor): A vector of length 2 containing the x and y offset of the image.

        Returns:
            None: The function is performed in-place, so it returns None.
        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            end_duplication_removal = torch.cuda.Event(enable_timing=True)
            end_mask_combination = torch.cuda.Event(enable_timing=True)
            start.record()
        self.boxes = torch.cat([p.boxes for p in predictions])  # Nx4
        self.confs = torch.cat([p.confs for p in predictions])  # N
        self.scales = [p.scale for p in predictions for _ in range(len(p))]  # N

        ## Duplicate removal ##
        # Calculate indices of non-duplicate boxes - prioritzed by resolution
        valid_indices = detect_duplicate_boxes(
            self.boxes,
            torch.tensor(self.scales, dtype=self.dtype, device=self.device),
            margin=self.BOX_IS_EQUAL_MARGIN, return_indices=True
        )
        # Subset the boxes and confidences to the valid indices
        self.boxes = self.boxes[valid_indices]
        self.confs = self.confs[valid_indices]
        # Divide the valid indices into each prediction object
        n_detections = [len(p) for p in predictions]
        # max_indices = cumsum(n_detections)
        max_indices = np.cumsum(n_detections).tolist()
        valid_chunked = [valid_indices[(valid_indices < max_indices[i]) & (valid_indices >= (max_indices[i - 1] if i > 0 else 0))] - (max_indices[i] - n_detections[i]) for i in range(len(predictions))]

        if self.time:
            end_duplication_removal.record()

        # For the remaining attributes we remove the duplicates before combining them
        self.masks = stack_masks([p.masks[nd] for p, nd in zip(predictions, valid_chunked)])  # NxMHxMW - MH and MW are proportional to the original image size
        self.mask_height, self.mask_width = self.masks.shape[1:]

        if self.time:
            end_mask_combination.record()

        self.masks.orig_shape = self.image.shape[1:]  # Set the target shape of the masks to the shape of the image passed to the TensorPredictions object

        self.polygons = [p._predictions.polygons[nd_i] for p, nd in zip(predictions, valid_chunked) for nd_i in nd]
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
        assert len(self) == len(self.boxes), RuntimeError(f"len(self) {len(self)} != len(self.boxes) {len(self.boxes)}")
        assert len(self) == len(self.confs), RuntimeError(f"len(self) {len(self)} != len(self.confs) {len(self.confs)}")
        assert len(self) == len(self.classes), RuntimeError(f"len(self) {len(self)} != len(self.classes) {len(self.classes)}")
        assert len(self) == len(self.scales), RuntimeError(f"len(self) {len(self)} != len(self.scales) {len(self.scales)}")
        if self.time:
            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end) / 1000
            duplication_removal = start.elapsed_time(end_duplication_removal) / 1000
            mask_combination = end_duplication_removal.elapsed_time(end_mask_combination) / 1000
            logger.info(
                f'Combining {len(predictions)} predictions into a single TensorPredictions object took {total:.3f} s |'
                f' Duplication removal: {duplication_removal:.3f} s | Mask combination: {mask_combination:.3f} s'
            )

    def offset_scale_pad(
            self, 
            offset: torch.Tensor, 
            scale: float, 
            pad: int = 0
        ) -> Self:
        """
        Since the image may be padded, the masks and boxes should be offset by the padding-width and scaled by 
        the `scale_before` factor to match the original image size. Also pads the boxes by pad pixels to be safe.

        Args:
            offset (`torch.Tensor`): A vector of length 2 containing the x and y offset of the image. Useful for removing image-padding effects.
            scale (`float`): The scale factor of the image.
            pad (`int`, optional): The number of pixels to pad the boxes by. Defaults to 0. (Not to be confused with image-padding, 
                this is about expanding the boxes a bit to ensure they cover the entire mask)
        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if any(offset > 0):
            raise NotImplementedError("Positive offsets are not implemented yet")

        if len(self) > 0:
            # Boxes is easy
            self.boxes = offset_box(self.boxes, offset)  # Add the offsets to the box-coordinates
            self.boxes[:, :4] = (self.boxes[:,
                                :4] * scale).round()  # Multiply the box-coordinates by the scale factor (round so it doesn't implicitly gets floored when cast to an integer later)
            # Pad the boxes a bit to be safe
            self.boxes[:, :2] -= pad
            self.boxes[:, 2:] += pad
            self.boxes = self.boxes.long()
            # Clamp the boxes to the image size
            self.boxes[:, 0:4:2] = self.boxes[:, 0:4:2].clamp(0, self.image.shape[2] - 1)
            self.boxes[:, 1:4:2] = self.boxes[:, 1:4:2].clamp(0, self.image.shape[1] - 1)

            self.polygons = [(poly + offset.unsqueeze(0)) * scale for poly in self.polygons]

            # However masks are more complicated since they don't have the same size as the image
            image_shape = torch.tensor([self.image.shape[1], self.image.shape[2]], device=self.device,
                                    dtype=self.dtype)  # Get the shape of the original image
            # Calculate the normalized offset (i.e. the offset as a fraction of the scaled and padded image size, here the scaled and padded image size is calculated from the original image shape, but it would probably be easier just to pass it...)
            offset_norm = -offset / (image_shape / scale - 2 * offset)
            orig_mask_shape = torch.tensor([self.masks.shape[1], self.masks.shape[2]], device=self.device, dtype=self.dtype) - 1  # Get the shape of the masks
            # Convert the normalized offset to the coordinates of the masks
            offset_mask_coords = offset_norm * orig_mask_shape
            # Round the coordinates to the nearest integer and convert to long (needed for indexing)
            offset_mask_coords = torch.round(offset_mask_coords).long()
            self.masks.data = self.masks.data[
                :, 
                offset_mask_coords[0]:(-(offset_mask_coords[0] + 1) if offset_mask_coords[0] != 0 else None), 
                offset_mask_coords[1]:(-(offset_mask_coords[1] + 1) if offset_mask_coords[1] != 0 else None)
            ]  # Slice out the padded parts of the masks

        if self.time:
            end.record()
            torch.cuda.synchronize()
            logger.info(f'Offsetting, scaling and padding took {start.elapsed_time(end) / 1000:.3f} s')

        return self

    def fix_boxes(self) -> Self:
        """
        This function simply sets the boxes to match the masks.

        It is not strictly needed, but can be used as a sanity check to see if the boxes match the masks.
        The discrepancy between the boxes and the masks comes about by all the scaling and smoothing of the masks.

        TODO: Should probably be removed.
        """
        if self.PREFER_POLYGONS:
            raise NotImplementedError("`fix_boxes` is not implemented for polygons")
        nonzero_indices = self.masks.data.nonzero()
        mask_size = torch.tensor([self.masks.data.shape[1], self.masks.data.shape[2]], device=self.device, dtype=self.dtype)
        image_size = torch.tensor([self.image.shape[1], self.image.shape[2]], device=self.device, dtype=self.dtype)
        mask_to_image_scale = image_size / mask_size
        for i in range(len(self)):
            this_mask_nz = nonzero_indices[nonzero_indices[:, 0] == i][:, 1:]
            if len(this_mask_nz) == 0:
                self.boxes[i] = torch.tensor([0, 0, 0, 0], device=self.device, dtype=self.dtype)
            else:
                self.boxes[i] = torch.tensor(
                    [this_mask_nz[:, 1].min(), this_mask_nz[:, 0].min(), this_mask_nz[:, 1].max(),
                     this_mask_nz[:, 0].max()], device=self.device, dtype=self.dtype) * mask_to_image_scale.repeat(2)
        self.boxes[:, :2] = self.boxes[:, :2].floor()
        self.boxes[:, 2:] = self.boxes[:, 2:].ceil()
        self.boxes[:, 0:4:2] = self.boxes[:, 0:4:2].clamp(0, self.image.shape[2])
        self.boxes[:, 1:4:2] = self.boxes[:, 1:4:2].clamp(0, self.image.shape[1])
        return self

    def non_max_suppression(
            self, 
            iou_threshold : float, 
            **kwargs
        ) -> Self:
        """
        Simply wraps the `nms_masks` function from yolo_helpers.py, and removes the duplicates from the `TensorPredictions` object.
        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            len_before = len(self)

        # Skip if there are no instances to remove
        if len(self) > 1:
            # Perform non-maximum suppression on the polygons or masks
            if self.PREFER_POLYGONS:
                nms_ind = nms_polygons(
                    polygons=self.polygons,
                    scores=self.confs,# * torch.tensor(self.scales, dtype=self.dtype, device=self.device),
                    iou_threshold=iou_threshold, 
                    return_indices=True, 
                    dtype=self.dtype,
                    boxes=self.boxes, 
                    **kwargs
                )
            else:
                image_to_mask_scale = torch.tensor(
                    [self.image.shape[1] / self.masks.data.shape[1], self.image.shape[2] / self.masks.data.shape[2]],
                    device=self.device, dtype=self.dtype
                )
                nms_ind = nms_masks(
                    masks=self.masks.data,
                    scores=self.confs,# * torch.tensor(self.scales, dtype=self.dtype, device=self.device),
                    iou_threshold=iou_threshold, 
                    return_indices=True,
                    boxes=self.boxes / image_to_mask_scale.repeat(2).unsqueeze(0), 
                    **kwargs
                )
            # Remove the instances that were not selected
            self = self[nms_ind.sort().values]
        
        if self.time:
            end.record()
            torch.cuda.synchronize()
            logger.info(
                f'Non-maximum suppression took {start.elapsed_time(end) / 1000:.3f}s '
                f'for removing {len_before - len(nms_ind)} elements of {len_before} elements'
            )
        return self

    @property
    def contours(self) -> List[torch.Tensor]:
        """
        This function wraps the openCV.findContours function, and uses openCV.contourArea to select the largest contour for each mask.
        """
        if self.PREFER_POLYGONS:
            return self.polygons
        else:
            return [
                self.contour_to_image_coordinates(find_contours(create_contour_mask(mask), largest_only=True, simplify=False)) 
                for mask in self.masks.data
            ]

    @contours.setter
    def contours(
            self, 
            value : List[torch.Tensor]
        ):
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
                        contour=value[i], 
                        scale=contour_scaling,
                        expand_by_one=True
                    )
                ).long().to(self.device)
            self.polygons = value
            self.masks = [torch.empty((0, 0), device=self.device, dtype=self.dtype) for _ in range(len(value))]  # Initialize empty masks
        else:
            self.masks = contours_to_masks(value, self.mask_height, self.mask_width).to(self.device)

    def contour_to_image_coordinates(
            self, 
            contour: torch.Tensor, 
            scale: float = 1
        ) -> torch.Tensor:
        """
        Converts a contour from mask coordinates to image coordinates. 

        Args:
            contour (`torch.Tensor`): The contour to convert.
            scale (`float`, optional): The scale factor to apply to the contour. Defaults to 1.

        Returns:
            `torch.Tensor`: The contour in image coordinates.
        """
        image_h, image_w = self.image.shape[1:]
        mask_to_image_scale = [(image_h - 1) / (self.mask_height - 1), (image_w - 1) / (self.mask_width - 1)]
        mask_to_image_scale = torch.tensor(mask_to_image_scale, device=self.device, dtype=torch.float32) * scale
        scaled_contour = scale_contour(contour.cpu().numpy(), mask_to_image_scale.cpu().numpy(), True)
        scaled_contour = simplify_contour(scaled_contour, (mask_to_image_scale / 2).mean().item())
        scaled_contour = torch.tensor(scaled_contour, device=self.device, dtype=torch.long).squeeze(1)

        return scaled_contour

    def flip(
            self, 
            direction : str="vertical"
        ) -> Self:
        """
        Flips the masks, polygons and boxes along the specified axis.

        Args:
            direction (`str`, optional): The axis to flip the masks, polygons and boxes along.
                Defaults to "vertical". Should be one of "vertical", "y", "horizontal" or "x".

        Returns:
            Self: The `TensorPredictions` object with the masks, polygons and boxes flipped.
        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        if direction == "vertical" or direction == "y":
            if self.masks.data.dim() == 3:
                self.masks.data = torch.flip(self.masks.data, [1])
            self.boxes[:, 1] = self.image.shape[1] - self.boxes[:, 1]
            self.boxes[:, 3] = self.image.shape[1] - self.boxes[:, 3]
            for i in range(len(self)):
                self.polygons[i][:, 1] = self.image.shape[1] - self.polygons[i][:, 1]
        elif direction == "horizontal" or direction == "x":
            if self.masks.data.dim() == 3:
                self.masks.data = torch.flip(self.masks.data, [2])
            self.boxes[:, 0] = self.image.shape[2] - self.boxes[:, 0]
            self.boxes[:, 2] = self.image.shape[2] - self.boxes[:, 2]
            for i in range(len(self)):
                self.polygons[i][:, 0] = self.image.shape[2] - self.polygons[i][:, 0]
        else:
            raise RuntimeError(f"Unknown direction `{direction}` - should be 'vertical'/'y' or 'horizontal'/'x'")

        if self.time:
            end.record()
            torch.cuda.synchronize()
            logger.info(f'Flipping masks, polygons and boxes {direction} took {start.elapsed_time(end) / 1000:.3f} s')

        return self

    def __len__(self) -> int:
        return len(self.polygons)

    def new(self):
        return TensorPredictions([], **{k: self.__dict__[k] for k in self.CONSTANTS if k in self.__dict__})

    def __getitem__(self, i):
        """
        Flexible indexing for TensorPredictions. Can be used to get a single element, a slice, or an iterable of indices (e.g. a list, tuple, tensor).
        """
        new_tp = self.new()
        for k, v in self.__dict__.items():
            if not k in self.CONSTANTS:
                # Check if 'i' is a slice
                if isinstance(i, slice):
                    new_value = v[i]
                # Check if 'i' is an iterable
                elif hasattr(i, "__iter__"):
                    if isinstance(i, torch.Tensor):
                        i = i.float().round().long().tolist()  # Just to be super safe we cast to float, then round, then cast to long, then to list
                    assert all([isinstance(j, int) for j in i]) or all([isinstance(j, float) and (j % 1) == 0 for j in i]), \
                        RuntimeError(f"Unknown type or non-integer float for {i}: {type(i)}")
                    i = [int(j) for j in i]
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
                            raise RuntimeError(f"Unknown type for {k}: {type(v)} does not support flexible indexing") from e
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
    
    def plot(self, *args, **kwargs):
        return self._plot_jpeg(*args, **kwargs)

    def _plot_svg(self, linewidth=2, masks=True, boxes=True, conf=True, outpath=None, scale=1):
        image = self.image.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        embed_jpeg = True
        if scale != 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        tmp_svg = tempfile.mktemp(suffix='.svg')

        try:
            height, width = image.shape[0:2]

            encoded_string = base64.b64encode(cv2.imencode('.jpg', image)[1])
            desc = ''

            with open(tmp_svg, 'w+') as f:
                f.write('<svg width="' + str(width) + '"' +
                        ' height="' + str(height) + '"' +
                        ' xmlns:xlink="http://www.w3.org/1999/xlink"' +
                        ' xmlns="http://www.w3.org/2000/svg"' +
                        ' >')
                #     f.write('<metadata  id="sticky_pi"> "%s" </metadata>' % str(self.metadata()))
                if embed_jpeg:
                    f.write('<image %s width="%i" height="%i" x="0" y="0" xlink:href="data:image/jpeg;base64,%s"/>' % \
                            (desc, width, height, str(encoded_string, 'utf-8')))

                for c, conf in zip(self.contours, self.confs):
                    f.write(self._contour_to_svg_element(c, scale=scale, confidence=conf))
                f.write('</svg>')

            if outpath:
                shutil.move(tmp_svg, outpath)

        except Exception as e:
            os.remove(tmp_svg)
            raise e

    def _contour_to_svg_element(
            self, 
            contour : Union["torch.Tensor", Any], 
            confidence : float, 
            scale : float=1.0
        ) -> str:
        d_list = []

        value = confidence
        stroke_colour = '#ff0000'
        fill_colour ='#0000ff'
        fill_opacity = 0.3
        for i in range(len(contour)):
            name=i
            x, y = contour[i][0] * scale
            d_list.append(str(x) + ',' + str(y))
        d_str = ' '.join(d_list)
        out = '<path name="%s" value="%i" style="stroke:%s;stroke-opacity:1;fill:%s;fill-opacity:%f" d="M%s Z"/>' % \
              (name, value, stroke_colour, fill_colour, fill_opacity, d_str)
        return out

    def _plot_jpeg(
            self, 
            linewidth : int=2, 
            masks : bool=True, 
            boxes : bool=True, 
            conf : bool=True, 
            outpath : Optional[str]=None, 
            scale : float=1
        ) -> Optional[cv2.UMat]:
        # Convert torch tensor to numpy array
        image = torchvision.transforms.ConvertImageDtype(torch.uint8)(self.image).permute(1, 2, 0).cpu().numpy()
        image : cv2.UMat = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if scale != 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        if len(self) > 0:
            # Draw masks
            if masks:
                contours = [simplify_contour((c * scale).round().to(torch.int32).cpu().numpy(), scale / 2) for c in self.contours]
                ih, iw = image.shape[:2]
                ALPHA = 0.3
                _alpha = int(255 * ALPHA)

                poly_alpha = np.zeros((ih, iw, 1), dtype=np.int32)
                for i, c in enumerate(contours):
                    this_poly_alpha = np.zeros((ih, iw, 1), dtype=np.uint8)
                    cv2.drawContours(this_poly_alpha, [c], -1, 1, -1)
                    poly_alpha += this_poly_alpha * _alpha
                poly_alpha = poly_alpha.clip(0, 255) / 255
                
                # Create a red fill for the polygons
                poly_fill = np.zeros_like(image)
                poly_fill[:, :, 2] = 255
                # Add the polygons to the image by blending the fill and the image using the alpha mask
                image = (image.astype(np.float32) * (1 - poly_alpha) + poly_fill * poly_alpha).round().astype(np.uint8)
                # Draw the contours
                for i, c in enumerate(contours):
                    cv2.drawContours(image, [c], -1, (0, 0, 255), linewidth)

            # Draw boxes and confidences
            if boxes:
                for box, conf in zip(self.boxes, self.confs):
                    box = (box * scale)
                    box[:2] = box[:2].floor()
                    box[2:] = box[2:].ceil()
                    box = box.long()
                    start_point = (int(box[0]), int(box[1]))
                    end_point = (int(box[2]), int(box[3]))
                    cv2.rectangle(image, start_point, end_point, (0, 0, 0), linewidth)  # Red box
                    if conf:
                        # Get the width and height of the text
                        (text_width, text_height), _ = cv2.getTextSize(
                            f"{conf * 100:.3g}%",
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1 * scale,
                            thickness=max(1, round(2 * scale))
                        )
                        # Calculate the text position
                        xp, yp = start_point[0], start_point[1] - text_height // 2
                        if yp < text_height:
                            yp = end_point[1] + text_height + 4 * linewidth
                        # Get the average color intensity behind the text
                        avg_color = np.mean(image[yp:yp + text_height, xp:xp + text_width])
                        # Draw the text
                        cv2.putText(
                            img=image,
                            text=f"{conf * 100:.3g}%",
                            org=(xp, yp), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1 * scale,
                            color=(0, 0, 0) if avg_color > 127 else (255, 255, 255),
                            thickness=max(1, round(2 * scale))
                        )

        # Save or show the image
        if outpath:
            cv2.imwrite(outpath, image)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    @property
    def crops(self) -> List[torch.Tensor]:
        return [self.image[:, y1:y2, x1:x2] for x1, y1, x2, y2 in self.boxes.long().tolist()]

    @property
    def crop_masks(self) -> List[torch.Tensor]:
        if self.PREFER_POLYGONS:
            return [
                contours_to_masks([contour.round().long() - box[:2]], box[3] - box[1], box[2] - box[0])
                for contour, box in zip(self.contours, self.boxes.long())
            ]
        else:
            return [
                resize_mask(mask, self.image.shape[1:])[box[1]:box[3], box[0]:box[2]]
                for mask, box in zip(self.masks, self.boxes.long())
            ]
        
    def _save_1_crop(
            self,
            crop : torch.Tensor,
            mask : Union[torch.Tensor, None],
            path : str,
        ) -> str:
        Image.fromarray(
            obj=chw2hwc_uint8(crop, mask).detach().cpu().numpy(),
            mode="RGB" if mask is None else "RGBA"
        ).save(path)
        return path

    def save_crops(
            self : Self, 
            outdir : str, 
            basename : Optional[str]=None, 
            mask : bool=False, 
            identifier : str=None
        ) -> List[Union[str, torch.Tensor]]:
        if outdir is None or not os.path.exists(outdir) or not os.path.isdir(outdir):
            raise RuntimeError(f"Invalid outdir {outdir}, does not exist or is not a directory")
        if basename is None:
            assert self.image_path is not None, RuntimeError("Cannot save crops without image_path")
            basename, _ = os.path.splitext(os.path.basename(self.image_path))
        _, image_ext = os.path.splitext(os.path.basename(self.image_path))
        if mask:
            image_ext = ".png"
        if identifier is None:
            identifier = "NONE"
        
        crops = self.crops
        if mask:
            crop_masks = self.crop_masks
        else:
            crop_masks = [None] * len(crops)
        crop_paths = [os.path.join(outdir, f"crop_{basename}_CROPNUMBER_{i}_UUID_{identifier}{image_ext}") for i in range(len(crops))]

        return [self._save_1_crop(crop, mask, path) for crop, mask, path in zip(crops, crop_masks, crop_paths)]
        
    
    @property
    def json_data(self):
        ## Clean up the data
        # 1. Convert the boxes to list
        boxes = self.boxes.cpu().tolist()
        # 2. Convert the masks to contours as lists 
        contours = [c.T.cpu().tolist() for c in self.contours]
        # 3. Convert the confidences to floats in a list
        confs = self.confs.float().cpu().tolist()
        # 4. Convert the classes to integers in a list
        classes = self.classes.cpu().long().tolist()
        # 5. Get the scales (already floats in a list)
        scales = self.scales
        return {
            "boxes": boxes,
            "contours": contours,
            "confs": confs,
            "classes": classes,
            "scales": scales,
            "image_path": self.image_path,
            "image_width": self.image.shape[2],
            "image_height": self.image.shape[1],
            "mask_width": self.image.shape[2] if self.PREFER_POLYGONS else self.masks.data.shape[2],
            "mask_height": self.image.shape[1] if self.PREFER_POLYGONS else self.masks.data.shape[1],
            "identifier": None
        }

    def serialize(
            self, 
            outpath: str = None, 
            save_json: bool = True, 
            save_pt: bool = False, 
            identifier: str = None
        ) -> None:
        """
        This function serializes the `TensorPredictions` object to a .pt file and/or a .json file. 
        The .pt file contains an exact copy of the `TensorPredictions` object, while the .json file contains the data in a more human-readable format, 
        which can be deserialized into a `TensorPredictions` object using the 'load' function.
        
        Args:
            outpath (str, optional): The path to save the serialized data to. Defaults to None.
            save_json (bool, optional): Whether to save the .json file. Defaults to True. Recommended.
            save_pt (bool, optional): Whether to save the .pt file. Defaults to False. Rather disk space wasteful.
            identifier (str, optional): An identifier for the serialized data. Defaults to None.
        """
        assert outpath is not None, RuntimeError("Cannot serialize without outpath")
        assert len(outpath) > 0, RuntimeError("Cannot serialize with empty outpath")
        assert os.path.exists(os.path.dirname(outpath)), RuntimeError(f"Invalide outpath {outpath}, directory does not exist")

        # Check for file-extension on the outpath, it should have none - not really necessary anymore due to the check for directory above
        outpath, ext = os.path.splitext(outpath)
        if ext != "" and len(ext) < 5:
            logger.warning(f"serializer outpath ({outpath}) should not have a file-extension for 'TensorPredictions.serialize'!")
        else:
            outpath = f"{outpath}{ext}"

        # Add the basename to the outpath
        pt_path = f'{outpath}.pt'
        json_path = f'{outpath}.json'

        if save_pt:
            if os.path.exists(pt_path):
                logger.warning(f"Pickle ({pt_path}) already exists, overwriting!")
            ### First serialize as .pt file
            torch.save(self, pt_path)

        if save_json:
            if os.path.exists(json_path):
                logger.warning(f"JSON ({json_path}) already exists, overwriting!")
            json_data = self.json_data
            json_data["identifier"] = identifier if identifier else self.image_path,
            with open(json_path, 'w') as f:
                json.dump(json_data, f)

    def load(
            self, 
            path: str, 
            device : Optional[DeviceLikeType]=None, 
            dtype : Optional[torch.types._dtype]=None
        ) -> Self:
        """
        Deserializes a TensorPredictions object from a .pt or .json file. OBS: Mutates and returns the current object.

        Args:
            path (str): The path to the file to load.
            device (Optional[DeviceLikeType], optional): The device to load the data to. Defaults to None. If None, the device is set to "cpu".
            dtype (Optional[torch.types._dtype], optional): The data type to load the data as. Defaults to None. If None, the data type is set to torch.float32.

        Returns:
            Self: This object with the deserialized data.
        """
        assert isinstance(path, str) and os.path.isfile(path), RuntimeError(f"Invalid path: {path}")
        # Check whether the path is a .pt file or a .json file
        _, ext = os.path.splitext(path)
        if ext == ".pt":
            # When loading from .pt we get an exact copy of the saved TensorPredictions object
            self = torch.load(path)
            return self
        elif ext == ".json":
            with open(path, 'r') as f:
                json_data = json.load(f)

            if device is None:
                device = torch.device("cpu")
            if dtype is None:
                dtype = torch.float32

            empty_image = torch.zeros((3, json_data["image_height"], json_data["image_width"]), device=device, dtype=dtype) + 255
            self.__init__(image=empty_image, device=device, dtype=dtype)
            setattr(self, "PREFER_POLYGONS", True) # Since we only store contours in the .json file, we prefer polygons on loading

            # Load constants
            for k, v in json_data.items():
                if k in self.CONSTANTS:
                    setattr(self, k, v)

            # Load the data
            for k, v in json_data.items():
                # Skip constants in second round
                if k in self.CONSTANTS:
                    continue
                # Skip the identifier 
                if k in ["identifier", "image_height", "image_width"]:
                    continue                
                # Catch attributes that don't need special treatment
                elif k in ["scales", "contours"]:
                    pass
                # Bounding boxes are easy (as usual)
                elif k == "boxes":
                    v = torch.tensor(v, device=self.device, dtype=self.dtype)
                # While masks are a bit more complicated
                # Confidences and classes are 1-d tensors (arrays)
                elif k in ["confs", "classes"]:
                    v = torch.tensor(v, device=self.device, dtype=self.dtype)
                else:
                    raise RuntimeError(f"Unknown key in json file: {k}")
                setattr(self, k, v)

            return self
        else:
            raise RuntimeError(f"Unknown file-extension: {ext} for path: {path}")

    def save(
            self, 
            output_directory: str, 
            overview: Union[bool, str]=True,
            crops: Union[bool, str]=True,
            metadata: Union[bool, str]=True,
            fast: bool=False,
            mask_crops: bool=False,
            identifier: Optional[str]=None, 
            basename: Optional[str]=None
        ) -> Optional[str]:
        """
        Saves the serialized prediction results, crops, and overview to the given output directory.

        TODO: Add the identifier to the names of the files, so that we can save multiple predictions for the same image or images with the same name.

        Args:
            output_directory (`str`): The directory to save the prediction results to.
            overview (`bool | str`, optional): Whether to save the overview image. Defaults to True. 
                If a string is given, it is interpreted as a path to a directory to save the overview image to.
            crops (`bool | str`, optional): Whether to save the crops. Defaults to True. 
                If a string is given, it is interpreted as a path to a directory to save the crops to.
            metadata (`bool | str`, optional): Whether to save the metadata. Defaults to True. 
                If a string is given, it is interpreted as a path to a directory to save the metadata to.
            fast (`bool`, optional): Whether to use the fast version of the overview image. Defaults to False. 
                Saves the overview image at half the resolution.
            mask_crops (`bool`, optional): Whether to mask the crops. Defaults to False.
            identifier (`str | None`, optional): An identifier for the serialized data. Defaults to None.
            basename (`str | None`, optional): The base name of the image. Defaults to None. 
                If None, the base name is extracted from the image path.
        
        Returns:
            `str`: The path to the directory containing the serialized data - the crops and overview image(s) are also saved here by default. \\
                If the standard location is not used at all, the directory is not created and None is returned instead.
        """
        if not os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory} does not exist")

        if basename is None:
            # Get the base name of the image
            basename = os.path.splitext(os.path.basename(self.image_path))[0]
        # Construct the prediction directory path
        prediction_directory = os.path.join(output_directory, basename)
        # Create the prediction directory if it does not exist and it is needed (i.e. if we are saving crops, overview, or metadata to a standard location)
        prediction_directory_is_used = (overview is True) or (crops is True) or (metadata is True)
        if prediction_directory_is_used:
            if not os.path.exists(prediction_directory):
                os.makedirs(prediction_directory)
        else:
            # If the prediction directory is not used set it to None
            prediction_directory = None

        # Save overview
        if overview:
            # Check if the overview path is overwritten and make sure the directory exists and is a directory
            if not isinstance(overview, str):
                overview_directory = prediction_directory
            else:
                if not os.path.exists(overview):
                    os.makedirs(overview)
                overview_directory = overview
            assert os.path.isdir(overview_directory), RuntimeError(f"Invalid path for overview: {overview_directory}")
            # The overview path is then constructed as a .jpg file in the overview directory with the name overview_{base_name}.jpg
            overview_path = os.path.join(overview_directory, f"overview_{basename}_UUID_{identifier}.jpg")
            # Save the overview image to the overview path
            if not fast:
                self.plot(outpath=overview_path, linewidth=2, scale=1)
            else:
                max_dim = max(self.image.shape[1:])
                fast_scale = min(1 / 2, 3072 / max_dim)
                self.plot(outpath=overview_path, linewidth=1, scale=fast_scale)

        # Save crops
        if crops:
            # Check if the crops path is overwritten and make sure the directory exists and is a directory
            if not isinstance(crops, str):
                crop_directory = os.path.join(prediction_directory, "crops")
            else:
                crop_directory = crops
            if not os.path.exists(crop_directory):
                os.makedirs(crop_directory)
            assert os.path.isdir(crop_directory), RuntimeError(f"Invalid path for crops: {crop_directory}")
            # Save the crops to the crops path
            self.save_crops(crop_directory, basename=basename, mask=mask_crops, identifier=identifier)

        # Save json
        if metadata:
            # Check if the metadata path is overwritten and make sure the directory exists and is a directory
            if not isinstance(metadata, str):
                metadata_directory = prediction_directory
            else:
                if not os.path.exists(metadata):
                    os.makedirs(metadata)
                metadata_directory = metadata
            assert os.path.isdir(metadata_directory), RuntimeError(f"Invalid path for metadata: {metadata_directory}")
            # The metadata path is then constructed as a .json file in the metadata directory with the name metadata_{base_name}_id_{identifier}.<EXT>
            metadata_path = os.path.join(metadata_directory, f'metadata_{basename}_UUID_{identifier}')
            # Serialize the data to the metadata path
            self.serialize(outpath=metadata_path, identifier=identifier)

        return prediction_directory

def _process_batch(
            image : torch.Tensor, 
            offsets : List[Tuple[Tuple[int, int], Tuple[int, int]]], 
            tile_size : int, 
            batch_start_idx : int, 
            batch_size : int,
            device : Optional[DeviceLikeType] = None,
            model : torch.nn.Module = None,
            time : bool = False,
            **kwargs : Any # Swallow any extra arguments
    ) -> Tuple[int, torch.Tensor, List]:
    if time:
        # Initialize batch timing calculations
        start_batch_event = torch.cuda.Event(enable_timing=True)
        end_fetch_event = torch.cuda.Event(enable_timing=True)
        end_forward_event = torch.cuda.Event(enable_timing=True)
        end_batch_event = torch.cuda.Event(enable_timing=True)
        current_device_stream = torch.cuda.current_stream(device=device)
        # Record batch start
        start_batch_event.record(current_device_stream)
    
    # Get the offsets for the current batch and extract and stack the corresponding tiles
    batch = torch.stack([
                image[:, o[0]: (o[0] + tile_size), o[1]: (o[1] + tile_size)] 
                for (m, n), o in offsets[batch_start_idx:min((batch_start_idx + batch_size), len(offsets))]
            ], dim=0)
    if time:
        # Record end of fetch
        end_fetch_event.record(current_device_stream)
    # Forward pass the model on the batch tiles
    with torch.no_grad():
        batch_outputs = model(batch)
    if time:
        # Record end of forward
        end_forward_event.record(current_device_stream)
        # Record batch end
        end_batch_event.record(current_device_stream)

        # Calculate timing
        torch.cuda.synchronize(device=device)
        batch_time = start_batch_event.elapsed_time(end_batch_event) / 1000  # Convert to seconds
        fetch_time = start_batch_event.elapsed_time(end_fetch_event) / 1000  # Convert to seconds
        forward_time = end_fetch_event.elapsed_time(end_forward_event) / 1000  # Convert to seconds
        # loggger.info(
        #   f'Batch time: {batch_time:.3f}s,'
        #   f' fetch time: {fetch_time:.3f}s,'
        #   f' forward time: {forward_time:.3f}s'
        # )
    # Return the postprocessed batch outputs and optionally the timing
    if time:
        return batch, batch_outputs, (batch_time, fetch_time, forward_time)
    else:
        return batch, batch_outputs, None

class Predictor(object):
    HYPERPARAMETERS : List[str] = CFG_PARAMS
    """
    The available hyperparameters for the predictor. \\
    These can be set using the `set_hyperparameters` class method.
    """

    # Hyperparameters, set to None so they are visible in the class
    MIN_MAX_OBJ_SIZE : Tuple[int, int] = None
    """
    Defines the minimum and maximum object size as seen in a single tile. \\
    Size is defined as the square root of the pixel area of the bounding box.
    """
    MAX_MASK_SIZE : int = None
    """
    Defines the maximum size of the segmentation masks. \\
    Only applies if PREFER_POLYGONS is False.
    """
    SCORE_THRESHOLD : float = None
    """
    The score threshold for the predictions. \\
    TODO: This should be called CONFIDENCE_THRESHOLD.
    """
    IOU_THRESHOLD: float = None
    """
    The IOU threshold used to determine if two instances are duplicates. \\
    """
    MINIMUM_TILE_OVERLAP : int = None
    """
    The minimum - but not necessarily the maximum - overlap between tiles \\
    in a single layer of the pyramid. Increasing this value will increase \\
    the computation time, but may improve the detection of large instances. 
    """
    EDGE_CASE_MARGIN : int = None
    """
    The margin to add to the edge of the image to catch instances that are \\
    split between tiles. The margin is added to the edge of the image, such \\
    that instances on the true edge of the images are not removed.
    """
    PREFER_POLYGONS : bool = None
    """
    Whether to prefer representing the instance segmentation using polygons \\
    instead of masks. This is a much more compact representation, but cannot \\
    represent complex shapes (like holes in the mask), only concave polygons.
    """
    EXPERIMENTAL_NMS_OPTIMIZATION : bool = None
    """
    Enables an experimental optimization for the NMS step. \\
    This optimization improves the performance of the NMS step when there are \\
    many instances in a large image and CUDA is available.
    """
    TIME : bool = None
    """
    Whether to time the different parts of the prediction process. \\
    Enabling this will print a verbose output of the timing of the different \\
    parts of the prediction process.
    """
    TILE_SIZE                       = None
    """
    The size of the tiles to split the image into. \\
    This is defined by the model and should probably not be changed.
    """
    BATCH_SIZE                      = None
    """
    The batch size to use for the prediction. \\
    This determines how many tiles are processed in parallel. \\
    Increasing this value may improve performance, but will also increase memory usage.
    """

    # Enable debug mode, only for development
    DEBUG = False

    def __init__(
            self, 
            model : Union[str, pathlib.Path], 
            cfg : Optional[Union[dict, str, os.PathLike]]=None, 
            device : Union[str, torch.device, int, List[Union[str, torch.device, int]]]=torch.device("cpu"), 
            dtype : Union[torch.types._dtype, str]=torch.float32
        ):
        if cfg is None:
            cfg = DEFAULT_CFG
        if isinstance(cfg, (str, os.PathLike)):
            cfg = read_cfg(cfg, strict=True)
        self.set_hyperparameters(**cfg)

        self._multi_gpu = isinstance(device, (list, tuple))
        self._devices = [torch.device(device)] if not self._multi_gpu else [torch.device(d) for d in device]
        if len(self._devices) > 1:
            raise NotImplementedError("Multi-GPU is not implemented yet")
        self._device = self._devices[0]
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
            raise ValueError(f"Dtype '{dtype}' is not supported.")
        self._dtype = dtype

        if isinstance(model, str):
            yolo = YOLO(model, "segment", verbose=True)
            pred = yolo._smart_load("predictor")
            class dict2attr:
                def __init__(self, d):
                    self.__dict__ = d
            args = dict2attr({
                "device": self._device,
                "half": self._dtype == torch.float16,
                "batch": self.BATCH_SIZE,
                "model": yolo.model,
                "fp16" : self._dtype == torch.float16,
                "dnn" : False,
                "data" : None # If we want to support multiclass inference, this needs to point to "Path to the additional data.yaml file containing class names. Optional." 
                              # see: https://github.com/ultralytics/ultralytics/blob/bc9fd45cdf10ebe8009037aaf8def2353761c9ed/ultralytics/nn/autobackend.py#L53
            })
            pred.args = args
            pred.setup_model(self=pred, model=yolo.model, verbose=True)
            self._model = pred.model
            self._model.to(self._device, dtype=self._dtype)
            self._model.eval()
        elif isinstance(model, torch.nn.Module):
            self._model = model
        else:
            raise RuntimeError(f"Unknown model type: {type(model)}")
        self._model = self._model.to(self._device, dtype=self._dtype)
        self._model.eval()

        self._yolo_predictor = None

    def set_hyperparameters(self, **kwargs) -> Self:
        """
        Mutably set the hyperparameters for the predictor. 

        Args:
            **kwargs: The hyperparameters to set.

        Returns:
            Self: This object (mutated with the new hyperparameters).
        """
        for k, v in kwargs.items():
            if k in self.HYPERPARAMETERS:
                setattr(self, k, v)
            else:
                raise ValueError(f"Unknown hyperparameter: {k}")
        return self
    
    def _detect_instances(
            self,
            image : torch.Tensor,
            scale : float=1.0,
            max_scale : bool = False
        ) -> Prepared_Results:
        TILE_SIZE = self.TILE_SIZE
        this_MIN_MAX_OBJ_SIZE = list(self.MIN_MAX_OBJ_SIZE)
        this_EDGE_CASE_MARGIN = self.EDGE_CASE_MARGIN
        # If we are at the top level, we don't want to remove large instances - since there are no layers above to detect them as small instances
        if max_scale:
            this_MIN_MAX_OBJ_SIZE[1] = 1e9
            this_EDGE_CASE_MARGIN = 0

        if self.TIME:
            # Initialize timing calculations
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
            image = torch.nn.functional.pad(image, pad_lrtb, mode="constant", value=0) # Pad with black
            h, w = image.shape[1:]

        # Tile calculation
        x_n_tiles = math.ceil(w / (TILE_SIZE - self.MINIMUM_TILE_OVERLAP)) if w != TILE_SIZE else 1
        y_n_tiles = math.ceil(h / (TILE_SIZE - self.MINIMUM_TILE_OVERLAP)) if h != TILE_SIZE else 1

        x_stride = TILE_SIZE - math.floor((TILE_SIZE * (x_n_tiles + 0) - w) / x_n_tiles) if x_n_tiles > 1 else TILE_SIZE
        y_stride = TILE_SIZE - math.floor((TILE_SIZE * (y_n_tiles + 0) - h) / y_n_tiles) if y_n_tiles > 1 else TILE_SIZE
        x_stride -= x_stride % 4
        y_stride -= y_stride % 4

        x_range = [i if (i + TILE_SIZE) < w else (w - TILE_SIZE - w % 4) for i in range(0, x_stride * x_n_tiles, x_stride)]
        y_range = [i if (i + TILE_SIZE) < h else (h - TILE_SIZE - h % 4) for i in range(0, y_stride * y_n_tiles, y_stride)]

        offsets = [((m, n), (j, i)) for n, j in enumerate(y_range) for m, i in enumerate(x_range)]

        hyperparams = {
            "image" : image,
            "batch_size" : self.BATCH_SIZE,
            "tile_size" : TILE_SIZE,
            "edge_case_margin" : this_EDGE_CASE_MARGIN,
            "score_threshold" : self.SCORE_THRESHOLD,
            "iou_threshold" : self.IOU_THRESHOLD,
            "min_max_object_size" : this_MIN_MAX_OBJ_SIZE,
            "time" : self.TIME
        }

        if self.TIME:
            # Initialize timing calculations
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            batch_times, fetch_times, forward_times, postprocess_times = [], [], [], []
            start_event.record(main_stream)
        
        postprocessed_results = [None for _ in range(len(offsets))]
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
                    imgs = batch,
                    max_det = 1000,
                    min_confidence = self.SCORE_THRESHOLD,
                    iou_threshold = self.IOU_THRESHOLD,
                    nms = 3,
                    valid_size_range = self.MIN_MAX_OBJ_SIZE,
                    edge_margin = self.EDGE_CASE_MARGIN,
                )
                for batch_index in range(len(this_postprocessed_results)):
                    postprocessed_results[batch_start_idx + batch_index] = Results(**this_postprocessed_results[batch_index])
                if self.TIME:
                    batch_times.append(timing[0])
                    fetch_times.append(timing[1])
                    forward_times.append(timing[2])
                    postprocess_end.record(main_stream)
                    torch.cuda.synchronize(device = self._device)
                    postprocess_times.append(postprocess_start.elapsed_time(postprocess_end) / 1000)
            
        if self.TIME:
            # Finish timing calculations
            end_event.record(main_stream)
            torch.cuda.synchronize(device = self._device)
            total_elapsed = start_event.elapsed_time(end_event) / 1000  # Convert to seconds
            fetch_time, forward_time, postprocess_time = sum(fetch_times), sum(forward_times), sum(postprocess_times)
            total_batch_time = sum(batch_times) + postprocess_time
            overhead_prop = (total_elapsed - total_batch_time) / total_elapsed
            fetch_prop, forward_prop, postprocess_prop = fetch_time / total_batch_time, forward_time / total_batch_time, postprocess_time / total_batch_time

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

        ## Combine the results from the tiles
        MASK_SIZE = 256  # Defined by the YOLOv8 model segmentation architecture
        MASK_TO_IMG_RATIO = MASK_SIZE / torch.tensor([TILE_SIZE, TILE_SIZE], dtype=torch.float32, device=self._device).unsqueeze(0)
        # For the boxes, we can simply add the offsets (and possibly subtract the padding)
        box_offsetters = torch.tensor([[o[1][0] - pad_lrtb[2], o[1][1] - pad_lrtb[0]] for o in offsets], dtype=torch.float32, device=self._device)
        # However for the masks, we need to create a new mask which can contain every tile, and then add the masks from each tile to the correct area - this will of course use some memory, but it's probably not too bad
        # Since the masks do not have the same size as the tiles, we need to scale the offsets
        mask_offsetters = box_offsetters * MASK_TO_IMG_RATIO
        # We also need to round the offsets, since they may not line up with the pixel-grid - RE: Now they do since I made sure the offsets are multiples of 4
        mask_offsetters = torch.round(mask_offsetters).long()
        # The padding must also be scaled and subtracted from the new mask size
        new_mask_size = (
            (mask_offsetters.max(dim=0).values + MASK_SIZE) - 
            torch.tensor(pad_lrtb[1::2][::-1], dtype=torch.long, device=self._device) * MASK_TO_IMG_RATIO[0]
        ).tolist()
        # Finally, we can merge the results - this function basically just does what I described above
        orig_img = image[:, pad_lrtb[2]:(-pad_lrtb[3] if pad_lrtb[3] != 0 else None),
                   pad_lrtb[0]:(-pad_lrtb[1] if pad_lrtb[1] != 0 else None)] if padded else image
        postprocessed_results = merge_tile_results(
            results = postprocessed_results, 
            orig_img = orig_img.permute(1, 2, 0), 
            box_offsetters = box_offsetters.to(self._dtype),
            mask_offsetters = mask_offsetters, 
            new_shape = new_mask_size,
            clamp_boxes = (h - sum(pad_lrtb[2:]), w - sum(pad_lrtb[:2])),
            max_mask_size = self.MAX_MASK_SIZE, 
            exclude_masks = self.PREFER_POLYGONS
        )

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

        if self.TIME:
            end_detect.record(main_stream)
            torch.cuda.synchronize(device=self._device)
            total_detect_time = start_detect.elapsed_time(end_detect) / 1000  # Convert to seconds
            pred_prop = total_elapsed / total_detect_time
            logger.info(
                f'Prediction time: {total_elapsed:.3f}s/{pred_prop * 100:.3g}%'
                f' (overhead: {overhead_prop * 100:.1f}) |'
                f' Fetch {fetch_prop * 100:.1f}% |'
                f' Forward {forward_prop * 100:.1f}% |'
                f' Postprocess {postprocess_prop * 100:.1f}%)'
            )
            if hasattr(self, "total_detection_time"):
                self.total_detection_time += total_detect_time
            if hasattr(self, "total_forward_time"):
                self.total_forward_time += forward_time
        return Prepared_Results(
            predictions = postprocessed_results, 
            scale = real_scale, 
            device = self._device, 
            dtype = self._dtype
        )

    def pyramid_predictions(
            self, 
            image : Union[torch.Tensor, str], 
            path : Optional[str]=None, 
            scale_increment : float=2/3, 
            scale_before : Union[float, int]=1, 
            single_scale : bool=False
        ) -> TensorPredictions:
        """
        Performs inference on an image at multiple scales and returns the predictions.
        
        Args:
            image (Union[torch.Tensor, str]): The image to run inference on. If a string is given, the image is read from the path.
                If it is a `torch.Tensor`, the path must be provided. \\
                We assume that floating point images are in the range [0, 1] and integer images are in the range [0, integer_type_max]. \\
                (see https://github.com/pytorch/vision/blob/6d7851bd5e2bedc294e40e90532f0e375fcfee04/torchvision/transforms/_functional_tensor.py#L66)
            path (Optional[str], optional): The path to the image. Defaults to None. Must be provided if `image` is a `torch.Tensor`.
            scale_increment (float, optional): The scale increment to use when resizing the image. Defaults to 2/3.
            scale_before (Union[float, int], optional): The scale to apply before running inference. Defaults to 1.
            single_scale (bool, optional): Whether to run inference on a single scale. Defaults to False.

        Returns:
            TensorPredictions: The predictions for the image.
        """
        if self.TIME:
            # Initialize timing calculations
            start_pyramid, end_pyramid = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_pyramid.record()

        if isinstance(image, str):
            path : str = image
            image : torch.Tensor = read_image(
                path=image, 
                mode=ImageReadMode.RGB, 
                apply_exif_orientation=True
            ).to(self._device)
        elif isinstance(image, torch.Tensor):
            assert path is not None, ValueError("Path must be provided if image is a tensor")
        else:
            raise TypeError(f"Unknown type for image: {type(image)}, expected str or torch.Tensor")

        c, h, w = image.shape
        transform_list = []
        # Check if the image has an integer data type
        if image.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            transform_list.append(transforms.ConvertImageDtype(self._dtype))

        if scale_before != 1:
            w, h = int(w * scale_before), int(h * scale_before)
            resize = transforms.Resize((h, w), antialias=True)
            transform_list.append(resize)

        # A border is always added now, to avoid edge-cases on the actual edge of the image. I.e. only detections on internal edges of tiles should be removed, not detections on the edge of the image.
        edge_case_margin_padding_multiplier = 2
        padding_offset = torch.tensor((self.EDGE_CASE_MARGIN, self.EDGE_CASE_MARGIN), dtype=self._dtype, device=self._device) * edge_case_margin_padding_multiplier
        if padding_offset.sum() > 0:
            # padding_for_edge_cases = transforms.Pad(
            #     padding=self.EDGE_CASE_MARGIN * edge_case_margin_padding_multiplier, 
            #     fill=0.5,
            #     padding_mode='constant'
            # )
            padding_for_edge_cases = InpaintPad(padding=self.EDGE_CASE_MARGIN * edge_case_margin_padding_multiplier)
            transform_list.append(padding_for_edge_cases)
        else:
            padding_offset[:] = 0
        if transform_list:
            transforms_composed = transforms.Compose(transform_list)

        transformed_image = transforms_composed(image) if transform_list else image

        # Check correct dimensions
        assert len(transformed_image.shape) == 3, RuntimeError(f"transformed_image.shape {transformed_image.shape} != 3") 
        # Check correct number of channels
        assert transformed_image.shape[0] == 3, RuntimeError(f"transformed_image.shape[0] {transformed_image.shape[0]} != 3")

        max_dim = max(transformed_image.shape[1:])
        min_dim = min(transformed_image.shape[1:])

        # fixme, what to do if the image is too small? - RE: Fixed by adding padding in _detect_instances
        scales = []

        if single_scale:
            scales.append(min(1, 1024 / min_dim))
        else:
            s = 1024 / max_dim

            if s > 1:
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
        all_preds = [self._detect_instances(transformed_image, scale=s, max_scale=s == min(scales)) for s in reversed(scales)]

        if self.TIME:
            logger.info(
                f'Total detection time: {self.total_detection_time:.3f}s'
                f' ({self.total_forward_time / self.total_detection_time * 100:.3g}% forward)'
            )

        all_preds = TensorPredictions(
            predictions     = all_preds,
            image           = image,
            image_path      = path,
            dtype           = self._dtype,
            device          = self._device,
            time            = self.TIME,
            PREFER_POLYGONS = self.PREFER_POLYGONS
        ).offset_scale_pad(
            offset  = -padding_offset,
            scale   = 1 / scale_before,
            pad     = 5  # pad the boxes a bit to ensure they encapsulate the masks
        ).non_max_suppression(
            iou_threshold   = self.IOU_THRESHOLD,
            # metric        = 'IoU', # Currently only IoU is supported and setting this will raise an error
            group_first     = self.EXPERIMENTAL_NMS_OPTIMIZATION
        )

        if self.TIME:
            # Finish timing calculations
            end_pyramid.record()
            torch.cuda.synchronize()
            total_pyramid_time = start_pyramid.elapsed_time(end_pyramid) / 1000
            logger.info(
                f'Total pyramid time: {total_pyramid_time:.3f}s'
                f' ({self.total_detection_time / total_pyramid_time * 100:.3g}% detection |'
                f' {self.total_forward_time / total_pyramid_time * 100:.3g}% forward)'
            )

        return all_preds
