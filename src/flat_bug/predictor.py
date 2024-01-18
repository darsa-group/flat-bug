import copy
import os.path
import torch
import math
import logging
import numpy as np
import PIL.Image
import exiftool
import cv2
import ultralytics
import json
from shapely.geometry import Polygon
from PIL import Image
import io
from flat_bug.ml_utils import iou_match_pairs, iou
from flat_bug.yolo_helpers import *
from flat_bug.geometry_simples import find_contours, contours_to_masks, interpolate_contour, create_contour_mask
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from torchvision.io import read_image
import torchvision.transforms as transforms

from tqdm import tqdm

from typing import Union

# Class for containing the results from a single _detect_instances call
class Prepared_Results:
    def __init__(self, predictions, scale, device, dtype):
        self._predictions = predictions
        self._predictions.boxes.data[:, :4] /= scale
        self.scale = scale
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self._predictions)
    
    def __getitem__(self, i):
        return Prepared_Results(self._predictions[i], self.scale, self.device, self.dtype)
    
    # Properties for accessing the data
    @property
    def contours(self):
        return self._predictions.masks.xy
    @property
    def masks(self):
        return self._predictions.masks
    @property
    def boxes(self):
        return self._predictions.boxes.xyxy
    @property
    def confs(self):
        return self._predictions.boxes.conf
    @property
    def classes(self):
        # Currently this function is pretty redundant, since the localizer only has a single class. 
        # If there were more classes, the function should do some kind of argmax on self._predictions.boxes.cls (I assume these are class probabilities).
        return torch.ones_like(self._predictions.boxes.cls)

# Class for containing the results from multiple _detect_instances calls
class TensorPredictions:
    BOX_IS_EQUAL_MARGIN = 10 # How many pixels the boxes can differ by and still be considered equal? Used for removing duplicates before merging overlapping masks.
    CONSTANTS = ["device", "dtype", "image", "image_path", "time", "CONSTANTS", "BOX_IS_EQUAL"] # Attributes that should not be changed after initialization - should 'contours' be here?

    def __init__(self, predictions : Union[list[Prepared_Results], None]=None, image : [torch.Tensor, None]=None, image_path = Union[str, None], device=None, dtype=None, antialias=False, time=False):
        # Timing could probably be hidden in a decorator...
        self.time = time
        if self.time and len(predictions) > 0:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        no_predictions = predictions is None or len(predictions) == 0
        if not no_predictions:
        # Check that all predictions have the same device and dtype
            if device is None:
                device = predictions[0].device
            if dtype is None:
                dtype = predictions[0].dtype
            for pi, p in enumerate(predictions):
                assert p.device == device, RuntimeError(f"predictions[{pi}].device {p.device} != device {device}")
                assert p.dtype == dtype, RuntimeError(f"predictions[{pi}].dtype {p.dtype} != dtype {dtype}")
        else:
            # If there are no predictions, set device and dtype to defaults (cpu/float32)
            if device is None:
                device = torch.device("cpu")
            if dtype is None:
                dtype = torch.float32
        # Set attributes
        self.device = device
        self.dtype = dtype
        self.image = image
        self.image_path = image_path
        # Combine the predictions
        if not no_predictions:
            self._combine_predictions(predictions, antialias=antialias)
        else:
            # If there are no predictions, set other attributes to None
            self.masks, self.boxes, self.confs, self.classes, self.scales = None, None, None, None, None
        
        if self.time and len(predictions) > 0:
            end.record()
            torch.cuda.synchronize()
            print(f'Initializing TensorPredictions took {start.elapsed_time(end)/1000:.3f} s')

    def _combine_predictions(self, predictions : list[Prepared_Results], antialias=False):
        """
        Combines a list of Prepared_Results from multiple _detect_instances calls into a single TensorPredictions object.

        Args:
            predictions (list[Prepared_Results]): A list of Prepared_Results objects.
            offset (torch.Tensor): A vector of length 2 containing the x and y offset of the image.
            antialias (bool): Whether to antialias the masks when combining them. Defaults to False. WIP!

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
        self.boxes = torch.cat([p.boxes for p in predictions]) # Nx4
        self.confs = torch.cat([p.confs for p in predictions]) # N
        
        ## Duplicate removal ##
        # Calculate indices of non-duplicate boxes
        non_duplicates = detect_duplicate_boxes(self.boxes, self.confs, margin=self.BOX_IS_EQUAL_MARGIN, return_indices=True) 
        # Remove the duplicate boxes
        self.boxes = self.boxes[non_duplicates]
        self.confs = self.confs[non_duplicates]
        # Divide the indices into each prediction object
        n_detections = [len(p) for p in predictions]
        max_indices = cumsum(n_detections)
        non_duplicates_chunked = [non_duplicates[(non_duplicates < max_indices[i]) & (non_duplicates >= (max_indices[i-1] if i > 0 else 0))] - (max_indices[i] - n_detections[i]) for i in range(len(predictions))]

        if self.time:
            end_duplication_removal.record()

        # For the remaining attributes we remove the duplicates before combining them
        self.masks = stack_masks([p.masks.data[nd] for p, nd in zip(predictions, non_duplicates_chunked)], antialias=antialias) # NxMHxMW - MH and MW are proportional to the original image size

        if self.time:
            end_mask_combination.record()

        self.masks.orig_shape = self.image.shape[1:] # Set the target shape of the masks to the shape of the image passed to the TensorPredictions object
        self.classes = torch.cat([p.classes[nd] for p, nd in zip(predictions, non_duplicates_chunked)]) # N
        self.scales = [predictions[i].scale for i, p in enumerate(non_duplicates_chunked) for _ in range(len(p))] # N
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
            print(f'Combining {len(predictions)} predictions into a single TensorPredictions object took {total:.3f} s | Duplication removal: {duplication_removal:.3f} s | Mask combination: {mask_combination:.3f} s')

            # print(f'Combining predictions took {start.elapsed_time(end)/1000:.3f} s')

    def offset_scale_pad(self, offset : torch.Tensor, scale : float, pad : int = 0) -> "TensorPredictions":
        """
        Since the image may be padded, the masks and boxes should be offset by the padding-width and scaled by the scale_before factor to match the original image size.
        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        
        if any(offset > 0):
            raise NotImplementedError("Positive offsets are not implemented yet")
        # Boxes is easy
        self.boxes = offset_box(self.boxes, offset) # Add the offsets to the box-coordinates
        self.boxes[:, :4] = (self.boxes[:, :4] * scale).round() # Multiply the box-coordinates by the scale factor (round so it doesn't implicitly gets floored when cast to an integer later)
        # Pad the boxes a bit to be safe
        self.boxes[:, :2] -= pad
        self.boxes[:, 2:] += pad
        self.boxes = self.boxes.long()
        # Clamp the boxes to the image size
        self.boxes[:, 0:4:2] = self.boxes[:, 0:4:2].clamp(0, self.image.shape[2] - 1)
        self.boxes[:, 1:4:2] = self.boxes[:, 1:4:2].clamp(0, self.image.shape[1] - 1)

        # However masks are more complicated since they don't have the same size as the image
        image_shape = torch.tensor([self.image.shape[1], self.image.shape[2]], device=self.device, dtype=self.dtype) # Get the shape of the original image
        # Calculate the normalized offset (i.e. the offset as a fraction of the scaled and padded image size, here the scaled and padded image size is calculated from the original image shape, but it would probably be easier just to pass it...)
        offset_norm = -offset / (image_shape / scale - 2 * offset) 
        orig_mask_shape = torch.tensor([self.masks.shape[1], self.masks.shape[2]], device=self.device, dtype=self.dtype) # Get the shape of the masks
        # Convert the normalized offset to the coordinates of the masks
        offset_mask_coords = offset_norm * orig_mask_shape 
        # Round the coordinates to the nearest integer and convert to long (needed for indexing)
        offset_mask_coords = torch.round(offset_mask_coords).long()
        self.masks.data = self.masks.data[:, offset_mask_coords[0]:-offset_mask_coords[0], offset_mask_coords[1]:-offset_mask_coords[1]] # Slice out the padded parts of the masks
        
        if self.time:
            end.record()
            torch.cuda.synchronize()
            print(f'Offsetting, scaling and padding took {start.elapsed_time(end)/1000:.3f} s')
        
        return self
    
    def fix_boxes(self):
        """
        This function simply sets the boxes to match the masks.

        It is *not* needed, but can be used as a sanity check to see if the boxes match the masks.
        The discrepancy between the boxes and the masks is due to how the contours are calculated from the masks.
        """
        nonzero_indices = self.masks.data.nonzero()
        mask_size = torch.tensor([self.masks.data.shape[1], self.masks.data.shape[2]], device=self.device, dtype=self.dtype)
        image_size = torch.tensor([self.image.shape[1], self.image.shape[2]], device=self.device, dtype=self.dtype)
        mask_to_image_scale = image_size / mask_size
        for i in range(len(self)):
            this_mask_nz = nonzero_indices[nonzero_indices[:, 0] == i][:, 1:]
            if len(this_mask_nz) == 0:
                self.boxes[i] = torch.tensor([0, 0, 0, 0], device=self.device, dtype=self.dtype)
            else:
                self.boxes[i] = torch.tensor([this_mask_nz[:, 1].min(), this_mask_nz[:, 0].min(), this_mask_nz[:, 1].max(), this_mask_nz[:, 0].max()], device=self.device, dtype=self.dtype) * mask_to_image_scale.repeat(2)
        self.boxes[:, :2] = self.boxes[:, :2].floor()
        self.boxes[:, 2:] = self.boxes[:, 2:].ceil()
        self.boxes[:, 0:4:2] = self.boxes[:, 0:4:2].clamp(0, self.image.shape[2])
        self.boxes[:, 1:4:2] = self.boxes[:, 1:4:2].clamp(0, self.image.shape[1])
        return self

    @property
    def contours(self):
        """
        This function is copied from ultralytics.engine.results.Masks.xy, which unfortunately uses caching, meaning that after updating the data in self.masks, the contours will not be updated (unless the cache is cleared, how?).
        """
        return [find_contours(m, largest_only=True) for m in self.masks.data]

    def contour_to_image_coordinates(self, contour : torch.Tensor, interpolate : bool=False):
        """
        Converts a contour from mask coordinates to image coordinates. 
        """
        mask_hw = self.masks.data.shape[1:]
        image_hw = self.image.shape[1:]
        # We use a fixed precision of torch.float32 to avoid numerical issues (unless the image is **really** large)
        mask_to_image_scale = torch.tensor([image_hw[0] / mask_hw[0], image_hw[1] / mask_hw[1]], device=self.device, dtype=torch.float32)
        # After scaling the coordinates, round and cast to long, we offset the contour by 1/2 before scaling to ensure that the scaled contour is in the "center" of the mask boundary
        center_offset = 1/2 * (mask_to_image_scale > 1).float() - 1 * (mask_to_image_scale < 1).float()
        scaled_contour = ((contour + center_offset) * mask_to_image_scale).round().long() 

        # Possibly interpolate the contour using integer linear interpolation
        if interpolate:
            scaled_contour = interpolate_contour(scaled_contour)
        return scaled_contour

    def __len__(self):
        return len(self.masks.data)
    
    def new(self):
        return TensorPredictions([], **{k : self.__dict__[k] for k in self.CONSTANTS if k in self.__dict__})
    
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
                        i = i.float().round().long().tolist() # Just to be super safe we cast to float, then round, then cast to long, then to list
                    assert all([isinstance(j, int) for j in i]) or all([isinstance(j, float) and (j % 1) == 0 for j in i]), RuntimeError(f"Unknown type or non-integer float for {i}: {type(i)}")
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
    
    def merge_overlapping_masks(self, iou_threshold, start=0):
        """
        Merges elements by union of their masks and boxes and maximum of their confidences and scales, if their IoU is above a threshold.
        
        OBS: This function is very slow though, it could perhaps be optimized by rewriting the function to use a loop instead of recursion,
        and calculate the indices of the merge-clusters for all elements, before merging them all at once at the end.
        
        Args:
            iou_threshold (float): IoU threshold for merging elements.

        Returns:
            TensorPredictions: The merged elements.
        """
        if start == 0:
            print(f'Merging overlapping masks (n={len(self)}) with IoU threshold {iou_threshold}')
        # Break if there are no more elements to merge. We can skip the last element, since it will have no possible other elements to merge with.
        if start >= (len(self) - 1):
            print(f'Finished merging with {len(self)} remaining elements')
            return self
        # Calculate the IoU between the element at start and all other elements
        iou = iou_masks_2sets(self.masks[[start]].data, self.masks[start+1:].data)
        # Get the indices of the elements that overlap with the element at start
        overlap_indices = (torch.where(iou > iou_threshold)[0] + start + 1).tolist() + [start]
        # Combine masks
        combined_mask = self.masks[overlap_indices].data.max(dim=0)[0]
        # Combine boxes
        considered_boxes = self.boxes[overlap_indices]
        combined_bottom_left = considered_boxes[:, :2].min(dim=0)[0]
        combined_top_right = considered_boxes[:, 2:].max(dim=0)[0]
        combined_boxes = torch.cat([combined_bottom_left, combined_top_right], dim=0)
        # Combine confidences
        combined_confs = self.confs[overlap_indices].max(dim=0)[0]
        # Combine classes
        assert (self.classes[overlap_indices] == self.classes[start]).all(), RuntimeError(f"Classes do not match for overlapping elements: {self.classes[overlap_indices]} != {self.classes[start]}")
        combined_classes = self.classes[start]
        # Combine scales
        combined_scales = max([self.scales[i] for i in overlap_indices])
        # Override the attributes at start
        self.masks.data[start] = combined_mask
        self.boxes.data[start] = combined_boxes
        self.confs[start] = combined_confs
        self.classes[start] = combined_classes
        self.scales[start] = combined_scales
        # Remove the other elements and merge again from the next element. 
        # Since IoU is symmetric we know that the next element cannot merge with any of the prior elements.
        non_overlapping_indices = [i for i in range(len(self)) if i not in overlap_indices or i == start]
        for k, v in self.__dict__.items():
            if not k in ["device", "dtype"]:
                if isinstance(v, torch.Tensor):
                    setattr(self, k, v[non_overlapping_indices])
                elif isinstance(v, list):
                    setattr(self, k, [v[i] for i in non_overlapping_indices])
                else:
                    try:
                        setattr(self, k, v[non_overlapping_indices])
                    except Exception as e:
                        raise RuntimeError(f"Unknown type for {k}: {type(v)} does not support flexible indexing") from e
        return self.merge_overlapping_masks(iou_threshold, start=start+1)

    def non_max_suppression(self, iou_threshold):
        """
        Simply wraps the nms_masks function from yolo_helpers.py, and removes the elements that were not selected.
        """
        ###     A significant optimization would be to do NMS before concatenating the predictions from multiple _detect_instances calls,     ### 
        ###     this would minimize the amount of time spent constructing and subsetting tensors. Annoying to implement though...             ###
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            len_before = len(self)

        # Skip if there are no elements to merge
        if len(self) <= 1:
            return self
        
        # To account for the case where a bigger mask overlaps a smaller mask, we adjust the confidences by the scale of the mask.
        sqrt_areas = self.masks.data.sum(dim=(1, 2)).to(dtype=self.dtype).sqrt()
        weights = sqrt_areas.min() / sqrt_areas     
        adjusted_confidences = self.confs * weights

        # Perform non-maximum suppression on the masks - This is very fast!!
        nms_ind = nms_masks(self.masks.data, adjusted_confidences, iou_threshold=iou_threshold, return_indices=True, dtype=self.dtype)
        # Remove the elements that were not selected
        self = self[nms_ind]
        if self.time:
            end.record()
            torch.cuda.synchronize()
            print(f'Non-maximum suppression took {start.elapsed_time(end)/1000:.3f} s for removing {len_before - len(nms_ind)} elements of {len_before} elements')
        return self
    
    def plot_matplotlib(self, masks : bool=True, boxes : bool=True, conf : bool=True, outpath : Union[str, None]=None, dpi=300, scale=1/2) -> None:
        """
        Plots the predictions in the TensorPredictions object. 
        
        Q: Is using matplotlib really the most efficient way to do this?

        Args:
            image (torch.Tensor): The image to plot the predictions on.
            masks (bool): Whether to plot the masks. Defaults to True.
            boxes (bool): Whether to plot the boxes. Defaults to True.
            conf (bool): Whether to plot the confidences. Defaults to True.
            outpath (Union[str, None], optional): If not None, saves the plot to the given path. Defaults to None.
        
        Returns:
            None
        """
        if conf and not boxes:
            raise RuntimeError("Cannot plot confidences without boxes!")
        # Get the data from the TensorPredictions object and convert to numpy
        _contours = self.contours # The property method for contours already converts to numpy
        _boxes = self.boxes.cpu().float().numpy()
        _confs = self.confs.cpu().float().numpy()
        _image = self.image.permute(1, 2, 0).cpu().to(dtype=torch.uint8).detach().numpy()

        # Setup the figure
        width, height = _image.shape[1], _image.shape[0]
        aspect_ratio = width / height
        figwidth = 10 # Can this be set dynamically in a better way or should it be a parameter?
        figheight = figwidth / aspect_ratio

        fig, ax = plt.subplots(1, 1, figsize=(figwidth, figheight), dpi=dpi)

        # Plot the image
        ax.imshow(_image)
        
        # Plot the boxes
        if boxes:
            for box, conf in zip(_boxes, _confs):
                rect = mpl.patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='red', facecolor='none', clip_on=True)
                ax.add_patch(rect)
                if conf:
                    ax.text(box[0], box[1] - 5, f"{conf*100:.3g}%", color='red', fontsize=12 * (100 / dpi) ** (1/2), clip_on=True)

        # Plot the masks
        if masks:
            for contour in _contours:
                contour = self.contour_to_image_coordinates(contour).cpu().float().numpy()
                ax.fill(contour[:, 1], contour[:, 0], color='red', alpha=0.15)

        # Remove axis margins and turn off the axis
        ax.margins(0)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

        # Adjust subplot parameters
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        
        # Save the plot if outpath is not None
        if outpath is not None:
            plt.savefig(outpath, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    def plot_opencv(self, linewidth=2, masks=True, boxes=True, conf=True, outpath=None, scale=1):
        # Convert torch tensor to numpy array
        image = self.image.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if scale != 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # Draw masks
        if masks:
            overlay = image.copy()
            contours = [None] * len(self.masks.data)
            for i, mask in enumerate(self.masks.data):
                contour = cv2.findContours(mask.to(torch.uint8).cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                # Calculate areas of each contour
                areas = np.array([cv2.contourArea(c) for c in contour])
                # Select the largest contour and convert it to a tensor
                contour = torch.tensor(contour[np.argmax(areas)], device=self.device).long().squeeze(1)
                # Convert contour to image coordinates
                contour = self.contour_to_image_coordinates(contour * scale, interpolate=False).cpu().numpy()
                # Append contour to list of contours
                contours[i] = contour
                # Fill the contours with a semi-transparent red overlay
                cv2.fillPoly(overlay, pts=[contour], color=(0, 0, 255))
            # Add the overlay to the image
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            # Draw the contours
            cv2.polylines(image, pts=contours, isClosed=True, color=(0, 0, 255), thickness=linewidth)

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
                    cv2.putText(
                        img =       image, 
                        text =      f"{conf*100:.3g}%", 
                        org =       (start_point[0], start_point[1] - round(10 * scale)),
                        fontFace =  cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1 * scale, 
                        color =     (0, 0, 0), 
                        thickness = max(1, round(2 * scale))
                    )

        # Save or show the image
        if outpath:
            cv2.imwrite(outpath, image)
        else:
            cv2.imshow('Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def plot_torch(self, masks=True, boxes=True, outpath=None, linewidth=3, scale=1):
        """
        Draws bounding boxes and masks on the image tensor and saves it to outpath.

        Only uses PyTorch operations, so it is a bit slower than plot_opencv, but it is also more flexible.

        Args:
            image (torch.Tensor): The image tensor to draw on.
            masks (bool): Whether to draw the masks. Defaults to True.
            boxes (bool): Whether to draw the boxes. Defaults to True.
            outpath (str, optional): If not None, saves the image to the given path. Defaults to None.
            lienwidth (int, optional): The width of the lines used to draw the boxes and masks. Defaults to 3.
            scale (float, optional): The scale of the image. Defaults to 1.
        
        Returns:
            None
        """
        if self.time:
            start = torch.cuda.Event(enable_timing=True)
            end_image_scaling = torch.cuda.Event(enable_timing=True)
            end_box_drawing = torch.cuda.Event(enable_timing=True)
            end_mask_drawing = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # Copy the image tensor
        image = self.image.clone()
        if scale != 1:
            image = torchvision.transforms.functional.resize(image, (int(image.shape[1] * scale), int(image.shape[2] * scale)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=False)
        # Convert to uint8
        image = image.round().to(torch.uint8)

        if self.time:
            end_image_scaling.record()
        
        h, w = torch.tensor(image.shape[1:], dtype=torch.long)
        box_linewidth = linewidth
        box_expansion = math.ceil((box_linewidth - 1) / 2)

        # For each crop in the image, draw the box by setting the pixels at the border to (0, 0, 0) and draw the mask by increasing the red channel by 85
        for box, mask in zip(self.boxes, self.masks):
            if boxes:
                # Draw the box
                x1, y1, x2, y2 = (box * scale).round().long().cpu().tolist()
                x1e, y1e, x2e, y2e = (max(0, x1 - box_expansion), min(w, x1 + box_expansion + 1)), (max(0, y1 - box_expansion), min(h, y1 + box_expansion + 1)), (max(0, x2 - box_expansion), min(w, x2 + box_expansion + 1)), (max(0, y2 - box_expansion), min(h, y2 + box_expansion + 1))
                image[:, y1e[0]:y2e[1], x2e[0]:x2e[1]] = 0
                image[:, y1e[0]:y1e[1], x1e[0]:x2e[1]] = 0
                image[:, y2e[0]:y2e[1], x1e[0]:x2e[1]] = 0
                image[:, y1e[0]:y2e[1], x1e[0]:x1e[1]] = 0
            if self.time:
                end_box_drawing.record()
            if masks:
                mask = torchvision.transforms.functional.resize(mask.data.unsqueeze(0), (h.item(), w.item()), interpolation=torchvision.transforms.InterpolationMode.NEAREST, antialias=False).squeeze(0).squeeze(0)
                # Draw the mask
                # Subtracting and adding from uint8 is a bit more complicated, since we need to ensure that the result is in the range [0, 255] so the numbers don't over/underflow
                image[0, mask] += (255 - image[0, mask]).clamp(max=50)
                image[1:, mask] -= image[1:, mask].clamp(max=30)
                # Draw the contour
                contour_mask = create_contour_mask(mask, width=linewidth)
                image[:, contour_mask] = 0
                image[0, contour_mask] = 255
            if self.time:
                end_mask_drawing.record()
                
        # Save the image (this is copied from the source-code for torchvision.utils.save_image, but bypassing the float-to-uint8 conversion, since we already have uint8)
        Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(outpath)
        if self.time:
            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end) / 1000
            image_scaling = start.elapsed_time(end_image_scaling) / 1000
            box_drawing = end_image_scaling.elapsed_time(end_box_drawing) / 1000
            mask_drawing = end_box_drawing.elapsed_time(end_mask_drawing) / 1000
            print(f'Drawing predictions took {total:.3f} s | Image scaling: {image_scaling:.3f} s |Box drawing: {box_drawing:.3f} s | Mask drawing: {mask_drawing:.3f} s')

    def save_crops(self, outdir=None, image_path=None):
        if image_path is None:
            assert self.image_path is not None, RuntimeError("Cannot save crops without image_path")
            image_path = self.image_path
        assert outdir is not None, RuntimeError("Cannot save crops without outpath")
        assert os.path.isdir(outdir), RuntimeError(f"outpath {outdir} is not a directory")
        image_name, image_ext = os.path.splitext(os.path.basename(self.image_path))

        # For each bounding box, save the corresponding crop
        for i, (box, mask, conf) in enumerate(zip(self.boxes, self.masks, self.confs)):
            # Define name of the crop using the pattern {image_name}_{x1}_{y1}_{x2}_{y2}_{confidence}.png
            x1, y1, x2, y2 = box.long().cpu().tolist()
            confidence = int(conf * 100)
            crop_name = f"crop_{image_name}_{x1}_{y1}_{x2}_{y2}_{confidence}.{image_ext}"
            # Extract the crop from the image tensor
            crop = self.image[:, y1:y2, x1:x2] / 255.0 # .to(torch.uint8) # Casting should probably be done for the whole image tensor instead of the crop, however I don't want side-effects and this would thus require either copying the tensor (not ideal) or re-casting the whole tensor back to the original dtype (also not ideal)
            # Save the crop
            torchvision.utils.save_image(crop, os.path.join(outdir, crop_name))

    def serialize(self, outpath : str=None, save_pt : bool=True, save_json : bool=True, save_image : bool=False, identifier : str=None, overwrite_basename : Union[str, None]=None, fast : bool=False):
        assert outpath is not None, RuntimeError("Cannot serialize without outpath")
        assert len(outpath) > 0, RuntimeError("Cannot serialize with empty outpath")
        assert os.path.isdir(outpath), RuntimeError(f"outpath {outpath} is not a directory")

        if overwrite_basename is not None:
            basename = overwrite_basename
        else:
            basename = os.path.splitext(os.path.basename(self.image_path))[0]

        # Check for file-extension on the outpath, it should have none - not really necessary anymore due to the check for directory above
        outpath, ext = os.path.splitext(outpath)
        if ext != "":
            print(f"WARNING: serializer outpath ({outpath}) should not have a file-extension for 'TensorPredictions.serialize'!")

        # Add the basename to the outpath
        outpath = os.path.join(outpath, basename)
        
        if save_pt:
            ## First serialize as .pt file
            pt_path = f'{outpath}.pt'
            torch.save(self, pt_path)

        if save_json:
            ## Then serialize as .json file
            json_path = f'{outpath}.json'
            ## Clean up the data
            # 1. Convert the boxes to list
            boxes = self.boxes.cpu().tolist()
            # 2. Convert the masks to contours as lists 
            if not fast:
                contours = [c.cpu().tolist() for c in self.contours]
            else:
                torch.save(self.masks.data, f'{outpath}_masks.pt')
                contours = None
            # 3. Convert the confidences to floats in a list
            confs = self.confs.float().cpu().tolist()
            # 4. Convert the classes to integers in a list
            classes = self.classes.cpu().long().tolist()
            # 5. Get the scales (already floats in a list)
            scales = self.scales
            json_data = {"boxes" : boxes, 
                         "contours" : contours, 
                         "confs" : confs, 
                         "classes" : classes, 
                         "scales" : scales, 
                         "identifier" : identifier if identifier is not None else "", 
                         "image_path" : self.image_path, 
                         "image_width" : self.image.shape[2], 
                         "image_height" : self.image.shape[1], 
                         "mask_width" : self.masks.data.shape[2], 
                         "mask_height" : self.masks.data.shape[1]}
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
        

    def load(self, path : str, device=None, dtype=None):
        assert isinstance(path, str) and os.path.isfile(path), RuntimeError(f"Invalid path: {path}")
        # Check whether the path is a .pt file or a .json file
        stripped_path, ext = os.path.splitext(path)
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

            empty_image = torch.zeros((3, json_data["image_height"], json_data["image_width"]), device=device, dtype=dtype)
            new_tp = TensorPredictions(image=empty_image, device=device, dtype=dtype)

            # Load the data
            for k, v in json_data.items():
                # Skip attributes in the json file that are not in the TensorPredictions object
                if k in ["identifier", "image_width", "image_height", "mask_width", "mask_height"]:
                    continue
                if k in ["boxes"]:
                    v = torch.tensor(v, device=new_tp.device, dtype=new_tp.dtype)
                elif k in ["contours"]:
                    # Check for masks in the same directory as the json file
                    masks_path = f'{stripped_path}_masks.pt'
                    if os.path.isfile(masks_path):
                        v = torch.load(masks_path).to(device=new_tp.device)
                    else:
                        v = contours_to_masks(v, (json_data["mask_heigth"], json_data["mask_width"]))
                    k = "masks"
                elif k in ["confs", "classes"]:
                    v = torch.tensor(v, device=new_tp.device, dtype=new_tp.dtype)
                setattr(new_tp, k, v)

            self = new_tp
            return self
        else:
            raise RuntimeError(f"Unknown file-extension: {ext} for path: {path}")
        
    def save(self, output_directory : str, overview : bool=True, crops : bool=True, metadata : bool=True, fast : bool=False) -> str:
        if not os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory} does not exist")
        
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        prediction_directory = os.path.join(output_directory, base_name)
        if not os.path.exists(prediction_directory):
            os.makedirs(prediction_directory)

        if crops:
            # Save crops
            crop_directory = os.path.join(prediction_directory, "crops")
            if not os.path.exists(crop_directory):
                os.makedirs(crop_directory)

            self.save_crops(crop_directory)
        if overview:
            # Save overview
            overview_directory = os.path.join(prediction_directory, f"overview_{base_name}.jpg")
            if not fast:
                self.plot_opencv(outpath=overview_directory, linewidth=2, scale=1) 
            else:
                self.plot_opencv(outpath=overview_directory, linewidth=1, scale=1/2)
        if metadata:
            # Save json
            self.serialize(prediction_directory, save_pt=False, save_image=False, fast=fast)

        return prediction_directory

class Predictions(object):
    def __init__(self, image, original_image_path, contours, confs, classes, class_dict):
        self._OVERVIEWS_DIR = "overviews"
        self._CROPS_DIR = "crops"
        assert len(classes) == len(contours) == len(confs)
        self._class_dict = class_dict
        self._contours = [np.array(c) for c in contours]
        self._classes = classes
        self._confs = confs
        self._image = image.permute(1, 2, 0).cpu().to(dtype=torch.uint8).detach().numpy()
        self._original_image_path = original_image_path
        if self._original_image_path is not None:
            self._dpis = self.get_dpis(self._original_image_path)
            array = cv2.imread(self._original_image_path)
            self._xy_scales = np.array([self._image.shape[1] / array.shape[1],
                                        self._image.shape[0] / array.shape[0]])
        else:
            self._dpis = None
            array = self._image
            self._xy_scales = (1, 1)

        # make bounding boxes for all contours fixme
        self._bboxes = []  # xywh
        self._centers = np.zeros((len(self._contours), 2), dtype=np.float32)
        for i, c in enumerate(self._contours):
            self._contours[i] = c / self._xy_scales
            self._contours[i] = self._contours[i].astype(np.int32)
            bb = cv2.boundingRect(self._contours[i])
            center = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2])
            self._centers[i] = center
            self._bboxes.append(bb)

        if self._original_image_path:
            assert os.path.isfile(self._original_image_path)
            self._img_name_prefix = os.path.splitext(os.path.basename(self._original_image_path))[0]
        else:
            # fixme has here?!
            self._img_name_prefix = "image"


        self._original_array = array

    @property
    def contours(self):
        return self._contours

    def compare(self, ref, iou_threshold,
                # obj_class # fixme, do that PER CLASS!!
                ):
        # print(self._class_dict.values())
        obj_class = 0 # fixme

        out = []
        if len(self) == 0:
            if len(ref) == 0:
                return {}
            pairs = [(i, None) for i, _ in enumerate(ref.contours)]

        elif len(ref) == 0:
            pairs = [(None, i) for i, _ in enumerate(self.contours)]

        else:
            arr = np.zeros((len(ref), len(self)), dtype=float)
            for m, g in enumerate(ref.contours):
                g_shape = Polygon(np.squeeze(g))
                for n, c in enumerate(self.contours):
                    i_shape = Polygon(np.squeeze(c))
                    # todo check bounding box overlap
                    iou_val = iou(g_shape, i_shape)
                    arr[m, n] = iou_val

            pairs = iou_match_pairs(arr, iou_threshold)

        for g_a, i_a in pairs:
            if i_a is not None:
                in_im = True
            else:
                in_im = False

            if g_a is not None:
                area = cv2.contourArea(ref.contours[g_a])
                in_gt = True

            else:
                in_gt = False
                area = cv2.contourArea(self.contours[i_a])

            assert sum([in_gt, in_im]) > 0, 'Occurence should exist either in gt or im!'

            out.append({'area': area,
                        'in_gt': in_gt,
                        'in_im': in_im,
                        'class': obj_class,
                        'filename': self._original_image_path})
        return out

    def __getitem__(self, i):
        return Predictions(self._image,
                           self._original_image_path,
                           [self._contours[i]]
                           [self._confs[i]],
                           [self._classes[i]],
                           self._class_dict,
                           )

    def __len__(self):
        return len(self._contours)

    def draw(self, figsize=(10, 10)):
        array = self._original_array

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(array[:,:,::-1])

        for i, (r, c) in enumerate(zip(self._bboxes, self._confs)):
            rect = mpl.patches.Rectangle((r[0], r[1]), r[2], r[3], linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(r[0], r[1], f"{i:0>4} ({c*100:.3g}%)", color='red', fontsize=12)

        for contour in self._contours:
            ax.fill(contour[:, 0], contour[:, 1], color='red', alpha=0.15)

        ax.set_title("Overview")
        ax.axis("off")
        plt.show()


    def get_dpis(self, input):
        # fixme, this is a fallback to use jfif instead of exif!

        try:
            with exiftool.ExifToolHelper() as et:
                metadata = et.get_metadata(input)
                x_res = metadata[0]['JFIF:XResolution']
                y_res = metadata[0]['JFIF:YResolution']
        except Exception as e:
            logging.error(e)
            x_res = y_res = 0
        if x_res == 0 or y_res == 0:
            return None
        return x_res, y_res

    def make_crops(self, out_dir, draw_all_preds=True, only_overview=False):

        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, "crops"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "overviews"), exist_ok=True)

        array = self._original_array
        if draw_all_preds:
            # Change to correct data structure
            overall_img = cv2.UMat(m=array)

            for i, r in enumerate(tqdm(self._bboxes, desc="Drawing overview", total=len(self._bboxes), leave=False, dynamic_ncols=True)):
                # r = [int(i) for i in [r[0] / self._xy_scales[1], r[1] / self._xy_scales[0], r[2] / self._xy_scales[1], r[3] / self._xy_scales[0]]]
                cv2.rectangle(img=overall_img, 
                              pt1=(r[0], r[1]),
                              pt2=(r[0] + r[2], r[1] + r[3]),
                              color=(0, 255, 255),
                              thickness=5, lineType=cv2.LINE_AA)
                cv2.rectangle(img=overall_img, 
                              pt1=(r[0], r[1]),
                              pt2=(r[0] + r[2], r[1] + r[3]),
                              color=(255, 0, 0),
                              thickness=3, lineType=cv2.LINE_AA)
                text = f"{i:0>4}"
                cv2.putText(img=overall_img, text=text, org=(r[0], r[1]), color=(255, 0, 0),
                            thickness=2, lineType=cv2.LINE_AA, fontScale=1, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            bottomLeftOrigin=False)
            
            # for x, y in self._centers:
            #     cv2.circle(img=overall_img, center=(int(x), int(y)), radius=5, color=(0, 255, 0), thickness=3,
            #                lineType=cv2.LINE_AA)

            cv2.drawContours(image=overall_img, contours=self._contours, contourIdx=-1, color=(0, 255, 255),
                             thickness=3, lineType=cv2.LINE_AA)
            cv2.drawContours(image=overall_img, contours=self._contours, contourIdx=-1, color=(255, 0, 0),
                             thickness=2, lineType=cv2.LINE_AA)

            basename = f"overview_{self._img_name_prefix}.jpg"
            out_file = os.path.join(out_dir, self._OVERVIEWS_DIR, basename)
            cv2.imwrite(out_file, overall_img)

        if not only_overview:
            for i, (bb, ct, cf, cl) in enumerate(tqdm(zip(self._bboxes, self._contours, self._confs, self._classes), desc="Extracting crops", total=len(self._bboxes), leave=False, dynamic_ncols=True)):
                x1, y1, w, h = bb
                x2 = x1 + w
                y2 = y1 + h

                x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                assert x1 < x2
                assert y1 < y2
                roi = np.copy(array[y1:y2, x1:x2]).astype(np.uint8)
                mask = np.zeros_like(roi, dtype=np.uint8)
                
                ct = ct.astype(np.int32)

                cv2.drawContours(image=mask, contours=[ct], contourIdx=-1, color=(255, 255, 255), thickness=-1,
                                lineType=cv2.LINE_8, offset=(-x1, -y1))

                if self._dpis:
                    area_sqr_in = np.count_nonzero(mask) / (self._dpis[0] * self._dpis[1])
                    area_sqr_mm = area_sqr_in * 645.16
                    area_sqr_mm = round(area_sqr_mm)
                else:
                    area_sqr_mm = 0
                basename = f"{self._img_name_prefix}_{i:0>4}_{x1}_{y1}_{area_sqr_mm:0>4}.png"
                out_file = os.path.join(out_dir, self._CROPS_DIR, basename)

                mask = PIL.Image.fromarray(mask).convert('L')
                im = PIL.Image.fromarray(roi[:,:,::-1])
                im.putalpha(mask)

                if self._dpis:
                    im.save(out_file, dpi=self._dpis, quality=95)
                else:
                    im.save(out_file, quality=95)

    def coco_entry(self):

        h, w, _ = self._original_array.shape
        annotations = []

        image_info = {
            "id": None,
            "file_name": os.path.basename(self._original_image_path),
            "height": h,
            "width": w
        }

        for ct, bb, cl, conf in zip(self._contours, self._bboxes, self._classes, self._confs):
            scaled_ct = np.divide(ct.astype(float), self._xy_scales).astype(np.int32)
            xs, ys = self._xy_scales
            scaled_bbox = np.divide(np.array(bb, float), [xs, ys, xs, ys]).astype(np.int32).tolist()
            area = cv2.contourArea(scaled_ct)
            segmentation = [scaled_ct.flatten().tolist()]

            # Calculate the bounding box

            annotation_info = {
                "id": None,
                "image_id": image_info["id"],
                "category_id": cl,
                "segmentation": segmentation,
                "area": area,
                "bbox": scaled_bbox,
                "confidence": conf,
                "iscrowd": 0  # Assuming all instances are not crowded
            }
            annotations.append(annotation_info)

        return image_info, annotations

class LabelPredictions(Predictions):
    def __init__(self, label, class_dict):
        # dict_keys(['im_file', 'cls', 'bboxes', 'segments', 'keypoints', 'normalized', 'bbox_format'])

        # self._CROPS_DIR = "crops"
        #
        # self._class_dict = class_dict
        # self._classes = label["cls"]
        #
        # self._original_image_path = label["im_file"]
        # make bounding boxes for all contours fixme

        label = copy.deepcopy(label)
        confs = np.ones((len(label["cls"])), float)
        im = cv2.imread(label["im_file"])
        cts = label["segments"]
        h, w = im.shape[0], im.shape[1]
        for c in cts:
            c *= (w, h)

        super().__init__(im, label["im_file"], cts, confs, label["cls"], class_dict)


class Predictor(object):
    MIN_MAX_OBJ_SIZE = (32, 2048)
    MINIMUM_TILE_OVERLAP = 384
    EDGE_CASE_MARGIN = 8
    SCORE_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.25
    MAX_MASK_SIZE = 768
    TIME = False
    DEBUG = False

    def __init__(self, model, cfg=None, device=torch.device("cpu"), dtype=torch.float32):
        if not cfg is None:
            raise NotImplementedError("cfg is not implemented yet")

        self._device = device
        self._dtype = dtype

        self._base_yolo = YOLO(model)
        self._base_yolo._load(model, "inference")
        self._base_yolo.load(model) 
        self._base_yolo.fuse()
        self._model = self._base_yolo.model.to(device=device, dtype=dtype)
        self._model.eval()

        self._yolo_predictor = None

    def _detect_instances(self, tensor : torch.Tensor, scale=1.0):
        if self.TIME:
            # Initialize timing calculations
            start_detect = torch.cuda.Event(enable_timing=True)
            end_detect = torch.cuda.Event(enable_timing=True)
            start_detect.record()
        orig_h, orig_w = tensor.shape[1:]
        w, h = orig_w, orig_h

        # Check dimensions and channels
        assert tensor.device == self._device, RuntimeError(f"tensor.device {tensor.device} != self._device {self._device}")
        assert tensor.dtype == self._dtype, RuntimeError(f"tensor.dtype {tensor.dtype} != self._dtype {self._dtype}")

        # Resize if scale is not 1
        if scale != 1:
            w, h = max(1024, int(orig_w * scale)), max(1024, int(orig_h * scale))
            box_scale = torch.tensor((w / orig_w, h / orig_h), device=self._device, dtype=self._dtype)
            resize = transforms.Resize((h, w), antialias=True)
            tensor = resize(tensor)

        # Tile calculation
        tile_size = 1024

        x_n_tiles = math.ceil(w / (tile_size - self.MINIMUM_TILE_OVERLAP)) if w != tile_size else 1
        y_n_tiles = math.ceil(h / (tile_size - self.MINIMUM_TILE_OVERLAP)) if h != tile_size else 1

        # x_stride = ((w - 1) // (x_n_tiles - 1)) if x_n_tiles > 1 else (tile_size - self.MINIMUM_TILE_OVERLAP)
        # y_stride = ((h - 1) // (y_n_tiles - 1)) if y_n_tiles > 1 else (tile_size - self.MINIMUM_TILE_OVERLAP)
        x_stride = tile_size - math.floor((tile_size * (x_n_tiles + 0) - w) / x_n_tiles) if x_n_tiles > 1 else tile_size
        y_stride = tile_size - math.floor((tile_size * (y_n_tiles + 0) - h) / y_n_tiles) if y_n_tiles > 1 else tile_size

        x_range = [i if (i + tile_size) < w else (w - tile_size) for i in range(0, x_stride * x_n_tiles, x_stride)]
        y_range = [i if (i + tile_size) < h else (h - tile_size) for i in range(0, y_stride * y_n_tiles, y_stride)]

        offsets = [((m, n), (j, i)) for n, j in enumerate(y_range) for m, i in enumerate(x_range)]
        ims = torch.stack([tensor[:, o[0]: (o[0] + 1024), o[1]: (o[1] + 1024)] for (m, n), o in offsets], dim=0)

        assert len(ims) == (x_n_tiles * y_n_tiles), RuntimeError(f"len(ims) {len(ims)} != (x_n_tiles * y_n_tiles) {x_n_tiles} * {y_n_tiles} ({x_n_tiles * y_n_tiles})")
        # ## DEBUG
        # # plot all the tiles in a grid
        # fig, axs = plt.subplots(y_n_tiles, x_n_tiles, figsize=(x_n_tiles * 3, y_n_tiles * 3))
        # axs = axs.flatten()
        # for i, im in enumerate(ims):
        #     axs[i].imshow(im.permute(1, 2, 0).float().cpu().detach().numpy())
        #     axs[i].axis("off")
        # plt.show()
        
        if self.TIME:
            # Initialize timing calculations
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            batch_times = []
            fetch_times = []
            forward_times = []
            postprocess_times = []

        ps = []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(ims), batch_size):
                if self.TIME:
                    # Initialize batch timing calculations
                    start_batch_event = torch.cuda.Event(enable_timing=True)
                    end_fetch_event = torch.cuda.Event(enable_timing=True)
                    end_forward_event = torch.cuda.Event(enable_timing=True)
                    end_postprocess_event = torch.cuda.Event(enable_timing=True)
                    end_batch_event = torch.cuda.Event(enable_timing=True)
                    # Record batch start
                    start_batch_event.record()

                batch = ims[i:min((i+batch_size), len(ims))]
                if self.TIME:
                    # Record end of fetch
                    end_fetch_event.record()
                if self._yolo_predictor: # Never used currently, but don't want to remove it yet
                    tps = self._yolo_predictor(batch, self._model, verbose=False)
                    raise NotImplementedError("This code has not been tested in a long time, and is probably broken")
                else:
                    tps = self._model(batch)
                    if self.TIME:
                        # Record end of forward
                        end_forward_event.record()
                    tps = postprocess(tps, batch, max_det=100, min_confidence=self.SCORE_THRESHOLD, iou_threshold=self.IOU_THRESHOLD, edge_margin=self.EDGE_CASE_MARGIN, nms=3) # Important to prune within each tile first, this avoids having to carry around a lot of data
                    if self.TIME:
                        # Record end of postprocess
                        end_postprocess_event.record()
                ps.extend(tps)
                if self.TIME:
                    # Record batch end
                    end_batch_event.record()

                    # Calculate timing
                    torch.cuda.synchronize(device=self._device)
                    batch_time = start_batch_event.elapsed_time(end_batch_event) / 1000 # Convert to seconds
                    fetch_time = start_batch_event.elapsed_time(end_fetch_event) / 1000 # Convert to seconds
                    forward_time = end_fetch_event.elapsed_time(end_forward_event) / 1000 # Convert to seconds
                    postprocess_time = end_forward_event.elapsed_time(end_postprocess_event) / 1000 # Convert to seconds
                    batch_times.append(batch_time)
                    fetch_times.append(fetch_time)
                    forward_times.append(forward_time)
                    postprocess_times.append(postprocess_time)
                    # print(f'Batch time: {batch_time:.3f}s, fetch time: {fetch_time:.3f}s, forward time: {forward_time:.3f}s, postprocess time: {postprocess_time:.3f}s')

        if self.TIME:
            # Finish timing calculations
            end_event.record()
            torch.cuda.synchronize(device=self._device)
            total_elapsed = start_event.elapsed_time(end_event) / 1000 # Convert to seconds
            total_batch_time = sum(batch_times)
            overhead_prop = (total_elapsed - total_batch_time) / total_elapsed
            fetch_time, forward_time, postprocess_time = sum(fetch_times), sum(forward_times), sum(postprocess_times)
            fetch_prop, forward_prop, postprocess_prop = fetch_time / total_batch_time, forward_time / total_batch_time, postprocess_time / total_batch_time
         
        if self.DEBUG:
            print(f'Number of tiles processed before merging and plotting: {len(ps)}')
            for i in range(len(ps)):
                ps[i].orig_img = (ps[i].orig_img.detach().contiguous() * 255).to(torch.uint8).cpu().numpy() # Needed for compatibility with the Results.plot function
            fig, axs = plt.subplots(y_n_tiles, x_n_tiles, figsize=(x_n_tiles * 3, y_n_tiles * 3))
            axs = axs.flatten()
            [axs[i].imshow(p.plot(pil=False, masks=False, probs=False, labels=False, kpt_line=False)) for i, p in enumerate(ps)]
            plt.show()
            for i in range(len(ps)):
                ps[i].orig_img = torch.tensor(ps[i].orig_img).squeeze(0).to(dtype=self._dtype, device=self._device) / 255.0 # Backtransform

        ## Combine the results from the tiles
        MASK_SIZE = 256 # Defined by the YOLOv8 model segmentation architecture
        # For the boxes, we can simply add the offsets
        box_offsetters = torch.tensor([o[1] for o in offsets], dtype=self._dtype, device=self._device)
        # However for the masks, we need to create a new mask which can contain every tile, and then add the masks from each tile to the correct area - this will of course use some memory, but it's probably not too bad
        # Since the masks are 256x256, while the tile sizes can be anything, we need to scale the offsets
        mask_offsetters = box_offsetters * (MASK_SIZE / torch.tensor([tile_size, tile_size], dtype=self._dtype, device=self._device).unsqueeze(0))
        # We also need to round the offsets, since they may not line up with the pixel-grid
        mask_offsetters = torch.round(mask_offsetters).long()
        # Then the new mask size is the maximum offset + 256 (this is kind of a hack, but it works, doing it properly would be a bit more complicated due to the tiles overlapping)
        max_mask_offsets = mask_offsetters.max(dim=0)[0] + MASK_SIZE
        new_mask_size = [i.item() for i in max_mask_offsets]
        # Finally, we can merge the results - this function basically just does what I described above
        orig_img = torch.zeros((orig_h, orig_w, 1))
        ps = merge_tile_results(ps, orig_img, box_offsetters=box_offsetters, mask_offsetters=mask_offsetters, new_shape=new_mask_size, clamp_boxes=(h, w), max_mask_size=self.MAX_MASK_SIZE)

        # Apply the size filters - This could be done before merging the tiles, but would require some scaling logic
        ps_bt = ps.boxes.xyxy
        ps_hw = ps_bt[:, 2:] - ps_bt[:, :2]
        ps_sqrt_area = ps_hw[:, 0].sqrt() * ps_hw[:, 1].sqrt()
        # Old criteria pruned images based on their minor dimension
        # big_enough = ((ps_hw > self.MIN_MAX_OBJ_SIZE[0]) & (ps_hw < self.MIN_MAX_OBJ_SIZE[1])).all(dim=1)
        # New criteria prunes images based on their area
        big_enough = (ps_sqrt_area > self.MIN_MAX_OBJ_SIZE[0]) & (ps_sqrt_area < self.MIN_MAX_OBJ_SIZE[1])
        ps = ps[big_enough]

        if self.TIME:
            end_detect.record()
            torch.cuda.synchronize(device=self._device)
            total_detect_time = start_detect.elapsed_time(end_detect) / 1000 # Convert to seconds
            pred_prop = total_elapsed / total_detect_time
            print(f'Prediction time: {total_elapsed:.3f}s/{pred_prop*100:.3g}% (overhead: {overhead_prop * 100:.1f}) | Fetch {fetch_prop * 100:.1f}% | Forward {forward_prop * 100:.1f}% | Postprocess {postprocess_prop * 100:.1f}%')
            self.total_detection_time += total_detect_time
            self.total_forward_time += forward_time
        return Prepared_Results(ps, scale=scale, device=self._device, dtype=self._dtype)

    def pyramid_predictions(self, image, path=None, scale_increment=2 / 3, scale_before=1, add_border=False):
        if self.TIME:
            # Initialize timing calculations
            start_pyramid = torch.cuda.Event(enable_timing=True)
            end_pyramid = torch.cuda.Event(enable_timing=True)
            start_pyramid.record()

        if isinstance(image, str):
            im = read_image(image)
            path = image
        elif isinstance(image, torch.Tensor):
            im = image
            assert path is not None, ValueError("Path must be provided if image is a tensor")
        else:
            raise TypeError(f"Unknown type for image: {type(image)}, expected str or torch.Tensor")

        c, h, w = im.shape
        transform_list = [transforms.Normalize(0, 255)] # add transforms.toDType(self._dtype) here? (probably slower than forcing the user to precast the image)

        if scale_before != 1:
            w, h = int(w * scale_before), int(h * scale_before)
            resize = transforms.Resize((h, w), antialias=True)
            transform_list.append(resize)
        
        # A border is always added now, to avoid edge-cases on the actual edge of the image. I.e. only detections on internal edges of tiles should be removed, not detections on the edge of the image.
        if add_border:
            raise NotImplementedError("add_border has been superceded, and its' functionality should be changed if it is to be meaningful again.")
            if w > h:
                top = (w - h) // 2
                bottom = math.ceil((w - h) / 2)
                transform_list.append(transforms.Pad(padding=(0, top, 0, bottom), fill=0))
                border_offset = (0, top)
            elif h > w:
                left = (h - w) // 2
                right = math.ceil((h - w) / 2)
                transform_list.append(transforms.Pad(padding=(left, 0, right, 0), fill=0))
                border_offset = (left, 0)
        edge_case_margin_padding_multiplier = 50
        padding_for_edge_cases = transforms.Pad(padding=self.EDGE_CASE_MARGIN * edge_case_margin_padding_multiplier, fill=0, padding_mode='constant')
        padding_offset = torch.tensor((self.EDGE_CASE_MARGIN, self.EDGE_CASE_MARGIN), dtype=self._dtype, device=self._device) * edge_case_margin_padding_multiplier
        transform_list.append(padding_for_edge_cases)
        if transform_list:
            transforms_composed = transforms.Compose(transform_list)
        
        im_b = transforms_composed(im) if transform_list else im

        # Check dimensions and channels
        assert im_b.dim() == 3, f"Image is not 3-dimensional"
        assert im_b.size(0) == 3, f"Image does not have 3 channels"

        min_dim = min(h, w)

        # fixme, what to do if the image is too small? - Not relevant anymore, the _detect_instances function simply upscales the image such that the smallest dimension is 1024
        # 0-pad
        scales = []
        s = 1024 / min_dim

        if s > 1:
            scales.append(s)
        else:
            while s <= 0.9: # Cut off at 90%, to avoid having s~1 and s=1.
                scales.append(s)
                s /= scale_increment
            if s != 1:
                scales.append(1.0)

        logging.info(f"Running inference on scales: {scales}")

        if self.TIME:
            self.total_detection_time = 0
            self.total_forward_time = 0
        all_preds = [self._detect_instances(im_b, scale=s) for s in reversed(scales)]
        if self.TIME:
            print(f'Total detection time: {self.total_detection_time:.3f}s ({self.total_forward_time / self.total_detection_time * 100:.3g}% forward)')

        all_preds = TensorPredictions(
            predictions = all_preds, 
            image       = im, 
            image_path  = path,
            dtype       = self._dtype,
            device      = self._device,
            time        = self.TIME
        ).non_max_suppression(
            iou_threshold = self.IOU_THRESHOLD
        ).offset_scale_pad(
            offset  = -padding_offset, 
            scale   = 1 / scale_before,
            pad     = 5 # pad the boxes a bit to ensure they encapsulate the masks
        )#.fix_boxes() # The boxes don't necessarily match the masks, but this does not fix it. The discrepancy arises due to the way the masks are converted to contours.

        if self.TIME:
            # Finish timing calculations
            end_pyramid.record()
            torch.cuda.synchronize(device=self._device)
            total_pyramid_time = start_pyramid.elapsed_time(end_pyramid) / 1000
            print(f'Total pyramid time: {total_pyramid_time:.3f}s ({self.total_detection_time / total_pyramid_time * 100:.3g}% detection | {self.total_forward_time / total_pyramid_time * 100:.3g}% forward)')

        return all_preds
        