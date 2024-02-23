import os.path
import torch
import math
import logging

import cv2
import json
from flat_bug.yolo_helpers import *
from flat_bug.geometry_simples import find_contours, contours_to_masks, simplify_contour, interpolate_contour, create_contour_mask, scale_contour
from ultralytics import YOLO

from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms

from shapely.geometry import Polygon
import shapely


from typing import Union

# Class for containing the results from a single _detect_instances call - This should probably not be its own class, but just a TensorPredictions object with a single element instead, but this would require altering the TensorPredictions._combine_predictions function to handle a single element differently or pass a flag or something
class Prepared_Results:
    def __init__(self, predictions : "ResultsWithTiles", scale : Tuple[float, float], device, dtype):
        self.wh_scale = torch.tensor(scale, device=device, dtype=dtype).unsqueeze(0)
        self._predictions = predictions
        self._predictions.boxes.data[:, :4] /= self.wh_scale.repeat(1, 2)
        self._predictions.polygons = [(poly + torch.roll(poly, 1, dims=0)) / (2 * self.wh_scale) for poly in self._predictions.polygons]
        # self._predictions.polygons = [torch.tensor(shapely.affinity.scale(Polygon(poly.cpu().numpy()), self.wh_scale[0][0].item(), self.wh_scale[0][1].item(), origin="centroid").exterior.coords, device=device, dtype=dtype) for poly in self._predictions.polygons]
        self.scale = sum(scale) / 2
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
    BOX_IS_EQUAL_MARGIN = 0 # How many pixels the boxes can differ by and still be considered equal? Used for removing duplicates before merging overlapping masks.
    PREFER_POLYGONS = False # If True, will use shapely Polygons instead of masks for NMS and drawing
    # These are simply initialized here to decrease clutter in the __init__ function and arguments
    mask_width = None 
    mask_height = None
    device = None
    dtype = None
    CONSTANTS = ["image", "image_path", "device", "dtype", "time", "mask_height", "mask_width", "CONSTANTS", "BOX_IS_EQUAL_MARGIN", "PREFER_POLYGONS"] # Attributes that should not be changed after initialization - should 'contours' be here?

    def __init__(self, predictions : Union[list[Prepared_Results], None]=None, image : Union[torch.Tensor, None]=None, image_path = Union[str, None], time=False, **kwargs):
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
                print(f"WARNING: Unknown keyword argument {k}={v} for TensorPredictions is ignored!")

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
            # If there are no predictions, set other attributes to None
            self.masks, self.polygons, self.boxes, self.confs, self.classes, self.scales = None, None, None, None, None, None
        
        if self.time and len(predictions) > 0:
            end.record()
            torch.cuda.synchronize()
            print(f'Initializing TensorPredictions took {start.elapsed_time(end)/1000:.3f} s')

    def _combine_predictions(self, predictions : list[Prepared_Results]):
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
        self.boxes = torch.cat([p.boxes for p in predictions]) # Nx4
        self.confs = torch.cat([p.confs for p in predictions]) # N
        self.scales = [p.scale for p in predictions for _ in range(len(p))] # N
        
        ## Duplicate removal ##
        # Calculate indices of non-duplicate boxes - prioritzed by resolution
        valid_indices = detect_duplicate_boxes(self.boxes, torch.tensor(self.scales, dtype=self.dtype, device=self.device), margin=self.BOX_IS_EQUAL_MARGIN, return_indices=True) 
        # Subset the boxes and confidences to the valid indices
        self.boxes = self.boxes[valid_indices]
        self.confs = self.confs[valid_indices]
        # Divide the valid indices into each prediction object
        n_detections = [len(p) for p in predictions]
        max_indices = cumsum(n_detections)
        valid_chunked = [valid_indices[(valid_indices < max_indices[i]) & (valid_indices >= (max_indices[i-1] if i > 0 else 0))] - (max_indices[i] - n_detections[i]) for i in range(len(predictions))]

        if self.time:
            end_duplication_removal.record()

        # For the remaining attributes we remove the duplicates before combining them
        self.masks = stack_masks([p.masks.data[nd] for p, nd in zip(predictions, valid_chunked)]) # NxMHxMW - MH and MW are proportional to the original image size
        self.mask_height, self.mask_width = self.masks.shape[1:]

        if self.time:
            end_mask_combination.record()

        self.masks.orig_shape = self.image.shape[1:] # Set the target shape of the masks to the shape of the image passed to the TensorPredictions object
        
        self.polygons = [p._predictions.polygons[nd_i] for p, nd in zip(predictions, valid_chunked) for nd_i in nd]
        self.classes = torch.cat([p.classes[nd] for p, nd in zip(predictions, valid_chunked)]) # N
        self.scales = [predictions[i].scale for i, p in enumerate(valid_chunked) for _ in range(len(p))] # N
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

    def offset_scale_pad(self, offset : torch.Tensor, scale : float, pad : int = 0) -> "TensorPredictions":
        """
        Since the image may be padded, the masks and boxes should be offset by the padding-width and scaled by the scale_before factor to match the original image size. Also pads the boxes by pad pixels to be safe.

        Args:
            offset (torch.Tensor): A vector of length 2 containing the x and y offset of the image. Useful for removing image-padding effects.
            scale (float): The scale factor of the image.
            pad (int, optional): The number of pixels to pad the boxes by. Defaults to 0. (Not to be confused with image-padding, this is about expanding the boxes a bit to ensure they cover the entire mask)
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

        self.polygons = [(poly + offset.unsqueeze(0)) * scale for poly in self.polygons]

        # However masks are more complicated since they don't have the same size as the image
        image_shape = torch.tensor([self.image.shape[1], self.image.shape[2]], device=self.device, dtype=self.dtype) # Get the shape of the original image
        # Calculate the normalized offset (i.e. the offset as a fraction of the scaled and padded image size, here the scaled and padded image size is calculated from the original image shape, but it would probably be easier just to pass it...)
        offset_norm = -offset / (image_shape / scale - 2 * offset) 
        orig_mask_shape = torch.tensor([self.masks.shape[1], self.masks.shape[2]], device=self.device, dtype=self.dtype) - 1 # Get the shape of the masks
        # Convert the normalized offset to the coordinates of the masks
        offset_mask_coords = offset_norm * orig_mask_shape 
        # Round the coordinates to the nearest integer and convert to long (needed for indexing)
        offset_mask_coords = torch.round(offset_mask_coords).long()
        self.masks.data = self.masks.data[:, offset_mask_coords[0]:(-(offset_mask_coords[0] + 1) if offset_mask_coords[0] != 0 else None), offset_mask_coords[1]:(-(offset_mask_coords[1] + 1) if offset_mask_coords[1] != 0 else None)] # Slice out the padded parts of the masks
        
        if self.time:
            end.record()
            torch.cuda.synchronize()
            print(f'Offsetting, scaling and padding took {start.elapsed_time(end)/1000:.3f} s')
        
        return self
    
    def fix_boxes(self):
        """
        This function simply sets the boxes to match the masks.

        It is not strictly needed, but can be used as a sanity check to see if the boxes match the masks.
        The discrepancy between the boxes and the masks comes about by all the scaling and smoothing of the masks.
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

    def non_max_suppression(self, iou_threshold, **kwargs):
        """
        Simply wraps the nms_masks function from yolo_helpers.py, and removes the elements that were not selected.
        """
        if self.time:
            # Initialize timing calculations
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            len_before = len(self)

        # Skip if there are no elements to merge
        if len(self) <= 1:
            return self

        # Perform non-maximum suppression on the masks, using the scales as weights, that is the highest resolution masks are given the highest priority
        image_to_mask_scale = torch.tensor([self.image.shape[1] / self.masks.data.shape[1], self.image.shape[2] / self.masks.data.shape[2]], device=self.device, dtype=self.dtype)
        if self.PREFER_POLYGONS:
            nms_ind = nms_polygons(self.polygons, torch.tensor(self.scales, dtype=self.dtype, device=self.device) * self.confs, iou_threshold=iou_threshold, return_indices=True, dtype=self.dtype, boxes=self.boxes / image_to_mask_scale.repeat(2).unsqueeze(0), **kwargs)
        else:
            nms_ind = nms_masks(self.masks.data, torch.tensor(self.scales, dtype=self.dtype, device=self.device) * self.confs, iou_threshold=iou_threshold, return_indices=True, boxes=self.boxes / image_to_mask_scale.repeat(2).unsqueeze(0), **kwargs)
        # Remove the elements that were not selected
        self = self[nms_ind]
        if self.time:
            end.record()
            torch.cuda.synchronize()
            print(f'Non-maximum suppression took {start.elapsed_time(end)/1000:.3f} s for removing {len_before - len(nms_ind)} elements of {len_before} elements')
        return self

    @property
    def contours(self):
        """
        This function wraps the openCV.findContours function, and uses openCV.contourArea to select the largest contour for each mask.
        """
        if self.PREFER_POLYGONS:
            return self.polygons
        else:
            return [self.contour_to_image_coordinates(find_contours(create_contour_mask(mask), largest_only=True, simplify=False)) for mask in self.masks.data]
    
    @contours.setter
    def contours(self, value):
        self.masks = contours_to_masks(value, self.mask_height, self.mask_width).to(self.device)

    def contour_to_image_coordinates(self, contour : torch.Tensor, scale : float=1, interpolate : bool=False):
        """
        Converts a contour from mask coordinates to image coordinates. 
        """
        mask_h, mask_w = self.masks.data.shape[1:]
        image_h, image_w = self.image.shape[1:]
        mask_to_image_scale = torch.tensor([(image_h - 1) / (mask_h - 1), (image_w - 1) / (mask_w - 1)], device=self.device, dtype=torch.float32) * scale
        scaled_contour = scale_contour(contour.cpu().numpy(), mask_to_image_scale.cpu().numpy(), True)
        scaled_contour = simplify_contour(scaled_contour, (mask_to_image_scale / 2).mean().item())
        scaled_contour = torch.tensor(scaled_contour, device=self.device, dtype=self.dtype)

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
    
    def plot(self, linewidth=2, masks=True, boxes=True, conf=True, outpath=None, scale=1):
        # Convert torch tensor to numpy array
        image = self.image.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
                    # image[create_contour_mask(resize_mask(self.masks.data[i], (ih, iw)), width=linewidth).cpu().numpy().astype(bool)] = (0, 0, 255)

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(20, 20))
            plt.imshow(image)
            plt.gca().axis('off')
            plt.show()

    def save_crops(self, outdir=None, basename=None, mask=False, identifier=None):
        if basename is None:
            assert self.image_path is not None, RuntimeError("Cannot save crops without image_path")
            basename, _ = os.path.splitext(os.path.basename(self.image_path))
        assert outdir is not None, RuntimeError("Cannot save crops without outpath")
        assert os.path.isdir(outdir), RuntimeError(f"outpath {outdir} is not a directory")
        _, image_ext = os.path.splitext(os.path.basename(self.image_path))
        if mask:
            image_ext = ".png"

        # For each bounding box, save the corresponding crop
        for i, (_box, _mask) in enumerate(zip(self.boxes, self.masks)):
            # Define name of the crop 
            x1, y1, x2, y2 = _box.long().cpu().tolist()
            crop_name = f"crop_{basename}_CROPNUMBER_{i}_UUID_{identifier}{image_ext}"
            # Extract the crop from the image tensor
            if mask:
                if self.PREFER_POLYGONS:
                    contour_offset = torch.tensor([x1, y1], device=self.device, dtype=self.dtype)
                    scaled_mask = contours_to_masks([self.contour_to_image_coordinates(self.contours[i], scale=1, interpolate=False) - contour_offset], y2 - y1, x2 - x1)
                else:
                    scaled_mask = resize_mask(_mask.data, self.image.shape[1:])[:, y1:y2, x1:x2]
                crop = torch.cat((self.image[:, y1:y2, x1:x2] / 255.0, scaled_mask.to(self.dtype)), dim=0)
            else:
               crop = self.image[:, y1:y2, x1:x2] / 255.0 
            # Save the crop
            torchvision.utils.save_image(crop, os.path.join(outdir, crop_name))

    def serialize(self, outpath : str=None, save_json : bool=True, save_pt : bool=False, readme : bool=True, identifier : str=None) -> None:
        """
        This function serializes the `TensorPredictions` object to a .pt file and/or a .json file. The .pt file contains an exact copy of the `TensorPredictions` object, while the .json file contains the data in a more human-readable format, which can be deserialized into a `TensorPredictions` object using the 'load' function.
        
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
        if ext != "":
            print(f"WARNING: serializer outpath ({outpath}) should not have a file-extension for 'TensorPredictions.serialize'!")

        # Add the basename to the outpath
        pt_path = f'{outpath}.pt'
        json_path = f'{outpath}.json'
        readme_path = f'{outpath}.md'
        
        if save_pt:
            if os.path.exists(pt_path):
                print(f"WARNING: Pickle ({pt_path}) already exists, overwriting!")
            ### First serialize as .pt file
            torch.save(self, pt_path)

        if save_json:
            if os.path.exists(json_path):
                print(f"WARNING: JSON ({json_path}) already exists, overwriting!")
            ### Then serialize as .json file
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
            json_data = {"boxes" : boxes, 
                         "contours" : contours, 
                         "confs" : confs, 
                         "classes" : classes, 
                         "scales" : scales, 
                         "identifier" : identifier if identifier else self.image_path, 
                         "image_path" : self.image_path, 
                         "image_width" : self.image.shape[2], 
                         "image_height" : self.image.shape[1], 
                         "mask_width" : self.image.shape[2] if self.PREFER_POLYGONS else self.masks.data.shape[2],
                         "mask_height" : self.image.shape[1] if self.PREFER_POLYGONS else self.masks.data.shape[1]}
            with open(json_path, 'w') as f:
                json.dump(json_data, f)

        if readme:
            # Add a readme file to the directory with some information about the serialized data
            if os.path.exists(readme_path):
                print(f"WARNING: README ({readme_path}) already exists, overwriting!")
            # TODO: Move the readme template to a separate file
            readme_text = \
f"""
# Localization results for `{identifier if identifier else self.image_path}`
This directory contains the localization predictions for the image found at `{self.image_path}`. For some pipelines, this path may be non-standard, please confer with the relevant developer for clarification.

## Files
The predictions are saved in two formats: .pt (`PyTorch` pickle) and .json (JSON).
The `PyTorch` file contains the pickled `TensorPredictions` object dictionary, while the JSON file contains the data serialized in a more human-readable format, which can reasonably be deserialized by anyone not familiar with the `TensorPredictions` object using any programming language with access to basic JSON libraries.

### JSON format
The JSON file contains the following data:
> - **`boxes`** (list of lists of integers):  
    The bounding boxes for each prediction in the format [x1, y1, x2, y2], where (x1, y1) is the bottom left corner and (x2, y2) is the top right corner. \\
    Coordinates are given in the "image pixel coordinate system".

> - **`contours`** (list of lists of lists of integers):  
    The contours for each prediction in the format [[x1, x2, ..., xn], [y1, y2, ..., yn]], where (x1, y1) is the first point, (x2, y2) is the second point, and so on. \\
    Coordinates are given in the "mask coordinate system" which is approximately proportional to the "image coordinate system". \\
    Points should be ordered in clockwise order, if not please contact the developers. 

> - **`confs`** (list of floats):  
    The confidences for each prediction.

> - **`classes`** (list of integers):  
    The classes for each prediction.

> - **`scales`** (list of floats):  
    The scale at which a given prediction was found.

> - **`identifier`** (string):  
    An identifier for the predictions.

> - **`image_path`** (string):  
    The path to the image that the predictions are for. May be non-standard.

> - **`image_width`** (integer):  
    The width of the image that the predictions are for.

> - **`image_height`** (integer):  
    The height of the image that the predictions are for.

> - **`mask_width`** (integer):  
    The width of the masks, where the contours are derived from.

> - **`mask_height`** (integer):  
    The height of the masks, where the contours are derived from.

The mask coordinates are given in the mask coordinate system, such that they must be scaled by the ratio between the image and the mask to get the image coordinates:
```
image_x = mask_x * (image_width / mask_width)
image_y = mask_y * (image_height / mask_height)
```
However care must be taken when rounding the scaled coordinates, since both coordinate systems are integer-grids.

The bounding box coordinates are given in the image coordinate system, so they do not need to be scaled to be used in the image.

### Image Coordinate System
The image coordinate system is simply the integer pixel coordinate system of the image, where the **top left** corner is (`0`; `0`) and the **bottom right** corner is (`image_width`; `image_height`).

## Deserializations
The .pt pickle file can be deserialized into a `TensorPredictions` object using `torch.load("{pt_path}")`. OBS: This may be deprecated in the future, since the .json file contains the same data in a more human-readable format, and serialization/deserialization is reasonably fast.

The JSON can be deserialized into a `TensorPredictions` object using `TensorPredictions().load("{json_path}")`.\
"""
            with open(readme_path, 'w') as f:
                f.write(readme_text)


    def load(self, path : str, device=None, dtype=None):
        """
        Deserializes a TensorPredictions object from a .pt or .json file.
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

            empty_image = torch.zeros((3, json_data["image_height"], json_data["image_width"]), device=device, dtype=dtype)
            new_tp = TensorPredictions(image=empty_image, device=device, dtype=dtype)

            # Load the data
            for k, v in json_data.items():
                # Skip attributes in the json file that are not in the TensorPredictions object
                if k in ["identifier", "image_path", "image_width", "image_height", "mask_width", "mask_height", "scales"]:
                    continue
                # Bounding boxes are easy (as usual)
                if k == "boxes":
                    v = torch.tensor(v, device=new_tp.device, dtype=new_tp.dtype)
                # While masks are a bit more complicated
                elif k == "contours":
                    # If no masks are found, we need to convert the contours to masks
                    v = [torch.tensor(vi, device=new_tp.device, dtype=torch.long).T for vi in v] # For compatibility reasons we convert to tensor, but we will convert to numpy in contours_to_masks anyway, since we are using openCV to reconstruct the masks
                    v = contours_to_masks(v, height=json_data["mask_height"], width=json_data["mask_width"])
                    # Change the attribute key to masks, since contours is a property method derived from masks and not a true property
                # Confidences and classes are 1-d tensors (arrays)
                elif k in ["confs", "classes"]:
                    v = torch.tensor(v, device=new_tp.device, dtype=new_tp.dtype)
                else:
                    raise RuntimeError(f"Unknown key in json file: {k}")
                setattr(new_tp, k, v)

            self = new_tp
            return self
        else:
            raise RuntimeError(f"Unknown file-extension: {ext} for path: {path}")
        
    def save(self, output_directory : str, overview : Union[bool, str]=True, crops : Union[bool, str]=True, metadata : Union[bool, str]=True, fast : bool=False, mask_crops : bool=False, identifier : Union[str, None]=None, basename : Union[str, None]=None) -> Union[str, None]:
        """
        Saves the serialized prediction results, crops, and overview to the given output directory.

        TODO: Add the identifier to the names of the files, so that we can save multiple predictions for the same image or images with the same name.

        Args:
            output_directory (str): The directory to save the prediction results to.
            overview (Union[bool, str], optional): Whether to save the overview image. Defaults to True. If a string is given, it is interpreted as a path to a directory to save the overview image to.
            crops (Union[bool, str], optional): Whether to save the crops. Defaults to True. If a string is given, it is interpreted as a path to a directory to save the crops to.
            metadata (Union[bool, str], optional): Whether to save the metadata. Defaults to True. If a string is given, it is interpreted as a path to a directory to save the metadata to.
            fast (bool, optional): Whether to use the fast version of the overview image. Defaults to False. Saves the overview image at half the resolution.
            mask_crops (bool, optional): Whether to mask the crops. Defaults to False.
            identifier (Union[str, None], optional): An identifier for the serialized data. Defaults to None.
            basename (Union[str, None], optional): The base name of the image. Defaults to None. If None, the base name is extracted from the image path.
        
        Returns:
            str: The path to the directory containing the serialized data - the crops and overview image(s) are also saved here by default. If the standard location is not used at all, the directory is not created and None is returned instead.
        """
        if not os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory} does not exist")
        
        if basename is None:
            # Get the base name of the image
            basename = os.path.splitext(os.path.basename(self.image_path))[0]
        # Construct the prediction directory path
        prediction_directory = os.path.join(output_directory, basename)
        # Create the prediction directory if it does not exist and it is needed (i.e. if we are saving crops, overview, or metadata to a non-standard location)
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
                fast_scale = min(1/2, 3072 / max_dim)
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

class Predictor(object):
    MIN_MAX_OBJ_SIZE = (8, 2048)
    MINIMUM_TILE_OVERLAP = 384
    EDGE_CASE_MARGIN = 64
    SCORE_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.25
    MAX_MASK_SIZE = 2048
    TIME = False
    EXPERIMENTAL_NMS_OPTIMIZATION = True
    PREFER_POLYGONS = False
    # DEBUG = False

    def __init__(self, model, cfg=None, device=torch.device("cpu"), dtype=torch.float32):
        if not cfg is None:
            raise NotImplementedError("cfg is not implemented yet")

        self._device = device
        self._dtype = dtype

        self._base_yolo = YOLO(model)
        self._base_yolo._load(model, "inference")
        self._base_yolo.load(model) 
        # self._base_yolo.fuse() # Seems to just be slower actually...
        self._model = self._base_yolo.model.to(device=device, dtype=dtype)
        self._model.eval()

        self._yolo_predictor = None

    def _detect_instances(self, tensor : torch.Tensor, scale=1.0, max_scale : bool=False):
        TILE_SIZE = 1024
        this_MIN_MAX_OBJ_SIZE = list(self.MIN_MAX_OBJ_SIZE)
        this_EDGE_CASE_MARGIN = self.EDGE_CASE_MARGIN
        # If we are at the top level, we don't want to remove large instances - since there are no layers above to detect them as small instances
        if max_scale:
            this_MIN_MAX_OBJ_SIZE[1] = 4096
            this_EDGE_CASE_MARGIN = 0

        if self.TIME:
            # Initialize timing calculations
            start_detect = torch.cuda.Event(enable_timing=True)
            end_detect = torch.cuda.Event(enable_timing=True)
            start_detect.record()
        orig_h, orig_w = tensor.shape[1:]
        w, h = orig_w, orig_h
        padded = False
        h_pad, w_pad = 0, 0
        pad_lrtb = 0, 0, 0, 0
        real_scale = 1, 1

        # Check dimensions and channels
        assert tensor.device == self._device, RuntimeError(f"tensor.device {tensor.device} != self._device {self._device}")
        assert tensor.dtype == self._dtype, RuntimeError(f"tensor.dtype {tensor.dtype} != self._dtype {self._dtype}")

        # Resize if scale is not 1
        if scale != 1:
            h, w = round(orig_h * scale / 4) * 4, round(orig_w * scale / 4) * 4
            real_scale = w / orig_w, h / orig_h
            resize = transforms.Resize((h, w), antialias=True) # Ensure that the width and height are even
            tensor = resize(tensor)
            h, w = tensor.shape[1:]
        # If any of the sides are smaller than the TILE_SIZE, pad to TILE_SIZE
        if w < TILE_SIZE or h < TILE_SIZE:
            padded = True
            w_pad = max(0, TILE_SIZE - w) // 2
            h_pad = max(0, TILE_SIZE - h) // 2
            pad_lrtb = w_pad, w_pad + (w % 2 == 1), h_pad, h_pad + (h % 2 == 1)
            tensor = F.pad(tensor, pad_lrtb, mode="constant", value=0) # Pad with black
            h, w = tensor.shape[1:]

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
        # ims = torch.stack([tensor[:, o[0]: (o[0] + TILE_SIZE), o[1]: (o[1] + TILE_SIZE)] for (m, n), o in offsets], dim=0)
        # assert len(ims) == (x_n_tiles * y_n_tiles), RuntimeError(f"len(ims) {len(ims)} != (x_n_tiles * y_n_tiles) {x_n_tiles} * {y_n_tiles} ({x_n_tiles * y_n_tiles})")
        
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
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(offsets), batch_size):
                if self.TIME:
                    # Initialize batch timing calculations
                    start_batch_event = torch.cuda.Event(enable_timing=True)
                    end_fetch_event = torch.cuda.Event(enable_timing=True)
                    end_forward_event = torch.cuda.Event(enable_timing=True)
                    end_postprocess_event = torch.cuda.Event(enable_timing=True)
                    end_batch_event = torch.cuda.Event(enable_timing=True)
                    # Record batch start
                    start_batch_event.record()

                # batch = ims[i:min((i+batch_size), len(ims))]
                batch = torch.stack([tensor[:, o[0]: (o[0] + TILE_SIZE), o[1]: (o[1] + TILE_SIZE)] for (m, n), o in offsets[i:min((i+batch_size), len(offsets))]], dim=0)
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
                    tps = postprocess(tps, batch, max_det=1000, min_confidence=self.SCORE_THRESHOLD, iou_threshold=self.IOU_THRESHOLD, edge_margin=this_EDGE_CASE_MARGIN, valid_size_range=this_MIN_MAX_OBJ_SIZE, nms=3, group_first=self.EXPERIMENTAL_NMS_OPTIMIZATION) # Important to prune within each tile first, this avoids having to carry around a lot of data
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
        
        ### DEBUG #####
        # if self.DEBUG:
        # print(f'Number of tiles processed before merging and plotting: {len(ps)}')
        # for i in range(len(ps)):
        #     ps[i].orig_img = (ps[i].orig_img.detach().contiguous() * 255).to(torch.uint8).cpu().numpy() # Needed for compatibility with the Results.plot function
        # fig, axs = plt.subplots(y_n_tiles, x_n_tiles, figsize=(x_n_tiles * 5, y_n_tiles * 5))
        # axs = axs.flatten() if len(offsets) > 1 else [axs]
        # [axs[i].imshow(p.plot(pil=False, masks=True, probs=False, labels=False, kpt_line=False)) for i, p in enumerate(ps)]
        # plt.savefig(f"debug_{scale:.3f}_fraw.png", dpi=300)
        # for i in range(len(ps)):
        #     ps[i].orig_img = torch.tensor(ps[i].orig_img).squeeze(0).to(dtype=self._dtype, device=self._device) / 255.0 # Backtransform
        #################

        ## Combine the results from the tiles
        MASK_SIZE = 256 # Defined by the YOLOv8 model segmentation architecture
        MASK_TO_IMG_RATIO = MASK_SIZE / torch.tensor([TILE_SIZE, TILE_SIZE], dtype=torch.float32, device=self._device).unsqueeze(0)
        # For the boxes, we can simply add the offsets (and possibly subtract the padding)
        box_offsetters = torch.tensor([[o[1][0] - pad_lrtb[2], o[1][1] - pad_lrtb[0]] for o in offsets], dtype=torch.float32, device=self._device)
        # However for the masks, we need to create a new mask which can contain every tile, and then add the masks from each tile to the correct area - this will of course use some memory, but it's probably not too bad
        # Since the masks do not have the same size as the tiles, we need to scale the offsets
        mask_offsetters = box_offsetters * MASK_TO_IMG_RATIO
        # We also need to round the offsets, since they may not line up with the pixel-grid - RE: Now they do since I made sure the offsets are multiples of 4
        mask_offsetters = torch.round(mask_offsetters).long()
        # The padding must also be scaled and subtracted from the new mask size
        new_mask_size = ((mask_offsetters.max(dim=0).values + MASK_SIZE) - torch.tensor(pad_lrtb[1::2][::-1], dtype=torch.long, device=self._device) * MASK_TO_IMG_RATIO[0]).tolist()
        # Finally, we can merge the results - this function basically just does what I described above
        orig_img = tensor[:, pad_lrtb[2]:(-pad_lrtb[3] if pad_lrtb[3] != 0 else None), pad_lrtb[0]:(-pad_lrtb[1] if pad_lrtb[1] != 0 else None)] if padded else tensor
        ps = merge_tile_results(ps, orig_img.permute(1,2,0), box_offsetters=box_offsetters.to(self._dtype), mask_offsetters=mask_offsetters, new_shape=new_mask_size, clamp_boxes=(h - sum(pad_lrtb[2:]), w - sum(pad_lrtb[:2])), max_mask_size=self.MAX_MASK_SIZE, exclude_masks=self.PREFER_POLYGONS)

        # # Apply the size filters - This could be done before merging the tiles, but would require some scaling logic
        # ps_sqrt_area = ((ps.boxes.data[:,2:4] - ps.boxes.data[:,:2]).log().sum(dim=1) / 2).exp()
        # # New criteria prunes images based on their area
        # big_enough = (ps_sqrt_area > self.MIN_MAX_OBJ_SIZE[0]) & (ps_sqrt_area < self.MIN_MAX_OBJ_SIZE[1])
        # ps = ps[big_enough]

        #### DEBUG #####
        # if self.DEBUG:
        # print(f'Number of tiles processed after merging and filtering: {len(ps)}')
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        # ps.orig_img = (ps.orig_img.detach().contiguous() * 255).to(torch.uint8).cpu().numpy() # Needed for compatibility with the Results.plot function
        # # ps.boxes.data[:, :4] /= scale
        # print(ps.orig_img.shape)
        # ax.imshow(ps.plot(pil=False, masks=True, probs=False, labels=False, kpt_line=False))
        # plt.savefig(f"debug_{scale:.3f}_merged.png", dpi=300)
        # ps.orig_img = torch.tensor(ps.orig_img).squeeze(0).to(dtype=self._dtype, device=self._device) / 255.0 # Backtransform
        # # ps.boxes.data[:, :4] *= scale
        #################

        if self.TIME:
            end_detect.record()
            torch.cuda.synchronize(device=self._device)
            total_detect_time = start_detect.elapsed_time(end_detect) / 1000 # Convert to seconds
            pred_prop = total_elapsed / total_detect_time
            print(f'Prediction time: {total_elapsed:.3f}s/{pred_prop*100:.3g}% (overhead: {overhead_prop * 100:.1f}) | Fetch {fetch_prop * 100:.1f}% | Forward {forward_prop * 100:.1f}% | Postprocess {postprocess_prop * 100:.1f}%)')
            self.total_detection_time += total_detect_time
            self.total_forward_time += forward_time
        return Prepared_Results(ps, scale=real_scale, device=self._device, dtype=self._dtype)

    def pyramid_predictions(self, image, path=None, scale_increment=2 / 3, scale_before=1, single_scale=False):
        if self.TIME:
            # Initialize timing calculations
            start_pyramid = torch.cuda.Event(enable_timing=True)
            end_pyramid = torch.cuda.Event(enable_timing=True)
            start_pyramid.record()

        if isinstance(image, str):
            im = read_image(image, ImageReadMode.RGB).to(device=self._device, dtype=self._dtype)
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
        edge_case_margin_padding_multiplier = 2
        padding_for_edge_cases = transforms.Pad(padding=self.EDGE_CASE_MARGIN * edge_case_margin_padding_multiplier, fill=0, padding_mode='constant')
        padding_offset = torch.tensor((self.EDGE_CASE_MARGIN, self.EDGE_CASE_MARGIN), dtype=self._dtype, device=self._device) * edge_case_margin_padding_multiplier
        if padding_offset.sum() > 0:
            transform_list.append(padding_for_edge_cases)
        if transform_list:
            transforms_composed = transforms.Compose(transform_list)
        
        im_b = transforms_composed(im) if transform_list else im

        # Check dimensions and channels
        assert im_b.dim() == 3, f"Image is not 3-dimensional"
        assert im_b.size(0) == 3, f"Image does not have 3 channels"

        max_dim = max(im_b.shape[1:])

        if single_scale:
            scales = [1]
        else:
            # fixme, what to do if the image is too small? - RE: Fixed by adding padding in _detect_instances
            scales = []
            s = 1024 / max_dim

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
        all_preds = [self._detect_instances(im_b, scale=s,max_scale=s==min(scales)) for s in reversed(scales)] # 

        if self.TIME:
            print(f'Total detection time: {self.total_detection_time:.3f}s ({self.total_forward_time / self.total_detection_time * 100:.3g}% forward)')

        all_preds = TensorPredictions(
            predictions     = all_preds, 
            image           = im, 
            image_path      = path,
            dtype           = self._dtype,
            device          = self._device,
            time            = self.TIME,
            PREFER_POLYGONS = self.PREFER_POLYGONS
        ).offset_scale_pad(
            offset  = -padding_offset, 
            scale   = 1 / scale_before,
            pad     = 5 # pad the boxes a bit to ensure they encapsulate the masks
        ).non_max_suppression(
            iou_threshold = self.IOU_THRESHOLD,
            # metric        = 'IoU', # Currently only IoU is supported and setting this will raise an error
            group_first = self.EXPERIMENTAL_NMS_OPTIMIZATION
        )

        if self.TIME:
            # Finish timing calculations
            end_pyramid.record()
            torch.cuda.synchronize(device=self._device)
            total_pyramid_time = start_pyramid.elapsed_time(end_pyramid) / 1000
            print(f'Total pyramid time: {total_pyramid_time:.3f}s ({self.total_detection_time / total_pyramid_time * 100:.3g}% detection | {self.total_forward_time / total_pyramid_time * 100:.3g}% forward)')

        return all_preds
        