import torch
import torch.nn.functional as F
import torchvision

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from ultralytics.utils import ops
from ultralytics.engine.results import Results, Masks

from functools import lru_cache

from typing import Union


class ResultsWithTiles(Results):
    def __init__(self, tiles=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiles = tiles

    def new(self):
        new = super().new()
        new.tiles = self.tiles
        return new

def offset_box(boxes, offset, max_x = None, max_y = None):
    m = 4 / offset.shape[0]
    assert m // 1 == m, f"4 must be divisible by the number of offsets ({offset.shape[0]})"
    boxes[:, :4] += offset.unsqueeze(0).repeat(1, int(m))
    if max_y is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, max_y - 1)
    if max_x is not None:
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, max_x - 1)
    return boxes

def offset_mask(mask, offset, new_shape=None, max_size=None):
    n, w, h = mask.shape
    if new_shape is not None: #isinstance(new_shape, tuple) or isinstance(new_shape, list) or isinstance(new_shape, torch.Tensor) and len(new_shape.shape) == 2:
        assert len(new_shape) == 2, f"new_shape must be a tuple or list of length 2, not {len(new_shape)}"
        new_shape = int(n), int(new_shape[0]), int(new_shape[1])
    elif new_shape is None:
        raise ValueError("new_shape must be specified")
    # Calculate the possible clamped size of the mask (if it needs to be clamped)
    clamp_factor = (max(new_shape[1:]) / max_size) if max_size is not None else 1
    clamped_size = [int(s // clamp_factor) for s in new_shape[1:]]
    # Edge case: If the mask is empty and the clamp_factor is greater than 1, return an empty mask at the clamped size
    if n == 0 and clamp_factor > 1:
        return torch.zeros([0] + clamped_size, dtype=torch.bool, device=mask.device)
    # Edge case: If the mask is empty and the clamp_factor is less than or equal to 1, return an empty mask at the new size
    new_mask = torch.zeros(new_shape, dtype=torch.bool, device=mask.device)
    if n == 0:
        return new_mask
    ## Check the offsets
    # Positivity check
    assert all([ofs >= 0 for ofs in offset]), f"offsets must be positive, not {offset}"
    # Bounds check
    # Width
    assert offset[0] + w <= new_shape[1], f"offset[0] ({offset[0]}) + w ({w}) must be less than or equal to new_shape[1] ({new_shape[1]})"
    # Height
    assert offset[1] + h <= new_shape[2], f"offset[1] ({offset[1]}) + h ({h}) must be less than or equal to new_shape[2] ({new_shape[2]})"

    new_mask[:, offset[0]:(offset[0]+w), offset[1]:(offset[1]+h)] = mask
    # Due to memory use, it is beneficial to restrict the maximum size of the masks. A 700x700 boolean tensor uses ~0.5 MB of memory
    if clamp_factor > 1:
        # Downsample the mask by the clamp_factor for each dimension, such that the largest dimension is clamped to max_size. The minor dimension may be smaller than max_size.
        resizer = torchvision.transforms.Resize(clamped_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        new_mask = resizer(new_mask)
    return new_mask

def merge_tile_results(results = list[Results], orig_img=None, box_offsetters=None, mask_offsetters=None, new_shape=None, clamp_boxes=(None, None), max_mask_size=(700, 700)):
    """
    Merges results from multiple images into a single Results object, possibly with a new image.
    """
    _device = results[0].boxes.data.device
    if orig_img is None:
        orig_img = results[0].orig_img
    assert isinstance(orig_img, torch.Tensor), f"orig_img must be a torch.Tensor, not {type(orig_img)}"
    if box_offsetters is None:
        box_offsetters = torch.zeros((len(results), 2), device=_device).int()
    else:
        box_offsetters = box_offsetters.int()
    if mask_offsetters is None:
        mask_offsetters = torch.zeros((len(results), 2), device=_device).int()
    else:
        mask_offsetters = mask_offsetters.int()
    mx, my = clamp_boxes
    path = results[0].path
    names = results[0].names
    tile_indices = torch.concatenate([torch.tensor([i] * len(r), dtype=torch.long, device=_device) for i, r in enumerate(results)])
    boxes = torch.cat([offset_box(r.boxes.data, o.flip(0), mx, my) for r, o in zip(results, box_offsetters)])
    masks = torch.cat([offset_mask(r.masks.data, o, new_shape, max_mask_size) for r, o in zip(results, mask_offsetters)]) # Instead of offsetting the masks statically, we instantiate a new OffsetMask object which offsets the masks dynamically
    if not all([r.probs is None for r in results]):
        raise NotImplementedError("'Probs' not implemented yet")
    if not all([r.keypoints is None for r in results]):
        raise NotImplementedError("'Keypoints' not implemented yet")
    return ResultsWithTiles(tiles=tile_indices, orig_img=orig_img, path=path, names=names, boxes=boxes, masks=masks, probs=None, keypoints=None)

def smooth_mask(mask, kernel_size=7):
    # Check kernel size parameter
    if kernel_size == 1:
        return mask
    elif kernel_size < 1:
        print(f"kernel_size ({kernel_size}) should be greater than or equal to 1")
        return mask
    elif kernel_size > 50:
        print(f'Very large kernel_size ({kernel_size}) may cause memory issues')
    
    def gaussian_kernel(size):
        if size % 2 == 0:
            raise ValueError("Size must be an odd number")
        w = size // 2
        sigma = w / 3

        # Create a coordinate grid
        x, y = torch.meshgrid(torch.arange(size) - w, torch.arange(size) - w)

        # Calculate the 2D Gaussian kernel
        g = torch.exp(-(x**2 + y**2) / (2*sigma**2))
        g[w, w] = 1

        # Normalize the kernel to ensure the sum is 1
        return g / g.sum()

    # Create a 2D Gaussian kernel
    kernel = gaussian_kernel(kernel_size).to(mask.device).unsqueeze(0).unsqueeze(0)

    # Apply convolution separately to each mask
    return torch.nn.functional.conv2d(mask.float().unsqueeze(1), kernel, padding=kernel_size // 2).squeeze(1) > 0.5


def stack_masks(masks, orig_shape=None, antialias=False):
    """
    Stacks a list of ultralytics.engine.results.Masks objects (or torch.Tensor) into a single ultralytics.engine.results.Masks object.

    If the masks are not all the same size, they are resized to the largest size in the list.

    Args:
        masks (list): A list of ultralytics.engine.results.Masks objects (or torch.Tensor).
        orig_shape (tuple, optional): The original shape of the image. Defaults to None. If None, the original shape is inferred from the first Masks object in the list if there is one, otherwise the original shape None.
        antialias (bool, optional): A flag to indicate whether to use antialiasing when resizing the masks. Defaults to False.

    Returns:
        ultralytics.engine.results.Masks: A Masks object containing the stacked masks.
    """
    assert isinstance(masks, list), f"'masks' must be a list, not {type(masks)}"
    for m in masks:
        if isinstance(m, Masks):
            orig_shape = m.orig_shape
            break
    masks = [m.data if isinstance(m, Masks) else m for m in masks]
    assert all([isinstance(m, torch.Tensor) for m in masks]), f"'masks' must be a list of torch.Tensor, not {type(masks[0])}"
    assert len(masks) != 0, f"'masks' ({masks}) must not be empty"
    _device = masks[0].device

    max_h = max([m.shape[1] for m in masks])
    max_w = max([m.shape[2] for m in masks])
    masks_in_each = [len(m) for m in masks]
    interpolation_method = torchvision.transforms.InterpolationMode.NEAREST
    resizer = torchvision.transforms.Resize((max_h, max_w), interpolation=interpolation_method, antialias=antialias)

    new_masks = torch.zeros((sum(masks_in_each), max_h, max_w), dtype=torch.bool, device=_device)

    i = 0
    for n, m in zip(masks_in_each, masks):
        if n == 0:
            continue
        if not (m.shape[1] == max_h and m.shape[2] == max_w):
            if not antialias:
                m = resizer(m)
            else:
                smooth_radius = torch.ceil(torch.tensor([max_h, max_w]) / torch.tensor(m.shape[1:])).max().long().item() * 4
                if smooth_radius % 2 == 0:
                    smooth_radius += 1
                m = smooth_mask(resizer(m), smooth_radius)
        new_masks[i:(i+n)] = m
        i += n

    return Masks(new_masks, orig_shape=orig_shape)

def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (torch.Tensor): [n, h, w] tensor of masks
        boxes (torch.Tensor): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        (torch.Tensor): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW <- This line has been changed from the original implementation, which had a superfluous type conversion which caused YOLOv8 to cast the masks to float32, this change simply removes the type conversion enabling support for other data types

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5).bool()

def fix2btlr(rects : torch.Tensor):
    """
    If a rectangle is not in the format [x_min, y_min, x_max, y_max], this function will fix it.
    """
    return torch.cat((torch.min(rects[:, :2], rects[:, 2:]), torch.max(rects[:, :2], rects[:, 2:])), dim=1, out=rects)

def check_bltr_validity(rects : torch.Tensor, strict=True):
    """
    This function checks if the rectangles are in the format [x_min, y_min, x_max, y_max].
    """
    if len(rects.shape) == 1 and not rects.shape[0] == 4 or len(rects.shape) == 2 and not rects.shape[1] == 4:
        raise ValueError(f"Rectangles must be of shape (n, 4), not {rects.shape}")
    if len(rects.shape) == 1:
        rects = rects.unsqueeze(0)
    # Check if the bottom left corner is less than the top right corner
    checks = (rects[:, 0] < rects[:, 2]) & (rects[:, 1] < rects[:, 3])
    if not (checks).all():
        if strict:
            raise ValueError(f"Bottom left corner ({rects[~checks, :2]}) must be less than top right corner ({rects[~checks, 2:]}) of rectangles {torch.where(~checks)[0].tolist()}")
        else:
            return False
    else:
        return True

def intersect(rect1, rect2s, area_only=False, debug=False):
    """
    Calculates the intersection of a rectangle with a set of rectangles.
    """
    if len(rect1.shape) == 1 and not rect1.shape[0] == 4 or len(rect1.shape) == 2 and not rect1.shape[1] == 4:
        raise ValueError(f"Rectangles must be of shape (n, 4), not {rect1.shape}")
    if len(rect2s.shape) == 1 and not rect2s.shape[0] == 4 or len(rect2s.shape) == 2 and not rect2s.shape[1] == 4:
        raise ValueError(f"Rectangles must be of shape (n, 4), not {rect2s.shape}")
    if len(rect1.shape) == 1:
        rect1 = rect1.unsqueeze(0)
    if len(rect2s.shape) == 1:
        rect2s = rect2s.unsqueeze(0)

    # Safer to enable this, but it is slower
    # # Check the validity of the rectangles
    # if not check_bltr_validity(rect1, debug):
    #     rect1 = fix2btlr(rect1)
    # if not check_bltr_validity(rect2s, debug):
    #     rect2s = fix2btlr(rect2s)

    # Calculate vectors from each corner of rect1 to each corner of rect2s
    blbltrtr = rect2s - rect1
    bl_to_bl = blbltrtr[:, :2]
    tr_to_tr = blbltrtr[:, 2:] 
    bltrtrbl = rect2s[:, [2, 3, 0, 1]] - rect1
    bl_to_tr = bltrtrbl[:, :2]
    tr_to_bl = bltrtrbl[:, 2:]
    
    # Determine if each corner of rect1 is inside each rect2
    inside_tr = (tr_to_tr[:, 0] >= 0) & (tr_to_tr[:, 1] >= 0) & (tr_to_bl[:, 0] <= 0) & (tr_to_bl[:, 1] <= 0)
    inside_bl = (bl_to_bl[:, 0] <= 0) & (bl_to_bl[:, 1] <= 0) & (bl_to_tr[:, 0] >= 0) & (bl_to_tr[:, 1] >= 0)
    inside_tl = (bl_to_bl[:, 0] <= 0) & (tr_to_tr[:, 1] >= 0) & (bl_to_tr[:, 0] >= 0) & (tr_to_bl[:, 1] <= 0)
    inside_br = (tr_to_tr[:, 0] >= 0) & (bl_to_bl[:, 1] <= 0) & (tr_to_bl[:, 0] <= 0) & (bl_to_tr[:, 1] >= 0)

    # Check for enclosure
    enclosure = (rect1[:, :2] <= rect2s[:, :2]) & (rect1[:, 2:] >= rect2s[:, 2:])

    # Check for intersection with the "infinitely" extended cross of rect1
    in_cross = ((bl_to_bl[:, 0] <= 0) & (bl_to_tr[:, 0] >= 0)) | ((tr_to_tr[:, 0] >= 0) & (tr_to_bl[:, 0] <= 0)), ((bl_to_bl[:, 1] <= 0) & (bl_to_tr[:, 1] >= 0)) | ((tr_to_tr[:, 1] >= 0) & (tr_to_bl[:, 1] <= 0))

    # Check for equality - if equal, return the original rectangles
    zero = torch.tensor(0, dtype=rect1.dtype, device=rect1.device)
    is_equal = (bl_to_bl.isclose(zero).all(dim=1)) & (tr_to_tr.isclose(zero).all(dim=1))

    # Check for no intersection - if no intersection, return the intersection rectangle [0, 0, 0, 0]
    is_intersecting = inside_tl | inside_br | inside_bl | inside_tr | (enclosure[:, 0] & in_cross[1]) | (enclosure[:, 1] & in_cross[0]) | (enclosure[:, 0] & enclosure[:, 1]) | is_equal

    if not area_only:
        intersections = is_intersecting.unsqueeze(1) * torch.cat((torch.max(rect1[:, :2], rect2s[:, :2]), torch.min(rect1[:, 2:], rect2s[:, 2:])), dim=1)
        intersections[is_equal] = rect1
    else:
        intersections = torch.zeros(rect2s.shape[0], dtype=rect1.dtype, device=rect1.device)
        intersections[is_intersecting] = (torch.min(rect1[:, 2:], rect2s[is_intersecting, 2:]) - torch.max(rect1[:, :2], rect2s[is_intersecting, :2])).abs().prod(dim=1)

    if debug:
        # Used for debugging
        return intersections, torch.stack((inside_tl, inside_tr, inside_br, inside_bl), dim=1), enclosure, in_cross
    else:
        return intersections
    
def iou_boxes(rectangles):
    """
    Calculates the intersection over union (IoU) of a set of rectangles.

    Args:
        rectangles (torch.Tensor): A tensor of shape (n, 4), where n is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles.

    Returns:
        torch.Tensor: A tensor of shape (n, n), where n is the number of rectangles, containing the IoU of each rectangle with each other rectangle.
    """
    if not len(rectangles.shape) == 2 and rectangles.shape[1] == 4:
        raise ValueError(f"Rectangles must be of shape (n, 4), not {rectangles.shape}")
    # Could improve stability, but would require making a copy of the boxes tensor
    rectangles = (rectangles / (rectangles[:, :4].max() + 1e-6)) * (10**(3/2)) # Normalize the box-coordinates to the square root of 1000 such that the areas are [0, 1000].
    # Calculate the areas of the rectangles
    areas = (rectangles[:, 2:] - rectangles[:, :2]).abs().prod(dim=1)
    # Calculate the intersections of the rectangles with each other
    intersections = torch.stack([intersect(rect, rectangles, area_only=True) for rect in rectangles])
    # Calculate the IoU
    ious = intersections / (areas.unsqueeze(1) + areas.unsqueeze(0) - intersections + 1e-3)
    return ious

def iou_boxes_2sets(rectangles1, rectangles2):
    """
    Calculates the intersection over union (IoU) of a set of rectangles with another set of rectangles.

    Args:
        rectangles1 (torch.Tensor): A tensor of shape (n, 4), where n is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles.
        rectangles2 (torch.Tensor): A tensor of shape (m, 4), where m is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles.

    Returns:
        torch.Tensor: A tensor of shape (n, m), where n is the number of rectangles in rectangles1 and m is the number of rectangles in rectangles2, containing the IoU of each rectangle in rectangles1 with each rectangle in rectangles2.
    """
    if not len(rectangles1.shape) == 2 or rectangles1.shape[1] != 4:
        if len(rectangles1.shape) == 1 and rectangles1.shape[0] == 4:
            rectangles1 = rectangles1.unsqueeze(0)
        else:
            raise ValueError(f"Rectangles must be of shape (n, 4), not {rectangles1.shape}")
    if not len(rectangles2.shape) == 2 or rectangles2.shape[1] != 4:
        if len(rectangles2.shape) == 1 and rectangles2.shape[0] == 4:
            rectangles2 = rectangles2.unsqueeze(0)
        else:
            raise ValueError(f"Rectangles must be of shape (n, 4), not {rectangles2.shape}")
    # Calculate the areas of the rectangles
    areas1 = (rectangles1[:, 2:] - rectangles1[:, :2]).abs().prod(dim=1)
    if rectangles1.shape[0] > 1:
        areas1 = areas1.T
    areas2 = (rectangles2[:, 2:] - rectangles2[:, :2]).abs().prod(dim=1)
    # Calculate the intersections of the rectangles with each other
    intersections = torch.stack([intersect(rect, rectangles2, area_only=True) for rect in rectangles1])

    ious = intersections / (areas1.unsqueeze(1) + areas2.unsqueeze(0) - intersections + 1e-3)

    return ious

def nms_(objects, iou_fun, scores, iou_threshold=0.5, strict=True, return_indices=False, **kwargs) -> Union[torch.Tensor, tuple]:
    """
    Implements the standard non-maximum suppression algorithm.

    Args:
        objects (any): An object which can be indexed by a tensor of indices.
        iou_fun (function): A function which takes an anchor object and a comparison set (not in the Python sense) of (different) objects and returns the IoU of the anchor object with each object in the comparison set as a tensor of shape (1, n). 
            The reason it is not just (n, ) is to allow for implementations of iou_fun functions between two sets, where the IoU is calculated between each pair of objects from distinct sets.
        scores: A tensor of shape (n, ) containing the "scores" of the objects, this can merely be though of as a priority score, where the higher the score, the higher the priority of the object - it does not have to be a probability/confidence.
        iou_threshold (float, optional): The IoU threshold for non-maximum suppression. Defaults to 0.5.
        strict (bool, optional): A flag to indicate whether to perform strict checks on the algorithm. Defaults to True.
        return_indices (bool, optional): A flag to indicate whether to return the indices of the picked objects or the objects themselves. Defaults to False. If True, both the picked objects and scores are returned.
        **kwargs: Additional keyword arguments to be passed to the iou_fun function.

    Returns:
        torch.Tensor: A tensor of shape (n, ) containing the indices of the picked objects.
            or
        tuple of length 2: A tuple containing the picked objects and their scores.
    """
    if len(objects) == 0 or len(objects) == 1:
        if return_indices:
            return torch.arange(len(objects))
        else:
            return objects, scores

    # Sort the boxes by score (implicitly)
    indices = torch.argsort(scores, descending=True)

    # Initialize tensors for winners (selected boxes), possible boxes and counters
    winners = []
    possible = torch.ones(objects.shape[0], dtype=torch.bool, device=objects.device)
    left = len(objects)
    i, n = 0, 0

    while True:
        # If there is only one box left, add it to the picked indices and break
        if left <= 1:
            if left == 1:
                i = possible.nonzero().min().item()
                possible[i] = False
                winners.append(i)
            break
        i = possible.nonzero().min().item()
        # Remove the current box from the possible boxes in the next iterations and add it to the picked indices
        possible[i] = False
        winners.append(i)
        # Calculate the IoU
        ious = iou_fun(objects[indices[i]], objects[indices[possible]], **kwargs).squeeze(0)
        # Get the indices of the boxes with an IoU greater than the threshold
        losers = ious > iou_threshold
        # Remove the boxes with an IoU greater than the threshold from the possible boxes
        possible[possible.clone()] &= ~losers

        # In/Decrement the counters
        increment = losers.sum().item() + 1
        left -= increment
        n += 1
        if strict:
            assert left == possible.sum().item(), f"left ({left}) != possible.sum() ({possible.sum().item()})"
            assert n == len(winners), f"n ({n}) != winners.sum() ({len(winners)})"


    # Map the indices back to the original indices and sort them (returns boxes, scores & indices in the original order of the input)
    winners = torch.tensor(winners, dtype=torch.long, device=objects.device)
    winners = indices[winners].sort().values 
    
    # Return the boxes and scores that were picked
    if return_indices:
        return winners
    else:
        return objects[winners], scores[winners]
    
def fancy_nms_boxes(objects, iou_fun, scores, iou_threshold=0.5, return_indices=False):
    """
    This is a 'fancy' implementation of non-maximum suppression. It is not as fast as the non-maximum suppression algorithm, nor does it follow the exact same algorithm, but it is more readable and easier to debug.

    The algorithm works as follows:
        1. Sort the objects by score (implicitly)
        2. Calculate the IoU matrix
        3. Create a boolean matrix where IoU > iou_threshold 
        4. Fold the boolean matrix sequentially (i.e. row_i = row_i + row_i-1 + ... + row_0)
           (The values on the diagonal of the matrix now correspond to the number of higher-priority objects that suppress the current object, including itself)
        5. objects which are suppressed only by themselves are returned.
    """
    if not len(objects.shape) == 2:
        raise ValueError(f"Boxes must be of shape (n, x), not {objects.shape}")
    if not len(scores.shape) == 1:
        raise ValueError(f"Scores must be of shape (n,), not {scores.shape}")
    if not objects.shape[0] == scores.shape[0]:
        raise ValueError(f"Boxes and scores must have the same number of boxes, not {objects.shape[0]} and {scores.shape[0]}")

    if len(objects) == 0 or len(objects) == 1:
        if return_indices:
            return torch.arange(len(objects))
        else:
            return objects, scores
    
    # Sort the boxes by score (implicitly)
    indices = torch.argsort(scores, descending=True)

    # Calculate the IoU matrix
    ious = iou_fun(objects[indices])

    # Fold the IoU matrix sequentially (i.e. row_i = row_i + row_i-1 + ... + row_0)
    ious = (ious > iou_threshold).cumsum(dim=1) <= 1

    # The boxes with an IoU greater than the threshold are the ones the elements on the diagonal of the folded IoU matrix which are one (suppressed only by itself)
    indices = indices[torch.where(ious.diagonal())[0]]

    if return_indices:
        return indices
    else:
        return objects[indices], scores[indices]
    
def intersect_masks_2sets(m1s, m2s, dtype=torch.int32):
    """
    Computes intersection between all pairs between two sets of masks
    """

    intersections = torch.zeros((m1s.shape[0], m2s.shape[0]), dtype=dtype, device=m1s.device)
    for i in range(m1s.shape[0]):
        intersections[i] = (m1s[i].unsqueeze(0) & m2s).sum(dim=(1, 2)).to(dtype)
    
    return intersections

def iou_masks_2sets(m1s, m2s, a1s = None, a2s = None, dtype=torch.float32):
    """
    Computes IoU between all pairs between two sets of masks
    """
    if len(m1s.shape) == 2:
        m1s = m1s.unsqueeze(0)
    if len(m2s.shape) == 2:
        m2s = m2s.unsqueeze(0)
    if a1s is None:
        a1s = m1s.sum(dim=(1, 2)).unsqueeze(0)
        if a1s.shape[0] > 0:
            a1s = a1s.T
    if a2s is None:
        a2s = m2s.sum(dim=(1, 2)).unsqueeze(0)
    a1s = a1s.to(dtype)
    a2s = a2s.to(dtype)
    intersections = intersect_masks_2sets(m1s, m2s, dtype)

    unions = a1s + a2s - intersections
    
    return intersections / unions

def ios_masks_2sets(m1s, m2s, a1s = None, a2s = None, dtype=torch.float32):
    """
    Computes IoU between all pairs between two sets of masks
    """
    if len(m1s.shape) == 2:
        m1s = m1s.unsqueeze(0)
    if len(m2s.shape) == 2:
        m2s = m2s.unsqueeze(0)
    if a1s is None:
        a1s = m1s.sum(dim=(1, 2)).unsqueeze(0)
        if a1s.shape[0] > 0:
            a1s = a1s.T
    if a2s is None:
        a2s = m2s.sum(dim=(1, 2)).unsqueeze(0)
    a1s = a1s.to(dtype)
    a2s = a2s.to(dtype)
    intersections = intersect_masks_2sets(m1s, m2s, dtype)

    smaller_area = torch.min(a1s, a2s)
    
    return intersections / smaller_area

def iou_masks(masks, areas = None, dtype=torch.float32):
    """
    Compute IoU between all pairs of masks
    """
    if areas is None:
        areas = masks.sum(dim=(1, 2))

    ious = torch.zeros((masks.shape[0], masks.shape[0]), dtype=dtype, device=masks.device)
    for i in range(masks.shape[0]):
        ious[i, i+1:] = iou_masks_2sets(masks[i].unsqueeze(0), masks[i+1:], areas[i].unsqueeze(0), areas[i+1:], dtype).squeeze(0)
    
    ious = ious + ious.T
    ious = ious.fill_diagonal_(1)
    return ious

def nms_masks(masks, scores, iou_threshold=0.5, return_indices=False, dtype=None, metric="IoU"):
    """
    Wrapper for the standard non-maximum suppression algorithm.
    """
    metric = metric.lower()
    if metric == "iou":
        metric = iou_masks_2sets
    elif metric == "ios":
        metric = ios_masks_2sets
    else:
        raise ValueError(f"metric must be one of ['IoU', 'IoS', 'IoS/D'], not {metric}")
    if dtype is None:
        raise ValueError("'dtype' must be specified for nms_masks")
    return nms_(masks, metric, scores, iou_threshold=iou_threshold, return_indices=return_indices, dtype=dtype)

def nms_boxes(boxes, scores, iou_threshold=0.5, strict=True, return_indices=False, debug=False):
    """
    Wrapper for the standard non-maximum suppression algorithm.
    """
    return nms_(boxes, iou_boxes_2sets, scores, iou_threshold=iou_threshold, strict=strict, return_indices=return_indices)

def detect_duplicate_boxes(boxes, scores, margin=9, return_indices=False):
    """
    Duplicate detection algorithm based on the standard non-maximum suppression algorithm.

    Algorithm overview:
        * Instead of IoU we use the maximum difference between the sides of the boxes as the metric for determining whether two boxes are duplicates.
        * To make this metric compatible with NMS we negate the metric and the threshold, such that large side difference are very negative and thus below the threshold.
    """
    def negated_max_side_difference(box1 : torch.Tensor, boxs : torch.Tensor):
        """
        Calculates the **NEGATED** maximum difference between the sides of box1 and boxs.

        Args:
            box1 (torch.Tensor): A tensor of shape (4, ) representing the box in the format [x_min, y_min, x_max, y_max].
            boxs (torch.Tensor): A tensor of shape (n, 4) representing the boxes in the format [x_min, y_min, x_max, y_max].

        Returns:
            torch.Tensor: A tensor of shape (n, ) representing the **NEGATED** maximum difference between the sides of box1 and each box in boxs.
        """
        return -(boxs - box1).abs().max(dim=1).values
    return nms_(boxes, negated_max_side_difference, scores, iou_threshold=-margin, return_indices=return_indices)

def cumsum(nums : list) -> list:
    """
    Calculates the cumulative sum of a list of numbers.

    Args:
        nums (list): A list of numbers.

    Returns:
        list: A list of the cumulative sums of the input list.
    """
    running_sum = 0
    sums = [None] * len(nums)
    for i in range(len(nums)):
        running_sum += nums[i]
        sums[i] = running_sum
    return sums
    
def postprocess(preds, imgs, max_det=300, min_confidence=0, iou_threshold=0.1, nms=0, edge_margin=None):
    """Postprocesses the predictions of the model.

    Args:
        preds (list): A list of predictions from the model.
        imgs (list): A list of images that were passed to the model.
        max_det (int, optional): The maximum number of detections to return. Defaults to 300.
        min_confidence (float, optional): The minimum confidence of the predictions to return. Defaults to 0.
        iou_threshold (float, optional): The IoU threshold for non-maximum suppression. Defaults to 0.1.
        nms (int, optional): The type of non-maximum suppression to use. Defaults to 0. 0 is no NMS, 1 is standard NMS, 2 is fancy NMS and 3 is mask NMS.
        edge_margin (int, optional): The minimum gap between the edge of the image and the bounding box in pixels for a prediction to be considered valid. Defaults to None (no edge margin).

    Returns:
        list: A list of postprocessed predictions.
    """
    p = preds[0]
    # Convert from xywh to xyxy
    p[:, :4, :] = torch.cat((
            p[:, 0:2, :] - p[:, 2:4, :] / 2,  # x_min, y_min
            p[:, 0:2, :] + p[:, 2:4, :] / 2   # x_max, y_max
        ),  
        dim=1)
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError("min_confidence must be between 0 and 1.")
    ## This cannot be done here since the number of boxes in each image must be the same, when they are stored in the same tensor
    # if min_confidence != 0:
    #     # Filter out the predictions with a confidence less than min_confidence
    #     p = p[..., (p[:, 4, :] > min_confidence).squeeze(0)]
    if max_det != 0:
        # Filter out the predictions with the lowest confidence
        p = p.gather(2, torch.argsort(p[:, 4, :], dim=1, descending=True)[:, :max_det].unsqueeze(1).expand(-1, p.size(1), -1))

    # Change shape from (batch, xyxy + cls + masks, n) to (batch, n, xyxy + cls + masks)
    p = p.transpose(-2, -1)
    
    results = []
    protos = preds[1][-1]
    for i, pred in enumerate(p):
        # Remove the predictions with a confidence less than min_confidence
        if min_confidence != 0:
            pred = pred[pred[:, 4] > min_confidence]
        boxes = ops.scale_boxes((1024, 1024), pred[:, :4], imgs[i].shape[-2:], padding=False)
        if edge_margin is not None:
            close_to_edge = (boxes[:, :2] < edge_margin).any(dim=1) | (boxes[:, 2:] > (1024 - edge_margin)).any(dim=1)
            pred = pred[~close_to_edge]
            boxes = boxes[~close_to_edge]
        if nms != 0:
            if nms == 1:
                nms_ind = nms_boxes(boxes, pred[:, 4], iou_threshold=iou_threshold, strict=False, return_indices=True)
            elif nms == 2:
                nms_ind = fancy_nms_boxes(boxes, pred[:, 4], iou_threshold=iou_threshold, return_indices=True)
            elif nms == 3:
                masks = process_mask(protos[i], pred[:, -32:], boxes, imgs[i].shape[-2:], False) # pred[:, -32:] - not sure this is correct for more than one class
                nms_ind = nms_masks(masks, pred[:, 4], iou_threshold=iou_threshold, return_indices=True, dtype=pred.dtype)
                masks = masks[nms_ind]
            else:
                raise ValueError(f"nms must be 0, 1, 2 or 3, not {nms}")
            pred = pred[nms_ind]
            boxes = boxes[nms_ind]
        if nms != 3:
            masks = process_mask(protos[i], pred[:, -32:], boxes, imgs[i].shape[-2:], False) # pred[:, -32:] - not sure this is correct for more than one class
        pred[:, :4] = boxes
        results.append(Results(imgs[i].permute(1,2,0), path="", names=["insect"], boxes=pred[:, :6], masks=masks))
    return results
    
def plot_boxes(proc_preds, extra = None, focus=None):
    """
    This function takes the processed predictions and plots the bounding boxes on the image.

    Args:
        proc_preds (ultralytics.engine.results.Results): The processed predictions.

    Returns:
        None
    """
    img = proc_preds.orig_img / 255
    if isinstance(img, torch.Tensor):
        img = img.detach().float().cpu()
    boxes = proc_preds.boxes.xyxy
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().float().cpu()
    confidences = proc_preds.boxes.conf
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.detach().float().cpu()

    _, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    for box, conf in zip(boxes, confidences):
        box = box.numpy()
        rect = mpl.patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="none", facecolor="r", alpha=conf.item() / 4)
        ax.add_patch(rect)
    if extra is not None:
        if isinstance(extra, torch.Tensor):
            if len(extra.shape) == 1:
                extra = extra.unsqueeze(0)
            if not extra.shape[1] == 4:
                raise ValueError(f"extra must be of shape (n, 4), not {extra.shape}")
            extra = extra.detach().float().cpu()
            for e in extra:
                rect = mpl.patches.Rectangle((e[0], e[1]), e[2] - e[0], e[3] - e[1], linewidth=1, edgecolor="none", facecolor="b", alpha=0.15)
                ax.add_patch(rect)
        else:
            raise ValueError(f"extra must be a torch.Tensor, not {type(extra)}")
    if focus is not None:
        if not isinstance(focus, int) or (isinstance(focus, list) or isinstance(focus, tuple) or isinstance(focus, torch.Tensor) or isinstance(focus, np.ndarray)) and len(focus) != 1:
            raise ValueError(f"focus must be an int, not {type(focus)}")
        if not isinstance(focus, int):
            focus = int(focus[0])
        box = boxes[focus].numpy()
        rect = mpl.patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor="g", facecolor="none", alpha=1)
        ax.add_patch(rect)
    plt.show()