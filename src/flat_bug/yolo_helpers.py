import torch
import torch.nn.functional as F
import torchvision

import numpy as np
from shapely.geometry import Polygon

import matplotlib as mpl
import matplotlib.pyplot as plt

from ultralytics.utils import ops
from ultralytics.engine.results import Results, Masks

from .geometry_simples import find_contours, resize_mask

from typing import Union, List, Tuple


class ResultsWithTiles(Results):
    def __init__(self, tiles=None, polygons=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiles = tiles
        self.polygons = polygons

    def new(self):
        new = super().new()
        new.tiles = self.tiles
        new.polygons = self.polygons
        return new

    def __getitem__(self, idx) -> 'ResultsWithTiles':
        new = super().__getitem__(idx)
        new.tiles = self.tiles[idx]
        if isinstance(idx, int) or isinstance(idx, slice):
            new.polygons = self.polygons[idx]
        elif isinstance(idx, torch.Tensor) and not idx.dtype == torch.bool or isinstance(idx, list) or isinstance(idx, tuple):
            new.polygons = [self.polygons[i] for i in idx]
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            new.polygons = [self.polygons[i] for i in torch.where(idx)[0]]
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

def offset_mask(mask, offset, new_shape=None, max_size=700):
    # Due to memory use, it is beneficial to restrict the maximum size of the masks. A 700x700 boolean tensor uses ~0.5 MB of memory
    n, h, w = mask.shape
    if new_shape is not None: #isinstance(new_shape, tuple) or isinstance(new_shape, list) or isinstance(new_shape, torch.Tensor) and len(new_shape.shape) == 2:
        assert len(new_shape) == 2, f"new_shape must be a tuple or list of length 2, not {len(new_shape)}"
        new_shape = int(n), int(new_shape[0]), int(new_shape[1])
    elif new_shape is None:
        raise ValueError("new_shape must be specified")
    new_mask = torch.zeros(new_shape, dtype=torch.bool, device=mask.device)
    
    # Calculate the possible clamped size of the mask (if it needs to be clamped)
    clamp_factor = (max(new_shape[1:]) / max_size) if max_size is not None else 1
    clamp_shape = [int(n), new_shape[1] / clamp_factor, new_shape[2] / clamp_factor]
    # And ensure that both direction are rounded in the same direction (down or up)
    clmp_delta = round(sum([c % 1 for c in clamp_shape[1:]]) / 2)
    clamp_shape[1:] = [int(c) + clmp_delta for c in clamp_shape[1:]]

    if n == 0:
        if clamp_factor <= 1:
            return new_mask
        else:
            return torch.zeros(clamp_shape, dtype=torch.bool, device=mask.device)

    # Calculate the overlap of the mask with the new mask (in the new mask's coordinate system)
    mask_overlap = [None, None]
    for i, (mask_d, new_mask_d, offset_d) in enumerate(zip([h, w], new_shape[1:], offset)):
        mask_overlap[i] = torch.arange(mask_d, device=mask.device) + offset_d
        mask_overlap[i] = mask_overlap[i][(mask_overlap[i] < new_mask_d) & (mask_overlap[i] >= 0)]
        mask_overlap[i] = mask_overlap[i][torch.tensor([0, -1], device=mask.device, dtype=torch.long)]

    # Insert the overlapping part of the old mask into the overlapping section of the new mask
    new_mask[:, mask_overlap[0][0]:mask_overlap[0][1], mask_overlap[1][0]:mask_overlap[1][1]] = mask[:, (mask_overlap[0][0] - offset[0]):(mask_overlap[0][1] - offset[0]), (mask_overlap[1][0] - offset[1]):(mask_overlap[1][1] - offset[1])]

    # If the mask is larger than the maximum size, clamp it by downscaling it such that the largest dimension is max_size
    if clamp_factor > 1:
        new_mask = resize_mask(new_mask, clamp_shape[1:]) # F.interpolate(new_mask.float().unsqueeze(0), clamp_shape[1:], mode='bilinear', align_corners=False, antialias=True).squeeze(0) > 0.25
    
    return new_mask

def merge_tile_results(results = List[Results], orig_img=None, box_offsetters=None, mask_offsetters=None, new_shape=None, clamp_boxes=(None, None), max_mask_size=700, exclude_masks=False):
    """
    Merges results from multiple images into a single Results object, possibly with a new image.
    """
    _device = results[0].boxes.data.device
    if orig_img is None:
        orig_img = results[0].orig_img
    assert isinstance(orig_img, torch.Tensor), f"orig_img must be a torch.Tensor, not {type(orig_img)}"
    if box_offsetters is None:
        box_offsetters = torch.zeros((len(results), 2), device=_device).int()
    if mask_offsetters is None:
        mask_offsetters = torch.zeros((len(results), 2), device=_device).int()
    else:
        mask_offsetters = mask_offsetters.int()
    mx, my = clamp_boxes
    path = results[0].path
    names = results[0].names
    tile_indices = torch.concatenate([torch.tensor([i] * len(r), dtype=torch.long, device=_device) for i, r in enumerate(results)])
    boxes = torch.cat([offset_box(r.boxes.data, o.flip(0), mx, my) for r, o in zip(results, box_offsetters)])
    polygons = [find_contours(resize_mask(mask, [256 * 3, 256 * 3]), True) * (1024 / 256) / 3 + o.flip(0).unsqueeze(0) for r, o in zip(results, box_offsetters) for mask in r.masks.data]
    if exclude_masks:
        masks = torch.cat([r.masks.data for r in results])
    else:
        masks = torch.cat([offset_mask(r.masks.data, o, new_shape, max_mask_size) for r, o in zip(results, mask_offsetters)])
    if len(masks.shape) == 2:
        masks = masks.unsqueeze(0)
    if not all([r.probs is None for r in results]):
        raise NotImplementedError("'Probs' not implemented yet")
    if not all([r.keypoints is None for r in results]):
        raise NotImplementedError("'Keypoints' not implemented yet")
    return ResultsWithTiles(tiles=tile_indices, orig_img=orig_img, path=path, names=names, boxes=boxes, masks=masks, polygons=polygons, probs=None, keypoints=None)

def stack_masks(masks, orig_shape=None):
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

    new_masks = torch.zeros((sum(masks_in_each), max_h, max_w), dtype=torch.bool, device=_device)

    i = 0
    for n, m in zip(masks_in_each, masks):
        if n == 0:
            continue
        for j in range(n):
            if m[[j]].shape[1] == max_h and m[[j]].shape[2] == max_w:
                new_masks[i + j] = m[[j]]
            else:
                new_masks[i + j] = resize_mask(m[[j]], (max_h, max_w))
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

    # masks = expand_bottom_right(masks)  # HW

    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    
    masks = masks.gt_(0.5).bool()

    return masks

def expand_bottom_right(mask):
    """
    Add an extra pixel above next to bottom/right edges of the region of 1s.

    Args:
        mask (torch.Tensor): A binary mask tensor of shape [h, w].

    Returns:
        (torch.Tensor): A binary mask tensor of shape [h, w], where an extra pixel is added above next to left/top edges of the region of 1s.
    """
    bottom_right_kernel = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, 1]], dtype=torch.float16, device=mask.device).t()
    bottom_right = F.conv2d(mask.to(torch.float16).unsqueeze(1), bottom_right_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(1).clamp(0)
    return mask + bottom_right
    
def iou_boxes(rectangles):
    """
    Calculates the intersection over union (IoU) of a set of rectangles.

    Args:
        rectangles (torch.Tensor): A tensor of shape (n, 4), where n is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles.

    Returns:
        torch.Tensor: A tensor of shape (n, n), where n is the number of rectangles, containing the IoU of each rectangle with each other rectangle.
    """
    return torchvision.ops.box_iou(rectangles, rectangles)

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
    return torchvision.ops.box_iou(rectangles1, rectangles2)
    
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
    
@torch.jit.script
def intersect_masks_2sets(m1s : torch.Tensor, m2s : torch.Tensor, dtype : torch.dtype=torch.float32) -> torch.Tensor:
    """
    Computes intersection between all pairs between two sets of masks
    """
    return (torch.matmul(m1s.reshape(m1s.shape[0], -1).to(dtype), m2s.reshape(m2s.shape[0], -1).t().to(dtype))).to(torch.int32)
    # intersections = torch.zeros((m1s.shape[0], m2s.shape[0]), dtype=dtype, device=m1s.device)
    # for i in range(m1s.shape[0]):
    #     intersections[i] = (m1s[i].unsqueeze(0) & m2s).sum(dim=(1, 2), dtype=dtype)
    
    # return intersections

@torch.jit.script
def iou_masks_2sets(m1s : torch.Tensor, m2s : torch.Tensor, a1s : Union[torch.Tensor, None]=None, a2s : Union[torch.Tensor, None]=None, dtype : torch.dtype=torch.float32) -> torch.Tensor:
    """
    Computes IoU between all pairs between two sets of masks
    """
    if len(m1s.shape) == 2:
        m1s = m1s.unsqueeze(0)
    if len(m2s.shape) == 2:
        m2s = m2s.unsqueeze(0)
    if a1s is None:
        a1s = m1s.sum(dim=(1, 2), dtype=torch.int32).unsqueeze(0)
        if a1s.shape[0] > 0:
            a1s = a1s.T
    else:
        a1s = a1s.to(torch.int32)
    if a2s is None:
        a2s = m2s.sum(dim=(1, 2), dtype=torch.int32).unsqueeze(0)
    else:
        a2s = a2s.to(torch.int32)
    
    intersections = intersect_masks_2sets(m1s, m2s)
    unions = a1s + a2s - intersections
    
    return (intersections / (unions + 1e-6)).to(dtype)

@torch.jit.script
def ios_masks_2sets(m1s : torch.Tensor, m2s : torch.Tensor, a1s : Union[torch.Tensor, None]=None, a2s : Union[torch.Tensor, None]=None, dtype : torch.dtype=torch.float32) -> torch.Tensor:
    """
    Computes IoS between all pairs between two sets of masks
    """
    if len(m1s.shape) == 2:
        m1s = m1s.unsqueeze(0)
    if len(m2s.shape) == 2:
        m2s = m2s.unsqueeze(0)
    if a1s is None:
        a1s = m1s.sum(dim=(1, 2), dtype=torch.int32).unsqueeze(0)
        if a1s.shape[0] > 0:
            a1s = a1s.T
    else:
        a1s = a1s.to(torch.int32)
    if a2s is None:
        a2s = m2s.sum(dim=(1, 2), dtype=torch.int32).unsqueeze(0)
    else:
        a2s = a2s.to(torch.int32)

    intersections = intersect_masks_2sets(m1s, m2s, dtype)
    smaller_area = torch.min(a1s, a2s)
    
    return intersections / (smaller_area + 1e-6)

def iou_masks(masks : torch.Tensor, areas : Union[torch.Tensor, None]= None, dtype :torch.dtype=torch.float32) -> torch.Tensor:
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
    
@torch.jit.script
def nms_masks_(masks : torch.Tensor, scores : torch.Tensor, iou_threshold : float=0.5) -> torch.Tensor:
    # Sort the boxes by score (implicitly)
    indices = torch.argsort(scores, descending=True)

    # Initialize tensors for winners (selected boxes), possible boxes and counters
    winners = -torch.ones(masks.shape[0], dtype=torch.long, device=masks.device)
    possible = torch.ones(masks.shape[0], dtype=torch.bool, device=masks.device)
    i = 0

    while True:
        possible_idx = possible.nonzero().squeeze()
        n_possible = possible_idx.numel()
        if n_possible < 2:
            if n_possible == 1:
                possible[possible_idx] = False
                winners[i] = possible_idx
                i += 1
            break
        # Pick the box with the highest score
        winners[i] = possible_idx[0]
        # Remove the picked box from the possible boxes
        possible[possible_idx[0]] = False
        # Calculate the IoU between the picked box and the remaining possible boxes
        ious = iou_masks_2sets(masks[indices[possible_idx[0]]], masks[indices[possible_idx[1:]]], dtype=torch.float32).squeeze(0)
        # Get the indices of the boxes with an IoU greater than the threshold
        winner_mask = ious <= iou_threshold
        # Remove the boxes with an IoU greater than the threshold from the possible boxes
        possible[possible_idx[1:]] = winner_mask
        i += 1

    # Map the indices back to the original indices and sort them (returns boxes, scores & indices in the original order of the input)
    winners = indices[winners[:i]].sort().values 
    
    # Return the winning indices
    return winners

def iou_polygons(polygons1, polygons2=None, dtype=torch.float32):
    if polygons2 is None:
        polygons2 = polygons1
    for polygon in polygons1 + polygons2:
        if len(polygon.shape) != 2 or polygon.shape[1] != 2:
            raise ValueError(f"Polygons must be of shape (n, 2), not {polygon.shape}: {polygon}")
    device = polygons1[0].device
    iou_mat = np.zeros((len(polygons1), len(polygons2)), dtype=np.float32)
    polygons1 = [Polygon(polygon.cpu().numpy()).buffer(0) for polygon in polygons1]
    polygons2 = [Polygon(polygon.cpu().numpy()).buffer(0) for polygon in polygons2]
    areas1 = np.array([polygon.area for polygon in polygons1], dtype=np.float32)
    areas2 = np.array([polygon.area for polygon in polygons2], dtype=np.float32)
    for i, polygon1 in enumerate(polygons1):
        areas1_i = areas1[i]
        for j, polygon2 in enumerate(polygons2):
            if polygon1.intersects(polygon2):
                intersection = polygon1.intersection(polygon2).area
                union = areas1_i + areas2[j] - intersection
                iou_mat[i, j] = intersection / (union + 1e-6)
    return torch.tensor(iou_mat, dtype=dtype, device=device)

def nms_polygons_(polys : List[torch.Tensor], scores : torch.Tensor, iou_threshold : float=0.5) -> torch.Tensor:
    if len(polys) == 0 or len(polys) == 1:
        return torch.arange(len(polys))
    if len(scores.shape) != 1:
        raise ValueError(f"Scores must be of shape (n,), not {scores.shape}")
    device = polys[0].device

    # Sort the boxes by score (implicitly)
    indices = torch.argsort(scores, descending=True)

    # Initialize tensors for winners (selected boxes), possible boxes and counters
    winners = -torch.ones(len(polys), dtype=torch.long, device=device)
    possible = torch.ones(len(polys), dtype=torch.bool, device=device)
    i = 0

    while True:
        possible_idx = possible.nonzero().squeeze()
        n_possible = possible_idx.numel()
        if n_possible < 2:
            if n_possible == 1:
                possible[possible_idx] = False
                winners[i] = possible_idx
                i += 1
            break
        # Pick the box with the highest score
        winners[i] = possible_idx[0]
        # Remove the picked box from the possible boxes
        possible[possible_idx[0]] = False
        # Calculate the IoU between the picked box and the remaining possible boxes
        ious = iou_polygons([polys[indices[possible_idx[0]].item()]], [polys[pi.item()] for pi in indices[possible_idx[1:]]]).squeeze(0)
        # Get the indices of the boxes with an IoU greater than the threshold
        winner_mask = ious <= iou_threshold
        # Remove the boxes with an IoU greater than the threshold from the possible boxes
        possible[possible_idx[1:]] = winner_mask
        i += 1

    # Map the indices back to the original indices and sort them (returns boxes, scores & indices in the original order of the input)
    winners = indices[winners[:i]].sort().values 

    # Return the winning indices
    return winners

@torch.jit.script
def compute_transitive_closure_cpu(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Computes the transitive closure of a boolean matrix.
    """
    # Convert the adjacency matrix to float16, this is just done to ensure that the values don't overflow when squaring the matrix before clamping - if there existed a "or-matrix multiplication" for boolean matrices, this would not be necessary
    closure = adjacency_matrix.to(torch.float16) 
    # Expand the adjacency matrix to the transitive closure matrix, by squaring the matrix and clamping the values to 1 - each step essentially corresponds to one step of parallel breadth-first search for all nodes
    last_max = 1
    for _ in range(int(torch.log2(torch.tensor(closure.shape[0], dtype=torch.float16)).ceil())):
        this_square = torch.matmul(closure, closure)
        this_max = this_square.max().item()
        if this_max == last_max:
            break
        closure[:] = this_square.clamp(max=1) # We don't need to worry about overflow, since overflow results in +inf, which is clamped to 1
        last_max = this_max
    # Convert the matrix back to boolean and return it
    return closure > 0.5

@torch.jit.script
def compute_transitive_closure_cuda(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Computes the transitive closure of a boolean matrix.
    """
    # torch._int_mm only supports matrices such that the output is larger than 32x32 and a multiple of 8
    if len(adjacency_matrix) < 32:
        padding = 32 - len(adjacency_matrix)
    elif len(adjacency_matrix) % 8 != 0:
        padding = 8 - len(adjacency_matrix) % 8
    else:
        padding = 0
    # Convert the adjacency matrix to float16, this is just done to ensure that the values don't overflow when squaring the matrix before clamping - if there existed a "or-matrix multiplication" for boolean matrices, this would not be necessary
    closure = F.pad(adjacency_matrix, (0, padding, 0, padding), value=0.).to(torch.int8) 
    # Expand the adjacency matrix to the transitive closure matrix, by squaring the matrix and clamping the values to 1 - each step essentially corresponds to one step of parallel breadth-first search for all nodes
    last_max = 1
    for _ in range(int(torch.log2(torch.tensor(adjacency_matrix.shape[0], dtype=torch.float16)).ceil())):
        this_square = torch._int_mm(closure, closure)
        this_max = this_square.max().item()
        if this_max == last_max:
            break
        closure[:] = this_square >= 1
        last_max = this_max
    # Convert the matrix back to boolean and return it
    return (closure > 0.5)[:adjacency_matrix.shape[0], :adjacency_matrix.shape[1]]

@torch.jit.script
def compute_transitive_closure(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(f"Matrix must be of shape (n, n), not {adjacency_matrix.shape}")
    # If the matrix is a 0x0, 1x1 or 2x2 matrix, the transitive closure is the matrix itself, since there are no transitive relations
    if len(adjacency_matrix) <= 2:
        return adjacency_matrix    
    # There can be a quite significant difference in performance between the CPU and GPU implementation, however this function is not the bottleneck, so it might not be noticeable in practice
    if adjacency_matrix.is_cuda and len(adjacency_matrix) > 32:
        return compute_transitive_closure_cuda(adjacency_matrix)
    else:
        return compute_transitive_closure_cpu(adjacency_matrix)

@torch.jit.script
def extract_components(transitive_closure : torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Extracts the connected components of a transitive closure matrix.

    Args:
        transitive_closure (torch.Tensor): A boolean matrix of shape (n, n), where n is the number of objects.

    Returns:
        List[torch.Tensor]: A list of tensors, where each tensor contains the indices of the objects in a cluster.
        torch.Tensor: A tensor of shape (n, ) containing the cluster index of each object.
    """
    n = len(transitive_closure)
    cluster_vec = -torch.ones(n, dtype=torch.long, device=transitive_closure.device)
    not_visited = torch.ones(n, dtype=torch.bool, device=transitive_closure.device)

    cluster_id = 0
    rounds = 0
    while not_visited.any() and rounds < n:
        rounds += 1
        pick = not_visited.nonzero().squeeze()
        if pick.numel() == 1:
            pick = pick
        else:
            pick = pick[0]
        visitors = transitive_closure[pick]
        not_visited &= ~visitors
        cluster_vec[visitors] = cluster_id # Profiling shows that this line is often the bottleneck
        cluster_id += 1

    clusters = [torch.where(cluster_vec == i)[0].sort().values for i in torch.unique(cluster_vec).sort().values]
    
    return clusters, cluster_vec

@torch.jit.script
def cluster_iou_boxes(boxes : torch.Tensor, iou_threshold : float=0.5, time : bool=False) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Computes the connected components of a set of boxes, where boxes are connected if their IoU is greater than the threshold.

    Args:
        boxes (any): A set of boxes with a __len__ method.
        iou_threshold (float): The IoU threshold for clustering. Defaults to 0.5.

    Returns:
        List[torch.Tensor]: A list of tensors, where each tensor contains the indices of the objects in a cluster.
        torch.Tensor: A tensor of shape (n, ) containing the cluster index of each object.
    """
    ## Due to the how torch.jit.script works, we have to record the timings even if we don't use them - so I have commented out the timing code for now
    # if time:
    #     stream = torch.cuda.current_stream(device=boxes.device)
    #     start = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
    #     end_adjacency = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
    #     end_transitive = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
    #     end_components = torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False)
    #     start.record(stream)
    
    adjacency_matrix = iou_boxes(boxes) >= iou_threshold
    # if time:
    #     end_adjacency.record(stream)
    
    transitive_closure = compute_transitive_closure(adjacency_matrix)
    # if time:
    #     end_transitive.record(stream)
    
    components = extract_components(transitive_closure)
    # if time:
    #     end_components.record(stream)
    #     torch.cuda.synchronize(device=boxes.device)
    #     total_time = start.elapsed_time(end_components)
    #     print()
    #     # F-strings are not compatible with torch.jit.script
    #     print("Adjacency Matrix:", str(round(start.elapsed_time(end_adjacency) / total_time * 100, 2)) + "%")
    #     print("Transitive Closure:", str(round(start.elapsed_time(end_transitive) / total_time * 100, 2)) + "%")
    #     print("Components:", str(round(start.elapsed_time(end_components) / total_time * 100, 2)) + "%")

    return components

# @torch.jit.script
def nms_masks(masks : torch.Tensor, scores : torch.Tensor, iou_threshold : float=0.5, return_indices : bool=False, group_first : bool=True, boxes : torch.Tensor=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Efficiently perform non-maximum suppression on a set of masks.
    """
    if not group_first or len(masks) < 10:
        return nms_masks_(masks=masks, scores=scores, iou_threshold=iou_threshold)
    else:
        if boxes is None:
            raise ValueError("'boxes' must be specified for nms_masks when 'group_first' is True")
        # We decrease the iou_threshold for the clustering, since there is no straight-forward relationship between the IoU of the boxes and the IoU of the polygons
        groups, _ = cluster_iou_boxes(boxes=boxes, iou_threshold=iou_threshold / 2, time=False)
        _nms_ind = [torch.empty(0) for i in range(len(groups))]
        for i, group in enumerate(groups):
            if len(group) == 1:
                _nms_ind[i] = group
            else:
                group_boxes = boxes[group].round().long()
                xmin, ymin, xmax, ymax = group_boxes[:, 0].min(), group_boxes[:, 1].min(), group_boxes[:, 2].max(), group_boxes[:, 3].max()
                _nms_ind[i] = group[nms_masks_(masks=masks[group, ymin:(ymax+1), xmin:(xmax+1)], scores=scores[group], iou_threshold=iou_threshold)]
        if len(_nms_ind) > 0:
            nms_ind = torch.cat(_nms_ind)
        else:
            nms_ind = torch.tensor([], dtype=torch.long, device=masks.device)
        if return_indices:
            return nms_ind
        else:
            return masks[nms_ind], scores[nms_ind]

def nms_polygons(polygons, scores, iou_threshold=0.5, return_indices=False, dtype=None, group_first : bool=True, boxes=None):
    """
    Efficiently perform non-maximum suppression on a set of polygons.
    """
    if dtype is None:
        raise ValueError("'dtype' must be specified for nms_masks")
    device = polygons[0].device
    if not group_first or len(polygons) < 10:
        return nms_polygons_(polys=polygons, scores=scores, iou_threshold=iou_threshold)
    else:
        if boxes is None:
            raise ValueError("'boxes' must be specified for nms_masks when 'group_first' is True")
        # We decrease the iou_threshold for the clustering, since there is no straight-forward relationship between the IoU of the boxes and the IoU of the polygons
        groups, _ = cluster_iou_boxes(boxes=boxes, iou_threshold=iou_threshold / 2, time=False) 
        nms_ind = [None for _ in range(len(groups))]
        for i, group in enumerate(groups):
            if len(group) == 1:
                nms_ind[i] = group
            else:
                nms_ind[i] = group[nms_polygons_(polys=[polygons[gi] for gi in group], scores=scores[group], iou_threshold=iou_threshold)]
        if len(nms_ind) > 0:
            nms_ind = torch.cat(nms_ind)
        else:
            nms_ind = torch.tensor([], dtype=torch.long, device=device)
        if return_indices:
            return nms_ind
        else:
            return [polygons[ni] for ni in nms_ind], scores[nms_ind]

def base_nms_(objects, iou_fun, scores : torch.Tensor, iou_threshold : float=0.5, strict : bool=True, return_indices : bool=False, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
    if len(scores.shape) != 1:
        raise ValueError(f"Scores must be of shape (n,), not {scores.shape}")

    # Sort the boxes by score (implicitly)
    indices = torch.argsort(scores, descending=True)

    # Initialize tensors for winners (selected boxes), possible boxes and counters
    winners = []
    possible = torch.ones(objects.shape[0], dtype=torch.bool, device=objects.device)
    left = len(objects)
    i, n = 0, 0

    while True:
        possible_idx = possible.nonzero().squeeze()
        n_possible = possible_idx.numel()
        if n_possible < 2:
            if n_possible == 1:
                possible[possible_idx] = False
                winners.append(possible_idx)
            break
        # Pick the box with the highest score
        winners.append(possible_idx[0])
        # Remove the picked box from the possible boxes
        possible[possible_idx[0]] = False
        # Calculate the IoU between the picked box and the remaining possible boxes
        ious = iou_fun(objects[indices[possible_idx[0]]], objects[indices[possible_idx[1:]]], **kwargs).squeeze(0)
        # Get the indices of the boxes with an IoU greater than the threshold
        winner_mask = ious <= iou_threshold
        # Remove the boxes with an IoU greater than the threshold from the possible boxes
        possible[possible_idx[1:]] = winner_mask

        if strict:
            # In/Decrement the counters
            increment = (~winner_mask).sum().item() + 1
            left -= increment
            n += 1
            assert left == (possible_idx.numel() - 1), f"left ({left}) != possible_idx.numel() - 1 ({possible_idx.numel() - 1})"
            assert n == len(winners), f"n ({n}) != winners.sum() ({len(winners)})"


    # Map the indices back to the original indices and sort them (returns boxes, scores & indices in the original order of the input)
    winners = torch.tensor(winners, dtype=torch.long, device=objects.device)
    winners = indices[winners].sort().values 
    
    # Return the boxes and scores that were picked
    if return_indices:
        return winners
    else:
        return objects[winners], scores[winners]

def nms_boxes(boxes, scores, iou_threshold=0.5):
    """
    Wrapper for the standard non-maximum suppression algorithm.
    """
    return torchvision.ops.nms(boxes, scores, iou_threshold)

def detect_duplicate_boxes(boxes, scores, margin=9, return_indices=False):
    """
    Duplicate detection algorithm based on the standard non-maximum suppression algorithm.

    Algorithm overview:
        * Instead of IoU we use the maximum difference between the sides of the boxes as the metric for determining whether two boxes are duplicates.
        * To make this metric compatible with NMS we negate the metric and the threshold, such that large side difference are very negative and thus below the threshold.
    """
    def negated_max_side_difference(box1 : torch.Tensor, boxs : torch.Tensor, dtype : None=None):
        """
        Calculates the **NEGATED** maximum difference between the sides of box1 and boxs.

        Args:
            box1 (torch.Tensor): A tensor of shape (4, ) representing the box in the format [x_min, y_min, x_max, y_max].
            boxs (torch.Tensor): A tensor of shape (n, 4) representing the boxes in the format [x_min, y_min, x_max, y_max].

        Returns:
            torch.Tensor: A tensor of shape (n, ) representing the **NEGATED** maximum difference between the sides of box1 and each box in boxs.
        """
        return -(boxs - box1).abs().max(dim=1).values
    return base_nms_(boxes, negated_max_side_difference, scores, iou_threshold=-margin, return_indices=return_indices, strict=False)

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
    
def postprocess(preds, imgs, max_det=300, min_confidence=0, iou_threshold=0.1, nms=0, valid_size_range=None, edge_margin=None, group_first=True):
    """Postprocesses the predictions of the model.

    Args:
        preds (list): A list of predictions from the model.
        imgs (list): A list of images that were passed to the model.
        max_det (int, optional): The maximum number of detections to return. Defaults to 300.
        min_confidence (float, optional): The minimum confidence of the predictions to return. Defaults to 0.
        iou_threshold (float, optional): The IoU threshold for non-maximum suppression. Defaults to 0.1.
        nms (int, optional): The type of non-maximum suppression to use. Defaults to 0. 0 is no NMS, 1 is standard NMS, 2 is fancy NMS and 3 is mask NMS.
        edge_margin (int, optional): The minimum gap between the edge of the image and the bounding box in pixels for a prediction to be considered valid. Defaults to None (no edge margin).
        group_first (bool, optional): **OBS: Disabled due to performance issues.** A flag to indicate whether to group the masks using the boxes before performing NMS. Only relevant when nms=3. Defaults to True.

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
    if min_confidence > 0:
        num_above_min_conf = (p[:, 4, :] > min_confidence).sum(dim=1)
        max_det = min(max_det, num_above_min_conf.max().item())
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
        if valid_size_range is not None:
            valid_size = ((boxes[:, 2:] - boxes[:, :2]).log() / 2).sum(dim=1).exp()
            valid = (valid_size >= valid_size_range[0]) & (valid_size <= valid_size_range[1])
            pred = pred[valid]
            boxes = boxes[valid]
        if edge_margin is not None:
            close_to_edge = (boxes[:, :2] < edge_margin).any(dim=1) | (boxes[:, 2:] > (1024 - edge_margin)).any(dim=1)
            pred = pred[~close_to_edge]
            boxes = boxes[~close_to_edge]
        if nms != 0:
            if nms == 1:
                nms_ind = nms_boxes(boxes, pred[:, 4], iou_threshold=iou_threshold)
            elif nms == 2:
                nms_ind = fancy_nms_boxes(boxes, iou_boxes, pred[:, 4], iou_threshold=iou_threshold, return_indices=True)
            elif nms == 3:
                masks = process_mask(protos[i], pred[:, -32:], boxes, imgs[i].shape[-2:], False) # pred[:, -32:] - not sure this is correct for more than one class
                nms_ind = nms_masks(masks, pred[:, 4], iou_threshold=iou_threshold, return_indices=True, boxes=boxes / 4, group_first=False)
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