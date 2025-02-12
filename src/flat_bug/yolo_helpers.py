from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics.engine.results import Masks, Results

from flat_bug.geometric import find_contours, resize_mask
from flat_bug.nms import fancy_nms, iou_boxes, nms_boxes, nms_masks


class ResultsWithTiles(Results):
    def __init__(self, tiles : List[int]=None, polygons=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiles = tiles
        self.polygons = polygons

    def new(self) -> 'ResultsWithTiles':
        new = super().new()
        new.tiles = self.tiles
        new.polygons = self.polygons
        return new

    def __getitem__(self, idx : Union[int, slice, List[int], Tuple[int], torch.Tensor]) -> 'ResultsWithTiles':
        new = super().__getitem__(idx)
        new.tiles = self.tiles[idx]
        if isinstance(idx, int) or isinstance(idx, slice):
            new.polygons = self.polygons[idx]
        elif isinstance(idx, torch.Tensor) and not idx.dtype == torch.bool or isinstance(idx, list) or isinstance(idx, tuple):
            new.polygons = [self.polygons[i] for i in idx]
        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            new.polygons = [self.polygons[i] for i in torch.where(idx)[0]]
        else:
            raise TypeError(f"idx must be an int, slice, list, tuple or torch.Tensor, not {type(idx)}")
        return new

def offset_box(
        boxes : torch.Tensor, 
        offset : torch.Tensor, 
        max_x : Optional[Union[int, float]] = None, 
        max_y : Optional[Union[int, float]] = None
    ) -> torch.Tensor:
    m = 4 / offset.shape[0]
    assert m // 1 == m, f"4 must be divisible by the number of offsets ({offset.shape[0]})"
    boxes[:, :4] += offset.unsqueeze(0).repeat(1, int(m))
    if max_y is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, max_y - 1)
    if max_x is not None:
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, max_x - 1)
    return boxes

def offset_mask(
        mask : torch.Tensor, 
        offset : torch.Tensor, 
        new_shape : Optional[Union[Tuple[int, int], List[int]]]=None, 
        max_size=700
    ) -> torch.Tensor:
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
    clamp_delta = round(sum([c % 1 for c in clamp_shape[1:]]) / 2)
    clamp_shape[1:] = [int(c) + clamp_delta for c in clamp_shape[1:]]

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

def merge_tile_results(
        results = List[Results], 
        orig_img : Optional["torch.Tensor"]=None,
        box_offsetters : Optional["torch.Tensor"]=None,
        mask_offsetters : Optional["torch.Tensor"]=None,
        new_shape : Union[Tuple[int, int], List[int]]=None,
        clamp_boxes : Union[Tuple[int, int], List[int]]=(None, None),
        max_mask_size : int =700,
        exclude_masks : bool=False
    ) -> "ResultsWithTiles":
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

def stack_masks(
        masks : List["Masks"], 
        orig_shape : Optional[Union[Tuple[int, int], List[int]]]=None
    ) -> "Masks":
    """
    Stacks a list of ultralytics.engine.results.Masks objects (or torch.Tensor) into a single ultralytics.engine.results.Masks object.

    If the masks are not all the same size, they are resized to the largest size in the list.

    Args:
        masks (`list`): A list of ultralytics.engine.results.Masks objects (or torch.Tensor).
        orig_shape (`tuple`, optional): The original shape of the image. Defaults to None. If None, the original shape is inferred from the first Masks object in the list if there is one, otherwise the original shape None.
        antialias (`bool`, optional): A flag to indicate whether to use antialiasing when resizing the masks. Defaults to False.

    Returns:
        out (`ultralytics.engine.results.Masks`): A Masks object containing the stacked masks.
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

def crop_mask(
        masks : "torch.Tensor", 
        boxes : "torch.Tensor"
    ) -> "torch.Tensor":
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (`torch.Tensor`): [n, h, w] tensor of masks
        boxes (`torch.Tensor`): [n, 4] tensor of bbox coordinates in relative point form

    Returns:
        out (`torch.Tensor`): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(
        protos : "torch.Tensor", 
        masks_in : "torch.Tensor", 
        bboxes : "torch.Tensor", 
        shape : Union[Tuple[int, int], List[int]], 
        upsample : bool=False
    ) -> "torch.Tensor":
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (`torch.Tensor`): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (`torch.Tensor`): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (`torch.Tensor`): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (`tuple`): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (`bool`, optional): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        out (`torch.Tensor`): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in.to(protos.dtype) @ protos.view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW <- This line has been changed from the original implementation, which had a superfluous type conversion which caused YOLOv8 to cast the masks to float32, this change simply removes the type conversion enabling support for other data types

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

def expand_bottom_right(mask : torch.Tensor):
    """
    Add an extra pixel above next to bottom/right edges of the region of 1s.

    Args:
        mask (`torch.Tensor`): A binary mask tensor of shape [h, w].

    Returns:
        out (`torch.Tensor`): A binary mask tensor of shape [h, w], where an extra pixel is added above next to left/top edges of the region of 1s.
    """
    bottom_right_kernel = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, 1]], dtype=torch.float16, device=mask.device).t()
    bottom_right = F.conv2d(mask.to(torch.float16).unsqueeze(1), bottom_right_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(1).clamp(0)
    return mask + bottom_right

## These are taken from ultralytics to avoid unnecessary dependencies
def clip_boxes(
        boxes : Union[torch.Tensor, np.ndarray], 
        shape : Tuple[int, int]
    ):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

    Args:
        boxes (`Union[torch.Tensor, np.ndarray]`): the bounding boxes to clip
        shape (`tuple`): The maximum x and y values for the bounding boxes.

    Returns:
        out (`Union[torch.Tensor, np.ndarray]`): Clipped boxes
    """
    if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
        boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
        boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
        boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
        boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes

def scale_boxes(
        img1_shape : Tuple[int, int], 
        boxes : torch.Tensor, 
        img0_shape : Tuple[int, int], 
        ratio_pad=None, 
        padding : bool=True, 
        xywh : bool=False
    ) -> torch.Tensor:
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (`tuple`): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (`torch.Tensor`): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (`tuple`): the shape of the target image, in the format of (height, width).
        ratio_pad (`Optional[Tuple[float, Tuple[int, int]]]`, optional): a tuple of (ratio, pad) for scaling the boxes. If None, the ratio and pad will be
            calculated based on the size difference between the two images. Defaults to None.
        padding (`bool`, optional): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling. Defaults to True.
        xywh (`bool`, optional): The box format is xywh or not. Defaults to False.

    Returns:
        boxes (`torch.Tensor`): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)

# Revised from ultralytics
def postprocess(
        preds, 
        imgs : List[torch.Tensor], 
        max_det : int=300, 
        min_confidence : float=0, 
        iou_threshold : float=0.1, 
        nms : int=0, 
        valid_size_range : Optional[Union[Tuple[int, int], List[int]]]=None, 
        edge_margin : Optional[int]=None
    ) -> List[Results]:
    """
    Postprocesses the predictions of the model.

    Args:
        preds (`list`): A list of predictions from the model.
        imgs (`List[torch.Tensor]`): A list of images that were passed to the model.
        max_det (`int`, optional): The maximum number of detections to return. Defaults to 300.
        min_confidence (`float`, optional): The minimum confidence of the predictions to return. Defaults to 0.
        iou_threshold (`float`, optional): The IoU threshold for non-maximum suppression. Defaults to 0.1.
        nms (`int`, optional): The type of non-maximum suppression to use. Defaults to 0. 0 is no NMS, 1 is standard NMS, 2 is fancy NMS and 3 is mask NMS.
        valid_size_range (`tuple`, optional): The range of valid sizes for the bounding boxes in pixels. Defaults to None (no valid size range).
        edge_margin (`int`, optional): The minimum gap between the edge of the image and the bounding box in pixels for a prediction to be considered valid. Defaults to None (no edge margin).

    Returns:
        out (`List[ultralytics.engine.results.Results]`): A list of postprocessed predictions.
    """
    tile_size = imgs[0].shape[-1]
    p : torch.Tensor = preds[0]
    # Convert from xywh to xyxy
    p[:, :4, :] = torch.cat((
            p[:, 0:2, :] - p[:, 2:4, :] / 2,  # x_min, y_min
            p[:, 0:2, :] + p[:, 2:4, :] / 2   # x_max, y_max
        ),  
        dim=1)
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError("min_confidence must be between 0 and 1.")
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
    if len(protos.shape) == 3:
        protos = protos.unsqueeze(0)
    for i, (pred, _) in enumerate(zip(p, range(len(imgs)))):
        # Remove the predictions with a confidence less than min_confidence
        if min_confidence != 0:
            pred = pred[pred[:, 4] > min_confidence]
        boxes = scale_boxes((tile_size, tile_size), pred[:, :4], imgs[i].shape[-2:], padding=False)
        if valid_size_range is not None and valid_size_range[0] > 0 and valid_size_range[1] > 0 and valid_size_range[1] < tile_size:
            valid_size = ((boxes[:, 2:] - boxes[:, :2]).log().sum(dim=1) / 2).exp()
            valid = (valid_size >= valid_size_range[0]) & (valid_size <= valid_size_range[1])
            pred = pred[valid]
            boxes = boxes[valid]
        if edge_margin is not None and edge_margin > 0:
            close_to_edge = (boxes[:, :2] < edge_margin).any(dim=1) | (boxes[:, 2:] > (tile_size - edge_margin)).any(dim=1)
            pred = pred[~close_to_edge]
            boxes = boxes[~close_to_edge]
        if nms != 0:
            if nms == 1:
                nms_ind = nms_boxes(boxes, pred[:, 4], iou_threshold=iou_threshold)
            elif nms == 2:
                nms_ind = fancy_nms(boxes, iou_boxes, pred[:, 4], iou_threshold=iou_threshold, return_indices=True)
            elif nms == 3:
                masks = process_mask(protos[min(i, len(protos)-1)], pred[:, -32:], boxes, imgs[i].shape[-2:], False) # pred[:, -32:] - not sure this is correct for more than one class
                nms_ind = nms_masks(masks, pred[:, 4], iou_threshold=iou_threshold, return_indices=True, boxes=boxes / 4, group_first=False)
                # group_first is True, because nms_masks has vectorized IoU, 
                # meaning that the overhead of doing connected-component clustering is larger than the time-loss from redundant IoU calculations
                masks = masks[nms_ind]
            else:
                raise ValueError(f"nms must be 0, 1, 2 or 3, not {nms}")
            pred = pred[nms_ind]
            boxes = boxes[nms_ind]
        if nms != 3:
            masks = process_mask(protos[i], pred[:, -32:], boxes, imgs[i].shape[-2:], False) # pred[:, -32:] - not sure this is correct for more than one class
        too_small = masks.sum(dim=[1, 2]) < 3
        pred = pred[~too_small]
        boxes = boxes[~too_small]
        masks = masks[~too_small]
        pred[:, :4] = boxes
        results.append({"orig_img" : imgs[i].clone().permute(1,2,0), "path" : "", "names" : ["insect"], "boxes" : pred[:, :6], "masks" : masks})
    return results