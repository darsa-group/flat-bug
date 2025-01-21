import math
from itertools import accumulate
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from flat_bug import logger

def equal_allocate_overlaps(total: int, segments: int, size: int) -> List[int]:
    """
    Generates cumulative positions for placing segments of a given size within a total length, with controlled overlaps.

    This function divides the specified `total` length into `segments` positions, ensuring each segment (of given `size`) fits
    evenly by introducing a small overlap between adjacent segments. The overlap is distributed uniformly, with the first few gaps 
    adjusted slightly to ensure the segments collectively sum to `total`.

    Args:
        total (int): The total length to be covered by the segments. This is the target cumulative length the segments should fit into.
        segments (int): The number of segments to place within the total length.
            Must be greater than or equal to 2.
        size (int): The desired size of each segment, used to determine the ideal spacing between segments.
        
    Returns:
        List[int]: A list of cumulative positions (starting from 0) where each segment should be placed.
            These positions are spaced with controlled overlaps to ensure they collectively cover the `total` length.
            
    Example:
        >>> equal_allocate_overlaps(1000, 5, 250)
        [0, 187, 374, 562, 750]
    """
    if segments < 2:
        return [0] * segments
    
    overlap = segments * size - total
    partial_overlap, remainder = divmod(overlap, segments - 1)
    distance = size - partial_overlap

    return list(accumulate([distance - (1 if i < remainder else 0) for i in range(segments - 1)], initial=0))

def calculate_tile_offsets(
        image_size=(int, int),
        tile_size=int,
        minimum_overlap=int
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    w, h = image_size
    x_n_tiles = math.ceil((w - minimum_overlap) / (tile_size - minimum_overlap)) if w != tile_size else 1
    y_n_tiles = math.ceil((h - minimum_overlap) / (tile_size - minimum_overlap)) if h != tile_size else 1
    
    x_range = equal_allocate_overlaps(w, x_n_tiles, tile_size)
    y_range = equal_allocate_overlaps(h, y_n_tiles, tile_size)

    return [((m, n), (j, i)) for n, j in enumerate(y_range) for m, i in enumerate(x_range)]

def intersect(
        rect1s : torch.Tensor, 
        rect2s : torch.Tensor, 
        area_only : bool=False
    ) -> torch.Tensor:
    """
    Calculates the intersections between two sets of rectangles. The rectangles are represented as tensors of shape (n, 4)
    where the 4 columns are the x and y coordinates of the top-left and bottom-right corners of the rectangles. 
    The intersection is calculated as the rectangle that covers the intersection of the two rectangles. 
    
    If `area_only` is True, only the area of the intersection(s) are/is returned, otherwise the intersecting rectangle(s) are/is returned.

    Args:
        rect1s (`torch.Tensor`): A tensor of shape (n_1, 4) representing the left hand set of rectangles.
        rect2s (`torch.Tensor`): A tensor of shape (n_2, 4) representing the right hand set of rectangles.
        area_only (`bool`, optional): Whether to return only the area of the intersection(s). Defaults to False.
    
    Returns:
        `torch.Tensor`: A tensor of shape (n_1, n_2, 4) representing the intersection(s) of the two sets of rectangles. \\ 
            If `area_only` is True, the tensor will have shape (n_1, n_2) instead. \\
            Empty intersections are represented as rectangles with area 0 and coordinates (0, 0, 0, 0).
    """
    # Shape checking
    if len(rect1s.shape) == 1 and not rect1s.shape[0] == 4 or len(rect1s.shape) == 2 and not rect1s.shape[1] == 4:
        raise ValueError(f"Rectangles must be of shape (n, 4), not {rect1s.shape}")
    if len(rect2s.shape) == 1 and not rect2s.shape[0] == 4 or len(rect2s.shape) == 2 and not rect2s.shape[1] == 4:
        raise ValueError(f"Rectangles must be of shape (n, 4), not {rect2s.shape}")
    # Ensure that the rectangles are of shape (n, 4) not (4,)
    if len(rect1s.shape) == 1:
        rect1s = rect1s.unsqueeze(0)
    if len(rect2s.shape) == 1:
        rect2s = rect2s.unsqueeze(0)
    # Manually broadcast the rectangles
    n1 = rect1s.shape[0]
    n2 = rect2s.shape[0]
    rect1s = rect1s.unsqueeze(1).repeat(1, n2, 1)
    rect2s = rect2s.unsqueeze(0).repeat(n1, 1, 1)

    # Calculate the intersections rectangle corners - the most top-right (i.e. max) of the bottom-left corners is the intersection's bottom-left, and vice versa for the top-right
    intersections_max = torch.max(rect1s[:, :, :2], rect2s[:, :, :2]) # Intersection's bottom-left corner
    intersections_min = torch.min(rect1s[:, :, 2:], rect2s[:, :, 2:]) # Intersection's top-right corner

    # Calculate the area of the intersections or the intersections themselves
    if area_only:
        intersections = (intersections_min - intersections_max).prod(dim=2)
    else:
        intersections = torch.zeros((n1, n2, 4), dtype=rect1s.dtype, device=rect1s.device)
        intersections[:, :, :2] = intersections_max
        intersections[:, :, 2:] = intersections_min
    
    # Check for no intersection - if the bottom-left corner is greater than the top-right corner in any dimension, the intersection is empty
    intersections[(intersections_min <= intersections_max).any(dim=2)] = 0

    return intersections

def create_contour_mask(
        mask: torch.Tensor, 
        width: int=1
    ) -> torch.Tensor:
    device = mask.device
    # Kernel to check for 8-neighbors
    kernel = torch.ones((3, 3), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

    # Convolve with the kernel to count neighbors
    neighbor_count = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()

    # Boundary pixels are those in the original mask with less than 9 neighbors
    contour_mask = (neighbor_count < 9) & mask
    if width == 0:
        return torch.zeros_like(contour_mask, dtype=torch.bool, device=device)
    elif width == 1:
        return contour_mask
    elif width > 1:
        # Expand the contour mask to include the neighbors (with a distance of less than or equal to width in either direction)
        expansion_kernel = torch.ones((1, 1, 1 + 2 * width, 1 + 2 * width), dtype=torch.float, device=device)
        expanded_contour_mask = F.conv2d(contour_mask.float().unsqueeze(0).unsqueeze(0), expansion_kernel, padding=width).squeeze() > 0.5
        return expanded_contour_mask
    else:
        raise ValueError(f"Invalid width: {width}")

def find_contours(
        mask : torch.Tensor, 
        largest_only : bool=True, 
        simplify : bool=True
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    contour = cv2.findContours(mask.to(torch.uint8).cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(contour) == 0:
        logger.info("No contours found; mask shape:", mask.shape, "mask sum:", mask.sum())
        return torch.tensor([[0, 0]], device=mask.device, dtype=torch.long)
    if largest_only:
        # Calculate areas of each contour
        areas = np.array([cv2.contourArea(c) for c in contour])
        # Select the largest contour and convert it to a tensor
        contour = contour[np.argmax(areas)]
    if simplify:
        contour = simplify_contour(contour, tolerance=1 if isinstance(simplify, bool) else simplify)
    # Convert to tensor
    if isinstance(contour, list):
        return [torch.tensor(c, dtype=torch.long, device=mask.device).squeeze(1) for c in contour]
    else:
        return torch.tensor(contour, dtype=torch.long, device=mask.device).squeeze(1)

def simplify_contour(
        contour : Union[torch.Tensor, np.ndarray], 
        tolerance : float=1.0
    ) -> Union[torch.Tensor, np.ndarray]:
    """
    Wrapper for cv2.approxPolyDP that simplifies a contour by reducing the number of points while keeping the shape of the contour.
    Only works for simple closed contours without holes.
    
    Args:
        contour (`Union[torch.Tensor, np.ndarray]`): The contour to simplify, represented as a Nx2 tensor or a Nx1x2 tensor.
        tolerance (`float`, optional): The maximum distance between the original contour and the simplified contour. Defaults to 1.0.

    Returns:
        `Union[torch.Tensor, np.ndarray]`: The simplified contour in the same format as the input.
    """
    if isinstance(contour, list):
        return [simplify_contour(c, tolerance) for c in contour]
    else:
        isTensor = isinstance(contour, torch.Tensor)
        if isTensor:
            device, dtype = contour.device, contour.dtype
            contour = contour.cpu().numpy().astype(np.int32)
        simplied_contour = cv2.approxPolyDP(contour, tolerance, True)
        if isTensor:
            simplied_contour = torch.tensor(simplied_contour, dtype=dtype, device=device).squeeze(1)
        return simplied_contour

def contours_to_masks(
        contours : List[torch.Tensor], 
        height : Union[int, torch.Tensor], 
        width : Union[int, torch.Tensor]
    ) -> torch.Tensor:
    """
    Takes a list of contours represented as (i, j) index-coordinates in a Xx2 tensor and returns a NxHxW tensor of boolean masks with the contours filled in.

    Args:
        contours (`List[torch.Tensor]`): List of contours represented as (i, j) index-coordinates in a Nx2 tensor (OBS: dtype=torch.long)
        height (`int | torch.Tensor`): The height of the masks
        width (`int | torch.Tensor`): The width of the masks

    Returns:
        `torch.Tensor`: NxHxW tensor of boolean masks with the contours filled in
    """
    device = contours[0].device
    N = len(contours)
    # Type checking
    assert all(c.dtype == torch.long for c in contours), "All contours must be of dtype=torch.long"
    assert all(c.device == device for c in contours), "All contours must be on the same device"
    assert all(len(c.shape) == 2 and c.shape[1] == 2 for c in contours), "All contours must be Xx2 tensors"
    if isinstance(height, torch.Tensor):
        assert height.numel() == 1, f"Height must be a scalar tensor not {height.shape}"
        height = height.item()
    assert isinstance(height, int), f"Height must be an integer not {height}"
    height = int(height)
    if isinstance(width, torch.Tensor):
        assert width.numel() == 1, f"Width must be a scalar tensor not {width.shape}"
        width = width.item()
    assert isinstance(width, int), f"Width must be an integer not {width}"
    assert height > 0 and width > 0, f"Height and width must be positive not {height} and {width}"

    # Initialize the masks as UMATs
    masks = np.zeros((N, height, width), dtype=np.uint8)
    # If there are no contours, return the empty masks
    if N == 0:
        # Convert to tensors
        return torch.tensor(masks, dtype=torch.bool, device=device)
    
    # Filling in the masks
    for i, contour in enumerate(contours):
        masks[i] = cv2.drawContours(masks[i], [contour.cpu().numpy()], -1, 1, -1)

    # Convert to tensors
    return torch.tensor(masks, dtype=torch.bool, device=device)

@torch.jit.script
def poly_area(poly : torch.Tensor) -> float:
    """
    Calculates the area of a 2D simple polygon represented by a positively oriented (counter clock wise) sequence of points.

    See https://en.wikipedia.org/wiki/Shoelace_formula#Shoelace_formula for details.

    Args:
        poly (torch.Tensor): A tensor of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the vertices.
    
    Returns:
        float: The area of the polygon
    """
    if len(poly) < 10e4:
        poly = poly.cpu()
    poly_r = poly.roll(1, 0)
    return (poly[:, 0] @ poly_r[:, 1] - poly[:, 1] @ poly_r[:, 0]).item() / 2

def poly_normals(polygon : torch.Tensor) -> torch.Tensor:
    """
    Calculates the normals of a polygon.

    Args:
        poly (torch.Tensor): A tensor of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the vertices.

    Returns:
        torch.Tensor: A tensor of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the normals.
    """
    v = np.roll(polygon, -1, axis=0) - polygon
    n = np.column_stack([v[:, 1], -v[:, 0]])
    n = (n + np.roll(n, 1, axis=0)) / 2
    return n

def linear_interpolate(
        poly : np.ndarray, 
        scale : int
    ) -> np.ndarray:
    """
    Linearly interpolates a N x 2 polygon to have N x scale vertices.
    """
    if scale < 1:
        raise ValueError(f"Scale must be at least 1, not {scale}")
    if len(poly) == 0:
        return poly
    if scale == 1:
        return poly

    new_poly = np.zeros((poly.shape[0] * scale, 2), dtype=np.float32)
    for i in range(poly.shape[0] - 1):
        new_poly[i*scale:(i+1)*scale] = np.linspace(poly[i], poly[i+1], scale, endpoint=False)
    new_poly[-scale:] = np.linspace(poly[-1], poly[0], scale, endpoint=False)
    return new_poly[~(new_poly == np.roll(new_poly, -1, axis=0)).all(axis=1)]

def scale_contour(
        contour : np.ndarray, 
        scale : Union[List[Union[float, int]], np.ndarray, float, int], 
        expand_by_one : bool=False
    ) -> np.ndarray:
    if len(contour.shape) != 2 or contour.shape[1] != 2:
        if contour.shape[0] == 2:
            contour = contour.reshape(1, 2)
        else:
            raise ValueError(f"Contour must be a Nx2 array, not {contour.shape}")
    if isinstance(scale, (int, float)):
        scale = [scale, scale]
    if isinstance(scale, list):
        scale = np.array(scale, dtype=np.float32)
    if len(scale) != 2:
        raise ValueError(f"Scale must be a scalar or a list of 2 scalars, not {scale}")

    if len(contour) == 0:
        return contour
    if len(contour) == 1:
        return np.round(contour * scale).astype(np.int32)
    if np.all(scale == 1):
        return contour
    contour = contour * scale
    centroid = contour.mean(axis=0)
    n_interp = max(1, int(np.ceil(scale.max())) * 2)
    contour = linear_interpolate(contour, n_interp)
    contour_normals = poly_normals(contour)
    if expand_by_one:
        expand_one = np.sign(contour_normals) * (np.abs(contour_normals) > 0)
        contour -= expand_one
    
    if scale[0] < 1:
        contour[:, 0] += contour_normals[:, 0] / scale[0] / 2
    if scale[1] < 1:
        contour[:, 1] += contour_normals[:, 1] / scale[1] / 2
    
    contour[contour_normals > 0] = np.floor(contour[contour_normals > 0])
    contour[contour_normals < 0] = np.ceil(contour[contour_normals < 0])
    contour = contour.round()
    drift = centroid - contour.mean(axis=0)
    return (contour + drift).round().astype(np.int32)[(n_interp // 2)::n_interp].copy()

def resize_mask(
        masks : torch.Tensor, 
        new_shape : Union[Tuple[int, int], List[int]]
    ) -> torch.Tensor:
    """
    Takes a mask (or a batch of masks) and resizes it by scaling the contour coordinates and snapping to the integer grid, 
    ensuring that snapping is always done towards the outside of the mask.

    Args:
        mask (`torch.Tensor`): A mask of shape (H, W) or (N, H, W) where N is the batch size.
        new_shape (`Tuple[int, int] | List[int]`): The new shape of the mask (H', W').

    Returns:
        `torch.Tensor`: The resized mask of shape (H', W') or (N, H', W').
    """
    # If the mask is a not a batch of masks, unsqueeze and call the function again
    if len(masks.shape) == 2:
        return resize_mask(masks.unsqueeze(0), new_shape).squeeze(0)
    # If the mask is already the target shape, return it
    if masks.shape[1:] == new_shape:
        return masks
    # If the target shape is smaller than 2x2, raise an error
    if new_shape[0] <= 1 or new_shape[1] <= 1:
        raise ValueError(f"Target shape must be at least 2x2, not {new_shape}")
    # Resize the mask
    return F.interpolate(masks.float()[None], new_shape, mode='nearest-exact', antialias=False)[0] > 0.5

_to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)

def chw2hwc_uint8(
        crop : torch.Tensor, 
        mask : torch.Tensor
    ) -> torch.Tensor:
    """
    Converts a crop from CHW to HWC format, and adds the mask as an alpha channel if it exists.

    Args:
        crop (`torch.Tensor`): The crop to convert from CHW to HWC format.
        mask (`torch.Tensor`): The mask to add as an alpha channel.

    Returns:
        `torch.Tensor`: The crop in HWC format with the mask as an alpha channel, if supplied.
    """
    crop = _to_uint8(crop)
    if mask is not None:
        mask = mask.bool().to(torch.uint8) * 255
        crop = torch.cat([crop, mask], dim=0)
    return crop.permute(1, 2, 0)