"""Geometric helper functions for flatbug."""
import math
from collections.abc import Sequence
from itertools import accumulate
from typing import Literal, TypeVar, overload

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from flat_bug import logger

V = TypeVar("V", bound=torch.Tensor | np.ndarray)


def equal_allocate_overlaps(total: int, segments: int, size: int) -> list[int]:
    """Generate cumulative positions for placing segments of a given size within a total length, with controlled overlaps.

    This function divides the specified `total` length into `segments` positions, ensuring each segment (of given `size`) fits
    evenly by introducing a small overlap between adjacent segments. The overlap is distributed uniformly, with the first few gaps 
    adjusted slightly to ensure the segments collectively sum to `total`.

    Args:
        total: The total length to be covered by the segments. This is the target cumulative length the segments should fit into.
        segments: The number of segments to place within the total length.
            Must be greater than or equal to 2.
        size: The desired size of each segment, used to determine the ideal spacing between segments.
        
    Returns:
        A listt of cumulative positions (starting from 0) where each segment should be placed.
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

def calculate_tile_offsets(  # noqa: D103
        image_size : tuple[int, int],
        tile_size : int,
        minimum_overlap : int
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    w, h = image_size
    x_n_tiles = math.ceil((w - minimum_overlap) / (tile_size - minimum_overlap)) if w != tile_size else 1
    y_n_tiles = math.ceil((h - minimum_overlap) / (tile_size - minimum_overlap)) if h != tile_size else 1
    
    x_range = equal_allocate_overlaps(w, x_n_tiles, tile_size)
    y_range = equal_allocate_overlaps(h, y_n_tiles, tile_size)

    return [((m, n), (j, i)) for n, j in enumerate(y_range) for m, i in enumerate(x_range)]

def create_contour_mask(
        mask: torch.Tensor, 
        width: int=1
    ) -> torch.Tensor:
    """Convert a binary mask for a filled polygon to a binary mask for the non-filled polygon.

    ```
        #      Before         After
        #
        #    ---------      ---------
        #    --#####--      --#####--
        #    -#######-  =>  -##---##-
        #    --#####--      --#####--
        #    ---------      ---------
        #
        # (here dashes "-" represent 0s and hashes "#" represent 1s)
    ```

    We call the result ("After") the "contour mask". 
    
    Optionally, the "linewidth" of the contour mask can be increased.
    
    Args:
        mask: a NxM binary tensor with 1s inside the "polygon".
        width: Width of the contour in the result. 
            Reasonable values are >= 1; Setting to 0 will result in all 0s in the output. Defaults to 1.

    Returns:
        A NxM binary tensor with 1s on the edge/border of the "polygon".

    """
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


@overload
def find_contours(mask : torch.Tensor, largest_only : Literal[True], simplify : bool=True) -> torch.Tensor: ...
@overload
def find_contours(mask : torch.Tensor, largest_only : Literal[False], simplify : bool=True) -> list[torch.Tensor]: ...
def find_contours(
        mask : torch.Tensor, 
        largest_only : bool=True, 
        simplify : bool=True
    ) -> torch.Tensor | list[torch.Tensor]:
    """Extract polygons from a boolean mask."""
    contour = cv2.findContours(mask.to(torch.uint8).cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(contour) == 0:
        logger.info("No contours found; mask shape:", mask.shape, "mask sum:", mask.sum())
        return torch.tensor([[0, 0]], device=mask.device, dtype=torch.long)
    if largest_only:
        # Calculate areas of each contour
        areas = np.array([cv2.contourArea(c) for c in contour])
        # Select the largest contour and convert it to a tensor
        contour = contour[np.argmax(areas).item()]
    if simplify:
        contour = simplify_contour(contour, tolerance=1 if isinstance(simplify, bool) else simplify)
    # Convert to tensor
    if isinstance(contour, list):
        return [torch.tensor(c, dtype=torch.long, device=mask.device).squeeze(1) for c in contour]
    else:
        return torch.tensor(contour, dtype=torch.long, device=mask.device).squeeze(1)

@overload
def simplify_contour(contour : V, tolerance : float=1.0) -> V: ...
@overload
def simplify_contour(contour : Sequence[V], tolerance : float=1.0) -> list[V]: ...
def simplify_contour(
        contour : V | Sequence[V], 
        tolerance : float=1.0
    ) -> V | list[V]:
    """Simplify one or more polygons via cv2.approxPolyDP.
    
    Wrapper for cv2.approxPolyDP that simplifies a contour by reducing the number of points while keeping the shape of the contour.
    Only works for simple closed contours without holes.
    
    Args:
        contour: The contour to simplify, represented as a Nx2 tensor or a Nx1x2 tensor.
        tolerance: The maximum distance between the original contour and the simplified contour. Defaults to 1.0.

    Returns:
        The simplified contour in the same format as the input.

    """
    if not isinstance(contour, (torch.Tensor, np.ndarray)) and hasattr(contour, "__iter__"):
        return [simplify_contour(c, tolerance) for c in contour]
    else:
        if isinstance(contour, torch.Tensor):
            device, dtype = contour.device, contour.dtype
            return torch.as_tensor(
                cv2.approxPolyDP(contour.cpu().numpy().astype(np.int32), tolerance, True),
                device=device, dtype=dtype
            )
        elif isinstance(contour, np.ndarray):
            return np.asarray(cv2.approxPolyDP(contour, tolerance, True))
    raise TypeError(
        f'Unable to simplify contour of type {type(contour).__name__}, '
        'expected a torch.Tensor or np.ndarray or an iterable of such.'
    )


def contours_to_masks(
        contours : list[torch.Tensor], 
        height : int | torch.Tensor, 
        width : int | torch.Tensor
    ) -> torch.Tensor:
    """Rasterize a list of countors to a NxHxW boolean tensor stack.
    
    Contours should be represented as (i, j) index-coordinates in a Xx2 tensor.

    Args:
        contours: List of contours represented as (i, j) index-coordinates in a Nx2 tensor (OBS: dtype=torch.long).
        height: The height of the masks.
        width: The width of the masks.

    Returns:
        NxHxW tensor of boolean masks with the contours filled in.

    """
    device = contours[0].device
    N = len(contours)
    # Type checking
    assert all(c.dtype == torch.long for c in contours), "All contours must be of dtype=torch.long"
    assert all(c.device == device for c in contours), "All contours must be on the same device"
    assert all(len(c.shape) == 2 and c.shape[1] == 2 for c in contours), "All contours must be Xx2 tensors"
    if isinstance(height, torch.Tensor):
        assert height.numel() == 1, f"Height must be a scalar tensor not {height.shape}"
        int_height = height.item()
    else:
        int_height = height
    int_height = int(int_height)
    if isinstance(width, torch.Tensor):
        assert width.numel() == 1, f"Width must be a scalar tensor not {width.shape}"
        int_width = width.item()
    else:
        int_width = width
    int_width = int(int_width)
    assert int_height > 0 and int_width > 0, f"Height and width must be positive not {int_height} and {int_width}"

    # Initialize the masks as UMATs
    masks = np.zeros((N, int_height, int_width), dtype=np.uint8)
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
    """Calculate the area of a 2D simple polygon represented by a positively oriented (counter clock wise) sequence of points.

    See https://en.wikipedia.org/wiki/Shoelace_formula#Shoelace_formula for details.

    Args:
        poly: A tensor of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the vertices.
    
    Returns:
        The area of the polygon

    """
    if len(poly) < 10e4:
        poly = poly.cpu()
    poly_r = poly.roll(1, 0)
    return (poly[:, 0] @ poly_r[:, 1] - poly[:, 1] @ poly_r[:, 0]).item() / 2

def poly_normals(polygon : torch.Tensor | np.ndarray) -> torch.Tensor:
    """Calculate the normals of a polygon.

    Args:
        polygon: A tensor of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the vertices.

    Returns:
        A tensor of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the normals.

    """
    v = np.roll(polygon, -1, axis=0) - polygon
    n = np.column_stack([v[:, 1], -v[:, 0]])
    n = (n + np.roll(n, 1, axis=0)) / 2
    return torch.as_tensor(n)

def linear_interpolate(
        poly : np.ndarray, 
        scale : int
    ) -> np.ndarray:
    """Linearly interpolates a N x 2 polygon to have N x scale vertices."""
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

def scale_contour(  # noqa: D103
        contour : np.ndarray, 
        scale : list[float | int] | np.ndarray | float | int, 
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

def resize_masks(
        masks : torch.Tensor, 
        new_shape : tuple[int, int] | list[int] | int
    ) -> torch.Tensor:
    """Resize a mask (or a batch of masks) by scaling the contour coordinates and snapping to the integer grid.
    
    Ensures that snapping is always done towards the outside of the mask.

    Args:
        masks: A mask of shape (H, W) or (N, H, W) where N is the batch size.
        new_shape: The new shape of the mask (H', W').

    Returns:
        The resized mask of shape (H', W') or (N, H', W').

    """
    # If the mask is a not a batch of masks, unsqueeze and call the function again
    if len(masks.shape) == 2:
        return resize_masks(masks.unsqueeze(0), new_shape).squeeze(0)
    # If the mask is already the target shape, return it
    if masks.shape[1:] == new_shape:
        return masks
    # If the target shape is smaller than 2x2, raise an error
    if not isinstance(new_shape, int) and (new_shape[0] <= 1 or new_shape[1] <= 1):
        raise ValueError(f"Target shape must be at least 2x2, not {new_shape}")
    # Resize the mask
    return F.interpolate(masks.float()[None], new_shape, mode='nearest-exact', antialias=False)[0] > 0.5

_to_uint8 = torchvision.transforms.ConvertImageDtype(torch.uint8)

def chw2hwc_uint8(
        crop : torch.Tensor, 
        mask : torch.Tensor | None
    ) -> torch.Tensor:
    """Convert a crop from CHW to HWC format, and adds the mask as an alpha channel if it exists.

    Args:
        crop: The crop to convert from CHW to HWC format.
        mask: The mask to add as an alpha channel.

    Returns:
        The crop in HWC format with the mask as an alpha channel, if supplied.

    """
    crop = _to_uint8(crop)
    if mask is not None:
        mask = mask.bool().to(torch.uint8) * 255
        crop = torch.cat([crop, mask], dim=0)
    return crop.permute(1, 2, 0)