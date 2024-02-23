import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

def duplicate_rows_and_columns(mask: torch.Tensor) -> torch.Tensor:
    """
    Expands a mask by a factor of two, by duplicating rows and columns.  

    Args:
        mask (`torch.Tensor`): The mask of size (H, W) to expand.

    Returns:
        `torch.Tensor`: The expanded mask of size (2H, 2W).
    """
    # Duplicate rows
    expanded_rows = torch.repeat_interleave(mask, 2, dim=0)
    # Duplicate columns
    expanded_mask = torch.repeat_interleave(expanded_rows, 2, dim=1)
    # Return the expanded mask
    return expanded_mask

def integer_lerp(coords: torch.Tensor):
    """
    Takes a tensor of coordinates which are assumed to be integer values and in order. Interpolates between the points such that the resulting tensor includes all points between each consecutive points which are all integer values and in order. 
    """
    dtype, device = coords.dtype, coords.device
    new_coords = []
    for start, end in zip(coords, coords.roll(-1, 0)):
        dir = (end - start).sign()
        dir[dir == 0] = 1
        x = torch.arange(start[0], end[0] + dir[0], dir[0], dtype=dtype, device=device)
        y = torch.arange(start[1], end[1] + dir[1], dir[1], dtype=dtype, device=device)
        if len(x) == 0 or len(y) == 0:
            raise RuntimeError(f"Something went wrong with the interpolation between {start} and {end}")
        if len(x) == len(y):
            new_coords.append(torch.stack([x, y], dim=1))
        else:
            max_n = max(len(x), len(y))
            repeat_x = math.ceil(max_n / len(x))
            repeat_y = math.ceil(max_n / len(y))
            x = x.repeat_interleave(repeat_x)[:max_n]
            y = y.repeat_interleave(repeat_y)[:max_n]
            new_coords.append(torch.stack([x, y], dim=1))

    return torch.cat(new_coords)

def interpolate_contour(contour : torch.Tensor) -> torch.Tensor:
    """
    Takes a contour represented as (i, j) index-coordinates in a Nx2 tensor and returns a contour with interpolated points. The interpolation is done by adding a points between each consecutive pair of points in the contour, assuming the contour is closed and the points are ordered clockwise.
    OBS: This functions mutates the input tensor.
    """
    return integer_lerp(contour)

def intersect(rect1, rect2s, area_only=False):
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
    
    intersections_max = torch.max(rect1[:, :2], rect2s[:, :2])
    intersections_min = torch.min(rect1[:, 2:], rect2s[:, 2:])
    if area_only:
        intersections = (intersections_min - intersections_max).abs().prod(dim=1)
    else:
        intersections = torch.zeros_like(rect2s)
        intersections[:, :2] = intersections_max
        intersections[:, 2:] = intersections_min
    # Check for no intersection
    intersections[(intersections_min < intersections_max).any(dim=1)] = 0

    return intersections

def draw_boxes(image : torch.Tensor, points : torch.Tensor, box_size : int, color : torch.Tensor = torch.tensor([255, 0, 0])) -> torch.Tensor:
    """
    Sets the values in an image tensor (CxHxW) to `color` at every index within `box_size` pixels of the points in `points`.
    
    Args:
        image (`torch.Tensor`): Image tensor of size CxHxW
        points (`torch.Tensor`): Points of size Nx2
        box_size (`int`): Width of the lines
        color (`torch.Tensor`, optional): Color of the lines. Defaults to torch.tensor([255, 0, 0]).

    Returns:
        `torch.Tensor`: Image tensor with the boxes drawn.
    """
    dtype, device = image.dtype, image.device
    color = color.to(dtype=dtype, device=device)

    assert len(image.shape) == 3, "Image must be a 3D tensor"
    assert points.dtype == torch.long, f"Points must be of dtype=`torch.long`, not `{points.dtype}`"

    C, H, W = image.shape
    N = points.shape[0]
    half_box = math.ceil(box_size / 2)

    # Create a grid of offsets
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-half_box, half_box + 1, dtype=torch.long, device=device), 
            torch.arange(-half_box, half_box + 1, dtype=torch.long, device=device),
            indexing="ij"
        ),
        dim=-1).reshape(-1, 2)

    # Broadcast and add the offsets to the points to get all indices
    all_indices = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 2)

    # Clip indices to image dimensions
    all_indices[:, 0] = all_indices[:, 0].clamp(0, H - 1)
    all_indices[:, 1] = all_indices[:, 1].clamp(0, W - 1)

    # Update the image
    image[:, all_indices[:, 0], all_indices[:, 1]] = color[:, None] # Set the colors in the image

    return image

def create_contour_mask(mask: torch.Tensor, width: int=1) -> torch.Tensor:
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

def find_contours(mask, largest_only=True, simplify=True):
    contour = cv2.findContours(mask.to(torch.uint8).cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(contour) == 0:
        print("No contours found; mask shape:", mask.shape, "mask sum:", mask.sum())
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
        contour = torch.tensor(contour, dtype=torch.long, device=mask.device).squeeze(1)
    return contour

def simplify_contour(contour, tolerance=1.0):
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

def contours_to_masks(contours : list[torch.Tensor], height : int, width : int) -> torch.Tensor:
    """
    Takes a list of contours represented as (i, j) index-coordinates in a Xx2 tensor and returns a NxHxW tensor of boolean masks with the contours filled in.

    Args:
        contours (`list[torch.Tensor]`): List of contours represented as (i, j) index-coordinates in a Nx2 tensor (OBS: dtype=torch.long)
        height (`int`): Height of the masks
        width (`int`): Width of the masks

    Returns:
        `torch.Tensor`: NxHxW tensor of boolean masks with the contours filled in
    """
    device = contours[0].device
    N = len(contours)
    # Type checking
    assert all(c.dtype == torch.long for c in contours), "All contours must be of dtype=torch.long"
    assert all(c.device == device for c in contours), "All contours must be on the same device"
    assert all(len(c.shape) == 2 and c.shape[1] == 2 for c in contours), "All contours must be Xx2 tensors"
    assert isinstance(height, int) and isinstance(width, int), "Height and width must be integers"
    assert height > 0 and width > 0, "Height and width must be positive"

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

def contour_sum(contours : list[torch.Tensor], height : int, width : int) -> torch.Tensor:
    """
    Takes a list of contours represented as (i, j) index-coordinates in a Xx2 tensor and returns a HxW tensor of the sum of the contours filled in.

    Args:
        contours (`list[torch.Tensor]`): List of contours represented as (i, j) index-coordinates in a Nx2 tensor (OBS: dtype=torch.long)
        height (`int`): Height of the masks
        width (`int`): Width of the masks

    Returns:
        `torch.Tensor`: HxW tensor of boolean masks with the sum of the contours filled in.
    """
    device = contours[0].device
    # Type checking
    assert all(c.dtype == torch.long for c in contours), "All contours must be of dtype=torch.long"
    assert all(c.device == device for c in contours), "All contours must be on the same device"
    assert all(len(c.shape) == 2 and c.shape[1] == 2 for c in contours), "All contours must be Xx2 tensors"
    assert isinstance(height, int) and isinstance(width, int), "Height and width must be integers"
    assert height > 0 and width > 0, "Height and width must be positive"

    # Initialize the mask as UMAT
    mask = np.zeros((height, width), dtype=np.int32)
    # If there are no contours, return the empty mask
    if len(contours) == 0:
        # Convert to tensors
        return mask
    
    for contour in contours:
        this_mask = np.zeros_like(mask, dtype=np.uint8)
        this_mask = cv2.fillPoly(this_mask, [contour.cpu().numpy()], True)
        mask += this_mask

    # Convert to tensors
    return torch.tensor(mask, device=device)
    
    
def poly_normals(poly):
    """
    Calculates the normals of a polygon.

    Args:
        poly (torch.Tensor): A numpy array of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the vertices.

    Returns:
        torch.Tensor: A PyTorch Tensor of shape (n, 2), where n is the number of vertices and the 2 columns are the x and y coordinates of the normals.
    """
    # Working version
    v = np.roll(poly, -1, axis=0) - poly
    n = np.column_stack([v[:, 1], -v[:, 0]])
    n = (n + np.roll(n, 1, axis=0)) / 2
    return n

def linear_interpolate(poly, scale):
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

def scale_contour(contour, scale, expand_by_one=False):
    if len(contour.shape) != 2 or contour.shape[1] != 2:
        if contour.shape[0] == 2:
            contour = contour.reshape(1, 2)
        else:
            raise ValueError(f"Contour must be a Nx2 array, not {contour.shape}")
    if len(contour) == 0:
        return contour
    if len(contour) == 1:
        return np.round(contour * scale).astype(np.int32)
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

def resize_mask(masks, new_shape) -> torch.Tensor:
    """
    Takes a mask (or a batch of masks) and resizes it by scaling the contour coordinates and snapping to the integer grid, ensuring that snapping is always done towards the outside of the mask.

    Args:
        mask (torch.Tensor): A mask of shape (H, W) or (N, H, W) where N is the batch size.
        new_shape (tuple[int, int]): The new shape of the mask (H', W').

    Returns:
        torch.Tensor: The resized mask of shape (H', W') or (N, H', W').
    """
    dtype, device = masks.dtype, masks.device
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