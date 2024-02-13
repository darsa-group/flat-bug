import torch
import torch.nn.functional as F
import numpy as np
import cv2
import math

## Helpers

def expand_mask(mask, n=1, dtype=torch.float16):
    """
    Useful for plotting the mask with PyTorch.
    """
    if n == 0:
        return mask
    neighbor_kernel = torch.ones(1, 1, 1+2*n, 1+2*n, device=mask.device, dtype=dtype)
    return torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0).to(dtype=dtype, device=mask.device), neighbor_kernel, padding=n).squeeze(0).squeeze(0) > 0.5

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
    contour = cv2.findContours(mask.to(torch.uint8).cpu().numpy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contour) == 0:
        print("No contours found; mask shape:", mask.shape, "mask sum:", mask.sum())
        return torch.tensor([0, 0], device=mask.device, dtype=torch.long)
    if largest_only:
        # Calculate areas of each contour
        areas = np.array([cv2.contourArea(c) for c in contour])
        # Select the largest contour and convert it to a tensor
        contour = contour[np.argmax(areas)]
    if simplify:
        contour = simplify_contour(contour, tolerance=0.5)
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
        return cv2.approxPolyDP(contour, tolerance, True)

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
        masks[i] = cv2.fillPoly(masks[i], [contour.cpu().numpy()], True)

    # Convert to tensors
    return torch.tensor(masks, dtype=torch.bool, device=device)


################################################################################################
################################################################################################
##############################                                    ##############################
##############################   Native PyTorch implementations   ##############################
##############################                                    ##############################
################################################################################################
################################################################################################

## Helpers

def is_in_2d(point: torch.Tensor, points: torch.Tensor) -> bool:
    """
    Checks if a point is in a set of points. Does this by checking the x-coordinates and y

    Args:
        point (`torch.Tensor`): The point of size (,2) to check.
        points (`torch.Tensor`): The set of points of size (N, 2) to check against.

    Returns:
        `torch.Tensor` (torch.bool): Boolean tensor of size 1. True if the point is in the set of points, False otherwise.
    """
    return (point == points).all(dim=1).any()

def order_points_clockwise(boundary_indices: torch.Tensor) -> torch.Tensor:
    """
    Orders the boundary points in a clockwise manner.

    Args:
        boundary_indices (`torch.Tensor`): Coordinates of the sparse boundary pixels of size (N, 2).

    Returns:
        `torch.Tensor`: Coordinates of the boundary pixels ordered clockwise of size (N, 2).
    """
    if len(boundary_indices) <= 1:
        return [boundary_indices]
    device = boundary_indices.device
    # Find the top-leftmost point as the starting point
    start_point = boundary_indices[boundary_indices[:, 0] == boundary_indices[:, 0].min()]
    start_point = start_point[start_point[:, 1].argmax()]

    # Directions to move: right, down, left, up (clockwise)
    directions = torch.tensor([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=torch.long, device=device)

    # Initialize helpers
    ordered_points = torch.ones_like(boundary_indices) * -1
    ordered_points[0] = start_point
    current_point = start_point
    dir_idx = 1
    i = 1
    other_regions = []

    while i < len(boundary_indices):
        found_next = False
        for d in range(4):
            this_dir = (dir_idx + d) % 4 # Always start with the current direction and go clockwise
            next_point = current_point + directions[this_dir]
            if is_in_2d(next_point, boundary_indices) and not is_in_2d(next_point, ordered_points[:i]):
                ordered_points[i] = next_point
                current_point = next_point
                found_next = True
                dir_idx = this_dir - 1 # Update to the new direction
                i += 1
                break
        if not found_next and i != len(boundary_indices):
            # This happens when there are more than one contiguous regions in the mask
            # Here we simply call recursively on the remaining points
            remaining_points = boundary_indices[~torch.stack([is_in_2d(p, ordered_points[:i]) for p in boundary_indices])]
            other_regions = order_points_clockwise(remaining_points)
            break
        if i == len(boundary_indices):
            break

    return [ordered_points[:i]] + other_regions

def remove_inserted_points(boundary_indices: torch.Tensor) -> torch.Tensor:
    """
    Removes the inserted points from the boundary indices by dividing the indices by 2 and removing duplicates.
    This version preserves the original order of the unique points.

    Args:
        boundary_indices (`torch.Tensor`): Coordinates of the sparse boundary pixels of size (N, 2).

    Returns:
        `torch.Tensor`: Coordinates of the sparse boundary pixels without the inserted points of size (M, 2), where M <= N.
    """
    device = boundary_indices.device
    # Divide by 2 to map back to original coordinates
    scaled_down_points = boundary_indices // 2

    # Get unique rows and their inverse indices
    _, inverse_indices = torch.unique(scaled_down_points, return_inverse=True, sorted=False, dim=0)
    
    # Create an array to keep track of first occurrences
    first_occurrences = torch.full((inverse_indices.max() + 1,), -1, dtype=torch.long, device=device)
    
    where_to_replace = first_occurrences[inverse_indices] == -1
    first_occurrences[inverse_indices[where_to_replace]] = torch.arange(len(inverse_indices), device=device)[where_to_replace]
    first_occurrences = first_occurrences[first_occurrences >= 0].sort().values

    # Extract and return the first occurrences in the original order
    return scaled_down_points[first_occurrences]

def remove_consecutive_duplicates(elements: torch.Tensor) -> torch.Tensor:
    return elements[(elements != elements.roll(1, 0)).any(dim=1)]

def remove_unconnected_points(contour : torch.Tensor) -> torch.Tensor:
    """
    Takes a contour represented as (i, j) index-coordinates in a Nx2 tensor and removes unconnected points. A point is connected if the next element in the contour is a neighbor of the current element in the cardinal directions.

    Args:
        contour (`torch.Tensor`): Contour represented as (i, j) index-coordinates in a Nx2 tensor

    Returns:
        `torch.Tensor`: Contour with unconnected points removed of size Mx2, where M <= N
    """
    # All points are "unconnected" if there are two or less points
    if len(contour) <= 2:
        return contour
    # Clone the contour to avoid modifying the original
    neighbors = (torch.cdist(contour.float(), contour.float(), p=1) == 1)
    keeper_mask = torch.ones(contour.shape[0], dtype=torch.bool)
    # Remove the unconnected points, which may expose new unconnected points
    for _ in range(len(contour)):
        # If there are no points left, break
        if ~keeper_mask.any():
            break
        # Find the current points with a single (or none) neighbor
        single_neighbors = ((neighbors == 1).sum(dim=1) == 1).nonzero().flatten()
        # Check if the single_neighbors tensor is empty
        if len(single_neighbors) == 0:
            break

        keeper_mask[single_neighbors] = False
        # Subtract the single neighbors from the neighbors tensor
        if len(single_neighbors) == 1:
            neighbors[single_neighbors, :] = False
            neighbors[:, single_neighbors] = False
        else:
            for s in single_neighbors:
                neighbors[s, :] = False
                neighbors[:, s] = False

    return contour[keeper_mask]

def remove_consecute_numbers(numbers : torch.Tensor) -> torch.Tensor:
    """
    Remove consecutive numbers from a list of numbers, alternating between the left and the right side of each segment. If there is an odd number of consecutive groups, the last group is removed.

    Args:
        numbers (`torch.Tensor`): List of numbers of size N

    Returns:
        `torch.Tensor`: List of numbers with consecutive numbers removed of size M, where M <= N
    """
    l = (numbers - numbers.roll(1)) != 1
    r = (numbers - numbers.roll(-1)) != -1
    alternating = torch.cat((numbers[l][::2], numbers[r][1::2])).sort().values
    if len(alternating) % 2 == 1:
        alternating = alternating[:-1]
    return alternating



## Main functions

def _find_contours(mask: torch.Tensor, largest_only: bool=False) -> torch.Tensor:
    """
    Efficiently finds the sparse boundary of a contiguous region in the mask.
    
    Args:
        mask (`torch.Tensor`): A 2D boolean tensor representing the mask.
        largest_only (`bool`, optional): If True, only the region with the longest boundary is returned. Defaults to `False`.

    Returns:
        `list[torch.Tensor]`: A list of tensors, where each tensor represents the boundary of a contiguous region in the mask.
            or
        `torch.Tensor`: Coordinates of the sparse boundary pixels.
    """
    device = mask.device
    # Type checking
    assert isinstance(mask, torch.Tensor), "'mask' must be a tensor"
    assert isinstance(largest_only, bool), "'largest_only' argument must be a boolean"
    if len(mask.shape) != 2:
        if len(mask.shape) == 3 and mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(f"The mask must be a 2D tensor not {mask.shape}")

    # Duplicate rows and columns to make sure that the all boundaries have two valid neighbors
    mask = duplicate_rows_and_columns(mask)

    # Convert the mask to a contour mask (i.e. a mask where only True values with one or more zero neighbors are True)
    boundary_mask = create_contour_mask(mask)

    # Extract coordinates of boundary pixels
    boundary_indices = torch.nonzero(boundary_mask, as_tuple=False)

    # Find the regions
    regions = order_points_clockwise(boundary_indices)

    # Remove the inserted points
    # regions = [remove_inserted_points(r) for r in regions]
    regions = [remove_consecutive_duplicates(r // 2) for r in regions]

    if largest_only:
        # Find the largest region - this is not a correct implementation as it only considers the boundary length, not the area
        region_sizes = torch.tensor([len(r) for r in regions], dtype=torch.long, device=device)
        largest_region_idx = region_sizes.argmax()
        return regions[largest_region_idx]
    else:
        return regions

def _contours_to_masks(contours : list[torch.Tensor], height : int, width : int) -> torch.Tensor:
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

    # Initialize the masks
    masks = torch.zeros((N, height, width), dtype=torch.bool, device=device)
    if N == 0:
        return masks

    # Filling in the masks
    for i, contour in enumerate(contours):
        # Remove unconnected points
        trimmed_contour = contour # is this necessary? Don't think so. remove_unconnected_points(contour)
        # Create a mask with the trimmed contour filled in
        contour_mask = torch.zeros((height, width), dtype=torch.bool, device=device)
        contour_mask[trimmed_contour[:, 0], trimmed_contour[:, 1]] = True
        # The smallest polygon with a hole has 9 points
        if len(trimmed_contour) > 9:
            # For each row, fill in the pixels between each consecutive pair of contour pixels
            for j, row in enumerate(contour_mask):
                # Find the indices of the contour pixels in this row
                contour_pixels = torch.nonzero(row, as_tuple=False).flatten()
                # Remove consecutive neighbors (following a horizontal line)
                contour_pixels_boundaries = remove_consecute_numbers(contour_pixels)
                # Fill in the pixels between each pair of contour pixels
                for k in range(0, len(contour_pixels_boundaries) - 1, 2):
                    xmin, xmax = contour_pixels_boundaries[k], contour_pixels_boundaries[k + 1] + 1
                    contour_mask[j, xmin:xmax] = True
                # Remember to add the points which were removed by remove_unconnected_points - why is this necessary?
                contour_mask[j, contour_pixels] = True
        # Remember to add the points which were removed by remove_unconnected_points
        contour_mask[contour[:, 0], contour[:, 1]] = True
        # Add the mask to the list of masks
        masks[i] = contour_mask

    return masks
