import torch
import torch.nn.functional as F
import math

# def find_contours(neighbors):
#     _device = neighbors[0].device
#     outer_idx = torch.tensor([i for i, n in enumerate(neighbors) if len(n) < 9], dtype=torch.long, device=_device)
#     inner_idx = torch.tensor([i for i in range(len(neighbors)) if i not in outer_idx], dtype=torch.long, device=_device)
#     outer_points = [neighbors[i][torch.isin(neighbors[i], outer_idx)] for i in outer_idx]
#     # Remap indices for outer points
#     outer_remap = torch.arange(len(neighbors), dtype=torch.long, device=_device)
#     outer_remap[outer_idx] = torch.arange(len(outer_idx), dtype=torch.long, device=_device)
#     outer_remap[inner_idx] = -1
#     outer_points = [outer_remap[o] for o in outer_points]
    
#     skippers = torch.zeros(len(neighbors), dtype=torch.bool, device=_device)
#     winners = skippers.clone()

#     group_ind = 0
#     while group_ind < len(outer_points):
#         if skippers[group_ind]:
#             group_ind += 1
#             continue
#         last_added = outer_points[group_ind]
#         skippers[group_ind] = True
#         winners[group_ind] = True
#         while True:
#             this = outer_points[group_ind]
            
#             mergers = torch.zeros(len(outer_points), dtype=torch.bool, device=_device)
#             new_neighbors = mergers.clone()
#             for i, o in enumerate(outer_points):
#                 if skippers[i]:
#                     continue
#                 old_neighbors = torch.isin(o, last_added, assume_unique=True)
#                 if not old_neighbors.any():
#                     continue
#                 mergers[i] = True
#                 skippers[i] = True
#                 if old_neighbors.all():
#                     continue
#                 new_neighbors[o[~old_neighbors]] = True

#             if not mergers.any():
#                 break
            
#             last_added = torch.where(new_neighbors)[0].unique()
#             outer_points[group_ind] = torch.cat([last_added, this])
#         group_ind += 1
#     return [outer_idx[n] for n, w in zip(outer_points, winners) if w], inner_idx

def find_neighbors(mask, pos):
    raise NotImplementedError("Seems to be some bug with this function, but I cannot reproduce it at the moment. Only happens on real data, as far as I have been able to find.")
    # nmask = torch.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=torch.long, device=mask.device)
    # nmask[*(pos + 1).T] = torch.arange(len(pos), device=mask.device, dtype=torch.long) + 1
    # nidx = torch.arange(3, device=mask.device, dtype=torch.long).unsqueeze(0).repeat(len(pos), 3) - 1
    # nidx += (pos[:, 0].unsqueeze(1) + 1) + (pos[:, 1].unsqueeze(1) + 1) * nmask.shape[0]
    # nidx[:, :3] -= nmask.shape[0]
    # nidx[:, -3:] += nmask.shape[0]
    # assert (nmask.flatten()[nidx[:, 4]].sort().values == torch.arange(len(pos), device=mask.device, dtype=torch.long) + 1).all(), f"Centers {nidx[:, 4].sort().values} do not match {nmask.flatten().nonzero(as_tuple=False).flatten().sort().values}"

    return [neighbors[neighbors != 0] - 1 for neighbors in nmask.flatten()[nidx]]

def find_neighbors_naive(mask):
    pos = mask.nonzero()
    return [torch.where(((pos[i].unsqueeze(0) - pos) ** 2).sum(dim=1).sqrt() < 1.5)[0] for i in range(len(pos))]

# def mask_to_contour(mask):
#     """
#     Takes a boolean mask, where we assume there is a single contiguous region of True values, and returns a tensor of N indices (i, j) with shape Nx2 that represent the contour of the mask in clockwise order.
#     """
#     if not mask.any():
#         return torch.zeros((0, 2), dtype=torch.long, device=mask.device)
#     if len(mask.shape) == 3:
#         if mask.shape[0] == 1:
#             mask = mask[0]
#         else:
#             raise NotImplementedError("Only implemented for 2D masks")
#     pos = mask.nonzero(as_tuple=False)


def find_contigs(mask):
    print("DEPRECATION WARNING: _find_contigs is deprecated and will be removed in a future version. Use find_contiguous_regions instead.")
    if len(mask.shape) == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise NotImplementedError("Only implemented for 2D masks")
    # start = time.time()

    ## Find the points in the mask
    pos = mask.nonzero(as_tuple=False)
    # neighbors = find_neighbors(mask, pos) # This does not work at the moment for some reason, but could perhaps be faster than the naive version
    
    ## For every point, find the neighbors
    neighbors = find_neighbors_naive(mask)
    
    # neighbor_finding_time = time.time() - start
    # start = time.time()

    ## Find the contigous contours and the inner points
    contours, inners = find_contours(neighbors)
    
    # contour_finding_time = time.time() - start
    # start = time.time()

    ## Convert to float for distance calculation
    pos = pos.float()
    contours = [pos[c] for c in contours]
    ## Assign inner points to contours
    inner_to_contour_min_dist = [(torch.cdist(pos[inners], c)).min(dim=1).values for c in contours]
    inner_to_contour_min_dist = torch.stack(inner_to_contour_min_dist)
    which_contour = inner_to_contour_min_dist.argmin(dim=0)
    ## Combine contours and inner points
    for i, c in enumerate(contours):
        c = torch.cat([c, pos[inners[which_contour == i]]])
        contours[i] = c.long()

    # inner_assigment_time = time.time() - start
    # start = time.time()
        
    ## Initialize the disjoint masks
    split_masks = torch.zeros((len(contours), *mask.shape), dtype=torch.bool, device=mask.device)
    ## Fill the contigous masks into separate disjoint masks
    for i, c in enumerate(contours):
        split_masks[i, c[:,0], c[:,1]] = True

    # mask_creation_time = time.time() - start
    # total_time = neighbor_finding_time + contour_finding_time + inner_assigment_time + mask_creation_time
    # print(f'Found {len(split_masks)} in {total_time:.2f} seconds | Neighbors {neighbor_finding_time:.2f} ({neighbor_finding_time/total_time*100:.3g}%) | Contours {contour_finding_time:.2f} ({contour_finding_time/total_time*100:.3g}%) | Inner Assignment {inner_assigment_time:.3f} ({inner_assigment_time/total_time*100:.3g}%) | Mask Creation {mask_creation_time:.3f} ({mask_creation_time/total_time*100:.3g}%)')
    
    return split_masks

def expand_mask(mask, n=1, dtype=torch.float16):
    """
    Useful for plotting the mask with PyTorch.
    """
    if n == 0:
        return mask
    neighbor_kernel = torch.ones(1, 1, 1+2*n, 1+2*n, device=mask.device, dtype=dtype)
    return torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0).to(dtype=dtype, device=mask.device), neighbor_kernel, padding=n).squeeze(0).squeeze(0) > 0.5

### Contour to mask conversion (complicated)

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

# def find_first_occurrences_1d(elements: torch.Tensor) -> torch.Tensor:
#     """
#     Finds the indices of the first occurrences of each unique element in a 1D tensor.

#     Args:
#         elements (`torch.Tensor`): The elements to find the first occurrences of.
    
#     Returns:
#         `torch.Tensor`: The indices of the first occurrences of each unique element.
#     """
#     unique_elements = elements.unique()
#     return torch.stack([torch.where(elements == e)[0][0] for e in unique_elements])

# def first_indices_of_unique_elements(elements_multidimensional: torch.Tensor, dim: int=0) -> torch.Tensor:
#     """
#     Finds the first indices of unique elements in a multidimensional tensor along a given dimension.

#     Args:
#         elements_multidimensional (`torch.Tensor`): The elements to find the first occurrences of.
#         dim (`int`, optional): The dimension along which to find the first occurrences. Defaults to 0.

#     Returns:
#         `torch.Tensor`: The indices of the first occurrences of each unique element.
#     """
#     # Find the indices of each group of unique elements along the given dimension
#     _, unique_idx = torch.unique(elements_multidimensional, return_inverse=True, dim=dim)
#     # Find the index of the first occurrence of each unique element group
#     return find_first_occurrences_1d(unique_idx)

# def remove_inserted_points(boundary_indices: torch.Tensor) -> torch.Tensor:
#     """
#     Removes the inserted points from the boundary indices. Simply divides the indices by 2 and removes duplicates. 
#     OBS: May lead to down-left bias, not sure though, depends on how the expanded mask is created.

#     Args:
#         boundary_indices (`torch.Tensor`): Coordinates of the sparse boundary pixels of size (N, 2).

#     Returns:
#         `torch.Tensor`: Coordinates of the sparse boundary pixels without the inserted points of size (M, 2), where M <= N.
#     """
#     # Divide by 2 to map back to original coordinates
#     scaled_down_points = boundary_indices // 2
#     # Remove duplicates that may have been created by scaling down
#     unique_idx = first_indices_of_unique_elements(scaled_down_points)
#     # Return the unique points
#     return scaled_down_points[unique_idx.sort().values]

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

def find_contours(mask: torch.Tensor, largest_only: bool=False) -> torch.Tensor:
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
    offsets = torch.stack(torch.meshgrid(torch.arange(-half_box, half_box + 1, dtype=torch.long, device=device), 
                                         torch.arange(-half_box, half_box + 1, dtype=torch.long, device=device)), -1).reshape(-1, 2)

    # Broadcast and add the offsets to the points to get all indices
    all_indices = (points[:, None, :] + offsets[None, :, :]).reshape(-1, 2)

    # Clip indices to image dimensions
    all_indices[:, 0] = all_indices[:, 0].clamp(0, H - 1)
    all_indices[:, 1] = all_indices[:, 1].clamp(0, W - 1)

    # Update the image
    image[:, all_indices[:, 0], all_indices[:, 1]] = color[:, None] # Set the colors in the image

    return image

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
