from typing import Union, List, Tuple, Optional, Any, Callable

import torch, torchvision
import numpy as np
from shapely.geometry import Polygon

def iou_boxes(
        rectangles : torch.Tensor,
        other_rectangles : Optional[torch.Tensor]=None
    ) -> torch.Tensor:
    """
    Calculates the intersection over union (IoU) of a set of rectangles.

    Args:
        rectangles (torch.Tensor): A tensor of shape (n, 4), where n is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles.
        other_rectangles (torch.Tensor, optional): A tensor of shape (m, 4), where m is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles. Defaults to None, in which case the symmetric IoU of the rectangles with themselves is calculated.

    Returns:
        torch.Tensor: A tensor of shape (n, n), where n is the number of rectangles, containing the IoU of each rectangle with each other rectangle.
    """
    if not isinstance(rectangles, torch.Tensor):
        raise ValueError(f"Rectangles must be a tensor, not {type(rectangles)}")
    elif not len(rectangles.shape) == 2 or rectangles.shape[1] != 4:
        raise ValueError(f"Rectangles must be of shape (n, 4), not {rectangles.shape}")
    if other_rectangles is None:
        pass
    elif not isinstance(other_rectangles, torch.Tensor):
        raise ValueError(f"Other rectangles must be a tensor, not {type(other_rectangles)}")
    elif not len(other_rectangles.shape) == 2 or other_rectangles.shape[1] != 4:
        raise ValueError(f"Other rectangles must be of shape (n, 4), not {other_rectangles.shape}")
    
    return torchvision.ops.box_iou(rectangles, rectangles if other_rectangles is None else other_rectangles)

def iou_boxes_2sets(
        rectangles1 : torch.Tensor, 
        rectangles2 : torch.Tensor
    ) -> torch.Tensor:
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
    
def fancy_nms(
        objects : Any, 
        iou_fun : Callable, 
        scores : torch.Tensor, 
        iou_threshold : Union[float, int]=0.5, 
        return_indices : bool=False
    ) -> Union[torch.Tensor, Tuple[Any, torch.Tensor]]:
    """
    This is a 'fancy' implementation of non-maximum suppression. It is not as fast as the non-maximum suppression algorithm, nor does it follow the exact same algorithm, but it is more readable and easier to debug.

    The algorithm works as follows:
        1. Sort the objects by score (implicitly)
        2. Calculate the IoU matrix
        3. Create a boolean matrix where IoU > iou_threshold 
        4. Fold the boolean matrix sequentially (i.e. row_i = row_i + row_i-1 + ... + row_0)
           (The values on the diagonal of the matrix now correspond to the number of higher-priority objects that suppress the current object, including itself)
        5. objects which are suppressed only by themselves are returned.

    
    Args:
        objects (Any): Any object collection that can be indexed by a tensor, where the first dimension corresponds to the objects.
        iou_fun (Callable): A function that calculates the symmetric IoU matrix of a set of objects returned as a `torch.Tensor` of shape (n, n), where n is the number of objects. The device should match the device of the scores.
        scores (torch.Tensor): A tensor of shape (n, ) containing the scores of the objects.
        iou_threshold (Union[float, int], optional): The IoU threshold for non-maximum suppression. Defaults to 0.5.
        return_indices (bool, optional): A flag to indicate whether to return the indices of the picked objects or the objects themselves. Defaults to False. If True, both the picked objects and scores are returned.

    Returns:
        torch.Tensor: A tensor of shape (m, ) containing the indices of the picked objects.
            or
        tuple of length 2: A tuple containing the picked objects and their scores.
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

    # The boxes with an IoU greater than the threshold are the elements on the diagonal of the folded IoU matrix which are one (suppressed only by itself)
    indices = indices[torch.where(ious.diagonal())[0]]

    if return_indices:
        return indices
    else:
        return objects[indices], scores[indices]
    
@torch.jit.script
def intersect_masks_2sets(
        m1s : torch.Tensor, 
        m2s : torch.Tensor, 
        dtype : torch.dtype=torch.float32
    ) -> torch.Tensor:
    """
    Computes intersection between all pairs between two sets of masks.

    Args:
        m1s (torch.Tensor): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        m2s (torch.Tensor): A tensor of shape (m, h, w), where m is the number of masks and h and w are the height and width of the masks.
        dtype (torch.dtype, optional): The data type of the output tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: A tensor of shape (n, m) containing the intersection (i.e. sum of elementwise product) of each pair of masks.
    """
    return (torch.matmul(m1s.reshape(m1s.shape[0], -1).to(dtype), m2s.reshape(m2s.shape[0], -1).t().to(dtype))).to(torch.int32)

@torch.jit.script
def iou_masks_2sets(
        m1s : torch.Tensor, 
        m2s : torch.Tensor, 
        a1s : Union[torch.Tensor, None]=None, 
        a2s : Union[torch.Tensor, None]=None, 
        dtype : torch.dtype=torch.float32
    ) -> torch.Tensor:
    """
    Computes IoU between all pairs between two sets of masks.

    The IoU is calculated using the formula: 
    
    `IoU[i,j] = intersection[i, j] / (m1s[i].sum() + m2s[j].sum() - intersection[i, j])`

    `intersection[i, j] = (m1s[i] * m2s[j]).sum()`

    The reason the intersection is calculated this way is that it can be vectorized and calculated in a single matrix multiplication for all pairs of masks.

    OBS: Results will only be valid for boolean or masks containing only 0s and 1s.

    Args:
        m1s (torch.Tensor): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        m2s (torch.Tensor): A tensor of shape (m, h, w), where m is the number of masks and h and w are the height and width of the masks.
        a1s (Union[torch.Tensor, None], optional): A tensor of shape (n, ) containing the areas of the masks in m1s. Defaults to None, in which case the areas are calculated.
        a2s (Union[torch.Tensor, None], optional): A tensor of shape (m, ) containing the areas of the masks in m2s. Defaults to None, in which case the areas are calculated.
        dtype (torch.dtype, optional): The data type of the output tensor. Defaults to torch.float32.
        
    Returns:
        torch.Tensor: A tensor of shape (n, m) containing the IoU of each pair of masks.
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
def ios_masks_2sets(
        m1s : torch.Tensor, 
        m2s : torch.Tensor, 
        a1s : Union[torch.Tensor, None]=None, 
        a2s : Union[torch.Tensor, None]=None, 
        dtype : torch.dtype=torch.float32
    ) -> torch.Tensor:
    """
    Computes IoS (Intersection over Smaller area) between all pairs between two sets of masks.

    The IoS is calculated using the formula:

    `IoS[i,j] = intersection[i, j] / (torch.min(m1s[i].sum(), m2s[j].sum()) + 1e-6)`

    `intersection[i, j] = (m1s[i] * m2s[j]).sum()`

    The reason the intersection is calculated this way is that it can be vectorized and calculated in a single matrix multiplication for all pairs of masks.

    OBS: Results will only be valid for boolean or masks containing only 0s and 1s.

    Args:
        m1s (torch.Tensor): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        m2s (torch.Tensor): A tensor of shape (m, h, w), where m is the number of masks and h and w are the height and width of the masks.
        a1s (Union[torch.Tensor, None], optional): A tensor of shape (n, ) containing the areas of the masks in m1s. Defaults to None, in which case the areas are calculated.
        a2s (Union[torch.Tensor, None], optional): A tensor of shape (m, ) containing the areas of the masks in m2s. Defaults to None, in which case the areas are calculated.
        dtype (torch.dtype, optional): The data type of the output tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: A tensor of shape (n, m) containing the IoS of each pair of masks.
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

def iou_masks(
        masks : torch.Tensor, 
        areas : Union[torch.Tensor, None]= None, 
        dtype :torch.dtype=torch.float32
    ) -> torch.Tensor:
    """
    Low-memory wrapper for `flat-bug.nms.iou_masks_2sets` that calculates the IoU of a set of masks with itself, in the symmetric case.

    Args:
        masks (torch.Tensor): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        areas (Union[torch.Tensor, None], optional): A tensor of shape (n, ) containing the areas of the masks. Defaults to None, in which case the areas are calculated.
        dtype (torch.dtype, optional): The data type of the output tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: A tensor of shape (n, n) containing the IoU of each pair of masks.
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
def nms_masks_(
        masks : torch.Tensor, 
        scores : torch.Tensor, 
        iou_threshold : float=0.5
    ) -> torch.Tensor:
    """
    Performs non-maximum suppression on a set of masks.
    
    Args:
        masks (torch.Tensor): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        scores (torch.Tensor): A tensor of shape (n, ) containing the scores of the masks.
        iou_threshold (float, optional): The IoU threshold for non-maximum suppression. Defaults to 0.5.

    Returns:
        torch.Tensor: A tensor containing the indices of the picked masks.
    """
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

def iou_polygons(
        polygons1 : List[torch.Tensor], 
        polygons2 : Optional[List[torch.Tensor]]=None, 
        dtype : torch.dtype=torch.float32
    ) -> torch.Tensor:
    """
    Calculates the intersection over union (IoU) of a set of polygons.

    The IoU is calculated using:
    
    `IoU[i,j] = intersection[i, j] / (area1[i] + area2[j] - intersection[i, j])`

    and then intersections and areas are calculated with the Shapely library.

    OBS: Invalid polygons are handled by using the buffer(0) method from Shapely, which ensures that the function does not crash, but the results are not guaranteed to be "correct" for invalid polygons.

    Args:
        polygons1 (List[torch.Tensor]): A list of tensors of shape (n, 2), where n is the number of vertices in the polygon and the 2 columns are the x and y coordinates of the vertices.
        polygons2 (List[torch.Tensor], optional): A list of tensors of shape (m, 2), where m is the number of vertices in the polygon and the 2 columns are the x and y coordinates of the vertices. Defaults to None, in which case the symmetric IoU of the polygons with themselves is calculated.

    Returns:
        torch.Tensor: A tensor of shape (n, m), where n is the number of polygons in polygons1 and m is the number of polygons in polygons2, containing the IoU of each polygon in polygons1 with each polygon in polygons2.
    """
    device = polygons1[0].device
    if polygons2 is None:
        # If polygons2 is None, we calculate the symmetric IoU of polygons1 with itself
        # This can be done slightly more efficiently than the non-symmetric case, as we only need to calculate the upper triangular part of the matrix

        # Initialize the IoU matrix
        iou_mat = torch.zeros((len(polygons1), len(polygons1)), dtype=dtype, device=device)
        # Calculate the upper triangular part of the matrix row by row
        for i in range(len(polygons1)):
            iou_mat[i, i+1:] = iou_polygons([polygons1[i]], polygons1[i+1:], dtype=dtype).squeeze(0)
        # Fold the matrix to make it symmetric
        iou_mat = iou_mat + iou_mat.T
        # Fill the diagonal with 1s
        iou_mat = iou_mat.fill_diagonal_(1)
        return iou_mat
    for polygon in polygons1 + polygons2:
        if len(polygon.shape) != 2 or polygon.shape[1] != 2:
            raise ValueError(f"Polygons must be of shape (n, 2), not {polygon.shape}: {polygon}")
    # Initialize the IoU matrix as a numpy array to minimize type conversions
    iou_mat = np.zeros((len(polygons1), len(polygons2)), dtype=np.float32)
    # Convert the tensors to Shapely polygons and calculate the areas
    polygons1 = [Polygon(polygon.cpu().numpy()).buffer(0) for polygon in polygons1]
    polygons2 = [Polygon(polygon.cpu().numpy()).buffer(0) for polygon in polygons2]
    areas1 = np.array([polygon.area for polygon in polygons1], dtype=np.float32)
    areas2 = np.array([polygon.area for polygon in polygons2], dtype=np.float32)
    # Naïvely loop through all pairs of polygons and calculate the IoU using Shapely
    for i, polygon1 in enumerate(polygons1):
        areas1_i = areas1[i]
        for j, polygon2 in enumerate(polygons2):
            # Check for intersection before calculating the intersection
            if polygon1.intersects(polygon2):
                intersection = polygon1.intersection(polygon2).area
                union = areas1_i + areas2[j] - intersection
                # Calculate the IoU and store it in the IoU matrix, we add a small epsilon to the denominator to avoid division by zero
                iou_mat[i, j] = intersection / (union + 1e-6)
    # Convert the IoU matrix to a torch tensor and return it
    return torch.tensor(iou_mat, dtype=dtype, device=device)

def nms_polygons_(
        polys : List[torch.Tensor], 
        scores : torch.Tensor, 
        iou_threshold : float=0.5
    ) -> torch.Tensor:
    """
    Performs non-maximum suppression on a set of polygons.

    Args:
        polys (List[torch.Tensor]): A list of tensors of shape (n, 2), where n is the number of vertices in the polygon and the 2 columns are the x and y coordinates of the vertices.
        scores (torch.Tensor): A tensor of shape (n, ) containing the scores of the polygons.
        iou_threshold (float, optional): The IoU threshold for non-maximum suppression. Defaults to 0.5.

    Returns:
        torch.Tensor: A tensor containing the indices of the picked polygons.
    """
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
def _compute_transitive_closure_cpu(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Computes the transitive closure of a boolean matrix.

    This function uses CPU compatible PyTorch operations and works with int16 for small matrices and int32 for larger matrices to avoid overflow.
    
    Note: Using matrices large enough that overflow with int16 matters for CPU matrix multiplication is extremely slow, so this is mostly included for compatibility reasons.

    Args:
        adjacency_matrix (torch.Tensor): A boolean matrix of shape (n, n), where n is the size of the graph represented by the matrix.

    Returns:
        torch.Tensor: A boolean matrix of shape (n, n), which is the transitive closure of the adjacency matrix.
    """
    csize = adjacency_matrix.shape[0]
    # Check for possible overflow
    if csize > 2 ** (32 - 1) - 1:
        raise ValueError(f"Matrix is too large ({csize}x{csize}) for CPU computation")
    elif csize > 2 ** (16 - 1) - 1:
        dtype = torch.int32
        # raise ValueError(f"Matrix is too large ({csize}x{csize}) for CPU computation")
    else:
        dtype = torch.int16
    # We convert to torch.int16 to avoid overflow when squaring the matrix and ensure torch compatibility
    closure = adjacency_matrix.to(dtype) 
    # Expand the adjacency matrix to the transitive closure matrix, by squaring the matrix and clamping the values to 1 - each step essentially corresponds to one step of parallel breadth-first search for all nodes
    last_max = torch.zeros(csize, dtype=dtype)
    for _ in range(int(torch.log2(torch.tensor(csize, dtype=torch.float32)).ceil())):
        this_square = torch.matmul(closure, closure)
        this_max = this_square.max(dim=0).values
        if (this_max == last_max).all():
            break
        closure[:] = this_square.clamp(max=1) # We don't need to worry about overflow, since overflow results in +inf, which is clamped to 1
        last_max = this_max
    # Convert the matrix back to boolean and return it
    return closure > 0.5

@torch.jit.script
def _compute_transitive_closure_cuda(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Computes the transitive closure of a boolean matrix.

    This function uses the torch._int_mm function, which is only available on CUDA devices and is significantly faster than the CPU implementation.

    Args:
        adjacency_matrix (torch.Tensor): A boolean matrix of shape (n, n), where n is the size of the graph represented by the matrix.

    Returns:
        torch.Tensor: A boolean matrix of shape (n, n), which is the transitive closure of the adjacency matrix.
    """
    # torch._int_mm only supports matrices such that the output is larger than 32x32 and a multiple of 8
    if len(adjacency_matrix) < 32:
        padding = 32 - len(adjacency_matrix)
    elif len(adjacency_matrix) % 8 != 0:
        padding = 8 - len(adjacency_matrix) % 8
    else:
        padding = 0
    # Convert the adjacency matrix to float16, this is just done to ensure that the values don't overflow when squaring the matrix before clamping - if there existed a "or-matrix multiplication" for boolean matrices, this would not be necessary
    closure = torch.nn.functional.pad(adjacency_matrix, (0, padding, 0, padding), value=0.).to(torch.int8) 
    # Expand the adjacency matrix to the transitive closure matrix, by squaring the matrix and clamping the values to 1 - each step essentially corresponds to one step of parallel breadth-first search for all nodes
    last_max = torch.zeros(len(closure), dtype=torch.int32, device=closure.device)
    for _ in range(int(torch.log2(torch.tensor(adjacency_matrix.shape[0], dtype=torch.float16)).ceil())):
        this_square = torch._int_mm(closure, closure)
        this_max = this_square.max(dim=0).values
        if (this_max == last_max).all():
            break
        closure[:] = this_square >= 1
        last_max = this_max
    # Convert the matrix back to boolean and remove the padding
    closure = (closure > 0.5)
    if padding > 0:
        closure = closure[:-padding, :-padding]
    return closure

@torch.jit.script
def compute_transitive_closure(adjacency_matrix : torch.Tensor) -> torch.Tensor:
    """
    Computes the transitive closure of a boolean matrix.

    Supports both CPU and CUDA devices, with performance and compatibility optimized sub-functions for each device.

    Args:
        adjacency_matrix (torch.Tensor): A boolean matrix of shape (n, n), where n is the size of the graph represented by the matrix.

    Returns:
        torch.Tensor: A boolean matrix of shape (n, n), which is the transitive closure of the adjacency matrix.
    """
    if len(adjacency_matrix.shape) != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError(f"Matrix must be of shape (n, n), not {adjacency_matrix.shape}")
    # If the matrix is a 0x0, 1x1 or 2x2 matrix, the transitive closure is the matrix itself, since there are no transitive relations
    if len(adjacency_matrix) <= 2:
        return adjacency_matrix    
    # There can be a quite significant difference in performance between the CPU and GPU implementation, however this function is not the bottleneck, so it might not be noticeable in practice
    if adjacency_matrix.is_cuda:
        return _compute_transitive_closure_cuda(adjacency_matrix)
    else:
        return _compute_transitive_closure_cpu(adjacency_matrix)

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
        pick = not_visited.nonzero()[0].squeeze()
        visitors = transitive_closure[pick]
        not_visited[visitors] = False
        cluster_vec[visitors] = cluster_id # Profiling shows that this line is often the bottleneck
        cluster_id += 1

    clusters = [torch.where(cluster_vec == i)[0].sort().values for i in torch.unique(cluster_vec).sort().values]
    
    return clusters, cluster_vec

@torch.jit.script
def cluster_iou_boxes(
        boxes : torch.Tensor, 
        iou_threshold : float=0.5, 
        time : bool=False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Computes the connected components of a set of boxes, where boxes are connected if their IoU is greater than the threshold.

    Args:
        boxes (any): A set of boxes with a __len__ method.
        iou_threshold (float): The IoU threshold for clustering. Defaults to 0.5.

    Returns:
        List[torch.Tensor]: A list of tensors, where each tensor contains the indices of the objects in a cluster.
        torch.Tensor: A tensor of shape (n, ) containing the cluster index of each object.
    """
    ## Due to the how torch.jit.script works, we can't use branched timing, so the code is commented out
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
def nms_masks(
        masks : torch.Tensor, 
        scores : torch.Tensor, 
        iou_threshold : float=0.5, 
        return_indices : bool=False, 
        group_first : bool=True, 
        boxes : torch.Tensor=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Efficiently perform non-maximum suppression on a set of masks.
    """
    if not group_first or len(masks) < 10:
        nms_ind = nms_masks_(masks=masks, scores=scores, iou_threshold=iou_threshold)
    else:
        if boxes is None:
            raise ValueError("'boxes' must be specified for nms_masks when 'group_first' is True")
        # We decrease the iou_threshold for the clustering, since there is no straight-forward relationship between the IoU of the boxes and the IoU of the polygons
        groups, _ = cluster_iou_boxes(boxes=boxes, iou_threshold=min(1, iou_threshold / 4), time=False)
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

def nms_polygons(
        polygons : List[torch.Tensor], 
        scores : torch.Tensor, 
        iou_threshold : Union[float, int]=0.5, 
        return_indices : bool=False, 
        dtype : torch.dtype=None, 
        group_first : bool=True, 
        boxes : Optional[torch.Tensor]=None
    ) -> Union[torch.Tensor, Tuple[List[torch.Tensor], torch.Tensor]]:
    """
    Efficiently perform non-maximum suppression on a set of polygons.
    """
    if dtype is None:
        raise ValueError("'dtype' must be specified for nms_masks")
    device = polygons[0].device
    if not group_first or len(polygons) < 10:
        nms_ind = nms_polygons_(polys=polygons, scores=scores, iou_threshold=iou_threshold)
    else:
        if boxes is None:
            raise ValueError("'boxes' must be specified for nms_masks when 'group_first' is True")
        # We decrease the iou_threshold for the clustering, since there is no straight-forward relationship between the IoU of the boxes and the IoU of the polygons
        groups, _ = cluster_iou_boxes(boxes=boxes, iou_threshold=min(1, iou_threshold / 4), time=False) 
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

def base_nms_(
        objects : Any, 
        iou_fun : Callable, 
        scores : torch.Tensor, 
        collate_fn : Callable=None, 
        iou_threshold : float=0.5, 
        strict : bool=True, 
        return_indices : bool=False, 
        **kwargs
    ) -> Union[torch.Tensor, Tuple[Any, torch.Tensor]]:
    """
    Implements the standard non-maximum suppression algorithm.

    Args:
        objects (any): An object which can be indexed by a tensor of indices.
        iou_fun (function): A function which takes an anchor object and a comparison set (not in the Python sense) of (different) objects and returns the IoU of the anchor object with each object in the comparison set as a tensor of shape (1, n). 
            The reason it is not just (n, ) is to allow for implementations of iou_fun functions between two sets, where the IoU is calculated between each pair of objects from distinct sets.
        scores: A tensor of shape (n, ) containing the "scores" of the objects, this can merely be though of as a priority score, where the higher the score, the higher the priority of the object - it does not have to be a probability/confidence.
        collate_fn (function): A function which takes a list of objects and returns a single combined object. Defaults to `torch.cat` if `objects` is a tensor and `lambda x : x` if `objects` is a list, otherwise it has to be specified.
        iou_threshold (float, optional): The IoU threshold for non-maximum suppression. Defaults to 0.5.
        strict (bool, optional): A flag to indicate whether to perform strict checks on the algorithm. Defaults to True.
        return_indices (bool, optional): A flag to indicate whether to return the indices of the picked objects or the objects themselves. Defaults to False. If True, both the picked objects and scores are returned.
        **kwargs: Additional keyword arguments to be passed to the iou_fun function.

    Returns:
        torch.Tensor: A tensor of shape (n, ) containing the indices of the picked objects.
            or
        tuple of length 2: A tuple containing the picked objects and their scores.
    """
    if collate_fn is None:
        if isinstance(objects, torch.Tensor):
            collate_fn = torch.cat
        elif isinstance(objects, list):
            collate_fn = lambda x : x
        else:
            raise ValueError(f"collate_fn must be specified for objects of type {type(objects)}")
    if len(scores.shape) != 1:
        raise ValueError(f"Scores must be of shape (n,), not {scores.shape}")

    if len(objects) == 0 or len(objects) == 1:
        if return_indices:
            return torch.arange(len(objects))
        else:
            return collate_fn([objects[i] for i in range(len(objects))]), scores
    
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
        return collate_fn([objects[ni] for ni in winners]), scores[winners]

def nms_boxes(
        boxes : torch.Tensor, 
        scores : torch.Tensor, 
        iou_threshold : Union[float, int]=0.5
    ) -> torch.Tensor:
    """
    Wrapper for the standard non-maximum suppression algorithm.
    """
    return torchvision.ops.nms(boxes, scores, iou_threshold)

def detect_duplicate_boxes(
        boxes : torch.Tensor, 
        scores : torch.Tensor, 
        margin : int=9, 
        return_indices : bool=False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Duplicate detection algorithm based on the standard non-maximum suppression algorithm.

    Algorithm overview:
        * Instead of IoU we use the maximum difference between the sides of the boxes as the metric for determining whether two boxes are duplicates.
        * To make this metric compatible with NMS we negate the metric and the threshold, such that large side difference are very negative and thus below the threshold.
    """
    def negated_max_side_difference(
            box : torch.Tensor, 
            boxs : torch.Tensor, 
            dtype : None=None
        ) -> torch.Tensor:
        """
        Calculates the **NEGATED** maximum difference between the sides of box1 and boxs.

        Args:
            box (torch.Tensor): A tensor of shape (4, ) representing the box in the format [x_min, y_min, x_max, y_max].
            boxs (torch.Tensor): A tensor of shape (n, 4) representing the boxes in the format [x_min, y_min, x_max, y_max].
            dtype (None, optional): OBS: Unused, only here for compatibility with the `iou_fun` signature.

        Returns:
            torch.Tensor: A tensor of shape (n, ) representing the **NEGATED** maximum difference between the sides of box1 and each box in boxs.
        """
        return -(boxs - box).abs().max(dim=1).values
    return base_nms_(boxes, negated_max_side_difference, scores, iou_threshold=-margin, return_indices=return_indices, strict=False)