from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import shapely
import torch
import torchvision

import scipy.sparse


def iou_boxes(
        rectangles : torch.Tensor,
        other_rectangles : Optional[torch.Tensor]=None
    ) -> torch.Tensor:
    """
    Calculates the intersection over union (IoU) of a set of rectangles.

    Args:
        rectangles (`torch.Tensor`): A tensor of shape (n, 4), where n is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles.
        other_rectangles (`Optional[torch.Tensor]`, optional): A tensor of shape (m, 4), where m is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles. 
            Defaults to None, in which case the symmetric IoU of the rectangles with themselves is calculated.

    Returns:
        out (`torch.Tensor`): A tensor of shape (n, n), where n is the number of rectangles, containing the IoU of each rectangle with each other rectangle.
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

# Check if 'fmt' is an argument in the current version of torchvision
try:
    torchvision.ops.boxes._box_inter_union(torch.empty((0,4)), torch.empty((0,4)), fmt="xyxy")
    _box_inter_union = partial(torchvision.ops.boxes._box_inter_union, fmt="xyxy")
except TypeError:
    _box_inter_union = torchvision.ops.boxes._box_inter_union

def ios_boxes(
        rectangles : torch.Tensor,
        other_rectangles : Optional[torch.Tensor]=None
    ) -> torch.Tensor:
    """
    Calculates the intersection over smaller (IoS) of a set of rectangles.

    Args:
        rectangles (`torch.Tensor`): A tensor of shape (n, 4), where n is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles.
        other_rectangles (`Optional[torch.Tensor]`, optional): A tensor of shape (m, 4), where m is the number of rectangles and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the rectangles. 
            Defaults to None, in which case the symmetric IoS of the rectangles with themselves is calculated.

    Returns:
        out (`torch.Tensor`): A tensor of shape (n, n), where n is the number of rectangles, containing the IoS of each rectangle with each other rectangle.
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
    other_rectangles = rectangles if other_rectangles is None else other_rectangles
    areas1 = torchvision.ops.box_area(rectangles)
    areas2 = torchvision.ops.box_area(other_rectangles)
    sareas = torch.minimum(
        areas1.unsqueeze(1).expand(-1, len(areas2)),
        areas2.unsqueeze(0).expand(len(areas1), -1)
    )
    intersections, unions = _box_inter_union(rectangles, other_rectangles)
    ios = intersections / (sareas + 1e-6)
    return ios

@torch.jit.script
def iou_masks(
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
        m1s (`torch.Tensor`): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        m2s (`torch.Tensor`): A tensor of shape (m, h, w), where m is the number of masks and h and w are the height and width of the masks.
        a1s (`Optional[torch.Tensor]`, optional): A tensor of shape (n, ) containing the areas of the masks in m1s. Defaults to None, in which case the areas are calculated.
        a2s (`Optional[torch.Tensor]`, optional): A tensor of shape (m, ) containing the areas of the masks in m2s. Defaults to None, in which case the areas are calculated.
        dtype (`torch.dtype`, optional): The data type of the output tensor. Defaults to torch.float32.
        
    Returns:
        out (`torch.Tensor`): A tensor of shape (n, m) containing the IoU of each pair of masks.
    """
    # 1. Standardize Inputs: Ensure batch dim and flatten spatial dims (N, H, W) -> (N, P)
    if m1s.dim() == 2:
        m1s = m1s.unsqueeze(0)
    if m2s.dim() == 2:
        m2s = m2s.unsqueeze(0)
        
    m1s_flat = m1s.flatten(1)
    m2s_flat = m2s.flatten(1)

    # 2. Pre-calculate Areas (if not provided) using the flattened view
    if a1s is None:
        a1s = m1s_flat.sum(dim=1).to(dtype)
    else:
        a1s = a1s.to(dtype)
        
    if a2s is None:
        a2s = m2s_flat.sum(dim=1).to(dtype)
    else:
        a2s = a2s.to(dtype)
    
    intersections = torch.mm(m1s_flat.to(dtype), m2s_flat.t().to(dtype))
    unions = a1s.unsqueeze(1) + a2s.unsqueeze(0) - intersections
    
    return intersections / (unions + 1e-6)

@torch.jit.script
def ios_masks(
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
        m1s (`torch.Tensor`): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        m2s (`torch.Tensor`): A tensor of shape (m, h, w), where m is the number of masks and h and w are the height and width of the masks.
        a1s (`Optional[torch.Tensor]`, optional): A tensor of shape (n, ) containing the areas of the masks in m1s. Defaults to None, in which case the areas are calculated.
        a2s (`Optional[torch.Tensor]`, optional): A tensor of shape (m, ) containing the areas of the masks in m2s. Defaults to None, in which case the areas are calculated.
        dtype (`torch.dtype`, optional): The data type of the output tensor. Defaults to torch.float32.

    Returns:
        out (`torch.Tensor`): A tensor of shape (n, m) containing the IoS of each pair of masks.
    """
    # 1. Standardize Inputs: Ensure batch dim and flatten spatial dims (N, H, W) -> (N, P)
    if m1s.dim() == 2:
        m1s = m1s.unsqueeze(0)
    if m2s.dim() == 2:
        m2s = m2s.unsqueeze(0)
        
    m1s_flat = m1s.flatten(1)
    m2s_flat = m2s.flatten(1)

    # 2. Pre-calculate Areas (if not provided) using the flattened view
    if a1s is None:
        a1s = m1s_flat.sum(dim=1).to(dtype)
    else:
        a1s = a1s.to(dtype)
        
    if a2s is None:
        a2s = m2s_flat.sum(dim=1).to(dtype)
    else:
        a2s = a2s.to(dtype)
    
    intersections = torch.mm(m1s_flat.to(dtype), m2s_flat.t().to(dtype))
    amin = torch.minimum(a1s.unsqueeze(1), a2s.unsqueeze(0))
    
    return intersections / (amin + 1e-6)

def iou_polygons(
        polygons1: Union[List[torch.Tensor], np.ndarray], 
        polygons2: Optional[Union[List[torch.Tensor], np.ndarray]] = None
    ) -> np.ndarray:
    
    if len(polygons1) == 0:
        return np.empty((0, 0 if polygons2 is None else len(polygons2)), dtype=np.float32)

    is_symmetric = polygons2 is None
    
    def ensure_geoms(objs: Any) -> np.ndarray:
        # If it's already an object-dtype numpy array, assume it's shapely geoms
        if isinstance(objs, np.ndarray) and objs.dtype == object:
            return objs
        # Otherwise, convert from List[torch.Tensor] or similar
        return np.array([shapely.polygons(p.cpu().numpy()).buffer(0) for p in objs])

    geoms1 = ensure_geoms(polygons1)
    geoms2 = geoms1 if is_symmetric else ensure_geoms(polygons2)

    areas1 = shapely.area(geoms1)
    areas2 = areas1 if is_symmetric else shapely.area(geoms2)
    
    intersections = shapely.area(shapely.intersection(geoms1[:, np.newaxis], geoms2[np.newaxis, :]))

    unions = areas1[:, np.newaxis] + areas2[np.newaxis, :] - intersections
    iou_mat = (intersections / (unions + 1e-6)).astype(np.float32)

    if is_symmetric:
        np.fill_diagonal(iou_mat, 1.0)

    return iou_mat


def ios_polygons(
        polygons1: Union[List[torch.Tensor], np.ndarray], 
        polygons2: Optional[Union[List[torch.Tensor], np.ndarray]] = None
    ) -> np.ndarray:
    
    if len(polygons1) == 0:
        return np.empty((0, 0 if polygons2 is None else len(polygons2)), dtype=np.float32)

    is_symmetric = polygons2 is None
    
    def ensure_geoms(objs: Any) -> np.ndarray:
        # If it's already an object-dtype numpy array, assume it's shapely geoms
        if isinstance(objs, np.ndarray) and objs.dtype == object:
            return objs
        # Otherwise, convert from List[torch.Tensor] or similar
        return np.array([shapely.polygons(p.cpu().numpy()).buffer(0) for p in objs])

    geoms1 = ensure_geoms(polygons1)
    geoms2 = geoms1 if is_symmetric else ensure_geoms(polygons2)

    areas1 = shapely.area(geoms1)
    areas2 = areas1 if is_symmetric else shapely.area(geoms2)
    
    intersections = shapely.area(shapely.intersection(geoms1[:, np.newaxis], geoms2[np.newaxis, :]))

    areas_min = np.minimum(areas1[:, np.newaxis], areas2[np.newaxis, :])
    ios_mat = (intersections / (areas_min + 1e-6)).astype(np.float32)

    if is_symmetric:
        np.fill_diagonal(ios_mat, 1.0)

    return ios_mat

def base_nms_(
        objects : Any, 
        overlap_fn : Callable, 
        scores : torch.Tensor, 
        collate_fn : Callable=None, 
        overlap_threshold : float=0.5, 
        strict : bool=True, 
        return_indices : bool=False, 
        **kwargs
    ) -> Union[torch.Tensor, Tuple[Any, torch.Tensor]]:
    """
    Implements the standard non-maximum suppression algorithm.

    Args:
        objects (`Any`): An object which can be indexed by a tensor of indices.
        overlap_fn (`Callable`): A function which takes an anchor object and a comparison set (not in the Python sense) of (different) objects and returns the IoU of the anchor object with each object in the comparison set as a tensor of shape (1, n). 
            The reason it is not just (n, ) is to allow for implementations of `overlap_fn` functions between two sets, where the IoU is calculated between each pair of objects from distinct sets.
        scores (`torch.Tensor`): A tensor of shape (n, ) containing the "scores" of the objects, this can merely be though of as a priority score, where the higher the score, the higher the priority of the object - it does not have to be a probability/confidence.
        collate_fn (`Callable`, optional): A function which takes a list of objects and returns a single combined object. Defaults to `torch.cat` if `objects` is a tensor and `lambda x : x` if `objects` is a list, otherwise it has to be specified.
        overlap_threshold (`float`, optional): The overlap (e.g. IoU) threshold for non-maximum suppression. Defaults to 0.5.
        strict (`bool`, optional): A flag to indicate whether to perform strict checks on the algorithm. Defaults to True.
        return_indices (`bool`, optional): A flag to indicate whether to return the indices of the picked objects or the objects themselves. Defaults to False. If True, both the picked objects and scores are returned.
        **kwargs: Additional keyword arguments to be passed to the overlap_fn function.
    
    Returns:
        out (`Union[torch.Tensor, Tuple[Any, torch.Tensor]]`):
            - `torch.Tensor`: A tensor of shape `(m,)` containing the indices of the picked objects.
            - `Tuple[Any, torch.Tensor]`: A tuple where the first element contains the picked objects and the second element is a tensor of their scores.
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
    possible = torch.ones((len(objects),), dtype=torch.bool, device=objects.device)
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
        # Calculate the overlaps (e.g. IoU) between the picked box and the remaining possible boxes
        overlaps = overlap_fn(objects[indices[possible_idx[0]]], objects[indices[possible_idx[1:]]], **kwargs).squeeze(0)
        # Get the indices of the boxes with an overlap greater than the threshold
        winner_mask = overlaps <= overlap_threshold
        # Remove the boxes with an overlap greater than the threshold from the possible boxes
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

def fancy_nms(
        objects : Any, 
        overlap_fn : Callable, 
        scores : torch.Tensor, 
        overlap_threshold : Union[float, int]=0.5, 
        return_indices : bool=False
    ) -> Union[torch.Tensor, Tuple[Any, torch.Tensor]]:
    """
    This is a 'fancy' implementation of non-maximum suppression. It is not as fast as the non-maximum suppression algorithm, nor does it follow the exact same algorithm, but it is more readable and easier to debug.

    The algorithm works as follows:
        1. Sort the objects by score (implicitly)
        2. Calculate the overlap (e.g. IoU) matrix
        3. Create a boolean matrix where overlap > overlap_threshold 
        4. Fold the boolean matrix sequentially (i.e. row_i = row_i + row_i-1 + ... + row_0)
           (The values on the diagonal of the matrix now correspond to the number of higher-priority objects that suppress the current object, including itself)
        5. objects which are suppressed only by themselves are returned.

    
    Args:
        objects (`Any`): Any object collection that can be indexed by a tensor, where the first dimension corresponds to the objects.
        overlap_fn (`Callable`): A function that calculates the symmetric overlap (e.g. IoU) matrix of a set of objects returned as a `torch.Tensor` of shape (n, n), where n is the number of objects. The device should match the device of the scores.
        scores (`torch.Tensor`): A tensor of shape (n, ) containing the scores of the objects.
        overlap_threshold (`Union[float, int]`, optional): The overlap (e.g. IoU) threshold for non-maximum suppression. Defaults to 0.5.
        return_indices (`bool`, optional): A flag to indicate whether to return the indices of the picked objects or the objects themselves. Defaults to False. If True, both the picked objects and scores are returned.

    Returns:
        out (`Union[torch.Tensor, Tuple[Any, torch.Tensor]]`):
            - `torch.Tensor`: A tensor of shape `(m,)` containing the indices of the picked objects.
            - `Tuple[Any, torch.Tensor]`: A tuple where the first element contains the picked objects and the second element is a tensor of their scores.
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

    # Calculate the overlap (e.g. IoU) matrix
    overlaps = overlap_fn(objects[indices])

    # Fold the overlap matrix sequentially (i.e. row_i = row_i + row_i-1 + ... + row_0)
    overlaps = (overlaps > overlap_threshold).cumsum(dim=1) <= 1

    # The boxes with an overlap greater than the threshold are the elements on the diagonal of the folded overlap matrix which are one (suppressed only by itself)
    indices = indices[torch.where(overlaps.diagonal())[0]]

    if return_indices:
        return indices
    else:
        return objects[indices], scores[indices]

# @torch.jit.script
def nms_masks_(
        masks : torch.Tensor, 
        scores : torch.Tensor, 
        overlap_threshold : float=0.5,
        overlap_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]=iou_masks
    ) -> torch.Tensor:
    """
    Performs non-maximum suppression on a set of masks.
    
    Args:
        masks (`torch.Tensor`): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        scores (`torch.Tensor`): A tensor of shape (n, ) containing the scores of the masks.
        overlap_threshold (`float`, optional): The overlap (e.g. IoU) threshold for non-maximum suppression. Defaults to 0.5.

    Returns:
        out (`torch.Tensor`): A tensor containing the indices of the picked masks.
    """
    N, device = len(scores), masks.device
    if N <= 1:
        return torch.arange(N, dtype=torch.long, device=device)

    indices = torch.argsort(scores, descending=True)

    # We MUST cast to float32 here because torch.mm (used in overlap_fn) does not support bool.
    masks = masks.flatten(1).to(dtype=torch.float32)[indices]
    areas = masks.sum(dim=1)

    winners = -torch.ones(N, dtype=torch.long, device=device)
    possible = torch.ones(N, dtype=torch.bool, device=device)
    i = 0

    for _ in range(N):
        possible_idx = possible.nonzero().squeeze(1)
        n_possible = possible_idx.numel()
        
        if n_possible < 2:
            if n_possible == 1:
                possible[possible_idx] = False
                winners[i] = possible_idx
                i += 1
            break
            
        winners[i] = possible_idx[0]
        possible[possible_idx[0]] = False
        
        overlaps = overlap_fn(
            masks[possible_idx[0:1]].unsqueeze(1), 
            masks[possible_idx[1:]].unsqueeze(1), 
            a1s=areas[possible_idx[0:1]], 
            a2s=areas[possible_idx[1:]], 
            dtype=torch.float32
        ).squeeze(0)
        
        winner_mask = overlaps <= overlap_threshold
        possible[possible_idx[1:]] = winner_mask
        i += 1

    return indices[winners[:i]].sort().values 

def nms_polygons_(
        polys : List[torch.Tensor], 
        scores : torch.Tensor, 
        overlap_threshold : float=0.5,
        overlap_fn : Callable[[np.ndarray, np.ndarray], np.ndarray]=iou_polygons
    ) -> torch.Tensor:
    N, device = len(scores), scores.device
    if N <= 1:
        return torch.arange(N, device=device)

    scores_np = scores.cpu().numpy()
    geoms = np.array([shapely.polygons(p.cpu().numpy()).buffer(0) for p in polys])

    indices = np.argsort(scores_np)[::-1] # Ascending sort -> reverse for descending
    geoms = geoms[indices]

    # int64 to ensure compatibility when converting back to torch.long later
    winners = np.full(N, -1, dtype=np.int64)
    possible = np.ones(N, dtype=bool)
    i = 0

    for _ in range(N):
        possible_idx = np.flatnonzero(possible)
        n_possible = possible_idx.size
        
        if n_possible < 2:
            if n_possible == 1:
                possible[possible_idx] = False
                winners[i] = possible_idx[0]
                i += 1
            break

        # Pick the winner
        curr_idx = possible_idx[0]
        winners[i] = curr_idx
        possible[curr_idx] = False

        overlaps = overlap_fn(
            geoms[curr_idx:curr_idx+1], 
            geoms[possible_idx[1:]]
        ).squeeze(0)
        
        # Logical masking in pure NumPy
        winner_mask = overlaps <= overlap_threshold
        possible[possible_idx[1:]] = winner_mask
        i += 1

    return torch.from_numpy(np.sort(indices[winners[:i]])).to(device=device)


def cluster_overlap_boxes(
        boxes: torch.Tensor, 
        overlap_threshold: float = 0.5,
        overlap_fn: Callable[[torch.Tensor], torch.Tensor] = iou_boxes,
        time: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Cluster boxes via connected components.
    
    Note: Implementation relies on ``overlap_fn`` being symmetric (e.g. IoU/IoS).
    """
    N, device = len(boxes), boxes.device
    if N <= 1:
        return [torch.arange(N, dtype=torch.long, device=device)], torch.zeros(N, dtype=torch.long, device=device)
    
    # Chuncked adjacency matrix (memory-to-compute tradeoff)
    CHUNK_SIZE = 2500 
    rows_list = []
    cols_list = []
    
    for i in range(0, N, CHUNK_SIZE):
        chunk_i = boxes[i : i + CHUNK_SIZE]
        for j in range(i, N, CHUNK_SIZE):
            chunk_j = boxes[j : j + CHUNK_SIZE]
            adj_chunk = overlap_fn(chunk_i, chunk_j) >= overlap_threshold
            local_edges = adj_chunk.nonzero().cpu().numpy()
            
            if local_edges.size > 0:
                # Offset the local indices to global indices
                rows_list.append(local_edges[:, 0] + i)
                cols_list.append(local_edges[:, 1] + j)

    # Build Sparse Graph (CPU)
    if not rows_list:
        # Degenerate case: all nodes are isolated
        labels = np.arange(N)
    else:
        row, col = np.concatenate(rows_list), np.concatenate(cols_list)
        data = np.ones(len(row), dtype=bool)
        
        sparse_graph = scipy.sparse.coo_matrix((data, (row, col)), shape=(N, N))
        
        # Find connected components (scipy handles the symmetry implicitly with directed=False)
        _, labels = scipy.sparse.csgraph.connected_components(
            sparse_graph, 
            directed=False, 
            return_labels=True
        )

    # Postprocess
    cluster_vec = torch.from_numpy(labels).to(device=device, dtype=torch.long)
    sorted_idx = torch.argsort(cluster_vec)
    sorted_labels = cluster_vec[sorted_idx]
    
    _, counts = torch.unique(sorted_labels, return_counts=True)
    groups = torch.split(sorted_idx, counts.tolist())

    return list(groups), cluster_vec


OVERLAP_FNS : dict[str, dict[str, Callable[[torch.Tensor], torch.Tensor]]] = {
    "polygon" : {
        "iou" : iou_polygons,
        "ios" : ios_polygons
    },
    "mask" : {
        "iou" : iou_masks,
        "ios" : ios_masks
    },
    "box" : {
        "iou" : iou_boxes,
        "ios" : ios_boxes
    }
}

def get_overlap_fn(geometry : str, metric : str):
    geometry, metric = geometry.lower().strip(), metric.lower().strip()
    if geometry not in OVERLAP_FNS:
        raise NotImplementedError(f'No overlap metrics implemented for geometry type: "{geometry}", valid options are [{", ".join(OVERLAP_FNS.keys())}]')
    options = OVERLAP_FNS[geometry]
    if metric not in options:
        raise NotImplementedError(f'Overlap metric: "{metric}" not implemented for geometry type: "{geometry}", valid options are [{", ".join(options.keys())}]')
    return options[metric]

# @torch.jit.script
def nms_masks(
        masks : torch.Tensor, 
        scores : torch.Tensor, 
        overlap_threshold : float=0.5, 
        return_indices : bool=False, 
        group_first : bool=True, 
        boxes : torch.Tensor=None,
        overlap_fn : Callable[[torch.Tensor], torch.Tensor] | str=iou_masks,
        overlap_fn_boxes : Optional[Union[Callable[[torch.Tensor], torch.Tensor], str]]=None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Efficiently perform non-maximum suppression on a set of boolean masks.

    Defaults to a modified two-stage NMS algorithm, that aims to minimize the number of mask intersection calculations needed.

    Args:
        masks (`torch.Tensor`): A tensor of shape (n, h, w), where n is the number of masks and h and w are the height and width of the masks.
        scores (`torch.Tensor`): A tensor of shape (n, ) containing the "scores" of the masks, this can merely be though of as a priority score, where the higher the score, the higher the priority of the object - it does not have to be a probability/confidence.
        overlap_threshold (`float`, optional): The overlap (e.g. IoU) threshold for non-maximum suppression. Defaults to 0.5.
        return_indices (`bool`, optional): A flag to indicate whether to return the indices of the picked objects or the objects themselves. Defaults to False. If True, both the picked objects and scores are returned.
        group_first (`bool`, optional): A flag to indicate whether two use the two-stage NMS method. Defaults to True.
        boxes (`Optional[torch.Tensor]`, optional): Bounding boxes for the masks. A tensor of shape (n, 4), where n is the number of masks and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the bounding boxes.
    
    Returns:
        out (`Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`):
            - `torch.Tensor`: A tensor of shape `(m,)` containing the indices of the picked objects.
            - `Tuple[torch.Tensor, torch.Tensor]`: A tuple where the first element contains the picked masks and the second element is a tensor of their scores.
    """
    if isinstance(overlap_fn_boxes, str):
        overlap_fn_boxes = get_overlap_fn("box", overlap_fn_boxes)
    if isinstance(overlap_fn, str):
        if overlap_fn_boxes is None:
            overlap_fn_boxes = get_overlap_fn("box", overlap_fn)
        overlap_fn = get_overlap_fn("mask", overlap_fn)
    if not group_first or len(masks) < 10:
        nms_ind = nms_masks_(masks=masks, scores=scores, overlap_threshold=overlap_threshold, overlap_fn=overlap_fn)
    else:
        if boxes is None:
            raise ValueError("'boxes' must be specified for nms_masks when 'group_first' is True")
        if overlap_fn_boxes is None:
            raise RuntimeError("If an overlap function is manually provided for masks, one must also be provided for boxes.")
        # We decrease the overlap_threshold for the clustering, since there is no straight-forward relationship between the IoU of the boxes and the IoU of the masks
        groups, _ = cluster_overlap_boxes(boxes=boxes, overlap_threshold=min(1, overlap_threshold / 4), overlap_fn=overlap_fn_boxes, time=False)
        _nms_ind = [torch.empty(0) for i in range(len(groups))]
        for i, group in enumerate(groups):
            if len(group) == 1:
                _nms_ind[i] = group
            else:
                group_boxes = boxes[group].round().long()
                xmin, ymin, xmax, ymax = group_boxes[:, 0].min(), group_boxes[:, 1].min(), group_boxes[:, 2].max(), group_boxes[:, 3].max()
                _nms_ind[i] = group[nms_masks_(masks=masks[group, ymin:(ymax+1), xmin:(xmax+1)], scores=scores[group], overlap_threshold=overlap_threshold, overlap_fn=overlap_fn)]
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
        overlap_threshold : Union[float, int]=0.5, 
        return_indices : bool=False, 
        group_first : bool=True, 
        boxes : Optional[torch.Tensor]=None,
        overlap_fn : Union[Callable[[List[torch.Tensor], List[torch.Tensor]], torch.Tensor], str]="IoU",
        overlap_fn_boxes : Optional[Union[Callable[[torch.Tensor], torch.Tensor], str]]=None,
    ) -> Union[torch.Tensor, Tuple[List[torch.Tensor], torch.Tensor]]:
    """
    Efficiently perform non-maximum suppression on a set of polygons.

    Defaults to a modified two-stage NMS algorithm, that aims to minimize the number of polygon intersection calculations needed (very expensive).

    Args:
        polygons (`List[torch.Tensor]`): A list of tensors of shape (n, 2), where n is the number of vertices in the polygon and the 2 columns are the x and y coordinates of the vertices.
        scores (`torch.Tensor`): A tensor of shape (n, ) containing the "scores" of the polygons, this can merely be though of as a priority score, where the higher the score, the higher the priority of the object - it does not have to be a probability/confidence.
        overlap_threshold (`float`, optional): The overlap (e.g. IoU) threshold for non-maximum suppression. Defaults to 0.5.
        return_indices (`bool`, optional): A flag to indicate whether to return the indices of the picked objects or the objects themselves. Defaults to False. If True, both the picked objects and scores are returned.
        group_first (`bool`, optional): A flag to indicate whether two use the two-stage NMS method. Defaults to True (recommended).
        boxes (`Optional[torch.Tensor]`, optional): Bounding boxes for the polygons. A tensor of shape (n, 4), where n is the number of polygons and the 4 columns are the x_min, y_min, x_max and y_max coordinates of the bounding boxes.
    
    Returns:
        out (`Union[torch.Tensor, Tuple[List[torch.Tensor], torch.Tensor]]`):
            - `torch.Tensor`: A tensor of shape `(m,)` containing the indices of the picked polygons.
            - `Tuple[List[torch.Tensor], torch.Tensor]`: A tuple where the first element contains the picked polygons and the second element is a tensor of their scores.
    """
    if isinstance(overlap_fn_boxes, str):
        overlap_fn_boxes = get_overlap_fn("box", overlap_fn_boxes)
    if isinstance(overlap_fn, str):
        if overlap_fn_boxes is None:
            overlap_fn_boxes = get_overlap_fn("box", overlap_fn)
        overlap_fn = get_overlap_fn("polygon", overlap_fn)
    else:
        if overlap_fn_boxes is None:
            raise RuntimeError("If an overlap function is manually provided for polygons, one must also be provided for boxes.")
    device = polygons[0].device
    if not group_first or len(polygons) < 10:
        nms_ind = nms_polygons_(polys=polygons, scores=scores, overlap_threshold=overlap_threshold, overlap_fn=overlap_fn)
    else:
        if boxes is None:
            raise ValueError("'boxes' must be specified for nms_masks when 'group_first' is True")
        # We decrease the overlap_threshold for the clustering, since there is no straight-forward relationship between the overlap of the boxes and the overlap of the polygons
        groups, _ = cluster_overlap_boxes(boxes=boxes, overlap_threshold=min(1, overlap_threshold / 4), overlap_fn=overlap_fn_boxes, time=False) 
        nms_ind = [None for _ in range(len(groups))]
        for i, group in enumerate(groups):
            if len(group) == 1:
                nms_ind[i] = group
            else:
                nms_ind[i] = group[nms_polygons_(polys=[polygons[gi] for gi in group], scores=scores[group], overlap_threshold=overlap_threshold, overlap_fn=overlap_fn)]
        if len(nms_ind) > 0:
            nms_ind = torch.cat(nms_ind)
        else:
            nms_ind = torch.tensor([], dtype=torch.long, device=device)
    if return_indices:
        return nms_ind
    else:
        return [polygons[ni] for ni in nms_ind], scores[nms_ind]

def nms_boxes(
        boxes : torch.Tensor, 
        scores : torch.Tensor, 
        overlap_threshold : Union[float, int]=0.5,
        overlap_fn : Optional[Union[Callable[[torch.Tensor], torch.Tensor], str]]=None,
    ) -> torch.Tensor:
    """
    Wrapper for `torchvision.ops.nms`; the standard non-maximum suppression algorithm.
    """
    if overlap_fn is None or isinstance(overlap_fn, str) and (overlap_fn := overlap_fn.strip().lower()) == "iou":
        with torch.autocast(device_type=boxes.device.type, dtype=boxes.dtype):
            return torchvision.ops.nms(boxes, scores, overlap_threshold).sort().values
    if isinstance(overlap_fn, str):
        overlap_fn = get_overlap_fn("box", overlap_fn)
    return base_nms_(boxes, overlap_fn=overlap_fn, scores=scores, overlap_threshold=overlap_fn, return_indices=True)