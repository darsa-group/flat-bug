import logging
from shapely.geometry import Polygon
import numpy as np
from typing import List, Tuple, Union, IO


def iou_match_pairs(arr: np.ndarray, iou_threshold: float) -> List[Tuple[int, int]]:
    """
    :param arr: a triangular 2d array containing iou values
    :param iou_threshold: the threshold under which two objects do not match
    :return: A list of matched object, by index. None for no match.
    """
    pairs = []
    arr[arr < iou_threshold] = 0

    gt_not_in_im = np.where(np.sum(arr, axis=1) == 0)[0]
    im_not_in_gt = np.where(np.sum(arr, axis=0) == 0)[0]

    for g in gt_not_in_im:
        pairs.append((g, None))

    for i in im_not_in_gt:
        pairs.append((None, i))

    while np.sum(arr) > 0:
        i, j = np.unravel_index(arr.argmax(), arr.shape)
        pairs.append((i, j))
        arr[i, :] = 0
        arr[:, j] = 0
    return pairs


def iou(poly1: Polygon, poly2: Polygon):
    try:
        inter = poly1.intersection(poly2).area
        if inter == 0:
            return 0
        return inter / poly1.union(poly2).area
    except Exception as e:
        logging.error(e)
        return 0
