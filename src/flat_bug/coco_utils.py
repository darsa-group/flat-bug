"""
Evaluation functions for FlatBug datasets.
"""

import os
from typing import Union, List, Tuple, Dict, Optional
import cv2
import numpy as np

# COCO Format:
#
#     {
#         "info": info,
#         "licenses": [license],
#         "images": [image],  // list of all images in the dataset
#         "annotations": [annotation],  // list of all annotations in the dataset
#         "categories": [category]  // list of all categories
#     }
#
#     where:
#
#     info = {
#         "year": int,
#         "version": str,
#         "description": str,
#         "contributor": str,
#         "url": str,
#         "date_created": datetime,
#     }
#
#     license = {
#         "id": int,
#         "name": str,
#         "url": str,
#     }
#
#     image = {
#         "id": int,
#         "width": int,
#         "height": int,
#         "file_name": str,
#         "license": int,  // the id of the license
#         "date_captured": datetime,
#     }
#
#     category = {
#         "id": int,
#         "name": str,
#         "supercategory": str,
#     }
#
#     annotation = {
#         "id": int,
#         "image_id": int,  // the id of the image that the annotation belongs to
#         "category_id": int,  // the id of the category that the annotation belongs to
#         "segmentation": RLE or [polygon],
#         "area": float,
#         "bbox": [x,y,width,height],
#         "iscrowd": int,  // 0 or 1,
#     }

# Flat-Bug Format:

#     {
#         "boxes": [[xmin, ymin, xmax, ymax], ...],
#         "contours": [[[x_1, x_2, ..., x_n], [y_1, y_2, ..., y_n]], ...],
#         "confs": [float, ...],
#         "classes": [int, ...],
#         "scales": [int, ...],
#         "identifier": "",
#         "image_path": "",
#         "image_width": int,
#         "image_height": int,
#         "mask_width": int,
#         "mask_height": int
#     }




def fb_to_coco(d: dict, coco: dict) -> dict:
    """
    Converts a FlatBug dataset to a COCO dataset.

    Args:
        d (dict): FlatBug dataset.
        coco (dict): An instantiated COCO dataset or an empty dictionary.

    Returns:
        dict: COCO dataset.
    """

    if len(coco) == 0:
        image_id = 1
        object_id_offset = 0
        coco.update(
            {
                "licenses": [
                    {
                        "name": "",
                        "id": 0,
                        "url": ""
                    }
                ],
                "info": {
                    "contributor": "",
                    "date_created": "",
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": ""
                },
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 1,
                        "name": "insect",
                        "supercategory": ""
                    }
                ]
            }
        )
    else:
        image_id = len(coco["images"]) + 1
        if len(coco["annotations"]) == 0:
            object_id_offset = 0
        else:
            object_id_offset = coco["annotations"][-1]["id"] + 1

    boxes, contours, confs, classes, scales = d["boxes"], d["contours"], d["confs"], d["classes"], d["scales"]
    identifier, image_path = d["identifier"], d["image_path"]
    image_width, image_height, mask_width, mask_height = d["image_width"], d["image_height"], d["mask_width"], d[
        "mask_height"]

    # Image
    image = {
        "id": image_id,
        "width": image_width,
        "height": image_height,
        "file_name": os.path.basename(image_path),
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    }
    coco["images"].append(image)

    # Boxes, contours, confs, classes, scales
    for i in range(len(boxes)):
        box, contour, conf, class_, scale = boxes[i], contours[i], confs[i], classes[i], scales[i]
        x1, y1, x2, y2 = box
        x,y,w,h = x1, y1, x2 - x1, y2 - y1
        box=[x,y,w,h]

        # Scale and restructure the contour
        m2i = [(mask_width - 1) / (image_width - 1), (mask_height - 1) / (image_height - 1)]  # Mask to image ratio
        contour = [[round(c / m2i[d]) for c in contour[d]] for d in range(2)]
        contour = [c for p in zip(contour[0], contour[1]) for c in p]

        # Annotation
        annotation = {
            "id": i + object_id_offset,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [contour],
            "area": 0.0,
            "bbox": box,
            "iscrowd": 0,
            "conf": conf
        }
        coco["annotations"].append(annotation)

    return coco


def format_contour(c) -> np.array:
    """
    Formats a contour to the OpenCV format.

    Args:
        c (list): Contour.

    Returns:
        np.array: Formatted contour.
    """
    c = c[0]
    return np.array([[c[i], c[i + 1]] for i in range(0, len(c), 2)], dtype=np.int32)


def contour_bbox(c: np.array) -> np.array:
    """
    Calculates the bounding box of a contour.

    Args:
        c (np.array): Contour.

    Returns:
        np.array: Bounding box.
    """
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])


def split_annotations(coco: Dict, strip_directories: bool = True) -> Dict[str, dict]:
    """
    Splits COCO annotations by image ID.

    Args:
        coco (Dict): COCO dataset.

    Returns:
        Dict[Dict]: Dict of COCO datasets, split by image ID and keyed by image name.
    """
    img_id = np.array([i["image_id"] for i in coco["annotations"]])
    ids = np.unique(np.array([i["id"] for i in coco["images"]]))
    groups = [np.where(img_id == i)[0] for i in ids]

    if strip_directories:
        for i in range(len(coco["images"])):
            coco["images"][i]["file_name"] = os.path.basename(coco["images"][i]["file_name"])

    result = {coco["images"][id - 1]["file_name"]: [coco["annotations"][i] for i in g] for id, g in zip(ids, groups)}
    
    # Ensure that all images are included in the result, even if they have no annotations/predictions
    for i in range(len(coco["images"])):
        image_name = coco["images"][i]["file_name"]
        if image_name not in result:
            result[image_name] = []
    
    return result


def annotations_2_contours(annotations: Dict[str, dict]) -> Dict[str, List[np.array]]:
    """
    Converts COCO annotations to contours.

    Args:
        annotations (Dict[str, dict]): COCO annotations.

    Returns:
        Dict[str, List[np.array]]: Contours.
    """
    return {k: [format_contour(i["segmentation"]) for i in v] for k, v in annotations.items()}


def contour_area(c: np.array) -> np.array:
    """
    Calculates the area of a contour.

    Args:
        c (np.array): Contour of shape (n, 2).

    Returns:
        np.array[np.int32]: Scalar area of the contour of shape (1,).
    """
    # return cv2.contourArea(c)
    min_xy = c.min(axis=0)
    c = c - min_xy
    max_xy = c.max(axis=0) + 1
    mask = np.zeros(max_xy[::-1], dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 1, thickness=cv2.FILLED)
    return np.sum(mask, dtype=np.int64)


def annotations_to_numpy(annotations: List[Dict[str, Union[int, List[int]]]]) -> Tuple[np.array, np.array]:
    """
    Converts COCO annotations to NumPy arrays.

    Args:
        annotations (List[Dict[str, Union[int, List[int]]]]): COCO annotations.

    Returns:
        Tuple[np.array, np.array]: Bounding boxes and contours.
    """
    contours = [format_contour(i["segmentation"]) for i in annotations]
    bboxes = np.array([contour_bbox(c) for c in contours])
    return bboxes, contours

def filter_coco(coco : Dict, confidence : Optional[float] = None, area : Optional[int] = None, verbose=False) -> Dict:
    """
    Filters COCO annotations by confidence.

    Args:
        coco (Dict): COCO dataset.
        confidence (float): Confidence threshold.
        area (int): Area threshold.

    Returns:
        Dict: Filtered COCO dataset.
    """
    filtered_annotations = []
    for a in coco["annotations"]:
        if confidence is not None and "conf" in a:
            if a["conf"] < confidence:
                continue
        if area is not None and "bbox" in a:
            _, _, w, h = a["bbox"]
            if (w * h) < area:
                if verbose:
                    print("SKIPPED")
                continue
        filtered_annotations += [a]

    return {
        "info": coco["info"],
        "licenses": coco["licenses"],
        "images": coco["images"],
        "annotations": filtered_annotations,
        "categories": coco["categories"]
    }