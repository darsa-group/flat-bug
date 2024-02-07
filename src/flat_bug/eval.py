"""
Evaluation functions for FlatBug datasets.
"""

# COCO Format:

#     {
#         "info": info,
#         "licenses": [license], 
#         "images": [image],  // list of all images in the dataset
#         "annotations": [annotation],  // list of all annotations in the dataset
#         "categories": [category]  // list of all categories
#     }

#     where:

#     info = {
#         "year": int, 
#         "version": str, 
#         "description": str, 
#         "contributor": str, 
#         "url": str, 
#         "date_created": datetime,
#     }

#     license = {
#         "id": int, 
#         "name": str, 
#         "url": str,
#     }

#     image = {
#         "id": int, 
#         "width": int, 
#         "height": int, 
#         "file_name": str, 
#         "license": int,  // the id of the license
#         "date_captured": datetime,
#     }

#     category = {
#         "id": int, 
#         "name": str, 
#         "supercategory": str,
#     }

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

import time
from typing import Union

def fb_to_coco(d : dict, coco : dict) -> dict:
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
    image_width, image_height, mask_width, mask_height = d["image_width"], d["image_height"], d["mask_width"], d["mask_height"]

    for i in range(len(boxes)):
        box, contour, conf, class_, scale = boxes[i], contours[i], confs[i], classes[i], scales[i]

        # Scale and restructure the contour
        m2i = [mask_width / image_width, mask_height / image_height] # Mask to image ratio
        contour = [[round(c / m2i[d], 2)  for c in contour[d]] for d in range(2)]
        contour = [c for p in zip(contour[0], contour[1]) for c in p]
        # Image
        image = {
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": image_path,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        coco["images"].append(image)

        # Annotation
        annotation = {
            "id": i + object_id_offset,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": contour,
            "area": 0.0,
            "bbox": box,
            "iscrowd": 0
        }
        coco["annotations"].append(annotation)

    return coco

import cv2

def format_contour(c) -> list:
    """
    Formats a contour to the OpenCV format.

    Args:
        c (list): Contour.

    Returns:
        list: Formatted contour.
    """
    return [[c[i], c[i + 1]] for i in range(0, len(c), 2)]

def contour_intersection(c1 : list, c2 : list) -> float:
    """
    Calculates the intersection of two contours.

    Contours should be providedd as [x1, y1, x2, y2, ..., xn, yn]

    Algorithm:
    1. Format the contours to OpenCV format.
    2. Calculate the area of the contours.
    3. If the area of either contour is 0, return 0.
    4. Calculate the bounding box of the contours.

    Args:
        c1 (list): Contour 1. 
        c2 (list): Contour 2.

    Returns:
        float: Intersection area.
    """

    c1, c2 = format_contour(c1), format_contour(c2)

    a1, a2 = cv2.contourArea(c1), cv2.contourArea(c2)

    if a1 == 0 or a2 == 0:
        return 0
    
    b1, b2 = cv2.boundingRect(c1), cv2.boundingRect(c2)

    


import os
from glob import glob
import json

test_fb = [json.load(open(p)) for p in glob("dev/**/**.json", recursive=True)]

coco = {}

[fb_to_coco(d, coco) for d in test_fb]

coco["annotations"][0]["segmentation"]
