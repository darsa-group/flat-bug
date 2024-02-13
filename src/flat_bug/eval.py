"""
Evaluation functions for FlatBug datasets.
"""

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

import time
import os
from typing import Union, List, Tuple, Dict
import cv2
import numpy as np

DEBUG_DIRECTORY = "dev"

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

    # Boxes, contours, confs, classes, scales
    for i in range(len(boxes)):
        box, contour, conf, class_, scale = boxes[i], contours[i], confs[i], classes[i], scales[i]

        # Scale and restructure the contour
        m2i = [(mask_width - 1) / (image_width - 1), (mask_height - 1) / (image_height - 1)] # Mask to image ratio
        contour = [[round(c / m2i[d])  for c in contour[d]] for d in range(2)]
        contour = [c for p in zip(contour[0], contour[1]) for c in p]

        # Annotation
        annotation = {
            "id": i + object_id_offset,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": [contour],
            "area": 0.0,
            "bbox": box,
            "iscrowd": 0
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

def contour_bbox(c : np.array) -> np.array:
    """
    Calculates the bounding box of a contour.

    Args:
        c (np.array): Contour.

    Returns:
        np.array: Bounding box.
    """
    return np.array([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()])

def split_annotations(coco : Dict) -> Dict[str, dict]:
    """
    Splits COCO annotations by image ID.

    Args:
        coco (Dict): COCO dataset.

    Returns:
        Dict[Dict]: Dict of COCO datasets, split by image ID and keyed by image name.
    """
    img_id = np.array([i["image_id"] for i in coco["annotations"]])
    ids = np.unique(img_id)
    groups = [np.where(img_id == i)[0] for i in ids]
    return {coco["images"][id - 1]["file_name"] : [coco["annotations"][i] for i in g] for id, g in zip(ids, groups)}

def annotations_2_contours(annotations : Dict[str, dict]) -> Dict[str, List[np.array]]:
    """
    Converts COCO annotations to contours.

    Args:
        annotations (Dict[str, dict]): COCO annotations.

    Returns:
        Dict[str, List[np.array]]: Contours.
    """
    return {k : [format_contour(i["segmentation"]) for i in v] for k, v in annotations.items()}

def bbox_intersect(b1, b2s):
    """
    Calculate the intersecting rectangle between two rectangles. The rectangles must be aligned with the axes.

    Args:
        b1 (np.array): Bounding box 1.
        b2s (np.array): Bounding boxes 2.

    Returns:
        np.array: Intersecting rectangles of shape (n, 4).
    """
    if len(b2s.shape) == 1:
        b2s = b2s.copy().reshape(1, 4)
    ix_max = np.maximum(b1[:2], b2s[:, :2])
    ix_min = np.minimum(b1[2:], b2s[:, 2:])
    ix = np.zeros_like(b2s)
    ix[:, :2] = ix_max
    ix[:, 2:] = ix_min
    # Check for no intersection
    ix[(ix_min < ix_max).any(axis=1)] = 0
    return ix

def bbox_intersect_area(b1, b2s):
    """
    Calculate the area of the intersecting rectangle between two rectangles. The rectangles must be aligned with the axes.

    Args:
        b1 (np.array): Bounding box 1.
        b2s (np.array): Bounding boxes 2.

    Returns:
        np.array: Area of the intersecting rectangles of shape (n,).
    """
    if len(b2s.shape) == 1:
        b2s = b2s.copy().reshape(1, 4)
    ix_max = np.maximum(b1[:2], b2s[:, :2])
    ix_min = np.minimum(b1[2:], b2s[:, 2:])
    return np.prod((ix_min - ix_max).clip(0), axis=1)

def contour_area(c : np.array) -> np.array:
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

def contour_intersection(contour1 : np.array, contour2 : np.array, box1 : np.array, box2 : np.array) -> np.array:
    """
    Calculates the intersection of two contours.

    Contours should be providedd as [x1, y1, x2, y2, ..., xn, yn]

    Algorithm:
    1. Format the contours to OpenCV format.
    2. Calculate the area of the contours.
    3. If the area of either contour is 0, return 0.
    4. Offset the contours such that the minimum x and y coordinates are 0.
    5. Create a mask for each contour, the masks have size (max_x, max_y).
    6. Calculate the intersection of the masks by multiplying them element-wise.
    7. Return the sum of the intersection mask.

    Args:
        contour1 (np.array): Contour 1. 
        contour2 (np.array): Contour 2.
        box1 (np.array): Bounding box 1.
        box2 (np.array): Bounding box 2.

    Returns:
        np.array[np.int64]: Scalar intersection of the contours.
    """
    # If any of the contours are empty, return 0
    if len(contour1) < 2 or len(contour2) < 2:
        return 0
    
    # Find the minimum x and y coordinates, which will be used to offset the contour coordinates
    min_xy = np.minimum(box1[:2], box2[:2])
    # Find the maximum x and y coordinates, which will be used to create the mask
    max_xy = np.maximum(box1[2:], box2[2:]) + 1 - min_xy
    
    # Offset the contour coordinates
    contour1 = contour1 - min_xy
    contour2 = contour2 - min_xy
    if contour1.min() < 0 or contour2.min() < 0:
        raise Exception("Negative contour coordinates")

    # Create the mask
    mask1, mask2 = np.zeros(max_xy[::-1], dtype=np.uint8), np.zeros(max_xy[::-1], dtype=np.uint8)
    # Draw the contours
    cv2.drawContours(mask1, [contour1], -1, 1, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 1, thickness=cv2.FILLED)
    # Calculate the intersection
    return (mask1 * mask2).sum(dtype=np.int64)

def pairwise_contour_intersection(contours1 : List[np.array], contours2 : Union[None, List[np.array]]=None, bboxes1 : Union[None, np.array]=None, bboxes2 : Union[None, np.array]=None, areas1 : Union[None, np.array]=None, areas2 : Union[None, np.array]=None) -> np.array:
    """
    Calculates the pairwise intersection of two groups of contours.

    Args:
        contours1 (List[np.array]): Contours 1.
        contours2 (Union[None, List[np.array]]): Contours 2. If not provided, symmetric intersection is calculated for contours1 instead.
        areas1 (Union[None, np.array]): Areas 1. Computed if not provided.
        areas2 (Union[None, np.array]): Areas 2. Computed if not provided.
        bboxes1 (Union[None, np.array]): Bounding boxes 1. Computed if not provided.
        bboxes2 (Union[None, np.array]): Bounding boxes 2. Computed if not provided.

    Returns:
        np.array: Intersection matrix of shape (n, m).
    """

    # If contours2 is not provided, set it to contours1
    if contours2 is None:
        contours2 = contours1

    # Calculate the number of contours in each group
    n = len(contours1)
    m = len(contours2)

    # Calculate the bounding boxes and areas of the contours
    if bboxes1 is None:
        bboxes1 = np.array([contour_bbox(c) for c in contours1])
    if bboxes2 is None:
        bboxes2 = np.array([contour_bbox(c) for c in contours2])
    if areas1 is None:
        areas1 = np.array([contour_area(c) for c in contours1])
    if areas2 is None:
        areas2 = np.array([contour_area(c) for c in contours2])

    # Initialize the intersection matrix
    intersections = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        bintersect = bbox_intersect_area(bboxes1[i], bboxes2)
        for j in np.where(bintersect > 0)[0]:
            intersections[i, j] = contour_intersection(contours1[i], contours2[j], bboxes1[i], bboxes2[j])
    
    # Calcute and return the IoU
    return intersections
    
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

def match_geoms(contours1 : List[np.array], contours2 : List[np.array], threshold : float=1/4, iou_mat : Union[None, np.array]=None, areas1 : Union[None, np.array]=None, areas2 : Union[None, np.array]=None) -> Tuple[np.array, int]:
    """
    Matches geometries in group 1 to geometries in group 2.

    Args:
        contours1 (List[np.array]): Geometries 1. List of length N, where each element is a Xx2 array of contour coordinates.
        contours2 (List[np.array]): Geometries 2. List of length M, where each element is a Xx2 array of contour coordinates.
        threshold (float, optional): IoU threshold. Defaults to 1/4.
        iou_mat (Union[None, np.array], optional): IoU matrix. Computed if not provided. Defaults to None.
        areas1 (Union[None, np.array], optional): Areas 1. Computed if not provided. Defaults to None.
        areas2 (Union[None, np.array], optional): Areas 2. Computed if not provided. Defaults to None.

    Returns:
        List[np.array, int] : Nx2 array of matched indices from group 1 and group 2, and the number of unmatched geometries in group 2.
    """
    # Calculate the number of contours in each group
    n = len(contours1)
    m = len(contours2)
    if iou_mat is None:
        # Calculate the IoU matrix
        intersections = pairwise_contour_intersection(contours1, contours2)
        intersections = intersections / (areas1.reshape(-1, 1) + areas2.reshape(1, -1) - intersections)
    else:
        iou = iou_mat.copy()
    # Check the shape of the IoU matrix
    if iou.shape != (n, m):
        raise ValueError(f'Expected IoU matrix of shape {(n, m)}, got {iou.shape}')
    # Initialize the matche array
    matches = np.zeros((n, 2), dtype=np.int32)
    matches[:, 0] = np.arange(n, dtype=np.int32)
    matches[:, 1] = -1 # No match
    # Match the geometries in group 1 to the geometries in group 2
    for i in np.argsort(iou.max(axis=1)):
        j = np.argmax(iou[i])
        if iou[i, j] > threshold:
            matches[i, 1] = j.astype(np.int32)
            # Set the intersection to 0 so it doesn't get matched again
            iou[:, j] = 0
    # Find the remaining unmatched geometries in group 2
    unmatched = np.where(iou.sum(axis=0) > 0)[0]
    matches_2 = np.zeros((len(unmatched), 2), dtype=np.int32)
    matches_2[:, 0] = -1
    matches_2[:, 1] = unmatched
    # Add the unmatched geometries to the matches array
    matches = np.concatenate([matches, matches_2], axis=0)
    return matches, len(unmatched)

def plot_heatmap(mat : np.array, breaks : int=25, dimensions : Union[None, Tuple[int, int]]=None, fast : bool=True, output_path : str=None):
    """
    Plots a heatmap of a matrix using OpenCV.

    Args:
        mat (np.array): Matrix.
        breaks (int, optional): Number of breaks on the colorbar. Defaults to 25.
        dimensions (Union[None, Tuple[int, int]], optional): Dimensions of the output image. Defaults to None.
        fast (bool, optional): Whether to use fast mode. Defaults to True.
        output_path (str, optional): Output path. Defaults to None.

    Returns:
        None
    """
    if dimensions is None:
        dimensions = tuple([m * 10 for m in mat.shape[::-1]])
    # Create a colormap for viridis
    colormap = cv2.applyColorMap((mat / mat.max() * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    # Expand colormap to 1000x1000
    colormap = cv2.resize(colormap, dimensions, interpolation=cv2.INTER_NEAREST_EXACT)
    # Get the height of the colormap
    cheight = colormap.shape[0]

    ## Prepare data for the colorbar
    # Add numbers to the colorbar
    cmin, cmax = mat.min(), mat.max()
    cmin, cmax = 10 ** np.floor(np.log10(cmin)) if cmin > 0 else cmin, 10 ** np.ceil(np.log10(cmax))
    cmin, cmax = int(cmin), int(cmax)

    # Add semi-equally spaced numbers to the colorbar at "nice" values, "nice" values are defined as integer multiples of powers of 10 to the power of the maximum value - the integer rounded 10 logarithm of the number of breaks
    raw_breaks = np.linspace(cmin, cmax, breaks)
    nice_multiple = 10 ** (np.log10(cmax) - np.ceil(np.log10(breaks)))
    nice_breaks = (raw_breaks / nice_multiple).round() * nice_multiple
    nice_breaks = nice_breaks[nice_breaks <= cmax]
    # Ensure that the minimum and maximum values are included, and remove the breaks if they are within 1 "nice_multiple" of any other break
    nice_breaks = nice_breaks[np.abs(nice_breaks - cmin) >= (nice_multiple * 0.9)]
    nice_breaks = nice_breaks[np.abs(nice_breaks - cmax) >= (nice_multiple * 0.9)]
    nice_breaks = np.concatenate([[cmin], nice_breaks, [cmax]])
    # Create the labels
    labels = [f'{(i * 100):.3g}%' for i in nice_breaks]

    # Define a target font height
    font_height_target = int(min(cheight / 50, max(1, ((cheight * 0.5) / breaks))))
    # Dynamically calculate the font size
    font_size = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, font_height_target, 3)
    # Get the size of the labels
    label_sizes = [cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 3)[0] for label in labels]
    # Get the maximum width of the labels
    max_width = max([i[0] for i in label_sizes])
    # Calculate the width of the colorbar
    colorbar_width = int(max_width * 1.25)

    # Create a colorbar
    colorbar = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1).repeat(colorbar_width, 1), cv2.COLORMAP_VIRIDIS)
    # Stretch the colorbar to the height of the colormap
    colorbar = cv2.resize(colorbar, (colorbar.shape[1], cheight), interpolation=cv2.INTER_LINEAR_EXACT)
    cbheight = colorbar.shape[0]
    # Add the breaks to the colorbar
    for i, b in enumerate(nice_breaks):
        label = labels[i]
        text_width, font_height = label_sizes[i]
        cv2.putText(colorbar, label, ((colorbar_width - text_width) // 2, int((i + 0.5) * cbheight / len(nice_breaks) + font_height / 2)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), (3 * font_height_target) // 15, cv2.LINE_AA)
        cv2.putText(colorbar, label, ((colorbar_width - text_width) // 2, int((i + 0.5) * cbheight / len(nice_breaks) + font_height / 2)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_height_target // 15, cv2.LINE_AA)

    # Concatenate the colormap and the colorbar
    colormap = cv2.hconcat([colormap, colorbar])

    if fast:
        # Rescale the colormap to 2x lower resolution
        colormap = cv2.resize(colormap, (colormap.shape[1] // 2, colormap.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
        _, output_ext = os.path.splitext(output_path)
        if not output_ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            print(f'WARNING: Expected output path to have a JPEG extension, got {output_ext}. May negate some of the speed benefits of the fast mode.') 
        # Save the image
        cv2.imwrite(output_path, colormap, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    else:
        # Save the image
        cv2.imwrite(output_path, colormap)

def plot_matches(matches : np.array, contours1 : list[np.array], contours2 : List[np.array], image_path : str, output_path : str):
    """
    Plots the matches between two groups of contours using OpenCV.

    Args:
        matches (np.array): Matches.
        contours1 (list[np.array]): Contours of group 1.
        contours2 (list[np.array]): Contours of group 2.
        image_path (str): Path to the image.
        output_path (str): Output path.

    Returns:
        None
    """
    # Check the output path extension
    _, out_ext = os.path.splitext(output_path)
    if not out_ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        raise ValueError(f'Expected output path to have a .JPG/.jpg/.JPEG/.jpeg extension, got {out_ext}')

    # Load the image
    image = cv2.imread(image_path)

    # Calculate the bounding boxes of the contours
    bboxes1 = np.array([contour_bbox(c) for c in contours1])
    bboxes2 = np.array([contour_bbox(c) for c in contours2])
    
    # Create two copies of the image
    # For contours in group 1
    cimg1 = image.copy()
    # For contours in group 2
    cimg2 = image.copy()

    # Draw the contours on the copies of the image, and the boxes with indices on the original image
    for idx, (i, j) in enumerate(matches):
        if i != -1:
            # Draw the first contour mask on the copy
            cv2.fillPoly(cimg1, [contours1[i]], (0, 255, 0))
            # Draw boxes around the contours 
            cv2.rectangle(image, (bboxes1[i][0], bboxes1[i][1]), (bboxes1[i][2], bboxes1[i][3]), (0, 255, 0), 8)
        if j != -1:
            # Draw the second contour mask on the copy
            cv2.fillPoly(cimg2, [contours2[j]], (255, 0, 0))
            # Draw boxes around the contours
            cv2.rectangle(image, (bboxes2[j][0], bboxes2[j][1]), (bboxes2[j][2], bboxes2[j][3]), (255, 0, 0), 8)
        
        # if i != -1:
        #     # Draw a blue outline around the contour on the copy
        #     cv2.drawContours(cimg1, [contours1[i]], -1, (0, 0, 255), 25)
        # if j != -1:
        #     # Draw a blue outline around the contour on the copy
        #     cv2.drawContours(cimg1, [contours1[i]], -1, (0, 0, 255), 25)
        
    # Blend the image copies with the contour masks together (makes the contours semi-transparent - alpha=0.5)
    cv2.addWeighted(cimg1, 0.5, cimg2, 0.5, 0, dst=cimg1)
    # Blend with the blended copies with original image
    cv2.addWeighted(image, 0.5, cimg1, 0.5, 0, dst=image)
    for idx, (i, j) in enumerate(matches):
        no_match = (i == -1) or (j == -1)
        if no_match:
            match_color = (0, 0, 255)
        else:
            match_color = (255, 255, 255)
        if i != -1:
            # Draw a text label next to the box
            cv2.putText(image, f"{idx}", (bboxes1[i][0], bboxes1[i][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 10, cv2.LINE_8)
            cv2.putText(image, f"{idx}", (bboxes1[i][0], bboxes1[i][1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, match_color, 5 if no_match else 2, cv2.LINE_AA)
        if j != -1:
            # Draw a text label next to the box
            cv2.putText(image, f"{idx}", (bboxes2[j][0], bboxes2[j][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 10, cv2.LINE_8)
            cv2.putText(image, f"{idx}", (bboxes2[j][0], bboxes2[j][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, match_color, 5 if no_match else 2, cv2.LINE_AA)
    # Downscale the image
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

    # Save the image
    cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

def compare_groups(group1 : list, group2 : list, threshold : float=1/10, plot : bool=True, image_path : str=None, output_identifier : str=None, output_directory : str=None) -> str:
    """
    Compares group 1 to group 2.

    Output is saved to a CSV file with the following columns:
        - Idx_1 (`int`) is the index of the geometry in group 1
        - Idx_2 (`int`) is the index of the matched geometry in group 2, or -1 if there is no match
        - IoU (`float`) is the intersection over union, or 0 if there is no match
        - contourArea_1 (`int`) is the area of the contour in group 1
        - contourArea_2 (`int`) is the area of the contour in group 2, or 0 if there is no match
        - bbox_1 (`list[int, int, int, int : xmin, ymin, xmax, ymax]`) is the bounding box of the geometry in group 1
        - bbox_2 (`list[int, int, int, int : xmin, ymin, xmax, ymax]`) is the bounding box of the matched geometry in group 2, or an empty list if there is no match
        - contour_1 (`list[list[int, int : x_i, y_i]]`) is the contour of the geometry in group 1
        - contour_2 (`list[list[int, int : x_i, y_i]]`) is the contour of the matched geometry in group 2, or an empty list if there is no match

    Args:
        group1 (list): Group 1.
        group2 (list): Group 2.

    Returns:
        str: Path to the CSV file.
    """
    # Type check the input
    if not isinstance(group1, list) or not isinstance(group2, list):
        raise ValueError(f'Expected group1 and group2 to be lists, got {type(group1)} and {type(group2)}')
    if not all([isinstance(i, dict) for i in group1]) or not all([isinstance(i, dict) for i in group2]):
        raise ValueError(f'Expected group1 and group2 to be lists of dictionaries, got {type(group1[0])} and {type(group2[0])}')
    if not isinstance(threshold, float):
        raise ValueError(f'Expected threshold to be a float, got {type(threshold)}')
    if not isinstance(plot, bool):
        raise ValueError(f'Expected plot to be a bool, got {type(plot)}')
    if not (isinstance(image_path, str) or image_path is None):
        raise ValueError(f'Expected image_path to be a string or None, got {type(image_path)}')
    if not isinstance(output_directory, str):
        raise ValueError(f'Expected output_directory to be a string, got {type(output_directory)}')
    elif not os.path.isdir(output_directory):
        raise ValueError(f'Expected output_directory to be a valid directory, got {output_directory}')
    if not isinstance(output_identifier, str):
        raise ValueError(f'Expected output_identifier to be a string, got {type(output_identifier)}')

    # Convert the annotations to NumPy arrays, and calculate bounding boxes and areas
    b1, c1 = annotations_to_numpy(group1)
    b2, c2 = annotations_to_numpy(group2)
    a1, a2 = np.array([contour_area(c) for c in c1]), np.array([contour_area(c) for c in c2])
    len_1, len_2 = len(c1), len(c2)

    # Calculate the IoU matrix
    intersection = pairwise_contour_intersection(c1, c2, b1, b2, a1, a2)
    union = a1.reshape(-1, 1) + a2.reshape(1, -1) - intersection
    iou = intersection / union
    # Match the geometries
    matches, misses = match_geoms(c1, c2, threshold, iou)
    # Plot the matches and the IoU matrix
    if plot:
        if not isinstance(image_path, str):
            raise ValueError(f'Expected path to be a string, got {type(image_path)}')
        elif not os.path.isfile(image_path):
            raise ValueError(f'Expected path to be a valid file, got {image_path}')
        plot_matches(matches, c1, c2, image_path, f'{output_directory}{os.sep}{output_identifier}_matches.jpg')
        plot_heatmap(iou, fast=False, output_path=f'{output_directory}{os.sep}{output_identifier}_heatmap.png')
    if misses != 0:
        print(f"Missed {misses} geometries")
    
    ## Gather the data for the output
    matched_1 = matches[:, 1] != -1
    matched_2 = matches[:, 0] != -1
    unmatched_1 = np.where(~matched_1)[0]
    unmatched_2 = np.where(~matched_2)[0]

    # Get the matched IoU
    matched_iou = iou[*matches[:len_1].T]
    # Set the IoU of unmatched geometries to 0
    matched_iou[unmatched_1] = 0
    matched_iou = np.concatenate([matched_iou, np.zeros(len(unmatched_2))])
    
    # Get the matched bounding boxes
    boxes1 = b1[matches[:, 0]]
    boxes2 = b2[matches[:, 1]]

    # Get the matched areas
    careas1 = a1[matches[:, 0]]
    careas2 = a2[matches[:, 1]]
    # Set the areas of unmatched geometries to 0
    careas1[unmatched_2] = 0
    careas2[unmatched_1] = 0

    # Convert the bounding boxes to lists
    boxes1 = boxes1.tolist()
    boxes2 = boxes2.tolist()
    for i in unmatched_2:
        boxes1[i] = []
    for i in unmatched_1:
        boxes2[i] = []

    # Get the matched contours
    contours1 = [c1[i].tolist() if i != -1 else [] for i in matches[:, 0]]
    contours2 = [c2[i].tolist() if i != -1 else [] for i in matches[:, 1]]

    # Get the indices of the geometries in group 1
    idx1, idx2 = matches.T

    # Get the lengths of all the data which are not single values
    len_idx1, len_idx2 = len(idx1), len(idx2)
    len_matched_iou = len(matched_iou)
    len_careas1, len_careas2 = len(careas1), len(careas2)
    len_boxes1, len_boxes2 = len(boxes1), len(boxes2)
    len_contours1, len_contours2 = len(contours1), len(contours2)
    # Check that the lengths are all the same
    data_length = set([len_idx1, len_idx2, len_matched_iou, len_careas1, len_careas2, len_boxes1, len_boxes2, len_contours1, len_contours2])
    if len(data_length) != 1:
        raise ValueError(f"Lengths of the data are not all the same: {len_idx1, len_idx2, len_matched_iou, len_careas1, len_careas2, len_boxes1, len_boxes2, len_contours1, len_contours2}")

    # Construct the output by combining the data
    output = {
        "idx_1" : idx1, 
        "idx_2" : idx2, 
        "IoU" : matched_iou, 
        "contourArea_1" : careas1, 
        "contourArea_2": careas2, 
        "bbox_1" : boxes1, 
        "bbox_2" : boxes2, 
        "contour_1" : contours1, 
        "contour_2" : contours2
    }

    # Write the output to a CSV file
    output_path = f"{output_directory}{os.sep}{output_identifier}.csv"
    separator = ";"
    with open(output_path, "w") as out:
        columns = list(output.keys())
        data = list(output.values())
        out.write(separator.join(columns) + "\n")
        for row in zip(*data):
            out.write(separator.join([str(i) for i in row]) + "\n")

    # Return the path to the output
    return output_path



import os
from glob import glob
import json
from tqdm import tqdm

files = sorted(glob("dev/**/**.json", recursive=True))
flat_bug_pred = [json.load(open(p)) for p in files]

pred_coco = {}
[fb_to_coco(d, pred_coco) for d in flat_bug_pred]

gt_coco = json.load(open("s3/CollembolAI/instances_default.json"))

images = [i["file_name"] for i in gt_coco["images"]]

gt_annotations, pred_annotations = split_annotations(gt_coco), split_annotations(pred_coco)


for image in tqdm(images):
    if not "train" in image:
        continue
    matches = compare_groups(gt_annotations[image], pred_annotations[f"s3/CollembolAI/{image}"], plot=True, image_path=f"s3/CollembolAI/{image}", output_identifier=image, output_directory="dev/eval", threshold=0.1)


# groups = split_annotations(coco)
# image_names = [i["file_name"] for i in coco["images"]]

# matches = compare_groups(groups[image_names[0]], groups[image_names[1]], plot=True, image_path="s3/CollembolAI/ctrain03.jpg", output_identifier="test", output_directory="dev")