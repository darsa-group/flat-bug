import os
import time
from typing import Union, List, Tuple
import cv2
from IPython.display import display, clear_output
import ipywidgets as widgets
import numpy as np
from flat_bug.coco_utils import contour_bbox, contour_area, annotations_to_numpy


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
        b2s = b2s.copy().reshape(-1, 4)
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
        b2s = b2s.copy().reshape(-1, 4)
    ix_max = np.maximum(b1[:2], b2s[:, :2])
    ix_min = np.minimum(b1[2:], b2s[:, 2:])
    return np.prod((ix_min - ix_max).clip(0), axis=1)


def contour_intersection(contour1: np.array, contour2: np.array, box1: np.array, box2: np.array) -> np.array:
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


def pairwise_contour_intersection(contours1: List[np.array], contours2: Union[None, List[np.array]] = None,
                                  bboxes1: Union[None, np.array] = None, bboxes2: Union[None, np.array] = None,
                                  areas1: Union[None, np.array] = None,
                                  areas2: Union[None, np.array] = None) -> np.array:
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


def match_geoms(contours1: List[np.array], contours2: List[np.array], threshold: float = 1 / 4,
                iou_mat: Union[None, np.array] = None, areas1: Union[None, np.array] = None,
                areas2: Union[None, np.array] = None) -> Tuple[np.array, int]:
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
    matches[:, 1] = -1  # No match
    # Match the geometries in group 1 to the geometries in group 2
    if n > 0 and m > 0:
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


def plot_heatmap(mat: np.array, axis_labels: Union[List[str], None] = None, breaks: int = 25,
                 dimensions: Union[None, Tuple[int, int]] = None, output_path: str = None, scale: float = 1) -> None:
    """
    Plots a heatmap of a matrix using OpenCV.

    Args:
        mat (np.array): Matrix.
        axis_labels (Union[List[str], None], optional): Axis labels. Defaults to None.
        breaks (int, optional): Number of breaks on the colorbar. Defaults to 25.
        dimensions (Union[None, Tuple[int, int]], optional): Dimensions of the output image. Defaults to None.
        output_path (str, optional): Output path. Defaults to None.
        scale (float, optional): Scale of the output image. Defaults to 1.

    Returns:
        None
    """
    if dimensions is None:
        dimensions = tuple([m * 10 for m in mat.shape[::-1]])
        min_dim = max(min(dimensions), 1)
        if min_dim < 1000:
            scale_dims = 1000 / min_dim
            dimensions = tuple([int(d * scale_dims) for d in dimensions])
    if mat.shape[0] == 0 or mat.shape[1] == 0:
        print('WARNING: Empty matrix. Cannot plot heatmap.')
        return

    # Create a colormap for viridis
    colormap = cv2.applyColorMap((mat / (mat.max() or 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    # Expand colormap to 1000x1000
    colormap = cv2.resize(colormap, dimensions, interpolation=cv2.INTER_NEAREST_EXACT)
    # Get the height of the colormap
    cheight = colormap.shape[0]

    ## Prepare data for the colorbar
    # Add numbers to the colorbar
    cmin, cmax = mat.min(), mat.max()

    cmin, cmax = 10 ** np.floor(np.log10(cmin)) if cmin > 0 else cmin, 10 ** np.ceil(np.log10(cmax)) if cmax > 0 else cmax
    cmin, cmax = int(cmin), int(cmax)
    if cmin == cmax:
        nice_breaks = np.array([cmin])
    else:
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
    colorbar = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1).repeat(colorbar_width, 1),
                                 cv2.COLORMAP_VIRIDIS)
    # Stretch the colorbar to the height of the colormap
    colorbar = cv2.resize(colorbar, (colorbar.shape[1], cheight), interpolation=cv2.INTER_LINEAR_EXACT)
    cbheight = colorbar.shape[0]
    # Add the breaks to the colorbar
    for i, b in enumerate(nice_breaks):
        label = labels[i]
        text_width, font_height = label_sizes[i]
        cv2.putText(colorbar, label, (
            (colorbar_width - text_width) // 2, int((i + 0.5) * cbheight / len(nice_breaks) + font_height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), (3 * font_height_target) // 15, cv2.LINE_AA)
        cv2.putText(colorbar, label, (
            (colorbar_width - text_width) // 2, int((i + 0.5) * cbheight / len(nice_breaks) + font_height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_height_target // 15, cv2.LINE_AA)

    # Concatenate the colormap and the colorbar
    colormap = cv2.hconcat([colormap, colorbar])

    # If axis labels are provided
    if axis_labels is not None:
        x_label, y_label = axis_labels
        # Axis boxes have height/width of 10% of the minimum dimension of the heatmap
        axis_box_size = int(min(dimensions) * 0.05)
        # Calculate the font size
        axis_label_font_size = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_COMPLEX, axis_box_size // 2, 3)

        # First create the x-axis box
        x_axis_box = np.zeros((axis_box_size, dimensions[0] + colorbar_width, 3), dtype=np.uint8) + 255
        # Then create the y-axis box, remembering to take into account the extra vertical space taken up by the x-axis label. It is instantiated in the flipped orientation.
        y_axis_box = np.zeros((axis_box_size, dimensions[1] + axis_box_size, 3), dtype=np.uint8) + 255
        # Calculate the midpoint on each box with respect to the heatmap
        x_label_width = cv2.getTextSize(x_label, cv2.FONT_HERSHEY_COMPLEX, axis_label_font_size, 3)[0][0]
        y_label_width = cv2.getTextSize(y_label, cv2.FONT_HERSHEY_COMPLEX, axis_label_font_size, 3)[0][0]
        x_midpoint = dimensions[0] // 2 - x_label_width // 2
        y_midpoint = dimensions[1] // 2 + axis_box_size - y_label_width // 2
        center_offset = axis_box_size // 2 + axis_box_size // 4
        # Add the x-axis label
        cv2.putText(x_axis_box, x_label, (x_midpoint, center_offset), cv2.FONT_HERSHEY_COMPLEX, axis_label_font_size,
                    (0, 0, 0), axis_box_size // 15, cv2.LINE_AA)
        # Add the y-axis label
        cv2.putText(y_axis_box, y_label, (y_midpoint, center_offset), cv2.FONT_HERSHEY_COMPLEX, axis_label_font_size,
                    (0, 0, 0), axis_box_size // 15, cv2.LINE_AA)
        # Flip the y-axis box
        y_axis_box = cv2.rotate(y_axis_box, cv2.ROTATE_90_CLOCKWISE)
        # Concatenate the boxes
        colormap = cv2.vconcat([colormap, x_axis_box])
        colormap = cv2.hconcat([y_axis_box, colormap])

    if scale != 1:
        # Rescale the colormap to 2x lower resolution
        colormap = cv2.resize(colormap, (int(colormap.shape[1] * scale), int(colormap.shape[0] * scale)),
                              interpolation=cv2.INTER_LINEAR)

    if output_path is not None:
        # Save the image
        cv2.imwrite(output_path, colormap, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        compatible_display(colormap)

def equal_spaced_cuts(k, start, end):
    """
    Generate k equal spaced cuts between start and end. 
    
    The edges are not included, and the distance between the left-most and right-most cut to the edges is half the distance between the cuts. 

    Args:
        k (int): Number of cuts.
        start (float): Start value.
        end (float): End value.

    Returns:
        np.array: Cuts.
    """
    return np.linspace(start + (end - start) / (k * 2), end - (end - start) / (k * 2), k)


def plot_matches(matches: np.array, contours1: list[np.array], contours2: List[np.array],
                 group_labels: Union[List[str], None] = None, image_path: Union[str, None] = None,
                 output_path: Union[str, None] = None, scale: float = 1, boxes: bool = True) -> None:
    """
    Plots the matches between two groups of contours using OpenCV.

    Args:
        matches (np.array): Matches.
        contours1 (list[np.array]): Contours of group 1.
        contours2 (list[np.array]): Contours of group 2.
        image_path (str): Path to the image.
        output_path (str): Output path.
        scale (float): Scale of the output image.

    Returns:
        None
    """
    GROUP_COLORS = [(184, 126, 55), (28, 26, 228)]
    # Type check the input
    if not isinstance(matches, np.ndarray):
        raise ValueError(f'Expected matches to be a NumPy array, got {type(matches)}')
    for i, c1 in enumerate(contours1):
        if not isinstance(c1, np.ndarray):
            raise ValueError(f'Expected contours1[{i}] to be a NumPy array, got {type(c1)}')
    for i, c2 in enumerate(contours2):
        if not isinstance(c2, np.ndarray):
            raise ValueError(f'Expected contours2[{i}] to be a NumPy array, got {type(c2)}')
    if not isinstance(group_labels, list) and not group_labels is None:
        raise ValueError(f'Expected group_labels to be a list or None, got {type(group_labels)}')
    elif isinstance(group_labels, list):
        for i, l in enumerate(group_labels):
            if not isinstance(l, str):
                raise ValueError(f'Expected group_labels[{i}] to be a string, got {type(l)}')
    if not isinstance(image_path, str) and not image_path is None:
        raise ValueError(f'Expected image_path to be a string or None, got {type(image_path)}')
    elif isinstance(image_path, str) and not os.path.exists(image_path):
        raise ValueError(f'Expected image_path to be a valid file, got {image_path}')
    if not isinstance(output_path, str) and not output_path is None:
        raise ValueError(f'Expected output_path to be a string or None, got {type(output_path)}')
    elif isinstance(output_path, str) and not os.path.exists(os.path.dirname(output_path)):
        raise ValueError(f'Output directory does not exist: {os.path.dirname(output_path)}')

    # If a output path is provided the image is saved, otherwise it is displayed with IPython
    save_plot = isinstance(output_path, str)
    if save_plot:
        # Check the output path extension
        _, out_ext = os.path.splitext(output_path)
        if not out_ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
            raise ValueError(f'Expected output path to have a .JPG/.jpg/.JPEG/.jpeg extension, got {out_ext}')

    # If the is image path is provided
    if isinstance(image_path, str):
        # Load the image
        image = cv2.imread(filename = image_path)
    else:
        # Otherwise, create a blank image. The dimensions are dynamically calculated to fit the contours
        xmax, ymax = 0, 0
        for c in contours1 + contours2:
            xmax = max(xmax, c[:, 0].max())
            ymax = max(ymax, c[:, 1].max())
        # Padded by 10 pixels
        xmax += 10
        ymax += 10
        # And set to even dimension sizes
        xmax = (xmax // 2) * 2
        ymax = (ymax // 2) * 2
        image = np.zeros((ymax, xmax, 3), dtype=np.uint8) + 255  # White background

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
            cv2.fillPoly(
                img = cimg1, 
                pts = [contours1[i]], 
                color = GROUP_COLORS[0]
            )
            cv2.drawContours(
                image = image, 
                contours = [contours1[i]], 
                contourIdx = -1,
                color = GROUP_COLORS[0],
                thickness = 4,
                lineType=cv2.LINE_AA
            )
            if boxes:
                # Draw boxes around the contours
                cv2.rectangle(
                    img = image, 
                    pt1 = (bboxes1[i][0], bboxes1[i][1]), 
                    pt2 = (bboxes1[i][2], bboxes1[i][3]), 
                    color = GROUP_COLORS[0], 
                    thickness = 8
                )
        if j != -1:
            # Draw the second contour mask on the copy
            cv2.fillPoly(
                img = cimg2, 
                pts = [contours2[j]], 
                color = GROUP_COLORS[1]
            )
            cv2.drawContours(
                image = image, 
                contours = [contours2[j]],
                contourIdx=-1, 
                color = GROUP_COLORS[1],
                thickness=4,
                lineType=cv2.LINE_AA
            )
            if boxes:
                # Draw boxes around the contours
                cv2.rectangle(
                    img = image, 
                    pt1 = (bboxes2[j][0], bboxes2[j][1]), 
                    pt2 = (bboxes2[j][2], bboxes2[j][3]), 
                    color = GROUP_COLORS[1], 
                    thickness = 8
                )

    # Blend the image copies with the contour masks together (makes the contours semi-transparent - alpha=0.5)
    cv2.addWeighted(
        src1 = cimg1, 
        alpha = 0.5, 
        src2 = cimg2, 
        beta = 0.5, 
        gamma = 0, 
        dst = cimg1
    )
    # Blend with the blended copies with original image
    cv2.addWeighted(
        src1 = image, 
        alpha = 0.5, 
        src2 = cimg1, 
        beta = 0.5, 
        gamma = 0, 
        dst = image
    )

    # Downscale the image
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    # And the bounding boxes
    bboxes1 = (bboxes1 / 2).astype(np.int32)
    bboxes2 = (bboxes2 / 2).astype(np.int32)

    # Label the objects
    label_font_height = max(8, image.shape[1] // 200)
    label_font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, label_font_height, 3)
    label_font_thickness = max(1, label_font_height // 15)
    for idx, (i, j) in enumerate(matches):
        no_match = (i == -1) or (j == -1)
        label_coord = []
        label_font_color = []
        if no_match:
            match_color = (0, 0, 255)
        else:
            match_color = (0, 0, 0)
        if i != -1:
            # Draw a text label next to the box
            label_width = \
                cv2.getTextSize(f"{idx}", cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_font_thickness)[0][0]
            label_font_color.append(GROUP_COLORS[0])
            label_coord.append((bboxes1[i][0] - label_width, bboxes1[i][1] + label_font_height))
        if j != -1:
            # Draw a text label next to the box
            label_font_color.append(GROUP_COLORS[1])
            label_coord.append((bboxes2[j][0], bboxes2[j][1] - label_font_height // 4))
        for coord, color in zip(label_coord, label_font_color):
            cv2.putText(
                img = image, 
                text = str(idx), 
                org = coord, 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = label_font_scale, 
                color = match_color,
                thickness = label_font_thickness * 3, 
                lineType = cv2.LINE_8
            )
            cv2.putText(
                img = image, 
                text = str(idx), 
                org = coord, 
                fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale = label_font_scale, 
                color = color,
                thickness = label_font_thickness, 
                lineType = cv2.LINE_AA
            )
            

    # Create a legend
    LEGEND_TEXT_Y_JUST = 1/2
    legend_font_height = max(min(6, image.shape[0] // 40), 24)
    legend_margin = max(1, int(image.shape[0] * 0.01))
    legend_font_size = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_COMPLEX, legend_font_height, 3)
    legend_label_widths = \
        [cv2.getTextSize(glabel, cv2.FONT_HERSHEY_COMPLEX, legend_font_size, 3)[0][0] for glabel in group_labels]
    legend_font_width = max(legend_label_widths)
    legend_box_height = int((legend_font_height * 1.6) * len(group_labels))
    legend_box_width = int(legend_font_width * 1.25)
    # Extract the legend box
    legend_box = image[
        legend_margin:(legend_box_height + legend_margin),
        -(legend_box_width + legend_margin):-legend_margin, 
        :
    ]
    # Whiten legend box
    whiten_frac = 0.5
    whiten_amount = ((255 - legend_box) * whiten_frac).astype(np.uint8)
    legend_box += whiten_amount
    # Add a black border to the legend box
    cv2.rectangle(
        img = legend_box, 
        pt1 = (1, 1), 
        pt2 = (legend_box_width - legend_font_height // 15 - 1, legend_box_height - legend_font_height // 15 - 1), 
        color = (0, 0, 0), 
        thickness = legend_font_height // 15,
        lineType=cv2.LINE_AA
    )
    # Add the legend labels and items
    legend_attributes = []
    item_cut_ys = equal_spaced_cuts(len(group_labels), 0, legend_box_height)
    for i, (glabel, label_width, item_cut_y) in enumerate(zip(group_labels, legend_label_widths, item_cut_ys)):
        item_label_y = int(item_cut_y + legend_font_height * LEGEND_TEXT_Y_JUST)
        # Labels
        label_x = int(legend_box_width * 0.975) - label_width
        
        # Items - positioned to the left of the labels with a margin of 'legend_margin'
        item_x = label_x // 2
        item_color = GROUP_COLORS[i]
        
        # Add the attributes to the legend_attributes list
        legend_attributes.append((glabel, item_x, item_color, label_x, item_label_y))

    # Set all item_x to the minimum of all item_x
    min_item_x = min([item_x for _, item_x, _, _, _ in legend_attributes])

    for glabel, _, item_color, label_x, item_label_y in legend_attributes:
        # Draw the legend item label
        cv2.putText(
            img = legend_box, 
            text = glabel, 
            org = (label_x, item_label_y), 
            fontFace = cv2.FONT_HERSHEY_COMPLEX, 
            fontScale = legend_font_size, 
            color = (0, 0, 0),
            thickness = legend_font_height // 15, 
            lineType = cv2.LINE_AA
        )
        # Fill the item circle with the color of the group
        cv2.circle(
            img = legend_box, 
            center = (min_item_x, item_label_y - legend_font_height // 2), 
            radius = legend_font_height // 2, 
            color = item_color,
            thickness=cv2.FILLED
        )
        # Add a black border to the item circle
        cv2.circle(
            img = legend_box,
            center = (min_item_x, item_label_y - legend_font_height // 2), 
            radius = legend_font_height // 2, 
            color = (0, 0, 0),
            thickness = legend_font_height // 30,
            lineType=cv2.LINE_AA
        )

    # Add the legend to the image
    image[legend_margin:(legend_box_height + legend_margin), -(legend_box_width + legend_margin):-legend_margin, :] = legend_box

    if scale != 1:
        # Scale the image
        cv2.resize(
            src = image, 
            dsize = (int(image.shape[1] * scale), int(image.shape[0] * scale)), 
            dst = image
        )

    if save_plot:
        # Save the image
        cv2.imwrite(
            filename = output_path, 
            img = image, 
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
    else:
        compatible_display(image)


def compatible_display(image: np.array):
    TIMEOUT = 5  # seconds
    # Check if the image is displayed in a Jupyter notebook
    if 'get_ipython' in globals():
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Create a button widget
        button = widgets.Button(description="Close Image")
        button_clicked = False

        # Display callback function
        def on_button_clicked(b):
            nonlocal button_clicked
            # Clear the output after button click
            clear_output()
            button_clicked = True

        # Attach the callback function to the button
        button.on_click(on_button_clicked)

        # Display the image
        display(widgets.Image(value=cv2.imencode('.jpg', image_rgb)[1].tobytes(), format='jpg'))
        # Display the button
        display(button)

        # Wait for the button to be clicked
        start_time = time.time()
        while not button_clicked and (time.time() - start_time) < TIMEOUT:
            time.sleep(0.01)
        print('Image display closed')
    else:
        # Check if a display is available
        if os.environ.get('DISPLAY', '') == '':
            print('No display found, unable to display the image')
        else:
            # Display the image
            cv2.imshow('Matches', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def compare_groups(group1: list, group2: list, group_labels: Union[str, None] = None, threshold: float = 1 / 10,
                   plot: bool = True, plot_scale: float = 1, plot_boxes: bool = True,
                   image_path: Union[str, None] = None, output_identifier: str = None, output_directory: str = None) -> \
        Union[str, dict]:
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
        group_labels (Union[str, None], optional): Group labels. Defaults to None.
        threshold (float, optional): IoU threshold. Defaults to 1/10.
        plot (bool, optional): Whether to plot the matches and the IoU matrix. Defaults to True.
        plot_scale (float, optional): Scale of the plot. Defaults to 1. Lower values will make the plot smaller, but may be faster.
        plot_boxes (bool, optional): Whether to plot the bounding boxes. Defaults to True.
        image_path (Union[str, None], optional): Path to the image. Defaults to None.
        output_identifier (str, optional): Output identifier. Defaults to None.
        output_directory (str, optional): Output directory. Defaults to None.

    Returns:
        str: Path to the CSV file.\n
        or\n
        dict: The data that would have been saved to the CSV file as a dictionary, where the keys are the column names and the values are the column values.
    """
    # Type check the input
    if not isinstance(group1, list) or not isinstance(group2, list):
        raise ValueError(f'Expected group1 and group2 to be lists, got {type(group1)} and {type(group2)}')
    if not all([isinstance(i, dict) for i in group1]) or not all([isinstance(i, dict) for i in group2]):
        raise ValueError(
            f'Expected group1 and group2 to be lists of dictionaries, got {type(group1[0])} and {type(group2[0])}')
    if not isinstance(threshold, float):
        raise ValueError(f'Expected threshold to be a float, got {type(threshold)}')
    if not isinstance(plot, bool):
        raise ValueError(f'Expected plot to be a bool, got {type(plot)}')
    if not (isinstance(image_path, str) or image_path is None):
        raise ValueError(f'Expected image_path to be a string or None, got {type(image_path)}')
    if not isinstance(output_directory, str) and not output_directory is None:
        raise ValueError(f'Expected output_directory to be a string or None, got {type(output_directory)}')
    elif isinstance(output_directory, str) and not os.path.isdir(output_directory):
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
        plot_matches(
            matches = matches, 
            contours1 = c1, 
            contours2 = c2, 
            group_labels = group_labels, 
            image_path = image_path,
            output_path = os.path.join(output_directory, f'{output_identifier}_matches.jpg') if not output_directory is None else None,
            scale = plot_scale, 
            boxes = plot_boxes
        )
        if not any([l == 0 for l in iou.shape]):
            plot_heatmap(
                mat = iou, 
                axis_labels = group_labels,
                output_path = os.path.join(output_directory, f'{output_identifier}_heatmap.jpg') if not output_directory is None else None,
                scale = plot_scale
            )

    ## Gather the data for the output
    matched_1 = matches[:, 1] != -1
    matched_2 = matches[:, 0] != -1
    unmatched_1 = np.where(~matched_1)[0]
    unmatched_2 = np.where(~matched_2)[0]

    # Get the matched IoU
    if all(iou.shape):
        matched_iou = iou[*matches[:len_1].T]
        # Set the IoU of unmatched geometries to 0
        matched_iou[unmatched_1] = 0
    else:
        matched_iou = np.zeros(len(matches))
    matched_iou = np.concatenate([matched_iou, np.zeros(len(unmatched_2))])

    # Get the confidences
    conf1 = ["NA" for _ in range(len(matches))]
    conf2 = ["NA" for _ in range(len(matches))]
    for i, m in enumerate(matches):
        if m[0] != -1 and "conf" in group1[m[0]]:
            conf1[i] = group1[m[0]]["conf"]
        if m[1] != -1 and "conf" in group2[m[1]]:
            conf2[i] = group2[m[1]]["conf"]

    # Get the matched bounding boxes
    boxes1 = b1[matches[:, 0]] if all(b1.shape) else -np.ones((len(matches), 4), dtype=np.int32)
    boxes2 = b2[matches[:, 1]] if all(b2.shape) else -np.ones((len(matches), 4), dtype=np.int32)

    # Get the matched areas
    careas1 = a1[matches[:, 0]] if all(a1.shape) else -np.ones(len(matches), dtype=np.int32)
    careas2 = a2[matches[:, 1]] if all(a2.shape) else -np.ones(len(matches), dtype=np.int32)
    # Set the areas of unmatched geometries to 0
    if all(careas1.shape) and len(unmatched_2) > 0:
        careas1[unmatched_2] = 0
    if all(careas2.shape) and len(unmatched_1) > 0:
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
    data_length = set(
        [len_idx1, len_idx2, len_matched_iou, len_careas1, len_careas2, len_boxes1, len_boxes2, len_contours1,
         len_contours2])
    if len(data_length) != 1:
        raise ValueError(
            f"Lengths of the data are not all the same: {len_idx1, len_idx2, len_matched_iou, len_careas1, len_careas2, len_boxes1, len_boxes2, len_contours1, len_contours2}")

    # Construct the output by combining the data
    output = {
        "idx_1": idx1,
        "idx_2": idx2,
        "conf1": conf1,
        "conf2": conf2,
        "IoU": matched_iou,
        "contourArea_1": careas1,
        "contourArea_2": careas2,
        "bbox_1": boxes1,
        "bbox_2": boxes2,
        "contour_1": contours1,
        "contour_2": contours2
    }

    if not output_directory is None:
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
    else:
        return output
